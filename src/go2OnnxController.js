/**
 * ONNX-based RL walking controller for Unitree Go2.
 * Uses a trained IsaacLab policy (49-dim obs → 12-dim action).
 *
 * Observation vector (49):
 *   [0:3]   base angular velocity (body frame)
 *   [3:6]   projected gravity (body frame)
 *   [6:9]   velocity commands [vx, vy, wz]
 *   [9:21]  joint positions - defaults (IsaacLab order)
 *   [21:33] joint velocities (IsaacLab order)
 *   [33:37] binary foot contacts [FL, FR, RL, RR]
 *   [37:49] previous actions (12)
 *
 * Actions (12): position targets in IsaacLab order.
 *   target = action * scale(0.25) + default
 *   torque = Kp * (target - q) - Kd * qdot
 *
 * IsaacLab joint order (grouped by type):
 *   FL_hip, FR_hip, RL_hip, RR_hip,
 *   FL_thigh, FR_thigh, RL_thigh, RR_thigh,
 *   FL_calf, FR_calf, RL_calf, RR_calf
 *
 * MuJoCo actuator order (grouped by leg):
 *   FL_hip(0), FL_thigh(1), FL_calf(2),
 *   FR_hip(3), FR_thigh(4), FR_calf(5),
 *   RL_hip(6), RL_thigh(7), RL_calf(8),
 *   RR_hip(9), RR_thigh(10), RR_calf(11)
 */

export class Go2OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;
    this.session = null; // ONNX session

    this.simDt = model.opt.timestep || 0.002;

    // Decimation: run policy at 50Hz (every 10 steps at 0.002s dt)
    this.decimation = 10;
    this.stepCount = 0;

    // PD gains (IsaacLab standard for Go2)
    this.Kp = 25.0;
    this.Kd = 0.5;

    // Action scale
    this.actionScale = 0.25;

    // IsaacLab default joint positions (IL order)
    // FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf
    this.defaultPos = new Float32Array([
      0, 0, 0, 0,         // hips
      1.1, 1.1, 1.1, 1.1, // thighs
      -1.8, -1.8, -1.8, -1.8, // calfs
    ]);

    // MuJoCo→IsaacLab index mapping
    // IL[i] gets its value from MuJoCo joint at MJ_TO_IL[i]
    this.MJ_TO_IL = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];

    // IsaacLab→MuJoCo ctrl mapping (same structure)
    // IL action[i] maps to MuJoCo ctrl[IL_TO_MJ[i]]
    this.IL_TO_MJ = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];

    // State
    this.lastAction = new Float32Array(12); // Previous raw actions
    this.currentTargets = new Float32Array(12); // Current position targets (IL order)
    for (let i = 0; i < 12; i++) this.currentTargets[i] = this.defaultPos[i];

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Joint indices (qpos/qvel)
    this.jntQpos = new Int32Array(12); // MuJoCo qpos indices for each MJ joint
    this.jntDof = new Int32Array(12);  // MuJoCo qvel indices for each MJ joint
    this.findJointIndices();

    // Foot geom IDs for contact detection
    this.footGeomIds = [-1, -1, -1, -1]; // FL, FR, RL, RR
    this.floorGeomId = -1;
    this.findFootGeoms();
  }

  findJointIndices() {
    const jointNames = [
      'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
      'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
      'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
      'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    ];
    for (let i = 0; i < 12; i++) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, jointNames[i]);
        if (jid >= 0) {
          this.jntQpos[i] = this.model.jnt_qposadr[jid];
          this.jntDof[i] = this.model.jnt_dofadr[jid];
        }
      } catch (e) { /* ignore */ }
    }
  }

  findFootGeoms() {
    // Go2 foot geoms are named "FL", "FR", "RL", "RR" (class="foot")
    const footGeomNames = ['FL', 'FR', 'RL', 'RR'];
    for (let i = 0; i < 4; i++) {
      try {
        const gid = this.mujoco.mj_name2id(this.model, 5, footGeomNames[i]); // mjOBJ_GEOM=5
        if (gid >= 0) this.footGeomIds[i] = gid;
      } catch (e) { /* ignore */ }
    }
    try {
      this.floorGeomId = this.mujoco.mj_name2id(this.model, 5, 'floor');
    } catch (e) { /* ignore */ }
  }

  async loadModel(modelPath) {
    if (typeof ort === 'undefined') {
      console.warn('ONNX Runtime Web not loaded');
      return false;
    }
    try {
      this.session = await ort.InferenceSession.create(modelPath);
      console.log('Go2 ONNX policy loaded:', modelPath);
      return true;
    } catch (e) {
      console.error('Failed to load ONNX model:', e);
      return false;
    }
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  /**
   * Get body angular velocity in body frame.
   */
  getBodyAngVel() {
    // Free joint: qvel[0:3]=linear vel (world), qvel[3:6]=angular vel (world)
    const wx = this.data.qvel[3];
    const wy = this.data.qvel[4];
    const wz = this.data.qvel[5];

    // Rotate world angular velocity to body frame using inverse quaternion
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];

    // Inverse quaternion rotation: q* v q
    // For inv(q) = [qw, -qx, -qy, -qz]
    return this.rotateByInvQuat(wx, wy, wz, qw, qx, qy, qz);
  }

  /**
   * Project gravity into body frame.
   */
  getProjectedGravity() {
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];

    // Gravity in world frame: [0, 0, -1] (normalized)
    return this.rotateByInvQuat(0, 0, -1, qw, qx, qy, qz);
  }

  /**
   * Rotate vector (vx,vy,vz) by inverse of quaternion (qw,qx,qy,qz).
   */
  rotateByInvQuat(vx, vy, vz, qw, qx, qy, qz) {
    // Inverse quaternion: conjugate for unit quaternion
    const iqx = -qx, iqy = -qy, iqz = -qz;

    // t = 2 * cross(q_xyz, v)
    const tx = 2 * (iqy * vz - iqz * vy);
    const ty = 2 * (iqz * vx - iqx * vz);
    const tz = 2 * (iqx * vy - iqy * vx);

    // result = v + qw * t + cross(q_xyz, t)
    return [
      vx + qw * tx + (iqy * tz - iqz * ty),
      vy + qw * ty + (iqz * tx - iqx * tz),
      vz + qw * tz + (iqx * ty - iqy * tx),
    ];
  }

  /**
   * Detect binary foot contacts (1 if foot touches floor, 0 otherwise).
   * Returns [FL, FR, RL, RR].
   */
  getFootContacts() {
    const contacts = [0, 0, 0, 0]; // FL, FR, RL, RR

    const ncon = this.data.ncon || 0;
    for (let c = 0; c < ncon; c++) {
      try {
        const contact = this.data.contact.get(c);
        const g1 = contact.geom1;
        const g2 = contact.geom2;

        // Check if one geom is floor
        const isFloor1 = g1 === this.floorGeomId;
        const isFloor2 = g2 === this.floorGeomId;
        if (!isFloor1 && !isFloor2) continue;

        const otherGeom = isFloor1 ? g2 : g1;

        // Check if the other geom is a foot
        for (let f = 0; f < 4; f++) {
          if (otherGeom === this.footGeomIds[f]) {
            contacts[f] = 1;
          }
        }
      } catch (e) { break; }
    }
    return contacts;
  }

  /**
   * Build the 49-dim observation vector.
   */
  buildObs() {
    const obs = new Float32Array(49);

    // [0:3] Body angular velocity (body frame)
    const angVel = this.getBodyAngVel();
    obs[0] = angVel[0];
    obs[1] = angVel[1];
    obs[2] = angVel[2];

    // [3:6] Projected gravity
    const grav = this.getProjectedGravity();
    obs[3] = grav[0];
    obs[4] = grav[1];
    obs[5] = grav[2];

    // [6:9] Velocity commands
    obs[6] = this.forwardSpeed;
    obs[7] = this.lateralSpeed;
    obs[8] = this.turnRate;

    // [9:21] Joint positions - defaults (IL order)
    for (let i = 0; i < 12; i++) {
      const mjIdx = this.MJ_TO_IL[i]; // MuJoCo joint index for IL position i
      const q = this.data.qpos[this.jntQpos[mjIdx]];
      obs[9 + i] = q - this.defaultPos[i];
    }

    // [21:33] Joint velocities (IL order)
    for (let i = 0; i < 12; i++) {
      const mjIdx = this.MJ_TO_IL[i];
      obs[21 + i] = this.data.qvel[this.jntDof[mjIdx]];
    }

    // [33:37] Binary foot contacts [FL, FR, RL, RR]
    const contacts = this.getFootContacts();
    obs[33] = contacts[0];
    obs[34] = contacts[1];
    obs[35] = contacts[2];
    obs[36] = contacts[3];

    // [37:49] Previous actions
    for (let i = 0; i < 12; i++) {
      obs[37 + i] = this.lastAction[i];
    }

    return obs;
  }

  /**
   * Apply PD control to track current position targets.
   * Called every physics step.
   */
  applyPD() {
    const ctrl = this.data.ctrl;
    for (let i = 0; i < 12; i++) {
      const mjIdx = this.IL_TO_MJ[i]; // MuJoCo actuator for IL joint i
      const q = this.data.qpos[this.jntQpos[mjIdx]];
      const qdot = this.data.qvel[this.jntDof[mjIdx]];
      const target = this.currentTargets[i];
      const torque = this.Kp * (target - q) - this.Kd * qdot;
      ctrl[mjIdx] = torque;
    }

    // Clamp to actuator ranges
    if (this.model.actuator_ctrlrange) {
      for (let i = 0; i < this.model.nu; i++) {
        const lo = this.model.actuator_ctrlrange[i * 2];
        const hi = this.model.actuator_ctrlrange[i * 2 + 1];
        ctrl[i] = Math.max(lo, Math.min(hi, ctrl[i]));
      }
    }
  }

  /**
   * Main step function. Called every physics step (synchronous).
   * Runs policy at decimated rate (async, fire-and-forget).
   * Applies PD every step using latest targets.
   */
  step() {
    if (!this.enabled) return;

    // Always apply PD control with current targets
    this.applyPD();

    // Run policy at decimated rate (async)
    this.stepCount++;
    if (this.stepCount % this.decimation !== 0) return;
    if (!this.session || this._inferring) return;

    this._runInference();
  }

  async _runInference() {
    this._inferring = true;
    try {
      const obs = this.buildObs();
      const input = new ort.Tensor('float32', obs, [1, 49]);
      const results = await this.session.run({ obs: input });
      const actions = results.actions.data;

      // Store raw actions and compute targets
      for (let i = 0; i < 12; i++) {
        this.lastAction[i] = actions[i];
        this.currentTargets[i] = actions[i] * this.actionScale + this.defaultPos[i];
      }
    } catch (e) {
      console.error('ONNX inference error:', e);
    }
    this._inferring = false;
  }

  reset() {
    this.stepCount = 0;
    this.lastAction.fill(0);
    for (let i = 0; i < 12; i++) {
      this.currentTargets[i] = this.defaultPos[i];
    }
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
  }
}
