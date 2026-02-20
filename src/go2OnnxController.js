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
 *
 * PD gains: Kp=50, Kd=1.5 — matches unitree_mujoco real-robot deployment
 * (Kp=50, Kd=3.5 in SDK, minus MuJoCo model's built-in joint damping=2).
 */

export class Go2OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;
    this.session = null;

    this.simDt = model.opt.timestep || 0.002;

    // Decimation: run policy at 50Hz (every 10 steps at 0.002s dt)
    this.decimation = 10;
    this.stepCount = 0;

    // PD gains — matched to unitree_mujoco deployment (Kp=50, Kd=3.5)
    // MuJoCo model has built-in joint damping=2, so we use Kd=1.5 → effective Kd≈3.5
    this.Kp = 50.0;
    this.Kd = 1.5;

    // Action scale (from IsaacLab training config)
    this.actionScale = 0.25;

    // IsaacLab default joint positions (IL order)
    this.defaultPos = new Float32Array([
      0, 0, 0, 0,                     // hips
      1.1, 1.1, 1.1, 1.1,             // thighs
      -1.8, -1.8, -1.8, -1.8,         // calfs
    ]);

    // MuJoCo-ordered defaults (for setting initial pose)
    // MJ order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, ...
    this.defaultPosMJ = new Float32Array([
      0, 1.1, -1.8,    // FL
      0, 1.1, -1.8,    // FR
      0, 1.1, -1.8,    // RL
      0, 1.1, -1.8,    // RR
    ]);

    // MuJoCo→IsaacLab index mapping
    this.MJ_TO_IL = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];
    this.IL_TO_MJ = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];

    // State
    this.lastAction = new Float32Array(12);
    this.currentTargets = new Float32Array(12);
    for (let i = 0; i < 12; i++) this.currentTargets[i] = this.defaultPos[i];

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Joint indices (qpos/qvel)
    this.jntQpos = new Int32Array(12);
    this.jntDof = new Int32Array(12);
    this.findJointIndices();

    // Foot geom IDs for contact detection
    this.footGeomIds = [-1, -1, -1, -1];
    this.floorGeomId = -1;
    this.findFootGeoms();

    this._inferring = false;
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
    const footGeomNames = ['FL', 'FR', 'RL', 'RR'];
    for (let i = 0; i < 4; i++) {
      try {
        const gid = this.mujoco.mj_name2id(this.model, 5, footGeomNames[i]);
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
   * Set initial joint positions to IsaacLab defaults.
   * Call before warm-up so the robot starts in the pose the policy expects.
   */
  setInitialPose() {
    for (let i = 0; i < 12; i++) {
      this.data.qpos[this.jntQpos[i]] = this.defaultPosMJ[i];
    }
    // Set body height slightly higher to let it settle
    this.data.qpos[2] = 0.35;
    // Reset velocities
    for (let i = 0; i < this.model.nv; i++) this.data.qvel[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
  }

  getBodyAngVel() {
    const wx = this.data.qvel[3];
    const wy = this.data.qvel[4];
    const wz = this.data.qvel[5];
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];
    return this.rotateByInvQuat(wx, wy, wz, qw, qx, qy, qz);
  }

  getProjectedGravity() {
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];
    return this.rotateByInvQuat(0, 0, -1, qw, qx, qy, qz);
  }

  rotateByInvQuat(vx, vy, vz, qw, qx, qy, qz) {
    const iqx = -qx, iqy = -qy, iqz = -qz;
    const tx = 2 * (iqy * vz - iqz * vy);
    const ty = 2 * (iqz * vx - iqx * vz);
    const tz = 2 * (iqx * vy - iqy * vx);
    return [
      vx + qw * tx + (iqy * tz - iqz * ty),
      vy + qw * ty + (iqz * tx - iqx * tz),
      vz + qw * tz + (iqx * ty - iqy * tx),
    ];
  }

  getFootContacts() {
    const contacts = [0, 0, 0, 0];
    // Use try-catch loop since data.ncon might not be available in WASM
    for (let c = 0; c < 100; c++) {
      try {
        const contact = this.data.contact.get(c);
        if (!contact) break;
        const g1 = contact.geom1;
        const g2 = contact.geom2;

        const isFloor1 = g1 === this.floorGeomId;
        const isFloor2 = g2 === this.floorGeomId;
        if (!isFloor1 && !isFloor2) continue;

        const otherGeom = isFloor1 ? g2 : g1;
        for (let f = 0; f < 4; f++) {
          if (otherGeom === this.footGeomIds[f]) contacts[f] = 1;
        }
      } catch (e) { break; }
    }
    return contacts;
  }

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
      const mjIdx = this.MJ_TO_IL[i];
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

    // Clip observations to prevent extreme values
    for (let i = 0; i < 49; i++) {
      obs[i] = Math.max(-100, Math.min(100, obs[i]));
    }

    return obs;
  }

  applyPD() {
    const ctrl = this.data.ctrl;
    for (let i = 0; i < 12; i++) {
      const mjIdx = this.IL_TO_MJ[i];
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

      // Clip actions to [-5, 5] (safety bound)
      for (let i = 0; i < 12; i++) {
        const a = Math.max(-5, Math.min(5, actions[i]));
        this.lastAction[i] = a;
        this.currentTargets[i] = a * this.actionScale + this.defaultPos[i];
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
    this._inferring = false;
  }
}
