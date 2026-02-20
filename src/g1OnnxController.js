/**
 * ONNX-based RL walking controller for Unitree G1.
 * Uses MuJoCo Playground policy (103-dim obs → 29-dim action).
 *
 * Observation vector (103):
 *   [0:3]   local linear velocity (body frame)
 *   [3:6]   angular velocity / gyro (body frame)
 *   [6:9]   projected gravity (body frame)
 *   [9:12]  velocity commands [vx, vy, wz]
 *   [12:41] joint positions - defaults (29)
 *   [41:70] joint velocities (29)
 *   [70:99] previous actions (29)
 *   [99:103] gait phase [cos(L), cos(R), sin(L), sin(R)]
 *
 * Actions (29): position targets.
 *   target = action * 0.5 + defaultAngles
 *   torque = kp * (target - q) - kd * qdot
 *
 * PD gains (from Playground position actuators):
 *   default: kp=75, kd=2
 *   ankle_pitch: kp=20, kd=1
 *   ankle_roll, wrists: kp=2, kd=0.2
 */

export class G1OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;
    this.session = null;

    this.simDt = model.opt.timestep || 0.002;
    this.numJoints = 29;

    // Decimation: 50Hz policy at 500Hz physics
    this.decimation = 10;
    this.stepCount = 0;

    // Action scale
    this.actionScale = 0.5;

    // Default joint angles (knees_bent keyframe from Playground)
    this.defaultAngles = new Float32Array([
      -0.312, 0, 0, 0.669, -0.363, 0,     // left leg
      -0.312, 0, 0, 0.669, -0.363, 0,     // right leg
      0, 0, 0.073,                          // waist
      0.2, 0.2, 0, 0.6, 0, 0, 0,          // left arm
      0.2, -0.2, 0, 0.6, 0, 0, 0,         // right arm
    ]);

    // PD gains per joint (matching Playground position actuators)
    this.kp = new Float32Array(29);
    this.kd = new Float32Array(29);
    for (let i = 0; i < 29; i++) {
      this.kp[i] = 75;
      this.kd[i] = 2;
    }
    // ankle_pitch (indices 4, 10)
    this.kp[4] = 20; this.kd[4] = 1;
    this.kp[10] = 20; this.kd[10] = 1;
    // ankle_roll (indices 5, 11)
    this.kp[5] = 2; this.kd[5] = 0.2;
    this.kp[11] = 2; this.kd[11] = 0.2;
    // wrists (indices 19-21, 26-28)
    for (const i of [19, 20, 21, 26, 27, 28]) {
      this.kp[i] = 2; this.kd[i] = 0.2;
    }

    // Gait phase (two-leg anti-phase)
    this.phase = [0.0, Math.PI];
    this.gaitFreq = 1.5;
    this.phaseDt = 2 * Math.PI * this.gaitFreq * (this.decimation * this.simDt);

    // State
    this.lastAction = new Float32Array(29);
    this.currentTargets = new Float32Array(29);
    for (let i = 0; i < 29; i++) this.currentTargets[i] = this.defaultAngles[i];

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Joint indices
    this.jntQpos = new Int32Array(29);
    this.jntDof = new Int32Array(29);
    this.findJointIndices();

    this._inferring = false;
  }

  findJointIndices() {
    const names = [
      'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
      'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
      'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
      'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
      'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
      'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
      'left_shoulder_yaw_joint', 'left_elbow_joint',
      'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
      'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
      'right_shoulder_yaw_joint', 'right_elbow_joint',
      'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
    ];
    for (let i = 0; i < 29; i++) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, names[i]);
        if (jid >= 0) {
          this.jntQpos[i] = this.model.jnt_qposadr[jid];
          this.jntDof[i] = this.model.jnt_dofadr[jid];
        }
      } catch (e) { /* ignore */ }
    }
  }

  async loadModel(modelPath) {
    if (typeof ort === 'undefined') {
      console.warn('ONNX Runtime Web not loaded');
      return false;
    }
    try {
      this.session = await ort.InferenceSession.create(modelPath);
      console.log('G1 ONNX policy loaded:', modelPath);
      return true;
    } catch (e) {
      console.error('Failed to load G1 ONNX model:', e);
      return false;
    }
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  setInitialPose() {
    // Set joints to knees_bent default
    for (let i = 0; i < 29; i++) {
      this.data.qpos[this.jntQpos[i]] = this.defaultAngles[i];
    }
    // Set body height (standing ~0.755m)
    this.data.qpos[2] = 0.755;
    for (let i = 0; i < this.model.nv; i++) this.data.qvel[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
  }

  // Rotate vector by inverse of body quaternion (world→body frame)
  rotateByInvQuat(vx, vy, vz) {
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];
    // Inverse quat = conjugate for unit quat
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

  buildObs() {
    const obs = new Float32Array(103);

    // [0:3] Local linear velocity (body frame)
    const linvel = this.rotateByInvQuat(this.data.qvel[0], this.data.qvel[1], this.data.qvel[2]);
    obs[0] = linvel[0]; obs[1] = linvel[1]; obs[2] = linvel[2];

    // [3:6] Angular velocity / gyro (body frame)
    const gyro = this.rotateByInvQuat(this.data.qvel[3], this.data.qvel[4], this.data.qvel[5]);
    obs[3] = gyro[0]; obs[4] = gyro[1]; obs[5] = gyro[2];

    // [6:9] Projected gravity (body frame)
    const grav = this.rotateByInvQuat(0, 0, -1);
    obs[6] = grav[0]; obs[7] = grav[1]; obs[8] = grav[2];

    // [9:12] Velocity commands
    obs[9] = this.forwardSpeed;
    obs[10] = this.lateralSpeed;
    obs[11] = this.turnRate;

    // [12:41] Joint positions - defaults
    for (let i = 0; i < 29; i++) {
      obs[12 + i] = this.data.qpos[this.jntQpos[i]] - this.defaultAngles[i];
    }

    // [41:70] Joint velocities
    for (let i = 0; i < 29; i++) {
      obs[41 + i] = this.data.qvel[this.jntDof[i]];
    }

    // [70:99] Previous actions
    for (let i = 0; i < 29; i++) {
      obs[70 + i] = this.lastAction[i];
    }

    // [99:103] Gait phase [cos(L), cos(R), sin(L), sin(R)]
    obs[99] = Math.cos(this.phase[0]);
    obs[100] = Math.cos(this.phase[1]);
    obs[101] = Math.sin(this.phase[0]);
    obs[102] = Math.sin(this.phase[1]);

    // Clip to prevent extreme values
    for (let i = 0; i < 103; i++) {
      obs[i] = Math.max(-100, Math.min(100, obs[i]));
    }

    return obs;
  }

  applyPD() {
    const ctrl = this.data.ctrl;
    for (let i = 0; i < 29; i++) {
      const q = this.data.qpos[this.jntQpos[i]];
      const qdot = this.data.qvel[this.jntDof[i]];
      const torque = this.kp[i] * (this.currentTargets[i] - q) - this.kd[i] * qdot;
      ctrl[i] = torque;
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

    this.applyPD();

    this.stepCount++;
    if (this.stepCount % this.decimation !== 0) return;
    if (!this.session || this._inferring) return;

    this._runInference();
  }

  async _runInference() {
    this._inferring = true;
    try {
      const obs = this.buildObs();
      const input = new ort.Tensor('float32', obs, [1, 103]);
      const results = await this.session.run({ obs: input });
      const actions = results.continuous_actions.data;

      for (let i = 0; i < 29; i++) {
        const a = Math.max(-5, Math.min(5, actions[i]));
        this.lastAction[i] = a;
        this.currentTargets[i] = a * this.actionScale + this.defaultAngles[i];
      }

      // Advance gait phase
      this.phase[0] = ((this.phase[0] + this.phaseDt + Math.PI) % (2 * Math.PI)) - Math.PI;
      this.phase[1] = ((this.phase[1] + this.phaseDt + Math.PI) % (2 * Math.PI)) - Math.PI;
    } catch (e) {
      console.error('G1 ONNX inference error:', e);
    }
    this._inferring = false;
  }

  reset() {
    this.stepCount = 0;
    this.lastAction.fill(0);
    for (let i = 0; i < 29; i++) this.currentTargets[i] = this.defaultAngles[i];
    this.phase = [0.0, Math.PI];
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this._inferring = false;
  }
}
