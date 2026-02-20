/**
 * ONNX-based RL walking controller for Unitree H1-2 (v2).
 * Uses a unitree_rl_gym LSTM policy (47-dim obs → 12-dim action).
 *
 * Observation vector (47):
 *   [0:3]   angular velocity (body frame)
 *   [3:6]   projected gravity (body frame)
 *   [6:9]   velocity commands [vx, vy, wz]
 *   [9:21]  joint positions - defaults (12)
 *   [21:33] joint velocities (12)
 *   [33:45] previous actions (12)
 *   [45:47] gait phase [sin(2πφ), cos(2πφ)]
 *
 * Actions (12): position targets for legs only (6 per leg).
 *   target = action * 0.25 + defaultAngles
 *   torque = kp * (target - q) - kd * qdot
 *
 * Key differences from H1:
 *   - 12 DOF legs (2-DOF ankles: pitch + roll)
 *   - Hip chain: yaw → pitch → roll (H1 is yaw → roll → pitch)
 *   - All joint names have "_joint" suffix
 *
 * LSTM hidden state (64-dim) maintained across steps.
 *
 * PD gains (from unitree_rl_gym training config):
 *   hip_yaw/roll: kp=150, kd=5
 *   hip_pitch/knee: kp=200, kd=5
 *   ankle_pitch: kp=40, kd=2
 *   ankle_roll: kp=20, kd=2
 */

export class H1_2OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;
    this.session = null;

    this.simDt = model.opt.timestep || 0.002;
    this.numJoints = 12;

    // Decimation: 50Hz policy at 500Hz physics
    this.decimation = 10;
    this.stepCount = 0;

    // Action scale
    this.actionScale = 0.25;

    // Default joint angles (standing pose)
    // Order: left_hip_yaw, left_hip_pitch, left_hip_roll, left_knee,
    //        left_ankle_pitch, left_ankle_roll,
    //        right_hip_yaw, right_hip_pitch, right_hip_roll, right_knee,
    //        right_ankle_pitch, right_ankle_roll
    this.defaultAngles = new Float32Array([
      0, -0.4, 0, 0.8, -0.4, 0,   // left leg
      0, -0.4, 0, 0.8, -0.4, 0,   // right leg
    ]);

    // PD gains per joint (matching unitree_rl_gym training config)
    this.kp = new Float32Array([
      150, 200, 150, 200, 40, 20,   // left leg
      150, 200, 150, 200, 40, 20,   // right leg
    ]);
    this.kd = new Float32Array([
      5, 5, 5, 5, 2, 2,   // left leg
      5, 5, 5, 5, 2, 2,   // right leg
    ]);

    // Gait phase
    this.phase = 0.0;
    this.gaitFreq = 1.25; // Hz
    this.ctrlDt = this.decimation * this.simDt; // 0.02s

    // LSTM hidden state (64-dim)
    this.hiddenSize = 64;
    this.h0 = new Float32Array(this.hiddenSize);
    this.c0 = new Float32Array(this.hiddenSize);

    // State
    this.lastAction = new Float32Array(12);
    this.currentTargets = new Float32Array(12);
    for (let i = 0; i < 12; i++) this.currentTargets[i] = this.defaultAngles[i];

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Joint indices
    this.jntQpos = new Int32Array(12);
    this.jntDof = new Int32Array(12);
    this.findJointIndices();

    this._inferring = false;
  }

  findJointIndices() {
    const names = [
      'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint',
      'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
      'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
      'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    ];
    for (let i = 0; i < 12; i++) {
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
      console.log('H1-2 ONNX policy loaded:', modelPath);
      return true;
    } catch (e) {
      console.error('Failed to load H1-2 ONNX model:', e);
      return false;
    }
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  setInitialPose() {
    for (let i = 0; i < 12; i++) {
      this.data.qpos[this.jntQpos[i]] = this.defaultAngles[i];
    }
    // Standing height ~1.03m
    this.data.qpos[2] = 1.03;
    for (let i = 0; i < this.model.nv; i++) this.data.qvel[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
  }

  // Rotate vector by inverse of body quaternion (world→body frame)
  rotateByInvQuat(vx, vy, vz) {
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];
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
    const obs = new Float32Array(47);

    // [0:3] Angular velocity (body frame)
    const gyro = this.rotateByInvQuat(this.data.qvel[3], this.data.qvel[4], this.data.qvel[5]);
    obs[0] = gyro[0]; obs[1] = gyro[1]; obs[2] = gyro[2];

    // [3:6] Projected gravity (body frame)
    const grav = this.rotateByInvQuat(0, 0, -1);
    obs[3] = grav[0]; obs[4] = grav[1]; obs[5] = grav[2];

    // [6:9] Velocity commands
    obs[6] = this.forwardSpeed;
    obs[7] = this.lateralSpeed;
    obs[8] = this.turnRate;

    // [9:21] Joint positions - defaults
    for (let i = 0; i < 12; i++) {
      obs[9 + i] = this.data.qpos[this.jntQpos[i]] - this.defaultAngles[i];
    }

    // [21:33] Joint velocities
    for (let i = 0; i < 12; i++) {
      obs[21 + i] = this.data.qvel[this.jntDof[i]];
    }

    // [33:45] Previous actions
    for (let i = 0; i < 12; i++) {
      obs[33 + i] = this.lastAction[i];
    }

    // [45:47] Gait phase [sin(2πφ), cos(2πφ)]
    obs[45] = Math.sin(2 * Math.PI * this.phase);
    obs[46] = Math.cos(2 * Math.PI * this.phase);

    // Clip to prevent extreme values
    for (let i = 0; i < 47; i++) {
      obs[i] = Math.max(-100, Math.min(100, obs[i]));
    }

    return obs;
  }

  applyPD() {
    const ctrl = this.data.ctrl;
    for (let i = 0; i < 12; i++) {
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
      const input = {
        obs: new ort.Tensor('float32', obs, [1, 47]),
        h0: new ort.Tensor('float32', this.h0, [1, 1, this.hiddenSize]),
        c0: new ort.Tensor('float32', this.c0, [1, 1, this.hiddenSize]),
      };
      const results = await this.session.run(input);
      const actions = results.actions.data;

      // Update LSTM hidden state
      this.h0 = new Float32Array(results.h_out.data);
      this.c0 = new Float32Array(results.c_out.data);

      for (let i = 0; i < 12; i++) {
        const a = Math.max(-5, Math.min(5, actions[i]));
        this.lastAction[i] = a;
        this.currentTargets[i] = a * this.actionScale + this.defaultAngles[i];
      }

      // Advance gait phase
      this.phase = (this.phase + this.gaitFreq * this.ctrlDt) % 1.0;
    } catch (e) {
      console.error('H1-2 ONNX inference error:', e);
    }
    this._inferring = false;
  }

  reset() {
    this.stepCount = 0;
    this.lastAction.fill(0);
    for (let i = 0; i < 12; i++) this.currentTargets[i] = this.defaultAngles[i];
    this.phase = 0.0;
    this.h0 = new Float32Array(this.hiddenSize);
    this.c0 = new Float32Array(this.hiddenSize);
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this._inferring = false;
  }
}
