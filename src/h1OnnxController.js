/**
 * ONNX-based RL walking controller for Unitree H1.
 * Uses a unitree_rl_gym LSTM policy (41-dim obs → 10-dim action).
 *
 * Observation vector (41):
 *   [0:3]   angular velocity (body frame) * 0.25
 *   [3:6]   projected gravity (body frame)
 *   [6:9]   velocity commands * [2.0, 2.0, 0.25]
 *   [9:19]  joint positions - defaults (10)
 *   [19:29] joint velocities (10) * 0.05
 *   [29:39] previous actions (10)
 *   [39:41] gait phase [sin(2πφ), cos(2πφ)]
 *
 * Actions (10): position targets for legs only.
 *   target = clip(action, -1, 1) * 0.25 + defaultAngles
 *   torque = kp * (target - q) - kd * qdot
 *
 * Upper body (torso + 8 arm joints) held at home pose via PD.
 * LSTM hidden state (64-dim) maintained across steps.
 *
 * H1 actuator layout (19):
 *   0-4: left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
 *   5-9: right leg
 *   10: torso
 *   11-14: left arm (shoulder_pitch/roll/yaw, elbow)
 *   15-18: right arm
 */

export class H1OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;
    this.session = null;

    this.simDt = model.opt.timestep || 0.002;
    this.numJoints = 10;

    // Decimation: 50Hz policy at 500Hz physics
    this.decimation = 10;
    this.stepCount = 0;

    // Action scale and clip
    this.actionScale = 0.25;
    this.actionClip = 1.0; // LSTM outputs small values; clip=5 causes instability

    // Default joint angles (from unitree_rl_gym h1_config.py)
    this.defaultAngles = new Float32Array([
      0, 0, -0.1, 0.3, -0.2,     // left: yaw, roll, pitch, knee, ankle
      0, 0, -0.1, 0.3, -0.2,     // right
    ]);

    // PD gains for legs
    this.kp = new Float32Array([150, 150, 150, 200, 40, 150, 150, 150, 200, 40]);
    this.kd = new Float32Array([2, 2, 2, 4, 2, 2, 2, 2, 4, 2]);

    // Observation scaling
    this.obsScales = {
      angVel: 0.25,
      dofPos: 1.0,
      dofVel: 0.05,
      cmdScale: [2.0, 2.0, 0.25],
    };

    // Gait phase
    this.phase = 0.0;
    this.gaitFreq = 1.25;
    this.ctrlDt = this.decimation * this.simDt;

    // LSTM hidden state
    this.hiddenSize = 64;
    this.h0 = new Float32Array(this.hiddenSize);
    this.c0 = new Float32Array(this.hiddenSize);

    // State
    this.lastAction = new Float32Array(10);
    this.currentTargets = new Float32Array(10);
    for (let i = 0; i < 10; i++) this.currentTargets[i] = this.defaultAngles[i];

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Joint indices for policy-controlled legs
    this.jntQpos = new Int32Array(10);
    this.jntDof = new Int32Array(10);
    this.findJointIndices();

    // Upper body PD hold (torso + arms: actuators 10-18)
    this.upperBody = [];
    this.findUpperBodyJoints();

    this._inferring = false;
  }

  findJointIndices() {
    const names = [
      'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
      'left_knee', 'left_ankle',
      'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
      'right_knee', 'right_ankle',
    ];
    for (let i = 0; i < 10; i++) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, names[i]);
        if (jid >= 0) {
          this.jntQpos[i] = this.model.jnt_qposadr[jid];
          this.jntDof[i] = this.model.jnt_dofadr[jid];
        }
      } catch (e) { /* ignore */ }
    }
  }

  findUpperBodyJoints() {
    // H1 upper body: torso, 4 left arm, 4 right arm (actuators 10-18)
    const upperNames = [
      { name: 'torso', target: 0, kp: 100, kd: 5 },
      { name: 'left_shoulder_pitch', target: 0, kp: 40, kd: 2 },
      { name: 'left_shoulder_roll', target: 0.2, kp: 40, kd: 2 },
      { name: 'left_shoulder_yaw', target: 0, kp: 40, kd: 2 },
      { name: 'left_elbow', target: -0.3, kp: 40, kd: 2 },
      { name: 'right_shoulder_pitch', target: 0, kp: 40, kd: 2 },
      { name: 'right_shoulder_roll', target: -0.2, kp: 40, kd: 2 },
      { name: 'right_shoulder_yaw', target: 0, kp: 40, kd: 2 },
      { name: 'right_elbow', target: -0.3, kp: 40, kd: 2 },
    ];

    for (const j of upperNames) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, j.name);
        if (jid < 0) continue;
        const qposadr = this.model.jnt_qposadr[jid];
        const dofadr = this.model.jnt_dofadr[jid];
        // Find actuator for this joint
        let actId = -1;
        for (let a = 0; a < this.model.nu; a++) {
          if (this.model.actuator_trnid[a * 2] === jid) { actId = a; break; }
        }
        if (actId >= 0) {
          this.upperBody.push({ actId, qposadr, dofadr, target: j.target, kp: j.kp, kd: j.kd });
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
      console.log('H1 ONNX policy loaded:', modelPath);
      return true;
    } catch (e) {
      console.error('Failed to load H1 ONNX model:', e);
      return false;
    }
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  setInitialPose() {
    // Use keyframe if available (H1 has "home" keyframe)
    if (this.model.nkey > 0) {
      this.data.qpos.set(this.model.key_qpos.slice(0, this.model.nq));
    } else {
      for (let i = 0; i < 10; i++) {
        this.data.qpos[this.jntQpos[i]] = this.defaultAngles[i];
      }
      this.data.qpos[2] = 1.0;
    }
    for (let i = 0; i < this.model.nv; i++) this.data.qvel[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
  }

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
    const obs = new Float32Array(41);
    const s = this.obsScales;

    const gyro = this.rotateByInvQuat(this.data.qvel[3], this.data.qvel[4], this.data.qvel[5]);
    obs[0] = gyro[0] * s.angVel;
    obs[1] = gyro[1] * s.angVel;
    obs[2] = gyro[2] * s.angVel;

    const grav = this.rotateByInvQuat(0, 0, -1);
    obs[3] = grav[0]; obs[4] = grav[1]; obs[5] = grav[2];

    obs[6] = this.forwardSpeed * s.cmdScale[0];
    obs[7] = this.lateralSpeed * s.cmdScale[1];
    obs[8] = this.turnRate * s.cmdScale[2];

    for (let i = 0; i < 10; i++) {
      obs[9 + i] = (this.data.qpos[this.jntQpos[i]] - this.defaultAngles[i]) * s.dofPos;
    }

    for (let i = 0; i < 10; i++) {
      obs[19 + i] = this.data.qvel[this.jntDof[i]] * s.dofVel;
    }

    for (let i = 0; i < 10; i++) {
      obs[29 + i] = this.lastAction[i];
    }

    obs[39] = Math.sin(2 * Math.PI * this.phase);
    obs[40] = Math.cos(2 * Math.PI * this.phase);

    for (let i = 0; i < 41; i++) {
      obs[i] = Math.max(-100, Math.min(100, obs[i]));
    }
    return obs;
  }

  applyPD() {
    const ctrl = this.data.ctrl;

    // Legs: policy targets
    for (let i = 0; i < 10; i++) {
      const q = this.data.qpos[this.jntQpos[i]];
      const qdot = this.data.qvel[this.jntDof[i]];
      ctrl[i] = this.kp[i] * (this.currentTargets[i] - q) - this.kd[i] * qdot;
    }

    // Upper body: hold at home pose
    for (const j of this.upperBody) {
      const q = this.data.qpos[j.qposadr];
      const qdot = this.data.qvel[j.dofadr];
      ctrl[j.actId] = j.kp * (j.target - q) - j.kd * qdot;
    }

    // Clamp all actuators
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
        obs: new ort.Tensor('float32', obs, [1, 41]),
        h0: new ort.Tensor('float32', this.h0, [1, 1, this.hiddenSize]),
        c0: new ort.Tensor('float32', this.c0, [1, 1, this.hiddenSize]),
      };
      const results = await this.session.run(input);
      const actions = results.actions.data;

      this.h0 = new Float32Array(results.h_out.data);
      this.c0 = new Float32Array(results.c_out.data);

      for (let i = 0; i < 10; i++) {
        const a = Math.max(-this.actionClip, Math.min(this.actionClip, actions[i]));
        this.lastAction[i] = a;
        this.currentTargets[i] = a * this.actionScale + this.defaultAngles[i];
      }

      this.phase = (this.phase + this.gaitFreq * this.ctrlDt) % 1.0;
    } catch (e) {
      console.error('H1 ONNX inference error:', e);
    }
    this._inferring = false;
  }

  reset() {
    this.stepCount = 0;
    this.lastAction.fill(0);
    for (let i = 0; i < 10; i++) this.currentTargets[i] = this.defaultAngles[i];
    this.phase = 0.0;
    this.h0 = new Float32Array(this.hiddenSize);
    this.c0 = new Float32Array(this.hiddenSize);
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this._inferring = false;
  }
}
