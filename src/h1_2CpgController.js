/**
 * CPG + Tricks + QWOP controller for Unitree H1-2 (v2).
 * Procedural gait with trick state machine and manual joint control.
 *
 * H1-2 actuator layout (27 torque motors):
 *   0: left_hip_yaw_joint       6: right_hip_yaw_joint     12: torso_joint
 *   1: left_hip_pitch_joint     7: right_hip_pitch_joint   13-19: left arm
 *   2: left_hip_roll_joint      8: right_hip_roll_joint    20-26: right arm
 *   3: left_knee_joint          9: right_knee_joint
 *   4: left_ankle_pitch_joint  10: right_ankle_pitch_joint
 *   5: left_ankle_roll_joint   11: right_ankle_roll_joint
 *
 * Key differences from H1 (v1):
 *   - 27 DOF (vs 19): 2-DOF ankles (pitch+roll), 3-DOF wrists per arm
 *   - Hip chain order: yaw -> pitch -> roll (H1 is yaw -> roll -> pitch)
 *   - All joint names have "_joint" suffix
 *
 * Tricks: 1=Jump, 2=Kick, 3=Wave, 4=Bow
 * QWOP: Q/W=hip pitch, A/S=knee, I/O=ankle, K/L=arms, Z/X=torso
 */

// ── Trick phase constants ────────────────────────────────────────────
const IDLE    = 'idle';
const CROUCH  = 'crouch';
const LAUNCH  = 'launch';
const AIR     = 'air';
const LAND    = 'land';
const RECOVER = 'recover';

const GAIN_SCALE = {
  [IDLE]:    { kp: 1.0, kd: 1.0 },
  [CROUCH]:  { kp: 1.3, kd: 1.0 },
  [LAUNCH]:  { kp: 1.8, kd: 0.5 },
  [AIR]:     { kp: 0.4, kd: 0.6 },
  [LAND]:    { kp: 1.0, kd: 2.0 },
  [RECOVER]: { kp: 1.0, kd: 1.0 },
};

const TRICKS = {
  jump: [
    { phase: CROUCH, steps: 80, targets: {
      left_hip_pitch_joint: -1.0, left_knee_joint: 2.0, left_ankle_pitch_joint: -1.0,
      right_hip_pitch_joint: -1.0, right_knee_joint: 2.0, right_ankle_pitch_joint: -1.0,
    }},
    { phase: LAUNCH, steps: 50, targets: {
      left_hip_pitch_joint: 0.15, left_knee_joint: 0.05, left_ankle_pitch_joint: 0.05,
      right_hip_pitch_joint: 0.15, right_knee_joint: 0.05, right_ankle_pitch_joint: 0.05,
    }},
    { phase: AIR, steps: 200, targets: {
      left_hip_pitch_joint: -0.8, left_knee_joint: 1.6, left_ankle_pitch_joint: -0.8,
      right_hip_pitch_joint: -0.8, right_knee_joint: 1.6, right_ankle_pitch_joint: -0.8,
    }},
    { phase: LAND, steps: 60, targets: {
      left_hip_pitch_joint: -0.7, left_knee_joint: 1.4, left_ankle_pitch_joint: -0.7,
      right_hip_pitch_joint: -0.7, right_knee_joint: 1.4, right_ankle_pitch_joint: -0.7,
    }},
    { phase: RECOVER, steps: 120, targets: {} },
  ],
  kick: [
    { phase: CROUCH, steps: 60, targets: {
      left_hip_roll_joint: -0.12, left_hip_pitch_joint: -0.5, left_knee_joint: 1.0, left_ankle_pitch_joint: -0.5,
      right_hip_pitch_joint: -0.6, right_knee_joint: 1.2,
    }},
    { phase: LAUNCH, steps: 40, targets: {
      left_hip_roll_joint: -0.15,
      right_hip_pitch_joint: -1.5, right_knee_joint: 0.2, right_ankle_pitch_joint: -0.1,
    }},
    { phase: AIR, steps: 100, gains: { kp: 1.2, kd: 1.0 }, targets: {
      left_hip_roll_joint: -0.15,
      right_hip_pitch_joint: -1.5, right_knee_joint: 0.2, right_ankle_pitch_joint: -0.1,
    }},
    { phase: LAND, steps: 60, targets: {
      right_hip_pitch_joint: -0.5, right_knee_joint: 1.0, right_ankle_pitch_joint: -0.5,
    }},
    { phase: RECOVER, steps: 100, targets: {} },
  ],
  wave: [
    { phase: CROUCH, steps: 50, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.6, right_elbow_joint: -1.0,
    }},
    { phase: LAUNCH, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.1, right_elbow_joint: -0.8,
    }},
    { phase: AIR, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.6, right_elbow_joint: -1.0,
    }},
    { phase: LAND, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.1, right_elbow_joint: -0.8,
    }},
    { phase: RECOVER, steps: 80, gains: { kp: 1, kd: 1 }, targets: {} },
  ],
  bow: [
    { phase: CROUCH, steps: 80, gains: { kp: 1.2, kd: 1 }, targets: {
      torso_joint: 0.3,
      left_hip_pitch_joint: -0.8, right_hip_pitch_joint: -0.8,
      left_shoulder_pitch_joint: 0.5, right_shoulder_pitch_joint: 0.5,
      left_shoulder_roll_joint: 0.05, right_shoulder_roll_joint: -0.05,
    }},
    { phase: LAUNCH, steps: 100, gains: { kp: 1.2, kd: 1 }, targets: {
      torso_joint: 0.3,
      left_hip_pitch_joint: -0.8, right_hip_pitch_joint: -0.8,
      left_shoulder_pitch_joint: 0.5, right_shoulder_pitch_joint: 0.5,
      left_shoulder_roll_joint: 0.05, right_shoulder_roll_joint: -0.05,
    }},
    { phase: AIR, steps: 40, gains: { kp: 1, kd: 1 }, targets: {} },
    { phase: LAND, steps: 40, gains: { kp: 1, kd: 1 }, targets: {} },
    { phase: RECOVER, steps: 80, targets: {} },
  ],
};

export class H1_2CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 1.2;
    this.phase = 0;

    // Amplitudes
    this.hipPitchAmp = 0.25;
    this.kneeAmp = 0.35;
    this.anklePitchAmp = 0.15;
    this.ankleRollAmp = 0.04;
    this.hipRollAmp = 0.03;

    // Arm swing
    this.armSwingGain = 0.3;

    // Balance
    this.balanceKp = 150.0;
    this.balanceKd = 10.0;

    // Home pose
    this.homeQpos = {
      left_hip_yaw_joint: 0, left_hip_pitch_joint: -0.4, left_hip_roll_joint: 0,
      left_knee_joint: 0.8, left_ankle_pitch_joint: -0.4, left_ankle_roll_joint: 0,
      right_hip_yaw_joint: 0, right_hip_pitch_joint: -0.4, right_hip_roll_joint: 0,
      right_knee_joint: 0.8, right_ankle_pitch_joint: -0.4, right_ankle_roll_joint: 0,
      torso_joint: 0,
      left_shoulder_pitch_joint: 0, left_shoulder_roll_joint: 0.2,
      left_shoulder_yaw_joint: 0, left_elbow_joint: -0.3,
      left_wrist_roll_joint: 0, left_wrist_pitch_joint: 0, left_wrist_yaw_joint: 0,
      right_shoulder_pitch_joint: 0, right_shoulder_roll_joint: -0.2,
      right_shoulder_yaw_joint: 0, right_elbow_joint: -0.3,
      right_wrist_roll_joint: 0, right_wrist_pitch_joint: 0, right_wrist_yaw_joint: 0,
    };

    // Actuator index map
    this.actIdx = {
      left_hip_yaw_joint: 0, left_hip_pitch_joint: 1, left_hip_roll_joint: 2,
      left_knee_joint: 3, left_ankle_pitch_joint: 4, left_ankle_roll_joint: 5,
      right_hip_yaw_joint: 6, right_hip_pitch_joint: 7, right_hip_roll_joint: 8,
      right_knee_joint: 9, right_ankle_pitch_joint: 10, right_ankle_roll_joint: 11,
      torso_joint: 12,
      left_shoulder_pitch_joint: 13, left_shoulder_roll_joint: 14,
      left_shoulder_yaw_joint: 15, left_elbow_joint: 16,
      left_wrist_roll_joint: 17, left_wrist_pitch_joint: 18, left_wrist_yaw_joint: 19,
      right_shoulder_pitch_joint: 20, right_shoulder_roll_joint: 21,
      right_shoulder_yaw_joint: 22, right_elbow_joint: 23,
      right_wrist_roll_joint: 24, right_wrist_pitch_joint: 25, right_wrist_yaw_joint: 26,
    };

    this.jntIdx = {};
    this.findJointIndices();

    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;

    // ── Trick state machine ──────────────────────────────────────────
    this.trickPhase = IDLE;
    this.trickName = null;
    this.trickIdx = 0;
    this.trickStep = 0;
    this.trickStart = {};

    // ── QWOP mode ────────────────────────────────────────────────────
    this.qwopMode = false;
    this.qwopKeys = {};
    this.qwopDelta = 0.5;
  }

  findJointIndices() {
    for (const name of Object.keys(this.actIdx)) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0 && this.model.jnt_qposadr) {
          this.jntIdx[name] = this.model.jnt_qposadr[jid];
        }
      } catch (e) { /* fallback */ }
    }
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  getTrunkOrientation() {
    const qw = this.data.qpos[3], qx = this.data.qpos[4];
    const qy = this.data.qpos[5], qz = this.data.qpos[6];
    const sinp = 2 * (qw * qy - qz * qx);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
    const roll = Math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
    return { pitch, roll };
  }

  pdTorque(jointName, target, kp, kd) {
    const idx = this.jntIdx[jointName];
    if (idx === undefined) return 0;
    let dofIdx;
    try {
      const jid = this.mujoco.mj_name2id(this.model, 3, jointName);
      dofIdx = this.model.jnt_dofadr[jid];
    } catch (e) { dofIdx = idx - 7 + 6; }
    const q = this.data.qpos[idx];
    const qdot = this.data.qvel[dofIdx] || 0;
    return kp * (target - q) - kd * qdot;
  }

  _gainsForJoint(name) {
    if (name.includes('knee')) return [500, 20];
    if (name.includes('ankle_pitch')) return [80, 5];
    if (name.includes('ankle_roll')) return [50, 3];
    if (name.includes('hip')) return [350, 15];
    if (name.includes('torso')) return [300, 12];
    if (name.includes('wrist')) return [12, 1];
    return [40, 2]; // shoulder/elbow
  }

  // ── Tricks ─────────────────────────────────────────────────────────
  triggerTrick(name) {
    if (this.trickPhase !== IDLE || !TRICKS[name]) return;
    this.trickName = name;
    this.trickIdx = 0;
    this.trickStep = 0;
    this.trickPhase = TRICKS[name][0].phase;
    this._snapshotCurrentPose();
    console.log(`H1-2 trick: ${name}`);
  }

  _snapshotCurrentPose() {
    this.trickStart = {};
    for (const name of Object.keys(this.actIdx)) {
      const idx = this.jntIdx[name];
      this.trickStart[name] = idx !== undefined ? this.data.qpos[idx] : (this.homeQpos[name] || 0);
    }
  }

  _stepTrick() {
    const seq = TRICKS[this.trickName];
    if (!seq) { this.trickPhase = IDLE; return; }
    const spec = seq[this.trickIdx];
    if (!spec) { this._endTrick(); return; }

    this.trickStep++;
    this.trickPhase = spec.phase;

    const p = Math.min(1.0, this.trickStep / spec.steps);
    const t = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;

    const gs = spec.gains || GAIN_SCALE[spec.phase];
    const fullTarget = { ...this.homeQpos, ...spec.targets };

    const ctrl = this.data.ctrl;
    for (const [name, actI] of Object.entries(this.actIdx)) {
      const start = this.trickStart[name] ?? this.homeQpos[name] ?? 0;
      const end = fullTarget[name] ?? this.homeQpos[name] ?? 0;
      const target = start + (end - start) * t;
      const [kp, kd] = this._gainsForJoint(name);
      ctrl[actI] = this.pdTorque(name, target, kp * gs.kp, kd * gs.kd);
    }

    if (spec.phase === AIR && this.trickStep > 40 && (this.trickName === 'jump' || this.trickName === 'kick')) {
      const z = this.data.qpos[2];
      const vz = this.data.qvel[2];
      if (z < 1.0 && vz < -0.1) {
        this._advanceTrick();
        return;
      }
    }

    if (this.trickStep >= spec.steps) this._advanceTrick();
    this._clampCtrl();
  }

  _advanceTrick() {
    this.trickIdx++;
    this.trickStep = 0;
    this._snapshotCurrentPose();
    const seq = TRICKS[this.trickName];
    if (!seq || this.trickIdx >= seq.length) {
      this._endTrick();
    } else {
      this.trickPhase = seq[this.trickIdx].phase;
    }
  }

  _endTrick() {
    this.trickPhase = IDLE;
    this.trickName = null;
    this.trickIdx = 0;
    this.trickStep = 0;
  }

  // ── QWOP ──────────────────────────────────────────────────────────
  _stepQwop() {
    const k = this.qwopKeys;
    const d = this.qwopDelta;
    const targets = { ...this.homeQpos };

    if (k['KeyQ']) targets.left_hip_pitch_joint = this.homeQpos.left_hip_pitch_joint - d;
    if (k['KeyW']) targets.right_hip_pitch_joint = this.homeQpos.right_hip_pitch_joint - d;
    if (k['KeyA']) targets.left_knee_joint = this.homeQpos.left_knee_joint + d;
    if (k['KeyS']) targets.right_knee_joint = this.homeQpos.right_knee_joint + d;
    if (k['KeyI']) targets.left_ankle_pitch_joint = this.homeQpos.left_ankle_pitch_joint + d * 0.5;
    if (k['KeyO']) targets.right_ankle_pitch_joint = this.homeQpos.right_ankle_pitch_joint + d * 0.5;
    if (k['KeyK']) targets.left_shoulder_pitch_joint = this.homeQpos.left_shoulder_pitch_joint - d * 1.5;
    if (k['KeyL']) targets.right_shoulder_pitch_joint = this.homeQpos.right_shoulder_pitch_joint - d * 1.5;
    if (k['KeyZ']) targets.torso_joint = this.homeQpos.torso_joint + d * 0.5;
    if (k['KeyX']) targets.torso_joint = this.homeQpos.torso_joint - d * 0.5;

    const ctrl = this.data.ctrl;
    for (const [name, actI] of Object.entries(this.actIdx)) {
      const [kp, kd] = this._gainsForJoint(name);
      ctrl[actI] = this.pdTorque(name, targets[name], kp, kd);
    }
    this._clampCtrl();
  }

  _clampCtrl() {
    if (this.model.actuator_ctrlrange) {
      const ctrl = this.data.ctrl;
      for (let i = 0; i < this.model.nu; i++) {
        const lo = this.model.actuator_ctrlrange[i * 2];
        const hi = this.model.actuator_ctrlrange[i * 2 + 1];
        ctrl[i] = Math.max(lo, Math.min(hi, ctrl[i]));
      }
    }
  }

  // ── Walking ───────────────────────────────────────────────────────
  _stepWalk() {
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI;

    const fwdMag = Math.abs(this.forwardSpeed);
    const latMag = Math.abs(this.lateralSpeed);
    const turnMag = Math.abs(this.turnRate);
    const anyCommand = fwdMag > 0.05 || latMag > 0.05 || turnMag > 0.05;
    const ampScale = anyCommand
      ? Math.max(0.25, fwdMag + latMag * 0.4 + turnMag * 0.3) : 0;
    const direction = Math.sign(this.forwardSpeed) || 1;

    const { pitch, roll } = this.getTrunkOrientation();
    const rawPR = (pitch - this.prevPitch) / this.simDt;
    const rawRR = (roll - this.prevRoll) / this.simDt;
    const pitchRate = Math.max(-8, Math.min(8, rawPR));
    const rollRate = Math.max(-8, Math.min(8, rawRR));
    this.prevPitch = pitch;
    this.prevRoll = roll;

    const hipKp = 350, hipKd = 15;
    const kneeKp = 500, kneeKd = 20;
    const apKp = 80, apKd = 5;
    const arKp = 50, arKd = 3;
    const torsoKp = 300, torsoKd = 12;
    const armKp = 40, armKd = 2;

    const ctrl = this.data.ctrl;

    // --- LEFT LEG ---
    const lS = Math.sin(leftPhase);
    const lSt = Math.max(0, -Math.sin(leftPhase));

    const lHipPitch = this.homeQpos.left_hip_pitch_joint + direction * this.hipPitchAmp * ampScale * lS;
    const lKnee = this.homeQpos.left_knee_joint + this.kneeAmp * ampScale * Math.max(0, Math.sin(leftPhase));
    const lAnkleP = this.homeQpos.left_ankle_pitch_joint - this.anklePitchAmp * ampScale * lS;
    const lAnkleR = this.homeQpos.left_ankle_roll_joint - this.ankleRollAmp * lSt;
    const lHipRoll = this.homeQpos.left_hip_roll_joint - this.hipRollAmp * lSt + this.lateralSpeed * 0.12;
    const lHipYaw = this.homeQpos.left_hip_yaw_joint + this.turnRate * 0.10 * lS;

    ctrl[this.actIdx.left_hip_yaw_joint] = this.pdTorque('left_hip_yaw_joint', lHipYaw, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_pitch_joint] = this.pdTorque('left_hip_pitch_joint', lHipPitch, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll_joint] = this.pdTorque('left_hip_roll_joint', lHipRoll, hipKp, hipKd);
    ctrl[this.actIdx.left_knee_joint] = this.pdTorque('left_knee_joint', lKnee, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle_pitch_joint] = this.pdTorque('left_ankle_pitch_joint', lAnkleP, apKp, apKd);
    ctrl[this.actIdx.left_ankle_roll_joint] = this.pdTorque('left_ankle_roll_joint', lAnkleR, arKp, arKd);

    // --- RIGHT LEG ---
    const rS = Math.sin(rightPhase);
    const rSt = Math.max(0, -Math.sin(rightPhase));

    const rHipPitch = this.homeQpos.right_hip_pitch_joint + direction * this.hipPitchAmp * ampScale * rS;
    const rKnee = this.homeQpos.right_knee_joint + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rAnkleP = this.homeQpos.right_ankle_pitch_joint - this.anklePitchAmp * ampScale * rS;
    const rAnkleR = this.homeQpos.right_ankle_roll_joint + this.ankleRollAmp * rSt;
    const rHipRoll = this.homeQpos.right_hip_roll_joint + this.hipRollAmp * rSt + this.lateralSpeed * 0.12;
    const rHipYaw = this.homeQpos.right_hip_yaw_joint + this.turnRate * 0.10 * rS;

    ctrl[this.actIdx.right_hip_yaw_joint] = this.pdTorque('right_hip_yaw_joint', rHipYaw, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_pitch_joint] = this.pdTorque('right_hip_pitch_joint', rHipPitch, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll_joint] = this.pdTorque('right_hip_roll_joint', rHipRoll, hipKp, hipKd);
    ctrl[this.actIdx.right_knee_joint] = this.pdTorque('right_knee_joint', rKnee, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle_pitch_joint] = this.pdTorque('right_ankle_pitch_joint', rAnkleP, apKp, apKd);
    ctrl[this.actIdx.right_ankle_roll_joint] = this.pdTorque('right_ankle_roll_joint', rAnkleR, arKp, arKd);

    // --- TORSO ---
    ctrl[this.actIdx.torso_joint] = this.pdTorque('torso_joint',
      this.homeQpos.torso_joint + this.turnRate * 0.2, torsoKp, torsoKd);

    // --- BALANCE CORRECTIONS ---
    const pCorr = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rCorr = -this.balanceKp * roll - this.balanceKd * rollRate;

    ctrl[this.actIdx.left_ankle_pitch_joint] += pCorr * 0.2;
    ctrl[this.actIdx.right_ankle_pitch_joint] += pCorr * 0.2;
    ctrl[this.actIdx.left_ankle_roll_joint] += rCorr * 0.15;
    ctrl[this.actIdx.right_ankle_roll_joint] -= rCorr * 0.15;
    ctrl[this.actIdx.left_hip_pitch_joint] += pCorr * 0.3;
    ctrl[this.actIdx.right_hip_pitch_joint] += pCorr * 0.3;
    ctrl[this.actIdx.left_hip_roll_joint] += rCorr * 0.2;
    ctrl[this.actIdx.right_hip_roll_joint] -= rCorr * 0.2;

    // --- ARMS ---
    const lArm = -this.armSwingGain * direction * ampScale * lS;
    const rArm = -this.armSwingGain * direction * ampScale * rS;

    ctrl[this.actIdx.left_shoulder_pitch_joint] = this.pdTorque('left_shoulder_pitch_joint',
      this.homeQpos.left_shoulder_pitch_joint + lArm, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_roll_joint] = this.pdTorque('left_shoulder_roll_joint',
      this.homeQpos.left_shoulder_roll_joint, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_yaw_joint] = this.pdTorque('left_shoulder_yaw_joint',
      this.homeQpos.left_shoulder_yaw_joint, armKp * 0.5, armKd);
    ctrl[this.actIdx.left_elbow_joint] = this.pdTorque('left_elbow_joint',
      this.homeQpos.left_elbow_joint, armKp, armKd);

    ctrl[this.actIdx.right_shoulder_pitch_joint] = this.pdTorque('right_shoulder_pitch_joint',
      this.homeQpos.right_shoulder_pitch_joint + rArm, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_roll_joint] = this.pdTorque('right_shoulder_roll_joint',
      this.homeQpos.right_shoulder_roll_joint, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_yaw_joint] = this.pdTorque('right_shoulder_yaw_joint',
      this.homeQpos.right_shoulder_yaw_joint, armKp * 0.5, armKd);
    ctrl[this.actIdx.right_elbow_joint] = this.pdTorque('right_elbow_joint',
      this.homeQpos.right_elbow_joint, armKp, armKd);

    // --- WRISTS ---
    const wKp = armKp * 0.3, wKd = armKd * 0.5;
    for (const side of ['left', 'right']) {
      for (const axis of ['roll', 'pitch', 'yaw']) {
        const j = `${side}_wrist_${axis}_joint`;
        ctrl[this.actIdx[j]] = this.pdTorque(j, this.homeQpos[j], wKp, wKd);
      }
    }

    // Clamp
    this._clampCtrl();
  }

  // ── Main step ─────────────────────────────────────────────────────
  step() {
    if (!this.enabled) return;

    if (this.qwopMode) {
      this._stepQwop();
    } else if (this.trickPhase !== IDLE) {
      this._stepTrick();
    } else {
      this._stepWalk();
    }
  }

  reset() {
    this.phase = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
    this._endTrick();
    this.qwopMode = false;
  }
}
