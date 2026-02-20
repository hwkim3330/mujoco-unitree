/**
 * CPG + Tricks + QWOP controller for Unitree H1.
 * Procedural gait with trick state machine and manual joint control.
 *
 * H1 actuator layout (19 torque motors):
 *  0: left_hip_yaw      5: right_hip_yaw     10: torso
 *  1: left_hip_roll      6: right_hip_roll     11: left_shoulder_pitch
 *  2: left_hip_pitch     7: right_hip_pitch    12: left_shoulder_roll
 *  3: left_knee          8: right_knee         13: left_shoulder_yaw
 *  4: left_ankle         9: right_ankle        14: left_elbow
 *                                              15: right_shoulder_pitch
 *                                              16: right_shoulder_roll
 *                                              17: right_shoulder_yaw
 *                                              18: right_elbow
 *
 * Home keyframe qpos (after free joint 7-DOF):
 *  left_leg:  [0, 0, -0.4, 0.8, -0.4]
 *  right_leg: [0, 0, -0.4, 0.8, -0.4]
 *  torso: 0
 *  left_arm:  [0, 0.2, 0, -0.3]
 *  right_arm: [0, -0.2, 0, -0.3]
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

// Default gain scaling per trick phase
const GAIN_SCALE = {
  [IDLE]:    { kp: 1.0, kd: 1.0 },
  [CROUCH]:  { kp: 1.3, kd: 1.0 },
  [LAUNCH]:  { kp: 1.8, kd: 0.5 },
  [AIR]:     { kp: 0.4, kd: 0.6 },
  [LAND]:    { kp: 1.0, kd: 2.0 },
  [RECOVER]: { kp: 1.0, kd: 1.0 },
};

// Trick definitions — targets are partial overrides of homeQpos
const TRICKS = {
  jump: [
    { phase: CROUCH, steps: 80, targets: {
      left_hip_pitch: -1.0, left_knee: 2.0, left_ankle: -1.0,
      right_hip_pitch: -1.0, right_knee: 2.0, right_ankle: -1.0,
    }},
    { phase: LAUNCH, steps: 50, targets: {
      left_hip_pitch: 0.15, left_knee: 0.05, left_ankle: 0.05,
      right_hip_pitch: 0.15, right_knee: 0.05, right_ankle: 0.05,
    }},
    { phase: AIR, steps: 200, targets: {
      left_hip_pitch: -0.8, left_knee: 1.6, left_ankle: -0.8,
      right_hip_pitch: -0.8, right_knee: 1.6, right_ankle: -0.8,
    }},
    { phase: LAND, steps: 60, targets: {
      left_hip_pitch: -0.7, left_knee: 1.4, left_ankle: -0.7,
      right_hip_pitch: -0.7, right_knee: 1.4, right_ankle: -0.7,
    }},
    { phase: RECOVER, steps: 120, targets: {} },
  ],
  kick: [
    { phase: CROUCH, steps: 60, targets: {
      left_hip_roll: -0.12, left_hip_pitch: -0.5, left_knee: 1.0, left_ankle: -0.5,
      right_hip_pitch: -0.6, right_knee: 1.2,
    }},
    { phase: LAUNCH, steps: 40, targets: {
      left_hip_roll: -0.15,
      right_hip_pitch: -1.5, right_knee: 0.2, right_ankle: -0.1,
    }},
    { phase: AIR, steps: 100, gains: { kp: 1.2, kd: 1.0 }, targets: {
      left_hip_roll: -0.15,
      right_hip_pitch: -1.5, right_knee: 0.2, right_ankle: -0.1,
    }},
    { phase: LAND, steps: 60, targets: {
      right_hip_pitch: -0.5, right_knee: 1.0, right_ankle: -0.5,
    }},
    { phase: RECOVER, steps: 100, targets: {} },
  ],
  wave: [
    { phase: CROUCH, steps: 50, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch: -1.5, right_shoulder_roll: -0.6, right_elbow: -1.0,
    }},
    { phase: LAUNCH, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch: -1.5, right_shoulder_roll: -0.1, right_elbow: -0.8,
    }},
    { phase: AIR, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch: -1.5, right_shoulder_roll: -0.6, right_elbow: -1.0,
    }},
    { phase: LAND, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch: -1.5, right_shoulder_roll: -0.1, right_elbow: -0.8,
    }},
    { phase: RECOVER, steps: 80, gains: { kp: 1, kd: 1 }, targets: {} },
  ],
  bow: [
    { phase: CROUCH, steps: 80, gains: { kp: 1.2, kd: 1 }, targets: {
      torso: 0.3,
      left_hip_pitch: -0.8, right_hip_pitch: -0.8,
      left_shoulder_pitch: 0.5, right_shoulder_pitch: 0.5,
      left_shoulder_roll: 0.05, right_shoulder_roll: -0.05,
    }},
    { phase: LAUNCH, steps: 100, gains: { kp: 1.2, kd: 1 }, targets: {
      torso: 0.3,
      left_hip_pitch: -0.8, right_hip_pitch: -0.8,
      left_shoulder_pitch: 0.5, right_shoulder_pitch: 0.5,
      left_shoulder_roll: 0.05, right_shoulder_roll: -0.05,
    }},
    { phase: AIR, steps: 40, gains: { kp: 1, kd: 1 }, targets: {} },
    { phase: LAND, steps: 40, gains: { kp: 1, kd: 1 }, targets: {} },
    { phase: RECOVER, steps: 80, targets: {} },
  ],
};

export class H1CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 1.2;
    this.phase = 0;

    // Amplitudes for leg joints
    this.hipPitchAmp = 0.25;
    this.kneeAmp = 0.35;
    this.ankleAmp = 0.15;
    this.hipRollAmp = 0.03;

    // Arm swing
    this.armSwingGain = 0.3;

    // PD gains for balance
    this.balanceKp = 150.0;
    this.balanceKd = 10.0;

    // Home pose
    this.homeQpos = {
      left_hip_yaw: 0, left_hip_roll: 0, left_hip_pitch: -0.4, left_knee: 0.8, left_ankle: -0.4,
      right_hip_yaw: 0, right_hip_roll: 0, right_hip_pitch: -0.4, right_knee: 0.8, right_ankle: -0.4,
      torso: 0,
      left_shoulder_pitch: 0, left_shoulder_roll: 0.2, left_shoulder_yaw: 0, left_elbow: -0.3,
      right_shoulder_pitch: 0, right_shoulder_roll: -0.2, right_shoulder_yaw: 0, right_elbow: -0.3,
    };

    // Actuator index map
    this.actIdx = {
      left_hip_yaw: 0, left_hip_roll: 1, left_hip_pitch: 2, left_knee: 3, left_ankle: 4,
      right_hip_yaw: 5, right_hip_roll: 6, right_hip_pitch: 7, right_knee: 8, right_ankle: 9,
      torso: 10,
      left_shoulder_pitch: 11, left_shoulder_roll: 12, left_shoulder_yaw: 13, left_elbow: 14,
      right_shoulder_pitch: 15, right_shoulder_roll: 16, right_shoulder_yaw: 17, right_elbow: 18,
    };

    // Joint qpos indices (after freejoint: offset 7)
    this.jntIdx = {};
    this.findJointIndices();

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Balance state
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
    const names = Object.keys(this.actIdx);
    for (const name of names) {
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
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];
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
    if (name.includes('ankle')) return [60, 4];
    if (name.includes('hip')) return [350, 15];
    if (name.includes('torso')) return [300, 12];
    return [40, 2]; // arms
  }

  // ── Tricks ─────────────────────────────────────────────────────────
  triggerTrick(name) {
    if (this.trickPhase !== IDLE || !TRICKS[name]) return;
    this.trickName = name;
    this.trickIdx = 0;
    this.trickStep = 0;
    this.trickPhase = TRICKS[name][0].phase;
    this._snapshotCurrentPose();
    console.log(`H1 trick: ${name}`);
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

    // Ease-in-out interpolation
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

    // Early landing detection during AIR for jump/kick
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

    // Q/W = left/right hip pitch (forward swing)
    if (k['KeyQ']) targets.left_hip_pitch = this.homeQpos.left_hip_pitch - d;
    if (k['KeyW']) targets.right_hip_pitch = this.homeQpos.right_hip_pitch - d;
    // A/S = left/right knee (bend more)
    if (k['KeyA']) targets.left_knee = this.homeQpos.left_knee + d;
    if (k['KeyS']) targets.right_knee = this.homeQpos.right_knee + d;
    // I/O = left/right ankle
    if (k['KeyI']) targets.left_ankle = this.homeQpos.left_ankle + d * 0.5;
    if (k['KeyO']) targets.right_ankle = this.homeQpos.right_ankle + d * 0.5;
    // K/L = left/right arm raise
    if (k['KeyK']) targets.left_shoulder_pitch = this.homeQpos.left_shoulder_pitch - d * 1.5;
    if (k['KeyL']) targets.right_shoulder_pitch = this.homeQpos.right_shoulder_pitch - d * 1.5;
    // Z/X = torso twist
    if (k['KeyZ']) targets.torso = this.homeQpos.torso + d * 0.5;
    if (k['KeyX']) targets.torso = this.homeQpos.torso - d * 0.5;

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
    // Advance phase
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
    const ankleKp = 60, ankleKd = 4;
    const torsoKp = 300, torsoKd = 12;
    const armKp = 40, armKd = 2;

    const ctrl = this.data.ctrl;

    // --- LEFT LEG ---
    const leftSwing = Math.sin(leftPhase);
    const leftStance = Math.max(0, -Math.sin(leftPhase));

    const leftHipPitchTarget = this.homeQpos.left_hip_pitch
      + direction * this.hipPitchAmp * ampScale * leftSwing;
    const leftKneeTarget = this.homeQpos.left_knee
      + this.kneeAmp * ampScale * Math.max(0, Math.sin(leftPhase));
    const leftAnkleTarget = this.homeQpos.left_ankle
      - this.ankleAmp * ampScale * leftSwing;
    const leftHipRollTarget = this.homeQpos.left_hip_roll
      - this.hipRollAmp * leftStance
      + this.lateralSpeed * 0.12;
    const leftHipYawTarget = this.homeQpos.left_hip_yaw
      + this.turnRate * 0.10 * leftSwing;

    ctrl[this.actIdx.left_hip_yaw] = this.pdTorque('left_hip_yaw', leftHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll] = this.pdTorque('left_hip_roll', leftHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_pitch] = this.pdTorque('left_hip_pitch', leftHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_knee] = this.pdTorque('left_knee', leftKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle] = this.pdTorque('left_ankle', leftAnkleTarget, ankleKp, ankleKd);

    // --- RIGHT LEG ---
    const rightSwing = Math.sin(rightPhase);
    const rightStance = Math.max(0, -Math.sin(rightPhase));

    const rightHipPitchTarget = this.homeQpos.right_hip_pitch
      + direction * this.hipPitchAmp * ampScale * rightSwing;
    const rightKneeTarget = this.homeQpos.right_knee
      + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rightAnkleTarget = this.homeQpos.right_ankle
      - this.ankleAmp * ampScale * rightSwing;
    const rightHipRollTarget = this.homeQpos.right_hip_roll
      + this.hipRollAmp * rightStance
      + this.lateralSpeed * 0.12;
    const rightHipYawTarget = this.homeQpos.right_hip_yaw
      + this.turnRate * 0.10 * rightSwing;

    ctrl[this.actIdx.right_hip_yaw] = this.pdTorque('right_hip_yaw', rightHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll] = this.pdTorque('right_hip_roll', rightHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_pitch] = this.pdTorque('right_hip_pitch', rightHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_knee] = this.pdTorque('right_knee', rightKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle] = this.pdTorque('right_ankle', rightAnkleTarget, ankleKp, ankleKd);

    // --- TORSO ---
    const torsoTarget = this.homeQpos.torso + this.turnRate * 0.2;
    ctrl[this.actIdx.torso] = this.pdTorque('torso', torsoTarget, torsoKp, torsoKd);

    // --- BALANCE CORRECTIONS ---
    const pitchCorrection = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorrection = -this.balanceKp * roll - this.balanceKd * rollRate;

    ctrl[this.actIdx.left_ankle] += pitchCorrection * 0.15;
    ctrl[this.actIdx.right_ankle] += pitchCorrection * 0.15;
    ctrl[this.actIdx.left_hip_pitch] += pitchCorrection * 0.3;
    ctrl[this.actIdx.right_hip_pitch] += pitchCorrection * 0.3;
    ctrl[this.actIdx.left_hip_roll] += rollCorrection * 0.2;
    ctrl[this.actIdx.right_hip_roll] -= rollCorrection * 0.2;

    // --- ARMS ---
    const leftArmSwing = -this.armSwingGain * direction * ampScale * leftSwing;
    const rightArmSwing = -this.armSwingGain * direction * ampScale * rightSwing;

    ctrl[this.actIdx.left_shoulder_pitch] = this.pdTorque('left_shoulder_pitch',
      this.homeQpos.left_shoulder_pitch + leftArmSwing, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_roll] = this.pdTorque('left_shoulder_roll',
      this.homeQpos.left_shoulder_roll, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_yaw] = this.pdTorque('left_shoulder_yaw',
      this.homeQpos.left_shoulder_yaw, armKp * 0.5, armKd);
    ctrl[this.actIdx.left_elbow] = this.pdTorque('left_elbow',
      this.homeQpos.left_elbow, armKp, armKd);

    ctrl[this.actIdx.right_shoulder_pitch] = this.pdTorque('right_shoulder_pitch',
      this.homeQpos.right_shoulder_pitch + rightArmSwing, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_roll] = this.pdTorque('right_shoulder_roll',
      this.homeQpos.right_shoulder_roll, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_yaw] = this.pdTorque('right_shoulder_yaw',
      this.homeQpos.right_shoulder_yaw, armKp * 0.5, armKd);
    ctrl[this.actIdx.right_elbow] = this.pdTorque('right_elbow',
      this.homeQpos.right_elbow, armKp, armKd);

    // Clamp all controls
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
