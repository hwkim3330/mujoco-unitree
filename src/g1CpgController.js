/**
 * CPG + Tricks + QWOP controller for Unitree G1.
 * Procedural gait with trick state machine and manual joint control.
 *
 * G1 actuator layout (29 torque motors):
 *   0: left_hip_pitch       6: right_hip_pitch     12: waist_yaw
 *   1: left_hip_roll        7: right_hip_roll      13: waist_roll
 *   2: left_hip_yaw         8: right_hip_yaw       14: waist_pitch
 *   3: left_knee            9: right_knee          15-21: left arm
 *   4: left_ankle_pitch    10: right_ankle_pitch   22-28: right arm
 *   5: left_ankle_roll     11: right_ankle_roll
 *
 * No keyframe — G1 starts at q=0 (T-pose).
 * G1 advantages: 2-DOF ankles, 3-DOF waist, lighter (28kg).
 *
 * Tricks: 1=Jump, 2=Kick, 3=Wave, 4=Bow
 * QWOP: Q/W=hip pitch, A/S=knee, I/O=ankle, K/L=arms, Z/X=waist
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

// G1 home: hip_pitch=-0.2, knee=0.4, ankle_pitch=-0.2
const TRICKS = {
  jump: [
    { phase: CROUCH, steps: 80, targets: {
      left_hip_pitch_joint: -0.7, left_knee_joint: 1.4, left_ankle_pitch_joint: -0.7,
      right_hip_pitch_joint: -0.7, right_knee_joint: 1.4, right_ankle_pitch_joint: -0.7,
    }},
    { phase: LAUNCH, steps: 50, targets: {
      left_hip_pitch_joint: 0.1, left_knee_joint: 0.05, left_ankle_pitch_joint: 0.05,
      right_hip_pitch_joint: 0.1, right_knee_joint: 0.05, right_ankle_pitch_joint: 0.05,
    }},
    { phase: AIR, steps: 200, targets: {
      left_hip_pitch_joint: -0.5, left_knee_joint: 1.0, left_ankle_pitch_joint: -0.5,
      right_hip_pitch_joint: -0.5, right_knee_joint: 1.0, right_ankle_pitch_joint: -0.5,
    }},
    { phase: LAND, steps: 60, targets: {
      left_hip_pitch_joint: -0.4, left_knee_joint: 0.8, left_ankle_pitch_joint: -0.4,
      right_hip_pitch_joint: -0.4, right_knee_joint: 0.8, right_ankle_pitch_joint: -0.4,
    }},
    { phase: RECOVER, steps: 120, targets: {} },
  ],
  kick: [
    { phase: CROUCH, steps: 60, targets: {
      left_hip_roll_joint: -0.1, left_hip_pitch_joint: -0.3, left_knee_joint: 0.6, left_ankle_pitch_joint: -0.3,
      right_hip_pitch_joint: -0.4, right_knee_joint: 0.8,
    }},
    { phase: LAUNCH, steps: 40, targets: {
      left_hip_roll_joint: -0.12,
      right_hip_pitch_joint: -1.2, right_knee_joint: 0.15, right_ankle_pitch_joint: -0.05,
    }},
    { phase: AIR, steps: 100, gains: { kp: 1.2, kd: 1.0 }, targets: {
      left_hip_roll_joint: -0.12,
      right_hip_pitch_joint: -1.2, right_knee_joint: 0.15, right_ankle_pitch_joint: -0.05,
    }},
    { phase: LAND, steps: 60, targets: {
      right_hip_pitch_joint: -0.3, right_knee_joint: 0.6, right_ankle_pitch_joint: -0.3,
    }},
    { phase: RECOVER, steps: 100, targets: {} },
  ],
  wave: [
    { phase: CROUCH, steps: 50, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.6, right_elbow_joint: 1.2,
    }},
    { phase: LAUNCH, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.1, right_elbow_joint: 0.8,
    }},
    { phase: AIR, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.6, right_elbow_joint: 1.2,
    }},
    { phase: LAND, steps: 30, gains: { kp: 1, kd: 1 }, targets: {
      right_shoulder_pitch_joint: -1.5, right_shoulder_roll_joint: -0.1, right_elbow_joint: 0.8,
    }},
    { phase: RECOVER, steps: 80, gains: { kp: 1, kd: 1 }, targets: {} },
  ],
  bow: [
    { phase: CROUCH, steps: 80, gains: { kp: 1.2, kd: 1 }, targets: {
      waist_pitch_joint: 0.4,
      left_hip_pitch_joint: -0.5, right_hip_pitch_joint: -0.5,
      left_shoulder_pitch_joint: 0.4, right_shoulder_pitch_joint: 0.4,
      left_shoulder_roll_joint: 0.15, right_shoulder_roll_joint: -0.15,
    }},
    { phase: LAUNCH, steps: 100, gains: { kp: 1.2, kd: 1 }, targets: {
      waist_pitch_joint: 0.4,
      left_hip_pitch_joint: -0.5, right_hip_pitch_joint: -0.5,
      left_shoulder_pitch_joint: 0.4, right_shoulder_pitch_joint: 0.4,
      left_shoulder_roll_joint: 0.15, right_shoulder_roll_joint: -0.15,
    }},
    { phase: AIR, steps: 40, gains: { kp: 1, kd: 1 }, targets: {} },
    { phase: LAND, steps: 40, gains: { kp: 1, kd: 1 }, targets: {} },
    { phase: RECOVER, steps: 80, targets: {} },
  ],
};

export class G1CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 1.4;
    this.phase = 0;

    // Amplitudes
    this.hipPitchAmp = 0.20;
    this.kneeAmp = 0.30;
    this.anklePitchAmp = 0.12;
    this.ankleRollAmp = 0.02;
    this.hipRollAmp = 0.03;

    // Arm swing
    this.armSwingGain = 0.25;

    // Waist balance gains
    this.waistPitchGain = 0.15;
    this.waistRollGain = 0.10;

    // Balance
    this.balanceKp = 100.0;
    this.balanceKd = 8.0;

    // Home pose
    this.homeQpos = {
      left_hip_pitch_joint: -0.2, left_hip_roll_joint: 0, left_hip_yaw_joint: 0,
      left_knee_joint: 0.4, left_ankle_pitch_joint: -0.2, left_ankle_roll_joint: 0,
      right_hip_pitch_joint: -0.2, right_hip_roll_joint: 0, right_hip_yaw_joint: 0,
      right_knee_joint: 0.4, right_ankle_pitch_joint: -0.2, right_ankle_roll_joint: 0,
      waist_yaw_joint: 0, waist_roll_joint: 0, waist_pitch_joint: 0,
      left_shoulder_pitch_joint: 0, left_shoulder_roll_joint: 0.3,
      left_shoulder_yaw_joint: 0, left_elbow_joint: 0.5,
      left_wrist_roll_joint: 0, left_wrist_pitch_joint: 0, left_wrist_yaw_joint: 0,
      right_shoulder_pitch_joint: 0, right_shoulder_roll_joint: -0.3,
      right_shoulder_yaw_joint: 0, right_elbow_joint: 0.5,
      right_wrist_roll_joint: 0, right_wrist_pitch_joint: 0, right_wrist_yaw_joint: 0,
    };

    // Actuator index map
    this.actIdx = {
      left_hip_pitch_joint: 0, left_hip_roll_joint: 1, left_hip_yaw_joint: 2,
      left_knee_joint: 3, left_ankle_pitch_joint: 4, left_ankle_roll_joint: 5,
      right_hip_pitch_joint: 6, right_hip_roll_joint: 7, right_hip_yaw_joint: 8,
      right_knee_joint: 9, right_ankle_pitch_joint: 10, right_ankle_roll_joint: 11,
      waist_yaw_joint: 12, waist_roll_joint: 13, waist_pitch_joint: 14,
      left_shoulder_pitch_joint: 15, left_shoulder_roll_joint: 16,
      left_shoulder_yaw_joint: 17, left_elbow_joint: 18,
      left_wrist_roll_joint: 19, left_wrist_pitch_joint: 20, left_wrist_yaw_joint: 21,
      right_shoulder_pitch_joint: 22, right_shoulder_roll_joint: 23,
      right_shoulder_yaw_joint: 24, right_elbow_joint: 25,
      right_wrist_roll_joint: 26, right_wrist_pitch_joint: 27, right_wrist_yaw_joint: 28,
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
    this.qwopDelta = 0.4;
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

  setStandingPose() {
    for (const [name, target] of Object.entries(this.homeQpos)) {
      const idx = this.jntIdx[name];
      if (idx !== undefined) this.data.qpos[idx] = target;
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
    if (name.includes('knee')) return [250, 15];
    if (name.includes('ankle_pitch')) return [70, 5];
    if (name.includes('ankle_roll')) return [70, 5];
    if (name.includes('hip')) return [150, 10];
    if (name.includes('waist')) return [150, 8];
    if (name.includes('wrist')) return [6, 0.5];
    return [20, 1.5]; // shoulder/elbow
  }

  // ── Tricks ─────────────────────────────────────────────────────────
  triggerTrick(name) {
    if (this.trickPhase !== IDLE || !TRICKS[name]) return;
    this.trickName = name;
    this.trickIdx = 0;
    this.trickStep = 0;
    this.trickPhase = TRICKS[name][0].phase;
    this._snapshotCurrentPose();
    console.log(`G1 trick: ${name}`);
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
      if (z < 0.7 && vz < -0.1) {
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
    if (k['KeyI']) targets.left_ankle_pitch_joint = this.homeQpos.left_ankle_pitch_joint + d * 0.4;
    if (k['KeyO']) targets.right_ankle_pitch_joint = this.homeQpos.right_ankle_pitch_joint + d * 0.4;
    if (k['KeyK']) targets.left_shoulder_pitch_joint = this.homeQpos.left_shoulder_pitch_joint - d * 1.5;
    if (k['KeyL']) targets.right_shoulder_pitch_joint = this.homeQpos.right_shoulder_pitch_joint - d * 1.5;
    if (k['KeyZ']) targets.waist_yaw_joint = this.homeQpos.waist_yaw_joint + d * 0.5;
    if (k['KeyX']) targets.waist_yaw_joint = this.homeQpos.waist_yaw_joint - d * 0.5;

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

    const hipKp = 150, hipKd = 10;
    const kneeKp = 250, kneeKd = 15;
    const ankleKp = 70, ankleKd = 5;
    const waistKp = 150, waistKd = 8;
    const armKp = 20, armKd = 1.5;

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

    ctrl[this.actIdx.left_hip_pitch_joint] = this.pdTorque('left_hip_pitch_joint', lHipPitch, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll_joint] = this.pdTorque('left_hip_roll_joint', lHipRoll, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_yaw_joint] = this.pdTorque('left_hip_yaw_joint', lHipYaw, hipKp, hipKd);
    ctrl[this.actIdx.left_knee_joint] = this.pdTorque('left_knee_joint', lKnee, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle_pitch_joint] = this.pdTorque('left_ankle_pitch_joint', lAnkleP, ankleKp, ankleKd);
    ctrl[this.actIdx.left_ankle_roll_joint] = this.pdTorque('left_ankle_roll_joint', lAnkleR, ankleKp, ankleKd);

    // --- RIGHT LEG ---
    const rS = Math.sin(rightPhase);
    const rSt = Math.max(0, -Math.sin(rightPhase));

    const rHipPitch = this.homeQpos.right_hip_pitch_joint + direction * this.hipPitchAmp * ampScale * rS;
    const rKnee = this.homeQpos.right_knee_joint + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rAnkleP = this.homeQpos.right_ankle_pitch_joint - this.anklePitchAmp * ampScale * rS;
    const rAnkleR = this.homeQpos.right_ankle_roll_joint + this.ankleRollAmp * rSt;
    const rHipRoll = this.homeQpos.right_hip_roll_joint + this.hipRollAmp * rSt + this.lateralSpeed * 0.12;
    const rHipYaw = this.homeQpos.right_hip_yaw_joint + this.turnRate * 0.10 * rS;

    ctrl[this.actIdx.right_hip_pitch_joint] = this.pdTorque('right_hip_pitch_joint', rHipPitch, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll_joint] = this.pdTorque('right_hip_roll_joint', rHipRoll, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_yaw_joint] = this.pdTorque('right_hip_yaw_joint', rHipYaw, hipKp, hipKd);
    ctrl[this.actIdx.right_knee_joint] = this.pdTorque('right_knee_joint', rKnee, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle_pitch_joint] = this.pdTorque('right_ankle_pitch_joint', rAnkleP, ankleKp, ankleKd);
    ctrl[this.actIdx.right_ankle_roll_joint] = this.pdTorque('right_ankle_roll_joint', rAnkleR, ankleKp, ankleKd);

    // --- WAIST (3 DOF) ---
    const waistYaw = this.homeQpos.waist_yaw_joint + this.turnRate * 0.15;
    const waistPitch = this.homeQpos.waist_pitch_joint - this.waistPitchGain * pitch;
    const waistRoll = this.homeQpos.waist_roll_joint - this.waistRollGain * roll;

    ctrl[this.actIdx.waist_yaw_joint] = this.pdTorque('waist_yaw_joint', waistYaw, waistKp, waistKd);
    ctrl[this.actIdx.waist_roll_joint] = this.pdTorque('waist_roll_joint', waistRoll, waistKp, waistKd);
    ctrl[this.actIdx.waist_pitch_joint] = this.pdTorque('waist_pitch_joint', waistPitch, waistKp, waistKd);

    // --- BALANCE CORRECTIONS ---
    const pCorr = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rCorr = -this.balanceKp * roll - this.balanceKd * rollRate;

    ctrl[this.actIdx.left_ankle_pitch_joint] += pCorr * 0.25;
    ctrl[this.actIdx.right_ankle_pitch_joint] += pCorr * 0.25;
    ctrl[this.actIdx.left_ankle_roll_joint] += rCorr * 0.2;
    ctrl[this.actIdx.right_ankle_roll_joint] -= rCorr * 0.2;
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
    for (const side of ['left', 'right']) {
      for (const axis of ['roll', 'pitch', 'yaw']) {
        const j = `${side}_wrist_${axis}_joint`;
        ctrl[this.actIdx[j]] = this.pdTorque(j, this.homeQpos[j], armKp * 0.3, armKd * 0.5);
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
