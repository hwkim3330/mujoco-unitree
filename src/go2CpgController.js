/**
 * CPG (Central Pattern Generator) trot controller for Unitree Go2.
 *
 * Go2 actuator layout (12 torque motors):
 *  0: FL_hip (abduction)   3: FR_hip    6: RL_hip    9: RR_hip
 *  1: FL_thigh (hip flex)   4: FR_thigh  7: RL_thigh  10: RR_thigh
 *  2: FL_calf  (knee)       5: FR_calf   8: RL_calf   11: RR_calf
 *
 * Torque limits: hip/thigh ±23.7 Nm, knee ±45.43 Nm
 * Joint damping: 2.0 (high passive damping)
 * Mass: ~15.2 kg
 * Home: hip=0, thigh=0.9, calf=-1.8, standing z≈0.27m
 */

// ── Trick phase constants ────────────────────────────────────────────
const IDLE    = 'idle';
const CROUCH  = 'crouch';
const LAUNCH  = 'launch';
const AIR     = 'air';
const LAND    = 'land';
const RECOVER = 'recover';

// Gain scaling per trick phase
const GAIN_SCALE = {
  [IDLE]:    { kp: 1.0, kd: 1.0 },
  [CROUCH]:  { kp: 1.5, kd: 1.0 },
  [LAUNCH]:  { kp: 2.0, kd: 0.5 },
  [AIR]:     { kp: 0.3, kd: 0.5 },
  [LAND]:    { kp: 0.8, kd: 2.5 },
  [RECOVER]: { kp: 1.0, kd: 1.0 },
};

// ── Trick definitions ────────────────────────────────────────────────
// Each phase: { phase, steps, targets: { FL:[h,t,c], FR, RL, RR } }
const HOME = [0, 0.9, -1.8];

const TRICKS = {
  jump: [
    { phase: CROUCH, steps: 60, targets: {
      FL: [0, 1.4, -2.6], FR: [0, 1.4, -2.6], RL: [0, 1.4, -2.6], RR: [0, 1.4, -2.6] } },
    { phase: LAUNCH, steps: 40, targets: {
      FL: [0, 0.3, -0.85], FR: [0, 0.3, -0.85], RL: [0, 0.3, -0.85], RR: [0, 0.3, -0.85] } },
    { phase: AIR, steps: 150, targets: {
      FL: [0, 1.2, -2.5], FR: [0, 1.2, -2.5], RL: [0, 1.2, -2.5], RR: [0, 1.2, -2.5] } },
    { phase: LAND, steps: 50, targets: {
      FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4], RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4] } },
    { phase: RECOVER, steps: 100, targets: { FL: HOME, FR: HOME, RL: HOME, RR: HOME } },
  ],
  frontflip: [
    { phase: CROUCH, steps: 50, targets: {
      FL: [0, 1.5, -2.65], FR: [0, 1.5, -2.65], RL: [0, 1.3, -2.5], RR: [0, 1.3, -2.5] } },
    { phase: LAUNCH, steps: 40, targets: {
      FL: [0, 0.6, -1.0], FR: [0, 0.6, -1.0], RL: [0, -0.2, -0.85], RR: [0, -0.2, -0.85] } },
    { phase: AIR, steps: 225, targets: {
      FL: [0, 1.5, -2.7], FR: [0, 1.5, -2.7], RL: [0, 1.5, -2.7], RR: [0, 1.5, -2.7] } },
    { phase: LAND, steps: 50, targets: {
      FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4], RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4] } },
    { phase: RECOVER, steps: 100, targets: { FL: HOME, FR: HOME, RL: HOME, RR: HOME } },
  ],
  backflip: [
    { phase: CROUCH, steps: 50, targets: {
      FL: [0, 1.3, -2.5], FR: [0, 1.3, -2.5], RL: [0, 1.5, -2.65], RR: [0, 1.5, -2.65] } },
    { phase: LAUNCH, steps: 40, targets: {
      FL: [0, -0.5, -0.85], FR: [0, -0.5, -0.85], RL: [0, 0.6, -1.0], RR: [0, 0.6, -1.0] } },
    { phase: AIR, steps: 225, targets: {
      FL: [0, 1.5, -2.7], FR: [0, 1.5, -2.7], RL: [0, 1.5, -2.7], RR: [0, 1.5, -2.7] } },
    { phase: LAND, steps: 50, targets: {
      FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4], RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4] } },
    { phase: RECOVER, steps: 100, targets: { FL: HOME, FR: HOME, RL: HOME, RR: HOME } },
  ],
  sideroll: [
    { phase: CROUCH, steps: 50, targets: {
      FL: [0, 1.4, -2.6], FR: [0, 1.4, -2.6], RL: [0, 1.4, -2.6], RR: [0, 1.4, -2.6] } },
    { phase: LAUNCH, steps: 40, targets: {
      FL: [0.8, 0.3, -0.85], FR: [-0.8, 0.6, -1.2], RL: [0.8, 0.3, -0.85], RR: [-0.8, 0.6, -1.2] } },
    { phase: AIR, steps: 200, targets: {
      FL: [0, 1.2, -2.5], FR: [0, 1.2, -2.5], RL: [0, 1.2, -2.5], RR: [0, 1.2, -2.5] } },
    { phase: LAND, steps: 50, targets: {
      FL: [0, 1.3, -2.4], FR: [0, 1.3, -2.4], RL: [0, 1.3, -2.4], RR: [0, 1.3, -2.4] } },
    { phase: RECOVER, steps: 100, targets: { FL: HOME, FR: HOME, RL: HOME, RR: HOME } },
  ],
};

// Gait constants
const DUTY_FACTOR = 0.6;
const SWING_HEIGHT = 0.35;
const BASE_FREQ = 1.8;
const MAX_FREQ = 3.3;

export class Go2CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait phase
    this.phase = 0;

    // Amplitudes
    this.thighAmp = 0.25;
    this.calfAmp = 0.20;
    this.hipAmp = 0.03;

    // PD gains (base values, scaled by trick phase)
    this.hipKp = 40;   this.hipKd = 2;
    this.thighKp = 45;  this.thighKd = 3;
    this.calfKp = 80;   this.calfKd = 5;

    // Balance feedback
    this.balanceKp = 30;
    this.balanceKd = 3;

    // Home pose
    this.homeHip = 0;
    this.homeThigh = 0.9;
    this.homeCalf = -1.8;

    // Trot: FL+RR (phase 0), FR+RL (phase π)
    this.legs = {
      FL: { act: [0, 1, 2],   phase: 0,         side: 1,  prefix: 'FL' },
      FR: { act: [3, 4, 5],   phase: Math.PI,    side: -1, prefix: 'FR' },
      RL: { act: [6, 7, 8],   phase: Math.PI,    side: 1,  prefix: 'RL' },
      RR: { act: [9, 10, 11], phase: 0,          side: -1, prefix: 'RR' },
    };

    this.jntQpos = {};
    this.jntDof = {};
    this.findJointIndices();

    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;

    // ── Trick state machine ──────────────────────────────────────────
    this.trickPhase = IDLE;
    this.trickName = null;
    this.trickIdx = 0;       // current phase index in trick sequence
    this.trickStep = 0;      // steps within current phase
    this.trickStart = {};     // snapshot of targets at phase start { FL:[h,t,c], ... }

    // Foot contact detection
    this.footGeomIds = {};
    this.floorBodyId = -1;
    this.findFootGeoms();

    // ── QWOP manual mode ─────────────────────────────────────────────
    // Key map: left hand = front legs, right hand = rear legs
    //   Q/A = FL thigh/calf,  W/S = FR thigh/calf
    //   I/K = RL thigh/calf,  O/L = RR thigh/calf
    // Hold key = swing joint away from home, release = spring back
    this.qwopMode = false;
    this.qwopKeys = {};  // set externally from main.js key state
    this.qwopThighDelta = 0.6;  // rad displacement per key press
    this.qwopCalfDelta = 0.8;
  }

  findJointIndices() {
    const jointNames = [
      'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
      'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
      'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
      'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    ];
    for (const name of jointNames) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0) {
          this.jntQpos[name] = this.model.jnt_qposadr[jid];
          this.jntDof[name] = this.model.jnt_dofadr[jid];
        }
      } catch (e) { /* ignore */ }
    }
  }

  findFootGeoms() {
    for (const name of ['FL', 'FR', 'RL', 'RR']) {
      try {
        const gid = this.mujoco.mj_name2id(this.model, 5, name);
        if (gid >= 0) this.footGeomIds[name] = gid;
      } catch (e) { /* ignore */ }
    }
    try {
      const fgid = this.mujoco.mj_name2id(this.model, 5, 'floor');
      if (fgid >= 0) this.floorBodyId = this.model.geom_bodyid[fgid];
    } catch (e) { /* ignore */ }
  }

  getFootContacts() {
    const contacts = { FL: 0, FR: 0, RL: 0, RR: 0 };
    const ncon = this.data.ncon || 0;
    for (let c = 0; c < ncon; c++) {
      try {
        const ct = this.data.contact.get(c);
        const b1 = this.model.geom_bodyid[ct.geom1];
        const b2 = this.model.geom_bodyid[ct.geom2];
        for (const [name, gid] of Object.entries(this.footGeomIds)) {
          const fb = this.model.geom_bodyid[gid];
          if ((b1 === fb || b2 === fb) && (this.floorBodyId < 0 || b1 === this.floorBodyId || b2 === this.floorBodyId)) {
            contacts[name] = 1;
          }
        }
      } catch (e) { break; }
    }
    return contacts;
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
    const qIdx = this.jntQpos[jointName];
    const dIdx = this.jntDof[jointName];
    if (qIdx === undefined) return 0;
    const q = this.data.qpos[qIdx];
    const qdot = this.data.qvel[dIdx] || 0;

    // Scale gains by trick phase
    const gs = GAIN_SCALE[this.trickPhase];
    return (kp * gs.kp) * (target - q) - (kd * gs.kd) * qdot;
  }

  // ── Trigger a trick ────────────────────────────────────────────────
  triggerTrick(name) {
    if (this.trickPhase !== IDLE) return;
    if (!TRICKS[name]) return;
    this.trickName = name;
    this.trickIdx = 0;
    this.trickStep = 0;
    this.trickPhase = TRICKS[name][0].phase;
    // Snapshot current joint positions as interpolation start
    this._snapshotCurrentPose();
    console.log(`Go2 trick: ${name}`);
  }

  _snapshotCurrentPose() {
    this.trickStart = {};
    for (const [name, leg] of Object.entries(this.legs)) {
      const h = this.jntQpos[`${leg.prefix}_hip_joint`];
      const t = this.jntQpos[`${leg.prefix}_thigh_joint`];
      const c = this.jntQpos[`${leg.prefix}_calf_joint`];
      this.trickStart[name] = [
        h !== undefined ? this.data.qpos[h] : this.homeHip,
        t !== undefined ? this.data.qpos[t] : this.homeThigh,
        c !== undefined ? this.data.qpos[c] : this.homeCalf,
      ];
    }
  }

  // ── Trick state machine step ───────────────────────────────────────
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

    const ctrl = this.data.ctrl;
    for (const [name, leg] of Object.entries(this.legs)) {
      const start = this.trickStart[name] || [this.homeHip, this.homeThigh, this.homeCalf];
      const end = spec.targets[name];
      const hipT   = start[0] + (end[0] - start[0]) * t;
      const thighT = start[1] + (end[1] - start[1]) * t;
      const calfT  = start[2] + (end[2] - start[2]) * t;

      ctrl[leg.act[0]] = this.pdTorque(`${leg.prefix}_hip_joint`, hipT, this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(`${leg.prefix}_thigh_joint`, thighT, this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(`${leg.prefix}_calf_joint`, calfT, this.calfKp, this.calfKd);
    }

    // Early landing detection during AIR
    if (spec.phase === AIR && this.trickStep > 30) {
      const fc = this.getFootContacts();
      if ((fc.FL + fc.FR + fc.RL + fc.RR) >= 2) {
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

  // ── Walking step (improved gait) ──────────────────────────────────
  _stepWalk() {
    const fwdMag = Math.abs(this.forwardSpeed);
    const latMag = Math.abs(this.lateralSpeed);
    const turnMag = Math.abs(this.turnRate);
    const speedMag = Math.sqrt(fwdMag * fwdMag + latMag * latMag);
    const anyCommand = fwdMag > 0.05 || latMag > 0.05 || turnMag > 0.05;

    // Speed-responsive frequency
    const freq = anyCommand ? BASE_FREQ + speedMag * (MAX_FREQ - BASE_FREQ) : BASE_FREQ;
    this.phase += 2 * Math.PI * freq * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    const ampScale = anyCommand ? Math.max(0.3, fwdMag + latMag * 0.5 + turnMag * 0.3) : 0;
    const direction = Math.sign(this.forwardSpeed) || 1;

    const { pitch, roll } = this.getTrunkOrientation();
    const rawPR = (pitch - this.prevPitch) / this.simDt;
    const rawRR = (roll - this.prevRoll) / this.simDt;
    const pitchRate = Math.max(-10, Math.min(10, rawPR));
    const rollRate = Math.max(-10, Math.min(10, rawRR));
    this.prevPitch = pitch;
    this.prevRoll = roll;

    const pitchCorr = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorr = -this.balanceKp * roll - this.balanceKd * rollRate;

    const ctrl = this.data.ctrl;

    for (const [name, leg] of Object.entries(this.legs)) {
      const legPhase = this.phase + leg.phase;
      const normalized = ((legPhase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / (2 * Math.PI);
      const isStance = normalized < DUTY_FACTOR;
      const isFront = name.startsWith('F');
      const isLeft = (leg.side === 1);
      const turnSign = isFront ? 1 : -1;

      // Differential turning
      let turnScale = 1.0;
      if (this.turnRate !== 0) {
        turnScale = isLeft ? 1.0 + this.turnRate * 0.5 : 1.0 - this.turnRate * 0.5;
        turnScale = Math.max(0.2, Math.min(2.0, turnScale));
      }

      let hipTarget, thighTarget, calfTarget;

      if (!anyCommand) {
        hipTarget = this.homeHip;
        thighTarget = this.homeThigh;
        calfTarget = this.homeCalf;
      } else if (isStance) {
        // Stance: foot pushes backward
        const sp = normalized / DUTY_FACTOR;
        thighTarget = this.homeThigh + direction * this.thighAmp * ampScale * turnScale * (0.5 - sp);
        calfTarget = this.homeCalf - 0.1; // slight extra bend for push
        hipTarget = this.homeHip + this.lateralSpeed * 0.15 * leg.side * (0.5 - sp)
          + this.turnRate * 0.03 * turnSign;
      } else {
        // Swing: foot lifts with bell-curve, thigh swings forward
        const swp = (normalized - DUTY_FACTOR) / (1 - DUTY_FACTOR);
        const lift = Math.sin(swp * Math.PI) * SWING_HEIGHT;
        thighTarget = this.homeThigh + direction * this.thighAmp * ampScale * turnScale * (-0.5 + swp) - lift * 0.5;
        calfTarget = this.homeCalf - lift;
        hipTarget = this.homeHip + this.lateralSpeed * 0.15 * leg.side * (-0.5 + swp)
          + this.turnRate * 0.03 * turnSign;
      }

      ctrl[leg.act[0]] = this.pdTorque(`${leg.prefix}_hip_joint`, hipTarget, this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(`${leg.prefix}_thigh_joint`, thighTarget, this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(`${leg.prefix}_calf_joint`, calfTarget, this.calfKp, this.calfKd);

      // Balance corrections
      ctrl[leg.act[1]] += pitchCorr * 0.15;
      ctrl[leg.act[0]] += rollCorr * 0.10 * leg.side;
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

  // ── QWOP manual step ────────────────────────────────────────────────
  // Key mapping:
  //   Q = FL thigh fwd    W = FR thigh fwd
  //   A = FL calf extend   S = FR calf extend
  //   I = RL thigh fwd    O = RR thigh fwd
  //   K = RL calf extend   L = RR calf extend
  //   Z = all hips left    X = all hips right
  _stepQwop() {
    const k = this.qwopKeys;
    const td = this.qwopThighDelta;
    const cd = this.qwopCalfDelta;

    const targets = {
      FL: [
        this.homeHip + (k['KeyZ'] ? 0.3 : 0) + (k['KeyX'] ? -0.3 : 0),
        this.homeThigh - (k['KeyQ'] ? td : 0),
        this.homeCalf + (k['KeyA'] ? cd : 0),
      ],
      FR: [
        this.homeHip + (k['KeyZ'] ? 0.3 : 0) + (k['KeyX'] ? -0.3 : 0),
        this.homeThigh - (k['KeyW'] ? td : 0),
        this.homeCalf + (k['KeyS'] ? cd : 0),
      ],
      RL: [
        this.homeHip + (k['KeyZ'] ? 0.3 : 0) + (k['KeyX'] ? -0.3 : 0),
        this.homeThigh - (k['KeyI'] ? td : 0),
        this.homeCalf + (k['KeyK'] ? cd : 0),
      ],
      RR: [
        this.homeHip + (k['KeyZ'] ? 0.3 : 0) + (k['KeyX'] ? -0.3 : 0),
        this.homeThigh - (k['KeyO'] ? td : 0),
        this.homeCalf + (k['KeyL'] ? cd : 0),
      ],
    };

    const ctrl = this.data.ctrl;
    for (const [name, leg] of Object.entries(this.legs)) {
      const t = targets[name];
      ctrl[leg.act[0]] = this.pdTorque(`${leg.prefix}_hip_joint`, t[0], this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(`${leg.prefix}_thigh_joint`, t[1], this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(`${leg.prefix}_calf_joint`, t[2], this.calfKp, this.calfKd);
    }

    this._clampCtrl();
  }

  // ── Main step ──────────────────────────────────────────────────────
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
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this._endTrick();
  }
}
