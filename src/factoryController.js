/**
 * Factory controller — manages multiple Go2 CPG controllers in a single MuJoCo world.
 * Each robot has independent gait phase, balance, and command inputs.
 *
 * Usage:
 *   const factory = new FactoryController(mujoco, model, data, 9);
 *   factory.enabled = true;
 *   factory.setCommand(0.5, 0, 0); // all walk forward
 *   // In loop: factory.step();
 */

class SingleRobotCPG {
  constructor(mujoco, model, data, prefix, index) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.prefix = prefix;
    this.index = index;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 2.5;
    this.phase = Math.random() * Math.PI * 2; // Randomize for variety

    // Amplitudes
    this.thighAmp = 0.25;
    this.calfAmp = 0.25;
    this.hipAmp = 0.04;

    // PD gains — tuned for Go2 ±23.7 Nm (hip/thigh), ±45.43 Nm (knee)
    this.hipKp = 40;   this.hipKd = 2;
    this.thighKp = 45;  this.thighKd = 3;
    this.calfKp = 80;   this.calfKd = 5;

    // Balance — gentle to avoid torque saturation
    this.balanceKp = 30;
    this.balanceKd = 3;

    // Home pose
    this.homeHip = 0;
    this.homeThigh = 0.9;
    this.homeCalf = -1.8;

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Balance state
    this.prevPitch = 0;
    this.prevRoll = 0;

    // Indices (found dynamically)
    this.baseQposAddr = 0;
    this.actBase = index * 12;
    this.jntQpos = {};
    this.jntDof = {};

    this.findIndices();
  }

  findIndices() {
    const p = this.prefix;

    // Find freejoint for base body orientation
    try {
      const fjId = this.mujoco.mj_name2id(this.model, 3, `${p}freejoint`);
      if (fjId >= 0) {
        this.baseQposAddr = this.model.jnt_qposadr[fjId];
      }
    } catch (e) { /* ignore */ }

    // Find hinge joints
    const suffixes = [
      'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
      'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
      'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
      'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    ];

    for (const s of suffixes) {
      const name = `${p}${s}`;
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0) {
          this.jntQpos[name] = this.model.jnt_qposadr[jid];
          this.jntDof[name] = this.model.jnt_dofadr[jid];
        }
      } catch (e) { /* ignore */ }
    }
  }

  getTrunkOrientation() {
    const a = this.baseQposAddr;
    const qw = this.data.qpos[a + 3];
    const qx = this.data.qpos[a + 4];
    const qy = this.data.qpos[a + 5];
    const qz = this.data.qpos[a + 6];

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
    return kp * (target - q) - kd * qdot;
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  step() {
    if (!this.enabled) return;

    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    const fwdMag = Math.abs(this.forwardSpeed);
    const latMag = Math.abs(this.lateralSpeed);
    const turnMag = Math.abs(this.turnRate);
    const anyCommand = fwdMag > 0.05 || latMag > 0.05 || turnMag > 0.05;
    const ampScale = anyCommand
      ? Math.max(0.3, fwdMag + latMag * 0.5 + turnMag * 0.3) : 0;
    const direction = Math.sign(this.forwardSpeed) || 1;

    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;

    const pitchCorr = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorr = -this.balanceKp * roll - this.balanceKd * rollRate;

    const ctrl = this.data.ctrl;
    const base = this.actBase;
    const p = this.prefix;

    // Legs: [actOffset, phaseOffset, side, legName, isFront]
    const legs = [
      [0, 0,         1,  'FL', true],
      [3, Math.PI,  -1,  'FR', true],
      [6, Math.PI,   1,  'RL', false],
      [9, 0,        -1,  'RR', false],
    ];

    for (const [actOff, legPhaseOff, side, legName, isFront] of legs) {
      const legPhase = this.phase + legPhaseOff;
      const swing = Math.sin(legPhase);
      const isSwing = swing > 0;

      const turnSign = isFront ? 1 : -1;

      const thighTarget = this.homeThigh
        - direction * this.thighAmp * ampScale * swing
        + this.turnRate * 0.15 * turnSign * side;

      const calfTarget = this.homeCalf
        - this.calfAmp * ampScale * (isSwing ? Math.sin(legPhase) : 0);

      const hipTarget = this.homeHip
        + side * this.hipAmp * (isSwing ? 1 : -1) * ampScale
        + this.lateralSpeed * 0.25 * side;

      const hipJoint = `${p}${legName}_hip_joint`;
      const thighJoint = `${p}${legName}_thigh_joint`;
      const calfJoint = `${p}${legName}_calf_joint`;

      ctrl[base + actOff + 0] = this.pdTorque(hipJoint, hipTarget, this.hipKp, this.hipKd);
      ctrl[base + actOff + 1] = this.pdTorque(thighJoint, thighTarget, this.thighKp, this.thighKd);
      ctrl[base + actOff + 2] = this.pdTorque(calfJoint, calfTarget, this.calfKp, this.calfKd);

      // Balance corrections
      ctrl[base + actOff + 1] += pitchCorr * 0.2;
      ctrl[base + actOff + 0] += rollCorr * 0.15 * side;
    }

    // Clamp to actuator ranges
    const ctrlRange = this.model.actuator_ctrlrange;
    if (ctrlRange) {
      for (let i = 0; i < 12; i++) {
        const idx = base + i;
        const lo = ctrlRange[idx * 2];
        const hi = ctrlRange[idx * 2 + 1];
        ctrl[idx] = Math.max(lo, Math.min(hi, ctrl[idx]));
      }
    }
  }

  reset() {
    this.phase = Math.random() * Math.PI * 2;
    this.prevPitch = 0;
    this.prevRoll = 0;
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
  }
}

export class FactoryController {
  constructor(mujoco, model, data, numRobots) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.numRobots = numRobots;
    this.enabled = false;
    this.robots = [];

    // Center robot body ID for camera follow
    const centerIdx = Math.floor(numRobots / 2);
    try {
      this.centerBodyId = mujoco.mj_name2id(model, 1, `r${centerIdx}_base`);
    } catch (e) {
      this.centerBodyId = 1;
    }

    for (let i = 0; i < numRobots; i++) {
      const robot = new SingleRobotCPG(mujoco, model, data, `r${i}_`, i);
      this.robots.push(robot);
    }
  }

  setCommand(forward, lateral, turn) {
    for (const robot of this.robots) {
      robot.setCommand(forward, lateral, turn);
    }
  }

  step() {
    if (!this.enabled) return;
    for (const robot of this.robots) {
      robot.enabled = true;
      robot.step();
    }
  }

  reset() {
    for (const robot of this.robots) {
      robot.reset();
    }
  }
}
