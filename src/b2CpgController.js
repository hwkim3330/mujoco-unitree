/**
 * CPG trot controller for Unitree B2.
 * Torque limits: hip/thigh ±200 Nm, knee ±300 Nm
 * Joint damping: 1.0, armature: 0.1
 * Mass: ~83kg
 */

export class B2CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    this.frequency = 2.0;
    this.phase = 0;

    this.thighAmp = 0.18;
    this.calfAmp = 0.18;
    this.hipAmp = 0.03;

    // PD gains — tuned for ±200/300 Nm limits
    this.hipKp = 200;  this.hipKd = 8;
    this.thighKp = 250; this.thighKd = 12;
    this.calfKp = 400;  this.calfKd = 18;

    this.balanceKp = 120;
    this.balanceKd = 10;

    this.homeHip = 0;
    this.homeThigh = 1.28;
    this.homeCalf = -2.84;

    // B2 actuator order: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
    this.legs = {
      FR: { act: [0, 1, 2],   phase: 0,         side: -1, prefix: 'FR' },
      FL: { act: [3, 4, 5],   phase: Math.PI,    side: 1,  prefix: 'FL' },
      RR: { act: [6, 7, 8],   phase: Math.PI,    side: -1, prefix: 'RR' },
      RL: { act: [9, 10, 11], phase: 0,          side: 1,  prefix: 'RL' },
    };

    this.jntQpos = {};
    this.jntDof = {};
    this.findJointIndices();

    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
  }

  findJointIndices() {
    const jointNames = [
      'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
      'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
      'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
      'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
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
    return kp * (target - q) - kd * qdot;
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
      const swing = Math.sin(legPhase);
      const isSwing = swing > 0;
      const isFront = name.startsWith('F');
      const turnSign = isFront ? 1 : -1;

      const thighTarget = this.homeThigh
        - direction * this.thighAmp * ampScale * swing
        + this.turnRate * 0.10 * turnSign * leg.side;

      const calfTarget = this.homeCalf
        - this.calfAmp * ampScale * (isSwing ? Math.sin(legPhase) : 0);

      const hipTarget = this.homeHip
        + leg.side * this.hipAmp * (isSwing ? 1 : -1) * ampScale
        + this.lateralSpeed * 0.12 * leg.side
        + this.turnRate * 0.03 * turnSign;

      const hipJoint = `${leg.prefix}_hip_joint`;
      const thighJoint = `${leg.prefix}_thigh_joint`;
      const calfJoint = `${leg.prefix}_calf_joint`;

      ctrl[leg.act[0]] = this.pdTorque(hipJoint, hipTarget, this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(thighJoint, thighTarget, this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(calfJoint, calfTarget, this.calfKp, this.calfKd);

      ctrl[leg.act[1]] += pitchCorr * 0.15;
      ctrl[leg.act[0]] += rollCorr * 0.10 * leg.side;
    }

    if (this.model.actuator_ctrlrange) {
      for (let i = 0; i < this.model.nu; i++) {
        const lo = this.model.actuator_ctrlrange[i * 2];
        const hi = this.model.actuator_ctrlrange[i * 2 + 1];
        ctrl[i] = Math.max(lo, Math.min(hi, ctrl[i]));
      }
    }
  }

  reset() {
    this.phase = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
  }
}
