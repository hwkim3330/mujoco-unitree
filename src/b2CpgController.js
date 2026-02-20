/**
 * CPG (Central Pattern Generator) trot controller for Unitree B2.
 *
 * B2 actuator layout (12 torque motors):
 *  0: FR_hip (abduction)   3: FL_hip    6: RR_hip    9: RL_hip
 *  1: FR_thigh (hip flex)   4: FL_thigh  7: RR_thigh  10: RL_thigh
 *  2: FR_calf  (knee)       5: FL_calf   8: RR_calf   11: RL_calf
 *
 * Trot gait: diagonal pairs in phase (FR+RL, FL+RR).
 *
 * Home keyframe: hip=0, thigh=1.28, calf=-2.84
 * Standing height: ~0.5m
 * Robot mass: ~83kg — requires significantly higher PD gains than Go2.
 */

export class B2CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 2.0;        // Hz (slower trot cadence for large robot)
    this.phase = 0;

    // Amplitudes
    this.thighAmp = 0.2;         // Hip flexion swing
    this.calfAmp = 0.2;          // Knee bend during swing
    this.hipAmp = 0.03;          // Abduction for lateral balance

    // PD gains — B2 is ~83kg, needs much higher gains than Go2
    this.hipKp = 500;   this.hipKd = 15;
    this.thighKp = 800;  this.thighKd = 25;
    this.calfKp = 1000;  this.calfKd = 30;

    // Balance feedback
    this.balanceKp = 200;
    this.balanceKd = 15;

    // Home pose
    this.homeHip = 0;
    this.homeThigh = 1.28;
    this.homeCalf = -2.84;

    // Leg definitions: [hipActIdx, thighActIdx, calfActIdx, phaseOffset, side]
    // B2 actuator order: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
    // Trot: FR+RL (phase 0), FL+RR (phase pi)
    this.legs = {
      FR: { act: [0, 1, 2],   phase: 0,         side: -1, prefix: 'FR' },
      FL: { act: [3, 4, 5],   phase: Math.PI,    side: 1,  prefix: 'FL' },
      RR: { act: [6, 7, 8],   phase: Math.PI,    side: -1, prefix: 'RR' },
      RL: { act: [9, 10, 11], phase: 0,          side: 1,  prefix: 'RL' },
    };

    // Joint indices (qpos/qvel)
    this.jntQpos = {};
    this.jntDof = {};
    this.findJointIndices();

    // Commands
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // Balance state
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
    const qIdx = this.jntQpos[jointName];
    const dIdx = this.jntDof[jointName];
    if (qIdx === undefined) return 0;

    const q = this.data.qpos[qIdx];
    const qdot = this.data.qvel[dIdx] || 0;
    return kp * (target - q) - kd * qdot;
  }

  step() {
    if (!this.enabled) return;

    // Advance phase
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    // Compute effective amplitude — activate gait for ANY command
    const fwdMag = Math.abs(this.forwardSpeed);
    const latMag = Math.abs(this.lateralSpeed);
    const turnMag = Math.abs(this.turnRate);
    const anyCommand = fwdMag > 0.05 || latMag > 0.05 || turnMag > 0.05;
    const ampScale = anyCommand
      ? Math.max(0.3, fwdMag + latMag * 0.5 + turnMag * 0.3) : 0;
    const direction = Math.sign(this.forwardSpeed) || 1;

    // Trunk orientation for balance
    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;

    const pitchCorr = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorr = -this.balanceKp * roll - this.balanceKd * rollRate;

    const ctrl = this.data.ctrl;

    for (const [name, leg] of Object.entries(this.legs)) {
      const legPhase = this.phase + leg.phase;
      const swing = Math.sin(legPhase);
      const isSwing = swing > 0;

      // Front/back distinction for turn
      const isFront = name.startsWith('F');
      const turnSign = isFront ? 1 : -1;

      // Thigh (hip flexion): swing forward/back
      const thighTarget = this.homeThigh
        - direction * this.thighAmp * ampScale * swing
        + this.turnRate * 0.12 * turnSign * leg.side;

      // Calf (knee): bend during swing + stance push-off
      const swingLift = isSwing ? Math.sin(legPhase) : 0;
      const stancePush = !isSwing ? Math.sin(legPhase + Math.PI) * 0.06 : 0;
      const calfTarget = this.homeCalf
        - this.calfAmp * ampScale * swingLift
        + stancePush * ampScale;

      // Hip (abduction): lateral movement + balance
      const hipTarget = this.homeHip
        + leg.side * this.hipAmp * (isSwing ? 1 : -1) * ampScale
        + this.lateralSpeed * 0.2 * leg.side
        + this.turnRate * 0.03 * turnSign;

      // PD torques
      const hipJoint = `${leg.prefix}_hip_joint`;
      const thighJoint = `${leg.prefix}_thigh_joint`;
      const calfJoint = `${leg.prefix}_calf_joint`;

      ctrl[leg.act[0]] = this.pdTorque(hipJoint, hipTarget, this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(thighJoint, thighTarget, this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(calfJoint, calfTarget, this.calfKp, this.calfKd);

      // Balance corrections
      ctrl[leg.act[1]] += pitchCorr * 0.2;
      ctrl[leg.act[0]] += rollCorr * 0.15 * leg.side;
      ctrl[leg.act[2]] += pitchCorr * 0.08;
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

  reset() {
    this.phase = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
  }
}
