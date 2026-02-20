/**
 * CPG (Central Pattern Generator) trot controller for Unitree Go2.
 *
 * Go2 actuator layout (12 torque motors):
 *  0: FL_hip (abduction)   3: FR_hip    6: RL_hip    9: RR_hip
 *  1: FL_thigh (hip flex)   4: FR_thigh  7: RL_thigh  10: RR_thigh
 *  2: FL_calf  (knee)       5: FR_calf   8: RL_calf   11: RR_calf
 *
 * Trot gait: diagonal pairs in phase (FL+RR, FR+RL).
 *
 * Home keyframe: hip=0, thigh=0.9, calf=-1.8
 * Standing height: ~0.27m (keyframe) to ~0.33m (natural stance)
 */

export class Go2CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 2.5;        // Hz (trot cadence)
    this.phase = 0;

    // Amplitudes
    this.thighAmp = 0.25;        // Hip flexion swing
    this.calfAmp = 0.25;         // Knee bend during swing
    this.hipAmp = 0.04;          // Abduction for lateral balance

    // PD gains
    this.hipKp = 60;   this.hipKd = 2;
    this.thighKp = 80; this.thighKd = 3;
    this.calfKp = 100;  this.calfKd = 4;

    // Balance feedback
    this.balanceKp = 30;
    this.balanceKd = 3;

    // Home pose
    this.homeHip = 0;
    this.homeThigh = 0.9;
    this.homeCalf = -1.8;

    // Leg definitions: [hipActIdx, thighActIdx, calfActIdx, phaseOffset, side]
    // Trot: FL+RR (phase 0), FR+RL (phase π)
    this.legs = {
      FL: { act: [0, 1, 2],   phase: 0,         side: 1,  prefix: 'FL' },
      FR: { act: [3, 4, 5],   phase: Math.PI,    side: -1, prefix: 'FR' },
      RL: { act: [6, 7, 8],   phase: Math.PI,    side: 1,  prefix: 'RL' },
      RR: { act: [9, 10, 11], phase: 0,          side: -1, prefix: 'RR' },
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

    const ampScale = Math.abs(this.forwardSpeed);
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
        + direction * this.thighAmp * ampScale * swing
        + this.turnRate * 0.06 * turnSign * leg.side;

      // Calf (knee): bend more during swing, extend during stance
      const calfTarget = this.homeCalf
        - this.calfAmp * ampScale * (isSwing ? Math.sin(legPhase) : 0);

      // Hip (abduction): lateral balance + lateral movement
      const hipTarget = this.homeHip
        + leg.side * this.hipAmp * (isSwing ? 1 : -1) * ampScale
        + this.lateralSpeed * 0.08 * leg.side;

      // PD torques
      const hipJoint = `${leg.prefix}_hip_joint`;
      const thighJoint = `${leg.prefix}_thigh_joint`;
      const calfJoint = `${leg.prefix}_calf_joint`;

      ctrl[leg.act[0]] = this.pdTorque(hipJoint, hipTarget, this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(thighJoint, thighTarget, this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(calfJoint, calfTarget, this.calfKp, this.calfKd);

      // Balance corrections on thigh and hip
      ctrl[leg.act[1]] += pitchCorr * 0.2;
      ctrl[leg.act[0]] += rollCorr * 0.15 * leg.side;
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
