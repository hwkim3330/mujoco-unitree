/**
 * CPG (Central Pattern Generator) walking controller for Unitree H1-2 (v2).
 * Procedural gait — no ONNX policy needed.
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
 */

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

    // Balance — ankle_pitch ±60, ankle_roll ±40 Nm
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

  step() {
    if (!this.enabled) return;

    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI;

    // Activate gait for ANY command
    const fwdMag = Math.abs(this.forwardSpeed);
    const latMag = Math.abs(this.lateralSpeed);
    const turnMag = Math.abs(this.turnRate);
    const anyCommand = fwdMag > 0.05 || latMag > 0.05 || turnMag > 0.05;
    const ampScale = anyCommand
      ? Math.max(0.25, fwdMag + latMag * 0.4 + turnMag * 0.3) : 0;
    const direction = Math.sign(this.forwardSpeed) || 1;

    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;

    // PD gains — tuned for ±200 (hip), ±300 (knee), ±60 (ankle_pitch), ±40 (ankle_roll)
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

    // Ankle is the bottleneck — keep corrections small
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
  }
}
