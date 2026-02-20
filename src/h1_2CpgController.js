/**
 * CPG (Central Pattern Generator) walking controller for Unitree H1-2 (v2).
 * Procedural gait — no ONNX policy needed.
 *
 * H1-2 actuator layout (27 torque motors):
 *   0: left_hip_yaw_joint       6: right_hip_yaw_joint     12: torso_joint
 *   1: left_hip_pitch_joint     7: right_hip_pitch_joint   13: left_shoulder_pitch_joint
 *   2: left_hip_roll_joint      8: right_hip_roll_joint    14: left_shoulder_roll_joint
 *   3: left_knee_joint          9: right_knee_joint        15: left_shoulder_yaw_joint
 *   4: left_ankle_pitch_joint  10: right_ankle_pitch_joint 16: left_elbow_joint
 *   5: left_ankle_roll_joint   11: right_ankle_roll_joint  17: left_wrist_roll_joint
 *                                                          18: left_wrist_pitch_joint
 *                                                          19: left_wrist_yaw_joint
 *                                                          20: right_shoulder_pitch_joint
 *                                                          21: right_shoulder_roll_joint
 *                                                          22: right_shoulder_yaw_joint
 *                                                          23: right_elbow_joint
 *                                                          24: right_wrist_roll_joint
 *                                                          25: right_wrist_pitch_joint
 *                                                          26: right_wrist_yaw_joint
 *
 * Key differences from H1 (v1):
 *   - 27 DOF (vs 19): 2-DOF ankles (pitch+roll), 3-DOF wrists per arm
 *   - Hip chain order: yaw -> pitch -> roll (H1 is yaw -> roll -> pitch)
 *   - All joint names have "_joint" suffix
 *   - Standing height: 1.03m
 *   - No keyframe — standing pose set procedurally
 *
 * Standing pose (home):
 *   left_leg:  [yaw=0, pitch=-0.4, roll=0, knee=0.8, ankle_pitch=-0.4, ankle_roll=0]
 *   right_leg: [yaw=0, pitch=-0.4, roll=0, knee=0.8, ankle_pitch=-0.4, ankle_roll=0]
 *   torso: 0
 *   left_arm:  [pitch=0, roll=0.2, yaw=0, elbow=-0.3, wrist_roll=0, wrist_pitch=0, wrist_yaw=0]
 *   right_arm: [pitch=0, roll=-0.2, yaw=0, elbow=-0.3, wrist_roll=0, wrist_pitch=0, wrist_yaw=0]
 */

export class H1_2CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 1.2;        // Hz (steps per second)
    this.phase = 0;              // Phase oscillator [0, 2*PI)

    // Amplitudes for leg joints
    this.hipPitchAmp = 0.25;     // Forward swing amplitude
    this.kneeAmp = 0.35;         // Knee bend during swing
    this.anklePitchAmp = 0.15;   // Ankle pitch compensation
    this.ankleRollAmp = 0.04;    // Ankle roll for lateral balance
    this.hipRollAmp = 0.03;      // Lateral balance via hip roll

    // Arm swing
    this.armSwingGain = 0.3;

    // PD gains for balance (applied as torque corrections)
    this.balanceKp = 300.0;
    this.balanceKd = 20.0;

    // Home pose (joint targets at stance) — all names have "_joint" suffix
    this.homeQpos = {
      left_hip_yaw_joint: 0,
      left_hip_pitch_joint: -0.4,
      left_hip_roll_joint: 0,
      left_knee_joint: 0.8,
      left_ankle_pitch_joint: -0.4,
      left_ankle_roll_joint: 0,

      right_hip_yaw_joint: 0,
      right_hip_pitch_joint: -0.4,
      right_hip_roll_joint: 0,
      right_knee_joint: 0.8,
      right_ankle_pitch_joint: -0.4,
      right_ankle_roll_joint: 0,

      torso_joint: 0,

      left_shoulder_pitch_joint: 0,
      left_shoulder_roll_joint: 0.2,
      left_shoulder_yaw_joint: 0,
      left_elbow_joint: -0.3,
      left_wrist_roll_joint: 0,
      left_wrist_pitch_joint: 0,
      left_wrist_yaw_joint: 0,

      right_shoulder_pitch_joint: 0,
      right_shoulder_roll_joint: -0.2,
      right_shoulder_yaw_joint: 0,
      right_elbow_joint: -0.3,
      right_wrist_roll_joint: 0,
      right_wrist_pitch_joint: 0,
      right_wrist_yaw_joint: 0,
    };

    // Actuator index map (from H1-2 XML actuator order)
    this.actIdx = {
      left_hip_yaw_joint: 0,
      left_hip_pitch_joint: 1,
      left_hip_roll_joint: 2,
      left_knee_joint: 3,
      left_ankle_pitch_joint: 4,
      left_ankle_roll_joint: 5,

      right_hip_yaw_joint: 6,
      right_hip_pitch_joint: 7,
      right_hip_roll_joint: 8,
      right_knee_joint: 9,
      right_ankle_pitch_joint: 10,
      right_ankle_roll_joint: 11,

      torso_joint: 12,

      left_shoulder_pitch_joint: 13,
      left_shoulder_roll_joint: 14,
      left_shoulder_yaw_joint: 15,
      left_elbow_joint: 16,
      left_wrist_roll_joint: 17,
      left_wrist_pitch_joint: 18,
      left_wrist_yaw_joint: 19,

      right_shoulder_pitch_joint: 20,
      right_shoulder_roll_joint: 21,
      right_shoulder_yaw_joint: 22,
      right_elbow_joint: 23,
      right_wrist_roll_joint: 24,
      right_wrist_pitch_joint: 25,
      right_wrist_yaw_joint: 26,
    };

    // Joint qpos indices (after freejoint: offset 7)
    this.jntIdx = {};
    this.findJointIndices();

    // Commands — start in standing mode (user presses W to walk)
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // State for balance
    this.prevPitch = 0;
    this.prevRoll = 0;
  }

  findJointIndices() {
    const names = Object.keys(this.actIdx);
    for (const name of names) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name); // mjOBJ_JOINT=3
        if (jid >= 0 && this.model.jnt_qposadr) {
          this.jntIdx[name] = this.model.jnt_qposadr[jid];
        }
      } catch (e) {
        // Will use fallback
      }
    }
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  /**
   * Extract trunk orientation from qpos quaternion.
   * Returns { pitch, roll } in radians.
   */
  getTrunkOrientation() {
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];

    // Euler angles from quaternion (ZYX convention)
    const sinp = 2 * (qw * qy - qz * qx);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
    const roll = Math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));

    return { pitch, roll };
  }

  /**
   * Compute PD torque for a joint to track a target position.
   * H1-2 uses torque actuators, so we compute: tau = kp*(target - q) - kd*qdot
   */
  pdTorque(jointName, target, kp, kd) {
    const idx = this.jntIdx[jointName];
    if (idx === undefined) return 0;

    // Find qvel index via jnt_dofadr
    let dofIdx;
    try {
      const jid = this.mujoco.mj_name2id(this.model, 3, jointName);
      dofIdx = this.model.jnt_dofadr[jid];
    } catch (e) {
      dofIdx = idx - 7 + 6; // fallback
    }

    const q = this.data.qpos[idx];
    const qdot = this.data.qvel[dofIdx] || 0;

    return kp * (target - q) - kd * qdot;
  }

  /**
   * Step the CPG and compute torques.
   * Called every physics step.
   */
  step() {
    if (!this.enabled) return;

    // Advance phase
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI; // Opposite leg

    // Scale amplitudes by forward command
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;

    // Get trunk orientation for balance
    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;

    // PD gains — must be high enough to resist gravity on 47kg robot
    // Knee gravity torque ~120 Nm, hip ~60 Nm
    const hipKp = 800, hipKd = 30;
    const kneeKp = 1200, kneeKd = 40;
    const anklePitchKp = 200, anklePitchKd = 10;
    const ankleRollKp = 100, ankleRollKd = 5;
    const torsoKp = 600, torsoKd = 25;
    const armKp = 50, armKd = 3;

    const ctrl = this.data.ctrl;

    // --- LEFT LEG ---
    const leftSwing = Math.sin(leftPhase);
    const leftStance = Math.max(0, -Math.sin(leftPhase)); // Positive during stance

    const leftHipPitchTarget = this.homeQpos.left_hip_pitch_joint
      + direction * this.hipPitchAmp * ampScale * leftSwing;
    const leftKneeTarget = this.homeQpos.left_knee_joint
      + this.kneeAmp * ampScale * Math.max(0, Math.sin(leftPhase));
    const leftAnklePitchTarget = this.homeQpos.left_ankle_pitch_joint
      - this.anklePitchAmp * ampScale * leftSwing;
    const leftAnkleRollTarget = this.homeQpos.left_ankle_roll_joint
      - this.ankleRollAmp * leftStance;  // Lean into stance leg via ankle roll
    const leftHipRollTarget = this.homeQpos.left_hip_roll_joint
      - this.hipRollAmp * leftStance  // Lean into stance leg
      + this.lateralSpeed * 0.05;
    const leftHipYawTarget = this.homeQpos.left_hip_yaw_joint
      + this.turnRate * 0.05 * leftSwing;

    ctrl[this.actIdx.left_hip_yaw_joint] = this.pdTorque('left_hip_yaw_joint', leftHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_pitch_joint] = this.pdTorque('left_hip_pitch_joint', leftHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll_joint] = this.pdTorque('left_hip_roll_joint', leftHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_knee_joint] = this.pdTorque('left_knee_joint', leftKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle_pitch_joint] = this.pdTorque('left_ankle_pitch_joint', leftAnklePitchTarget, anklePitchKp, anklePitchKd);
    ctrl[this.actIdx.left_ankle_roll_joint] = this.pdTorque('left_ankle_roll_joint', leftAnkleRollTarget, ankleRollKp, ankleRollKd);

    // --- RIGHT LEG ---
    const rightSwing = Math.sin(rightPhase);
    const rightStance = Math.max(0, -Math.sin(rightPhase));

    const rightHipPitchTarget = this.homeQpos.right_hip_pitch_joint
      + direction * this.hipPitchAmp * ampScale * rightSwing;
    const rightKneeTarget = this.homeQpos.right_knee_joint
      + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rightAnklePitchTarget = this.homeQpos.right_ankle_pitch_joint
      - this.anklePitchAmp * ampScale * rightSwing;
    const rightAnkleRollTarget = this.homeQpos.right_ankle_roll_joint
      + this.ankleRollAmp * rightStance;  // Lean into stance leg via ankle roll
    const rightHipRollTarget = this.homeQpos.right_hip_roll_joint
      + this.hipRollAmp * rightStance
      + this.lateralSpeed * 0.05;
    const rightHipYawTarget = this.homeQpos.right_hip_yaw_joint
      + this.turnRate * 0.05 * rightSwing;

    ctrl[this.actIdx.right_hip_yaw_joint] = this.pdTorque('right_hip_yaw_joint', rightHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_pitch_joint] = this.pdTorque('right_hip_pitch_joint', rightHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll_joint] = this.pdTorque('right_hip_roll_joint', rightHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_knee_joint] = this.pdTorque('right_knee_joint', rightKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle_pitch_joint] = this.pdTorque('right_ankle_pitch_joint', rightAnklePitchTarget, anklePitchKp, anklePitchKd);
    ctrl[this.actIdx.right_ankle_roll_joint] = this.pdTorque('right_ankle_roll_joint', rightAnkleRollTarget, ankleRollKp, ankleRollKd);

    // --- TORSO ---
    // Keep torso upright, with yaw control
    const torsoTarget = this.homeQpos.torso_joint + this.turnRate * 0.1;
    ctrl[this.actIdx.torso_joint] = this.pdTorque('torso_joint', torsoTarget, torsoKp, torsoKd);

    // --- BALANCE CORRECTIONS ---
    const pitchCorrection = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorrection = -this.balanceKp * roll - this.balanceKd * rollRate;

    // Ankle pitch strategy (primary): ankle torque shifts center of pressure
    ctrl[this.actIdx.left_ankle_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.right_ankle_pitch_joint] += pitchCorrection * 0.4;

    // Ankle roll strategy: lateral balance via 2-DOF ankles (H1-2 advantage)
    ctrl[this.actIdx.left_ankle_roll_joint] += rollCorrection * 0.25;
    ctrl[this.actIdx.right_ankle_roll_joint] -= rollCorrection * 0.25;

    // Hip strategy (secondary): hip torque for larger perturbations
    ctrl[this.actIdx.left_hip_pitch_joint] += pitchCorrection * 0.5;
    ctrl[this.actIdx.right_hip_pitch_joint] += pitchCorrection * 0.5;

    // Roll corrections via hip roll
    ctrl[this.actIdx.left_hip_roll_joint] += rollCorrection * 0.3;
    ctrl[this.actIdx.right_hip_roll_joint] -= rollCorrection * 0.3;

    // --- ARMS (opposite to legs) ---
    const leftArmSwing = -this.armSwingGain * direction * ampScale * leftSwing;
    const rightArmSwing = -this.armSwingGain * direction * ampScale * rightSwing;

    ctrl[this.actIdx.left_shoulder_pitch_joint] = this.pdTorque('left_shoulder_pitch_joint',
      this.homeQpos.left_shoulder_pitch_joint + leftArmSwing, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_roll_joint] = this.pdTorque('left_shoulder_roll_joint',
      this.homeQpos.left_shoulder_roll_joint, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_yaw_joint] = this.pdTorque('left_shoulder_yaw_joint',
      this.homeQpos.left_shoulder_yaw_joint, armKp * 0.5, armKd);
    ctrl[this.actIdx.left_elbow_joint] = this.pdTorque('left_elbow_joint',
      this.homeQpos.left_elbow_joint, armKp, armKd);

    ctrl[this.actIdx.right_shoulder_pitch_joint] = this.pdTorque('right_shoulder_pitch_joint',
      this.homeQpos.right_shoulder_pitch_joint + rightArmSwing, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_roll_joint] = this.pdTorque('right_shoulder_roll_joint',
      this.homeQpos.right_shoulder_roll_joint, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_yaw_joint] = this.pdTorque('right_shoulder_yaw_joint',
      this.homeQpos.right_shoulder_yaw_joint, armKp * 0.5, armKd);
    ctrl[this.actIdx.right_elbow_joint] = this.pdTorque('right_elbow_joint',
      this.homeQpos.right_elbow_joint, armKp, armKd);

    // --- WRISTS (held at zero) ---
    const wristKp = armKp * 0.3;
    const wristKd = armKd * 0.5;

    ctrl[this.actIdx.left_wrist_roll_joint] = this.pdTorque('left_wrist_roll_joint',
      this.homeQpos.left_wrist_roll_joint, wristKp, wristKd);
    ctrl[this.actIdx.left_wrist_pitch_joint] = this.pdTorque('left_wrist_pitch_joint',
      this.homeQpos.left_wrist_pitch_joint, wristKp, wristKd);
    ctrl[this.actIdx.left_wrist_yaw_joint] = this.pdTorque('left_wrist_yaw_joint',
      this.homeQpos.left_wrist_yaw_joint, wristKp, wristKd);

    ctrl[this.actIdx.right_wrist_roll_joint] = this.pdTorque('right_wrist_roll_joint',
      this.homeQpos.right_wrist_roll_joint, wristKp, wristKd);
    ctrl[this.actIdx.right_wrist_pitch_joint] = this.pdTorque('right_wrist_pitch_joint',
      this.homeQpos.right_wrist_pitch_joint, wristKp, wristKd);
    ctrl[this.actIdx.right_wrist_yaw_joint] = this.pdTorque('right_wrist_yaw_joint',
      this.homeQpos.right_wrist_yaw_joint, wristKp, wristKd);

    // Clamp all controls to their ranges
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
