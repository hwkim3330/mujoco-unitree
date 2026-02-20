/**
 * CPG (Central Pattern Generator) walking controller for Unitree G1.
 * Procedural gait — no ONNX policy needed.
 *
 * G1 actuator layout (29 torque motors):
 *   0: left_hip_pitch       6: right_hip_pitch     12: waist_yaw
 *   1: left_hip_roll        7: right_hip_roll      13: waist_roll
 *   2: left_hip_yaw         8: right_hip_yaw       14: waist_pitch
 *   3: left_knee            9: right_knee          15-21: left arm
 *   4: left_ankle_pitch    10: right_ankle_pitch   22-28: right arm
 *   5: left_ankle_roll     11: right_ankle_roll
 *
 * Left arm (15-21): shoulder P/R/Y, elbow, wrist R/P/Y
 * Right arm (22-28): shoulder P/R/Y, elbow, wrist R/P/Y
 *
 * No keyframe — G1 starts at q=0 (T-pose).
 * Standing pose set by controller:
 *   hip_pitch: -0.2, knee: 0.4, ankle_pitch: -0.2
 *   left_shoulder_roll: 0.3, right_shoulder_roll: -0.3
 *   elbow: 0.5, all others: 0
 *
 * G1 advantages over H1:
 *   - 2-DOF ankles (pitch + roll) for lateral balance
 *   - 3-DOF waist (yaw, roll, pitch) for trunk stabilization
 *   - Lighter (28kg vs 47kg) so lower PD gains
 */

export class G1CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 1.4;        // Hz — slightly faster cadence for shorter legs
    this.phase = 0;              // Phase oscillator [0, 2*PI)

    // Amplitudes for leg joints
    this.hipPitchAmp = 0.20;     // Forward swing amplitude (smaller than H1)
    this.kneeAmp = 0.30;         // Knee bend during swing
    this.anklePitchAmp = 0.12;   // Ankle pitch compensation
    this.ankleRollAmp = 0.02;    // Ankle roll for lateral balance
    this.hipRollAmp = 0.03;      // Lateral balance via hip roll

    // Arm swing
    this.armSwingGain = 0.25;

    // Waist balance gains
    this.waistPitchGain = 0.15;  // Forward lean compensation
    this.waistRollGain = 0.10;   // Lateral lean compensation

    // PD gains for balance (applied as torque corrections)
    this.balanceKp = 200.0;      // Lower than H1 — lighter robot
    this.balanceKd = 15.0;

    // Home pose (joint targets at stance — no keyframe in G1)
    this.homeQpos = {
      left_hip_pitch_joint: -0.2,   left_hip_roll_joint: 0,   left_hip_yaw_joint: 0,
      left_knee_joint: 0.4,         left_ankle_pitch_joint: -0.2,  left_ankle_roll_joint: 0,
      right_hip_pitch_joint: -0.2,  right_hip_roll_joint: 0,  right_hip_yaw_joint: 0,
      right_knee_joint: 0.4,        right_ankle_pitch_joint: -0.2, right_ankle_roll_joint: 0,
      waist_yaw_joint: 0,           waist_roll_joint: 0,      waist_pitch_joint: 0,
      left_shoulder_pitch_joint: 0,  left_shoulder_roll_joint: 0.3,
      left_shoulder_yaw_joint: 0,    left_elbow_joint: 0.5,
      left_wrist_roll_joint: 0,      left_wrist_pitch_joint: 0,    left_wrist_yaw_joint: 0,
      right_shoulder_pitch_joint: 0, right_shoulder_roll_joint: -0.3,
      right_shoulder_yaw_joint: 0,   right_elbow_joint: 0.5,
      right_wrist_roll_joint: 0,     right_wrist_pitch_joint: 0,   right_wrist_yaw_joint: 0,
    };

    // Actuator index map (from G1 XML actuator order)
    this.actIdx = {
      left_hip_pitch_joint: 0,   left_hip_roll_joint: 1,   left_hip_yaw_joint: 2,
      left_knee_joint: 3,        left_ankle_pitch_joint: 4, left_ankle_roll_joint: 5,
      right_hip_pitch_joint: 6,  right_hip_roll_joint: 7,  right_hip_yaw_joint: 8,
      right_knee_joint: 9,       right_ankle_pitch_joint: 10, right_ankle_roll_joint: 11,
      waist_yaw_joint: 12,       waist_roll_joint: 13,     waist_pitch_joint: 14,
      left_shoulder_pitch_joint: 15,  left_shoulder_roll_joint: 16,
      left_shoulder_yaw_joint: 17,    left_elbow_joint: 18,
      left_wrist_roll_joint: 19,      left_wrist_pitch_joint: 20,  left_wrist_yaw_joint: 21,
      right_shoulder_pitch_joint: 22, right_shoulder_roll_joint: 23,
      right_shoulder_yaw_joint: 24,   right_elbow_joint: 25,
      right_wrist_roll_joint: 26,     right_wrist_pitch_joint: 27, right_wrist_yaw_joint: 28,
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

  /**
   * Set the standing pose into qpos.
   * G1 has no keyframe, so this must be called after loading / resetting.
   */
  setStandingPose() {
    for (const [name, target] of Object.entries(this.homeQpos)) {
      const idx = this.jntIdx[name];
      if (idx !== undefined) {
        this.data.qpos[idx] = target;
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
   * G1 uses torque actuators, so we compute: tau = kp*(target - q) - kd*qdot
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

    // PD gains — G1 is lighter than H1 (~28kg vs 47kg)
    const hipKp = 400, hipKd = 15;
    const kneeKp = 600, kneeKd = 20;
    const ankleKp = 100, ankleKd = 5;
    const waistKp = 300, waistKd = 12;
    const armKp = 30, armKd = 2;

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
      - this.ankleRollAmp * leftStance; // Roll inward during stance for balance
    const leftHipRollTarget = this.homeQpos.left_hip_roll_joint
      - this.hipRollAmp * leftStance   // Lean into stance leg
      + this.lateralSpeed * 0.05;
    const leftHipYawTarget = this.homeQpos.left_hip_yaw_joint
      + this.turnRate * 0.05 * leftSwing;

    ctrl[this.actIdx.left_hip_pitch_joint] = this.pdTorque('left_hip_pitch_joint', leftHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll_joint] = this.pdTorque('left_hip_roll_joint', leftHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_yaw_joint] = this.pdTorque('left_hip_yaw_joint', leftHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_knee_joint] = this.pdTorque('left_knee_joint', leftKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle_pitch_joint] = this.pdTorque('left_ankle_pitch_joint', leftAnklePitchTarget, ankleKp, ankleKd);
    ctrl[this.actIdx.left_ankle_roll_joint] = this.pdTorque('left_ankle_roll_joint', leftAnkleRollTarget, ankleKp, ankleKd);

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
      + this.ankleRollAmp * rightStance; // Roll inward during stance (opposite sign)
    const rightHipRollTarget = this.homeQpos.right_hip_roll_joint
      + this.hipRollAmp * rightStance
      + this.lateralSpeed * 0.05;
    const rightHipYawTarget = this.homeQpos.right_hip_yaw_joint
      + this.turnRate * 0.05 * rightSwing;

    ctrl[this.actIdx.right_hip_pitch_joint] = this.pdTorque('right_hip_pitch_joint', rightHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll_joint] = this.pdTorque('right_hip_roll_joint', rightHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_yaw_joint] = this.pdTorque('right_hip_yaw_joint', rightHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_knee_joint] = this.pdTorque('right_knee_joint', rightKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle_pitch_joint] = this.pdTorque('right_ankle_pitch_joint', rightAnklePitchTarget, ankleKp, ankleKd);
    ctrl[this.actIdx.right_ankle_roll_joint] = this.pdTorque('right_ankle_roll_joint', rightAnkleRollTarget, ankleKp, ankleKd);

    // --- WAIST (3 DOF: yaw, roll, pitch) ---
    // Yaw: turning command
    const waistYawTarget = this.homeQpos.waist_yaw_joint + this.turnRate * 0.1;
    // Pitch: lean forward slightly when walking, compensate trunk pitch
    const waistPitchTarget = this.homeQpos.waist_pitch_joint
      - this.waistPitchGain * pitch;  // Counter trunk pitch error
    // Roll: compensate trunk roll during single-support
    const waistRollTarget = this.homeQpos.waist_roll_joint
      - this.waistRollGain * roll;    // Counter trunk roll error

    ctrl[this.actIdx.waist_yaw_joint] = this.pdTorque('waist_yaw_joint', waistYawTarget, waistKp, waistKd);
    ctrl[this.actIdx.waist_roll_joint] = this.pdTorque('waist_roll_joint', waistRollTarget, waistKp, waistKd);
    ctrl[this.actIdx.waist_pitch_joint] = this.pdTorque('waist_pitch_joint', waistPitchTarget, waistKp, waistKd);

    // --- BALANCE CORRECTIONS ---
    const pitchCorrection = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorrection = -this.balanceKp * roll - this.balanceKd * rollRate;

    // Ankle pitch strategy (primary): ankle torque shifts center of pressure
    ctrl[this.actIdx.left_ankle_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.right_ankle_pitch_joint] += pitchCorrection * 0.4;

    // Ankle roll strategy: G1 has 2-DOF ankles — use roll for lateral balance
    ctrl[this.actIdx.left_ankle_roll_joint] += rollCorrection * 0.3;
    ctrl[this.actIdx.right_ankle_roll_joint] -= rollCorrection * 0.3;

    // Hip strategy (secondary): hip torque for larger perturbations
    ctrl[this.actIdx.left_hip_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.right_hip_pitch_joint] += pitchCorrection * 0.4;

    // Roll corrections via hip roll
    ctrl[this.actIdx.left_hip_roll_joint] += rollCorrection * 0.3;
    ctrl[this.actIdx.right_hip_roll_joint] -= rollCorrection * 0.3;

    // --- ARMS (opposite to legs for natural gait) ---
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

    // --- WRISTS (hold at zero — passive) ---
    ctrl[this.actIdx.left_wrist_roll_joint] = this.pdTorque('left_wrist_roll_joint',
      this.homeQpos.left_wrist_roll_joint, armKp * 0.3, armKd * 0.5);
    ctrl[this.actIdx.left_wrist_pitch_joint] = this.pdTorque('left_wrist_pitch_joint',
      this.homeQpos.left_wrist_pitch_joint, armKp * 0.3, armKd * 0.5);
    ctrl[this.actIdx.left_wrist_yaw_joint] = this.pdTorque('left_wrist_yaw_joint',
      this.homeQpos.left_wrist_yaw_joint, armKp * 0.3, armKd * 0.5);

    ctrl[this.actIdx.right_wrist_roll_joint] = this.pdTorque('right_wrist_roll_joint',
      this.homeQpos.right_wrist_roll_joint, armKp * 0.3, armKd * 0.5);
    ctrl[this.actIdx.right_wrist_pitch_joint] = this.pdTorque('right_wrist_pitch_joint',
      this.homeQpos.right_wrist_pitch_joint, armKp * 0.3, armKd * 0.5);
    ctrl[this.actIdx.right_wrist_yaw_joint] = this.pdTorque('right_wrist_yaw_joint',
      this.homeQpos.right_wrist_yaw_joint, armKp * 0.3, armKd * 0.5);

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
