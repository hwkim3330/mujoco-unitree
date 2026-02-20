/**
 * Evolution runner — manages a population of H1 humanoid robots,
 * each with independently tunable CPG parameters, driven by a
 * genetic algorithm that optimizes walking ability.
 *
 * Each generation:
 *   1. Apply genome parameters to each robot's CPG
 *   2. All robots walk forward for evalSteps
 *   3. Measure fitness (distance + upright bonus - fall penalty)
 *   4. Select, crossover, mutate → next generation
 *   5. Reset all robots, repeat
 */

import { EvolutionController } from './evolutionController.js';

// ─── Single H1 CPG (multi-robot aware) ─────────────────────────────

class SingleH1CPG {
  constructor(mujoco, model, data, prefix, index) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.prefix = prefix;
    this.index = index;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // ─── Evolvable parameters (set by genome) ───
    this.frequency = 1.2;
    this.hipPitchAmp = 0.25;
    this.kneeBend = 0.35;
    this.anklePitchAmp = 0.15;
    this.hipRollAmp = 0.03;
    this.armSwingAmp = 0.3;
    this.balanceKp = 300;
    this.balanceKd = 20;
    this.stanceWidth = 0.05;
    this.leanForward = 0.05;
    this.legKp = 800;

    // ─── Fixed parameters ───
    this.phase = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;

    // Home pose
    this.homeQpos = {
      left_hip_yaw: 0, left_hip_roll: 0, left_hip_pitch: -0.4,
      left_knee: 0.8, left_ankle: -0.4,
      right_hip_yaw: 0, right_hip_roll: 0, right_hip_pitch: -0.4,
      right_knee: 0.8, right_ankle: -0.4,
      torso: 0,
      left_shoulder_pitch: 0, left_shoulder_roll: 0.2,
      left_shoulder_yaw: 0, left_elbow: -0.3,
      right_shoulder_pitch: 0, right_shoulder_roll: -0.2,
      right_shoulder_yaw: 0, right_elbow: -0.3,
    };

    // Actuator base index (19 actuators per H1)
    this.actBase = index * 19;
    this.numAct = 19;

    // Relative actuator offsets (within this robot's 19 actuators)
    this.actOff = {
      left_hip_yaw: 0, left_hip_roll: 1, left_hip_pitch: 2,
      left_knee: 3, left_ankle: 4,
      right_hip_yaw: 5, right_hip_roll: 6, right_hip_pitch: 7,
      right_knee: 8, right_ankle: 9,
      torso: 10,
      left_shoulder_pitch: 11, left_shoulder_roll: 12,
      left_shoulder_yaw: 13, left_elbow: 14,
      right_shoulder_pitch: 15, right_shoulder_roll: 16,
      right_shoulder_yaw: 17, right_elbow: 18,
    };

    // Joint indices
    this.baseQposAddr = 0;
    this.jntQpos = {};
    this.jntDof = {};
    this.findIndices();
  }

  findIndices() {
    const p = this.prefix;
    // Find freejoint
    try {
      const fjId = this.mujoco.mj_name2id(this.model, 3, `${p}freejoint`);
      if (fjId >= 0) this.baseQposAddr = this.model.jnt_qposadr[fjId];
    } catch (e) { /* ignore */ }

    // Find hinge joints
    for (const name of Object.keys(this.actOff)) {
      const fullName = `${p}${name}`;
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, fullName);
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

  step() {
    if (!this.enabled) return;

    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI;

    // Always walk forward during evolution
    const ampScale = 0.8;
    const direction = 1;

    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;

    // PD gains derived from genome
    const hipKp = this.legKp;
    const hipKd = this.legKp * 0.04;
    const kneeKp = this.legKp * 1.5;
    const kneeKd = kneeKp * 0.04;
    const ankleKp = this.legKp * 0.25;
    const ankleKd = ankleKp * 0.05;
    const torsoKp = this.legKp * 0.75;
    const torsoKd = torsoKp * 0.04;
    const armKp = 50;
    const armKd = 3;

    const ctrl = this.data.ctrl;
    const base = this.actBase;

    // --- LEFT LEG ---
    const lSwing = Math.sin(leftPhase);
    const lStance = Math.max(0, -Math.sin(leftPhase));

    const lHipPitch = this.homeQpos.left_hip_pitch
      + this.leanForward
      + direction * this.hipPitchAmp * ampScale * lSwing;
    const lKnee = this.homeQpos.left_knee
      + this.kneeBend * ampScale * Math.max(0, Math.sin(leftPhase));
    const lAnkle = this.homeQpos.left_ankle
      - this.anklePitchAmp * ampScale * lSwing;
    const lHipRoll = this.homeQpos.left_hip_roll
      - this.stanceWidth
      - this.hipRollAmp * lStance;
    const lHipYaw = this.homeQpos.left_hip_yaw;

    ctrl[base + this.actOff.left_hip_yaw] = this.pdTorque('left_hip_yaw', lHipYaw, hipKp, hipKd);
    ctrl[base + this.actOff.left_hip_roll] = this.pdTorque('left_hip_roll', lHipRoll, hipKp, hipKd);
    ctrl[base + this.actOff.left_hip_pitch] = this.pdTorque('left_hip_pitch', lHipPitch, hipKp, hipKd);
    ctrl[base + this.actOff.left_knee] = this.pdTorque('left_knee', lKnee, kneeKp, kneeKd);
    ctrl[base + this.actOff.left_ankle] = this.pdTorque('left_ankle', lAnkle, ankleKp, ankleKd);

    // --- RIGHT LEG ---
    const rSwing = Math.sin(rightPhase);
    const rStance = Math.max(0, -Math.sin(rightPhase));

    const rHipPitch = this.homeQpos.right_hip_pitch
      + this.leanForward
      + direction * this.hipPitchAmp * ampScale * rSwing;
    const rKnee = this.homeQpos.right_knee
      + this.kneeBend * ampScale * Math.max(0, Math.sin(rightPhase));
    const rAnkle = this.homeQpos.right_ankle
      - this.anklePitchAmp * ampScale * rSwing;
    const rHipRoll = this.homeQpos.right_hip_roll
      + this.stanceWidth
      + this.hipRollAmp * rStance;
    const rHipYaw = this.homeQpos.right_hip_yaw;

    ctrl[base + this.actOff.right_hip_yaw] = this.pdTorque('right_hip_yaw', rHipYaw, hipKp, hipKd);
    ctrl[base + this.actOff.right_hip_roll] = this.pdTorque('right_hip_roll', rHipRoll, hipKp, hipKd);
    ctrl[base + this.actOff.right_hip_pitch] = this.pdTorque('right_hip_pitch', rHipPitch, hipKp, hipKd);
    ctrl[base + this.actOff.right_knee] = this.pdTorque('right_knee', rKnee, kneeKp, kneeKd);
    ctrl[base + this.actOff.right_ankle] = this.pdTorque('right_ankle', rAnkle, ankleKp, ankleKd);

    // --- TORSO ---
    ctrl[base + this.actOff.torso] = this.pdTorque('torso', 0, torsoKp, torsoKd);

    // --- BALANCE CORRECTIONS ---
    const pitchCorr = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorr = -this.balanceKp * roll - this.balanceKd * rollRate;

    ctrl[base + this.actOff.left_ankle] += pitchCorr * 0.4;
    ctrl[base + this.actOff.right_ankle] += pitchCorr * 0.4;
    ctrl[base + this.actOff.left_hip_pitch] += pitchCorr * 0.5;
    ctrl[base + this.actOff.right_hip_pitch] += pitchCorr * 0.5;
    ctrl[base + this.actOff.left_hip_roll] += rollCorr * 0.3;
    ctrl[base + this.actOff.right_hip_roll] -= rollCorr * 0.3;

    // --- ARMS ---
    const lArmSwing = -this.armSwingAmp * direction * ampScale * lSwing;
    const rArmSwing = -this.armSwingAmp * direction * ampScale * rSwing;

    ctrl[base + this.actOff.left_shoulder_pitch] = this.pdTorque('left_shoulder_pitch',
      this.homeQpos.left_shoulder_pitch + lArmSwing, armKp, armKd);
    ctrl[base + this.actOff.left_shoulder_roll] = this.pdTorque('left_shoulder_roll',
      this.homeQpos.left_shoulder_roll, armKp, armKd);
    ctrl[base + this.actOff.left_shoulder_yaw] = this.pdTorque('left_shoulder_yaw',
      this.homeQpos.left_shoulder_yaw, armKp * 0.5, armKd);
    ctrl[base + this.actOff.left_elbow] = this.pdTorque('left_elbow',
      this.homeQpos.left_elbow, armKp, armKd);

    ctrl[base + this.actOff.right_shoulder_pitch] = this.pdTorque('right_shoulder_pitch',
      this.homeQpos.right_shoulder_pitch + rArmSwing, armKp, armKd);
    ctrl[base + this.actOff.right_shoulder_roll] = this.pdTorque('right_shoulder_roll',
      this.homeQpos.right_shoulder_roll, armKp, armKd);
    ctrl[base + this.actOff.right_shoulder_yaw] = this.pdTorque('right_shoulder_yaw',
      this.homeQpos.right_shoulder_yaw, armKp * 0.5, armKd);
    ctrl[base + this.actOff.right_elbow] = this.pdTorque('right_elbow',
      this.homeQpos.right_elbow, armKp, armKd);

    // Clamp this robot's actuators
    const ctrlRange = this.model.actuator_ctrlrange;
    if (ctrlRange) {
      for (let i = 0; i < this.numAct; i++) {
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
  }
}

// ─── Evolution Runner ───────────────────────────────────────────────

export class EvolutionRunner {
  /**
   * @param {object} mujoco
   * @param {object} model
   * @param {object} data
   * @param {number} numRobots - population size
   * @param {number} evalSeconds - evaluation time per generation (seconds)
   */
  constructor(mujoco, model, data, numRobots, evalSeconds = 8) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.numRobots = numRobots;
    this.enabled = false;

    const dt = model.opt.timestep || 0.002;
    this.evalSteps = Math.round(evalSeconds / dt);

    // Create GA controller
    this.evo = new EvolutionController('humanoid', numRobots, this.evalSteps);

    // Create N independent CPG controllers
    this.robots = [];
    for (let i = 0; i < numRobots; i++) {
      this.robots.push(new SingleH1CPG(mujoco, model, data, `r${i}_`, i));
    }

    // Center robot for camera follow
    const centerIdx = Math.floor(numRobots / 2);
    try {
      this.centerBodyId = mujoco.mj_name2id(model, 1, `r${centerIdx}_pelvis`);
    } catch (e) {
      this.centerBodyId = 1;
    }

    // State
    this.evaluating = false;
    this.stepCount = 0;

    // Apply initial genomes and start first evaluation
    this.applyGenomes();
    this.startEvaluation();
  }

  /**
   * Apply genome parameters from EvolutionController to each robot's CPG.
   */
  applyGenomes() {
    for (let i = 0; i < this.numRobots; i++) {
      const r = this.robots[i];
      r.frequency = this.evo.getParam(i, 'frequency');
      r.hipPitchAmp = this.evo.getParam(i, 'hipPitchAmp');
      r.kneeBend = this.evo.getParam(i, 'kneeBend');
      r.anklePitchAmp = this.evo.getParam(i, 'anklePitchAmp');
      r.hipRollAmp = this.evo.getParam(i, 'hipRollAmp');
      r.armSwingAmp = this.evo.getParam(i, 'armSwingAmp');
      r.balanceKp = this.evo.getParam(i, 'balanceKp');
      r.balanceKd = this.evo.getParam(i, 'balanceKd');
      r.stanceWidth = this.evo.getParam(i, 'stanceWidth');
      r.leanForward = this.evo.getParam(i, 'leanForward');
      r.legKp = this.evo.getParam(i, 'legKp');
    }
  }

  /**
   * Record start positions and begin a new evaluation period.
   */
  startEvaluation() {
    // Record start positions
    this.evo.recordStartPositions(this.robots);
    this.stepCount = 0;
    this.evaluating = true;

    // Enable all robots
    for (const r of this.robots) {
      r.enabled = true;
    }
  }

  /**
   * Reset all robots to home pose at grid positions.
   */
  resetRobots() {
    // Reset to keyframe
    if (this.model.nkey > 0) {
      this.data.qpos.set(this.model.key_qpos.slice(0, this.model.nq));
      for (let i = 0; i < this.model.nv; i++) this.data.qvel[i] = 0;
      for (let i = 0; i < this.model.nu; i++) this.data.ctrl[i] = 0;
      this.mujoco.mj_forward(this.model, this.data);
    }

    // Reset CPG states
    for (const r of this.robots) {
      r.reset();
    }

    // Warm-up: let PD settle for 200 steps
    for (const r of this.robots) r.enabled = true;
    for (let i = 0; i < 200; i++) {
      for (const r of this.robots) r.step();
      this.mujoco.mj_step(this.model, this.data);
    }
  }

  /**
   * Step the evolution. Called every physics step.
   */
  step() {
    if (!this.enabled || !this.evaluating) return;

    // Step all robots
    for (const r of this.robots) r.step();
    this.stepCount++;
    this.evo.currentStep = this.stepCount;

    // Check for fallen robots
    this.evo.checkFallen(this.robots);

    // End of evaluation period
    if (this.stepCount >= this.evalSteps) {
      this.evaluating = false;

      // Evaluate fitness
      const result = this.evo.evaluateFitness(this.robots);
      console.log(
        `Gen ${this.evo.generation}: best=${result.bestFit.toFixed(1)}, ` +
        `avg=${result.avg.toFixed(1)}, worst=${result.worst.toFixed(1)}`
      );

      // Evolve
      this.evo.evolve();

      // Reset and start next generation
      this.resetRobots();
      this.applyGenomes();
      this.startEvaluation();
    }
  }

  /**
   * Get status text for display.
   */
  getStatusText() {
    return this.evo.getStatusText();
  }

  /**
   * Get history for charting.
   */
  getHistory() {
    return this.evo.history;
  }

  setCommand() {
    // No-op: evolution robots walk automatically
  }

  reset() {
    this.resetRobots();
    this.applyGenomes();
    this.startEvaluation();
  }
}
