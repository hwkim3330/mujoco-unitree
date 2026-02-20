/**
 * Evolutionary learning controller — uses genetic algorithm to optimize
 * CPG walking parameters across a population of robots.
 *
 * Each robot in the population has a different set of CPG "genes" (parameters).
 * After an evaluation period, fitness is measured (distance walked + upright bonus),
 * the best performers are selected, and their genes are crossed over and mutated
 * to create the next generation.
 *
 * Works with any robot that has a compatible CPG controller.
 */

// ─── Gene definitions ──────────────────────────────────────────────
// Each gene has: name, min, max, default
const QUADRUPED_GENES = [
  { name: 'frequency',  min: 1.0,  max: 4.0,  def: 2.5  },
  { name: 'thighAmp',   min: 0.05, max: 0.5,  def: 0.25 },
  { name: 'calfAmp',    min: 0.05, max: 0.5,  def: 0.25 },
  { name: 'hipAmp',     min: 0.01, max: 0.1,  def: 0.04 },
  { name: 'balanceKp',  min: 20,   max: 120,  def: 60   },
  { name: 'balanceKd',  min: 1,    max: 15,   def: 5    },
  { name: 'homeThigh',  min: 0.5,  max: 1.3,  def: 0.9  },
  { name: 'homeCalf',   min: -2.2, max: -1.2, def: -1.8 },
  { name: 'hipKp',      min: 50,   max: 250,  def: 120  },
  { name: 'thighKp',    min: 80,   max: 400,  def: 200  },
  { name: 'calfKp',     min: 100,  max: 500,  def: 250  },
];

const HUMANOID_GENES = [
  { name: 'frequency',     min: 0.5,  max: 3.0,  def: 1.2  },
  { name: 'hipPitchAmp',   min: 0.05, max: 0.6,  def: 0.3  },
  { name: 'kneeBend',      min: 0.1,  max: 0.8,  def: 0.4  },
  { name: 'anklePitchAmp', min: 0.05, max: 0.5,  def: 0.2  },
  { name: 'hipRollAmp',    min: 0.01, max: 0.15, def: 0.05 },
  { name: 'armSwingAmp',   min: 0.0,  max: 0.5,  def: 0.2  },
  { name: 'balanceKp',     min: 100,  max: 800,  def: 300  },
  { name: 'balanceKd',     min: 5,    max: 50,   def: 15   },
  { name: 'stanceWidth',   min: 0.0,  max: 0.15, def: 0.05 },
  { name: 'leanForward',   min: -0.1, max: 0.2,  def: 0.05 },
  { name: 'legKp',         min: 200,  max: 1500, def: 800  },
];

// ─── Genetic Algorithm ─────────────────────────────────────────────

function randomGenome(geneDefs) {
  return geneDefs.map(g => g.min + Math.random() * (g.max - g.min));
}

function defaultGenome(geneDefs) {
  return geneDefs.map(g => g.def);
}

function crossover(parent1, parent2) {
  const child = new Float64Array(parent1.length);
  const crossPoint = Math.floor(Math.random() * parent1.length);
  for (let i = 0; i < parent1.length; i++) {
    child[i] = i < crossPoint ? parent1[i] : parent2[i];
  }
  return child;
}

function mutate(genome, geneDefs, rate = 0.2, strength = 0.15) {
  const result = new Float64Array(genome);
  for (let i = 0; i < result.length; i++) {
    if (Math.random() < rate) {
      const range = geneDefs[i].max - geneDefs[i].min;
      result[i] += (Math.random() - 0.5) * 2 * strength * range;
      result[i] = Math.max(geneDefs[i].min, Math.min(geneDefs[i].max, result[i]));
    }
  }
  return result;
}

function tournamentSelect(population, fitnesses, k = 3) {
  let bestIdx = Math.floor(Math.random() * population.length);
  for (let i = 1; i < k; i++) {
    const idx = Math.floor(Math.random() * population.length);
    if (fitnesses[idx] > fitnesses[bestIdx]) bestIdx = idx;
  }
  return population[bestIdx];
}

// ─── Evolution Manager ─────────────────────────────────────────────

export class EvolutionController {
  /**
   * @param {string} robotType - 'quadruped' or 'humanoid'
   * @param {number} populationSize - number of robots
   * @param {number} evalSteps - simulation steps per evaluation (e.g., 2500 = 5s at 500Hz)
   */
  constructor(robotType, populationSize, evalSteps = 2500) {
    this.robotType = robotType;
    this.geneDefs = robotType === 'humanoid' ? HUMANOID_GENES : QUADRUPED_GENES;
    this.populationSize = populationSize;
    this.evalSteps = evalSteps;

    // GA state
    this.generation = 0;
    this.currentStep = 0;
    this.bestFitness = -Infinity;
    this.bestGenome = null;
    this.bestEverFitness = -Infinity;
    this.bestEverGenome = null;

    // Population: array of genomes
    this.population = [];
    this.fitnesses = new Float64Array(populationSize);
    this.startPositions = new Float64Array(populationSize * 3); // x, y, z per robot
    this.fallen = new Uint8Array(populationSize); // 1 if robot has fallen

    // Initialize population: first robot gets defaults, rest are random
    this.population.push(defaultGenome(this.geneDefs));
    for (let i = 1; i < populationSize; i++) {
      this.population.push(randomGenome(this.geneDefs));
    }

    // Stats history for display
    this.history = []; // { gen, best, avg, worst }

    this.enabled = false;
    this.evaluating = false;
  }

  /**
   * Get a named parameter from a robot's genome.
   */
  getParam(robotIdx, paramName) {
    const idx = this.geneDefs.findIndex(g => g.name === paramName);
    if (idx < 0) return 0;
    return this.population[robotIdx][idx];
  }

  /**
   * Record starting positions at the beginning of an evaluation.
   */
  recordStartPositions(robots) {
    for (let i = 0; i < this.populationSize; i++) {
      const addr = robots[i].baseQposAddr;
      const data = robots[i].data;
      this.startPositions[i * 3 + 0] = data.qpos[addr + 0];
      this.startPositions[i * 3 + 1] = data.qpos[addr + 1];
      this.startPositions[i * 3 + 2] = data.qpos[addr + 2];
    }
    this.fallen.fill(0);
  }

  /**
   * Check if a robot has fallen (base z too low or tilted too much).
   */
  checkFallen(robots) {
    for (let i = 0; i < this.populationSize; i++) {
      if (this.fallen[i]) continue;
      const addr = robots[i].baseQposAddr;
      const data = robots[i].data;
      const z = data.qpos[addr + 2];
      // Check if z is too low (fallen)
      const threshold = this.robotType === 'humanoid' ? 0.4 : 0.1;
      if (z < threshold) this.fallen[i] = 1;
    }
  }

  /**
   * Evaluate fitness for all robots after evaluation period.
   * Fitness = forward distance + upright bonus - lateral penalty
   */
  evaluateFitness(robots) {
    let bestFit = -Infinity;
    let bestIdx = 0;

    for (let i = 0; i < this.populationSize; i++) {
      const addr = robots[i].baseQposAddr;
      const data = robots[i].data;

      const startX = this.startPositions[i * 3 + 0];
      const startY = this.startPositions[i * 3 + 1];
      const endX = data.qpos[addr + 0];
      const endY = data.qpos[addr + 1];
      const endZ = data.qpos[addr + 2];

      // Forward distance (in X direction, robot's initial heading)
      const fwdDist = endX - startX;

      // Lateral deviation penalty
      const latDist = Math.abs(endY - startY);

      // Upright bonus (higher z = more upright)
      const uprightBonus = this.fallen[i] ? 0 : endZ * 2;

      // Fitness
      let fitness = fwdDist * 10 + uprightBonus - latDist * 2;
      if (this.fallen[i]) fitness -= 5; // penalty for falling

      this.fitnesses[i] = fitness;
      if (fitness > bestFit) { bestFit = fitness; bestIdx = i; }
    }

    this.bestFitness = bestFit;
    this.bestGenome = [...this.population[bestIdx]];

    if (bestFit > this.bestEverFitness) {
      this.bestEverFitness = bestFit;
      this.bestEverGenome = [...this.population[bestIdx]];
    }

    // Record history
    const avg = this.fitnesses.reduce((a, b) => a + b, 0) / this.populationSize;
    const worst = Math.min(...this.fitnesses);
    this.history.push({
      gen: this.generation,
      best: bestFit,
      avg: avg,
      worst: worst,
    });

    return { bestFit, bestIdx, avg, worst };
  }

  /**
   * Create next generation using tournament selection, crossover, and mutation.
   * Elitism: best genome passes unchanged.
   */
  evolve() {
    const newPop = [];

    // Elitism: keep the best genome unchanged
    if (this.bestEverGenome) {
      newPop.push([...this.bestEverGenome]);
    } else {
      newPop.push([...this.population[0]]);
    }

    // Fill rest with offspring
    while (newPop.length < this.populationSize) {
      const parent1 = tournamentSelect(this.population, this.fitnesses);
      const parent2 = tournamentSelect(this.population, this.fitnesses);
      let child = crossover(parent1, parent2);

      // Increase mutation rate in early generations for exploration
      const mutRate = this.generation < 10 ? 0.35 : 0.2;
      const mutStrength = this.generation < 10 ? 0.2 : 0.12;
      child = mutate(child, this.geneDefs, mutRate, mutStrength);
      newPop.push(child);
    }

    this.population = newPop;
    this.generation++;
    this.currentStep = 0;
  }

  /**
   * Apply genome parameters to a robot's CPG controller.
   */
  applyGenome(robot, genomeIdx) {
    const genome = this.population[genomeIdx];

    if (this.robotType === 'quadruped') {
      robot.frequency = genome[0];
      robot.thighAmp = genome[1];
      robot.calfAmp = genome[2];
      robot.hipAmp = genome[3];
      robot.balanceKp = genome[4];
      robot.balanceKd = genome[5];
      robot.homeThigh = genome[6];
      robot.homeCalf = genome[7];
      robot.hipKp = genome[8];
      robot.thighKp = genome[9];
      robot.calfKp = genome[10];
    } else {
      // Humanoid params — applied by the humanoid CPG controller
      robot.frequency = genome[0];
      robot.hipPitchAmp = genome[1];
      robot.kneeBend = genome[2];
      robot.anklePitchAmp = genome[3];
      robot.hipRollAmp = genome[4];
      robot.armSwingAmp = genome[5];
      robot.balanceKp = genome[6];
      robot.balanceKd = genome[7];
      robot.stanceWidth = genome[8];
      robot.leanForward = genome[9];
      robot.legKp = genome[10];
    }
  }

  /**
   * Get status text for display.
   */
  getStatusText() {
    const progress = Math.floor(this.currentStep / this.evalSteps * 100);
    const best = this.bestEverFitness > -Infinity
      ? this.bestEverFitness.toFixed(1) : '--';
    return `Gen ${this.generation} | ${progress}% | Best: ${best}`;
  }

  /**
   * Get gene names and values for a specific robot (for debug/display).
   */
  getGenomeInfo(robotIdx) {
    const genome = this.population[robotIdx];
    return this.geneDefs.map((g, i) => ({
      name: g.name,
      value: genome[i],
      min: g.min,
      max: g.max,
    }));
  }

  // ─── Persistence (localStorage) ─────────────────────────────────

  static STORAGE_KEY = 'mujoco-evo-h1';

  /**
   * Serialize GA state to a plain object for JSON storage.
   */
  serialize() {
    return {
      robotType: this.robotType,
      generation: this.generation,
      bestEverFitness: this.bestEverFitness,
      bestEverGenome: this.bestEverGenome ? [...this.bestEverGenome] : null,
      population: this.population.map(g => [...g]),
      history: this.history.slice(-200), // keep last 200 gens
    };
  }

  /**
   * Restore GA state from a serialized object.
   * Returns true if successfully restored.
   */
  deserialize(saved) {
    if (!saved || saved.robotType !== this.robotType) return false;
    if (!saved.population || saved.population.length !== this.populationSize) return false;
    if (!saved.population[0] || saved.population[0].length !== this.geneDefs.length) return false;

    this.generation = saved.generation || 0;
    this.bestEverFitness = saved.bestEverFitness ?? -Infinity;
    this.bestEverGenome = saved.bestEverGenome ? [...saved.bestEverGenome] : null;
    this.population = saved.population.map(g => [...g]);
    this.history = saved.history || [];
    return true;
  }

  /**
   * Save current state to localStorage.
   */
  save() {
    try {
      const json = JSON.stringify(this.serialize());
      localStorage.setItem(EvolutionController.STORAGE_KEY, json);
    } catch (e) {
      console.warn('Evolution save failed:', e);
    }
  }

  /**
   * Load state from localStorage. Returns true if restored.
   */
  load() {
    try {
      const raw = localStorage.getItem(EvolutionController.STORAGE_KEY);
      if (!raw) return false;
      const saved = JSON.parse(raw);
      return this.deserialize(saved);
    } catch (e) {
      console.warn('Evolution load failed:', e);
      return false;
    }
  }
}
