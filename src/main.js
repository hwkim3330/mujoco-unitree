/**
 * main.js — MuJoCo WASM + Three.js Unitree robot playground.
 * Supports: Go2, B2 (quadruped CPG), H1, H1-2, G1 (humanoid CPG),
 *           Go2 RL, Go2 Factory (9x), Evolution mode.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import load_mujoco from 'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js';
import { buildScene, getPosition, getQuaternion } from './meshBuilder.js';
import { loadSceneAssets } from './assetLoader.js';
import { Go2CpgController } from './go2CpgController.js';
import { Go2OnnxController } from './go2OnnxController.js';
import { H1CpgController } from './h1CpgController.js';
import { B2CpgController } from './b2CpgController.js';
import { G1CpgController } from './g1CpgController.js';
import { H1_2CpgController } from './h1_2CpgController.js';
import { FactoryController } from './factoryController.js';
import { generateFactoryXML } from './factoryScene.js';
import { EvolutionRunner } from './evolutionRunner.js';
import { generateH1EvolutionXML } from './evolutionScene.js';

// ─── DOM ────────────────────────────────────────────────────────────
const statusEl = document.getElementById('status');
const sceneSelect = document.getElementById('scene-select');
const resetBtn = document.getElementById('btn-reset');
const controllerBtn = document.getElementById('btn-controller');
const speedBtn = document.getElementById('btn-speed');
const helpOverlay = document.getElementById('help-overlay');
const evolvePanel = document.getElementById('evolve-panel');
const evolveStatus = document.getElementById('evolve-status');

// ─── Three.js Setup ─────────────────────────────────────────────────
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11151d);
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.0));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(3, 5, 3);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(1.5, 1.0, 1.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.3, 0);
controls.enableDamping = true;

// ─── State ──────────────────────────────────────────────────────────
let mujoco;
let model;
let data;
let bodies = {};
let mujocoRoot = null;

let activeController = null; // 'go2' | 'go2rl' | 'h1' | 'b2' | 'g1' | 'h1_2' | 'factory' | null
let go2Controller = null;
let go2RlController = null;
let h1Controller = null;
let b2Controller = null;
let g1Controller = null;
let h1_2Controller = null;
let factoryController = null;
let evolveRunner = null;

let paused = false;
let cameraFollow = true;
let simSpeed = 1.0;
const SIM_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

// Keyboard state
const keys = {};

// Touch / joystick state
let touchX = 0;
let touchY = 0;
let touchRotL = false;
let touchRotR = false;
const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

// Step counter
let stepCounter = 0;

// ─── Obstacle System ─────────────────────────────────────────────────
const NUM_BALLS = 5;
const NUM_BOXES = 5;
const NUM_OBSTACLES = NUM_BALLS + NUM_BOXES;
const HIDE_Z = -50;

let obstacleQposBase = -1;
let obstacleQvelBase = -1;
let nextBall = 0;
let nextBox = 0;
let currentObstacleScale = 1;

// ─── Scene Config ───────────────────────────────────────────────────
const SCENES = {
  'unitree_go2/scene.xml': {
    controller: 'go2',
    camera: { pos: [1.5, 1.0, 1.5], target: [0, 0.25, 0] },
  },
  'unitree_go2/scene.xml|rl': {
    controller: 'go2rl',
    scenePath: 'unitree_go2/scene.xml',
    onnxModel: './assets/models/go2_flat_policy.onnx',
    camera: { pos: [1.5, 1.0, 1.5], target: [0, 0.25, 0] },
  },
  'unitree_go2/scene_stairs.xml': {
    controller: 'go2',
    camera: { pos: [2.0, 1.5, 2.0], target: [1.2, 0.15, 0] },
  },
  'unitree_go2/scene_rough.xml': {
    controller: 'go2',
    camera: { pos: [1.5, 1.0, 1.5], target: [0.5, 0.1, 0] },
  },
  'unitree_b2/scene.xml': {
    controller: 'b2',
    camera: { pos: [2.5, 1.5, 2.5], target: [0, 0.35, 0] },
  },
  'unitree_h1/scene.xml': {
    controller: 'h1',
    camera: { pos: [3.0, 2.0, 3.0], target: [0, 0.9, 0] },
  },
  'unitree_h1/scene_stairs.xml': {
    controller: 'h1',
    camera: { pos: [4.0, 2.5, 3.5], target: [2.0, 0.5, 0] },
  },
  'unitree_g1/scene.xml': {
    controller: 'g1',
    camera: { pos: [2.5, 1.8, 2.5], target: [0, 0.7, 0] },
  },
  'unitree_h1_2/scene.xml': {
    controller: 'h1_2',
    camera: { pos: [3.0, 2.0, 3.0], target: [0, 0.9, 0] },
  },
  'unitree_go2/scene_factory': {
    controller: 'factory',
    numRobots: 9,
    spacing: 1.5,
    camera: { pos: [4, 3, 4], target: [0, 0.25, 0] },
  },
  'unitree_h1/scene_evolve': {
    controller: 'evolve',
    numRobots: 16,
    spacing: 2.5,
    evalSeconds: 12,
    camera: { pos: [8, 6, 8], target: [0, 0.9, 0] },
  },
};

let currentScenePath = 'unitree_go2/scene.xml';
let currentSceneKey = 'unitree_go2/scene.xml';

// ─── Obstacle XML Generation ────────────────────────────────────────
function generateArenaXML(sceneXml, scale) {
  let xml = sceneXml;

  const colors = [
    '0.95 0.25 0.2 1', '0.2 0.55 0.95 1', '0.2 0.85 0.35 1',
    '0.95 0.75 0.15 1', '0.85 0.3 0.7 1',
  ];

  let obsXml = '\n  <worldbody>\n';
  for (let i = 0; i < NUM_BALLS; i++) {
    const r = (0.015 + i * 0.004) * scale;
    const m = (0.03 * scale * scale).toFixed(3);
    obsXml += `    <body name="obstacle_${i}" pos="0 0 ${HIDE_Z}"><freejoint name="obs_fj_${i}"/><geom type="sphere" size="${r.toFixed(4)}" rgba="${colors[i]}" mass="${m}" contype="1" conaffinity="1"/></body>\n`;
  }
  for (let i = 0; i < NUM_BOXES; i++) {
    const r = (0.012 + i * 0.003) * scale;
    const m = (0.04 * scale * scale).toFixed(3);
    const idx = NUM_BALLS + i;
    obsXml += `    <body name="obstacle_${idx}" pos="0 0 ${HIDE_Z}"><freejoint name="obs_fj_${idx}"/><geom type="box" size="${r.toFixed(4)} ${r.toFixed(4)} ${r.toFixed(4)}" rgba="${colors[i]}" mass="${m}" contype="1" conaffinity="1"/></body>\n`;
  }
  obsXml += '  </worldbody>\n';
  xml = xml.replace('</mujoco>', obsXml + '</mujoco>');

  // Extend keyframe qpos
  const obsQpos = Array(NUM_OBSTACLES).fill(`0 0 ${HIDE_Z} 1 0 0 0`).join(' ');
  xml = xml.replace(
    /(qpos\s*=\s*")([\s\S]*?)(")/,
    (m, pre, content, post) => pre + content.trimEnd() + ' ' + obsQpos + '\n    ' + post
  );

  return xml;
}

function findObstacleIndices() {
  obstacleQposBase = -1;
  obstacleQvelBase = -1;
  try {
    const bodyId = mujoco.mj_name2id(model, 1, 'obstacle_0');
    if (bodyId < 0) return;
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_bodyid[j] === bodyId) {
        obstacleQposBase = model.jnt_qposadr[j];
        obstacleQvelBase = model.jnt_dofadr[j];
        break;
      }
    }
  } catch (e) {
    console.warn('Could not find obstacle indices:', e);
  }
}

function spawnObstacle(type) {
  if (obstacleQposBase < 0 || !model || !data) return;

  let idx;
  if (type === 'box') {
    idx = NUM_BALLS + (nextBox % NUM_BOXES);
    nextBox++;
  } else {
    idx = nextBall % NUM_BALLS;
    nextBall++;
  }

  const rx = data.qpos[0];
  const ry = data.qpos[1];
  const rz = data.qpos[2];

  const angle = Math.random() * Math.PI * 2;
  const dist = (0.15 + Math.random() * 0.25) * currentObstacleScale;

  const base = obstacleQposBase + idx * 7;
  data.qpos[base + 0] = rx + Math.cos(angle) * dist;
  data.qpos[base + 1] = ry + Math.sin(angle) * dist;
  data.qpos[base + 2] = rz + 0.2 * currentObstacleScale;
  data.qpos[base + 3] = 1;
  data.qpos[base + 4] = 0;
  data.qpos[base + 5] = 0;
  data.qpos[base + 6] = 0;

  const vbase = obstacleQvelBase + idx * 6;
  for (let v = 0; v < 6; v++) data.qvel[vbase + v] = 0;
  mujoco.mj_forward(model, data);
}

// ─── Functions ──────────────────────────────────────────────────────
function setStatus(text) {
  statusEl.textContent = text;
}

function clearScene() {
  if (mujocoRoot) {
    scene.remove(mujocoRoot);
    mujocoRoot = null;
  }
  bodies = {};
}

function getObstacleScale(sceneKey) {
  if (sceneKey.includes('h1') || sceneKey.includes('g1')) return 3.5;
  if (sceneKey.includes('b2')) return 4.0;
  return 2.5;
}

async function loadScene(sceneKey) {
  setStatus(`Loading: ${sceneKey}`);

  const cfg = SCENES[sceneKey] || {};
  let loadPath;

  if (cfg.controller === 'evolve') {
    // Evolution mode: load H1 assets, then generate multi-robot XML
    await loadSceneAssets(mujoco, 'unitree_h1/scene.xml', setStatus);
    setStatus('Generating evolution arena...');
    const evoXml = generateH1EvolutionXML(mujoco, cfg.numRobots, cfg.spacing);
    mujoco.FS.writeFile('/working/unitree_h1/scene_evolve.xml', evoXml);
    loadPath = '/working/unitree_h1/scene_evolve.xml';
  } else if (cfg.controller === 'factory') {
    // Factory mode: load Go2 assets, then generate multi-robot XML
    await loadSceneAssets(mujoco, 'unitree_go2/scene.xml', setStatus);
    setStatus('Generating factory...');
    const factoryXml = generateFactoryXML(mujoco, cfg.numRobots, cfg.spacing);
    mujoco.FS.writeFile('/working/unitree_go2/scene_factory.xml', factoryXml);
    loadPath = '/working/unitree_go2/scene_factory.xml';
  } else {
    // Normal mode: load assets and add arena obstacles
    const scenePath = cfg.scenePath || sceneKey;
    await loadSceneAssets(mujoco, scenePath, setStatus);
    currentObstacleScale = getObstacleScale(sceneKey);
    const originalXml = new TextDecoder().decode(mujoco.FS.readFile('/working/' + scenePath));
    const arenaXml = generateArenaXML(originalXml, currentObstacleScale);
    const arenaPath = scenePath.replace('.xml', '_arena.xml');
    mujoco.FS.writeFile('/working/' + arenaPath, arenaXml);
    loadPath = '/working/' + arenaPath;
  }

  clearScene();
  if (data) { data.delete(); data = null; }
  if (model) { model.delete(); model = null; }

  model = mujoco.MjModel.loadFromXML(loadPath);
  data = new mujoco.MjData(model);

  console.log(`Model loaded: nq=${model.nq}, nv=${model.nv}, nu=${model.nu}, nbody=${model.nbody}`);

  // Apply home keyframe
  if (model.nkey > 0) {
    data.qpos.set(model.key_qpos.slice(0, model.nq));
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    if (model.key_ctrl) data.ctrl.set(model.key_ctrl.slice(0, model.nu));
    mujoco.mj_forward(model, data);
  }

  // Build Three.js scene
  const built = buildScene(model);
  mujocoRoot = built.mujocoRoot;
  bodies = built.bodies;
  scene.add(mujocoRoot);

  // Setup controller
  activeController = null;
  go2Controller = null;
  go2RlController = null;
  h1Controller = null;
  b2Controller = null;
  g1Controller = null;
  h1_2Controller = null;
  factoryController = null;
  evolveRunner = null;
  stepCounter = 0;

  // Reset turbo speed when leaving evolution mode
  if (cfg.controller !== 'evolve' && simSpeed > 4.0) {
    simSpeed = 1.0;
    updateSpeedBtn();
  }

  model.opt.iterations = 30;

  if (cfg.controller === 'evolve') {
    evolveRunner = new EvolutionRunner(mujoco, model, data, cfg.numRobots, cfg.evalSeconds);
    evolveRunner.enabled = true;
    activeController = 'evolve';
    // Auto-turbo: set 8x speed for faster evolution
    simSpeed = 8.0;
    updateSpeedBtn();
  } else if (cfg.controller === 'go2') {
    go2Controller = new Go2CpgController(mujoco, model, data);
    go2Controller.enabled = true;
    for (let i = 0; i < 200; i++) {
      go2Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = 'go2';
  } else if (cfg.controller === 'go2rl') {
    go2RlController = new Go2OnnxController(mujoco, model, data);
    setStatus('Loading RL policy...');
    const loaded = await go2RlController.loadModel(cfg.onnxModel);
    if (loaded) {
      go2RlController.setInitialPose();
      go2RlController.enabled = true;
      for (let i = 0; i < 500; i++) {
        go2RlController.applyPD();
        mujoco.mj_step(model, data);
      }
      activeController = 'go2rl';
    } else {
      setStatus('RL policy load failed, falling back to CPG');
      go2Controller = new Go2CpgController(mujoco, model, data);
      go2Controller.enabled = true;
      for (let i = 0; i < 200; i++) {
        go2Controller.step();
        mujoco.mj_step(model, data);
      }
      activeController = 'go2';
    }
  } else if (cfg.controller === 'b2') {
    b2Controller = new B2CpgController(mujoco, model, data);
    b2Controller.enabled = true;
    for (let i = 0; i < 200; i++) {
      b2Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = 'b2';
  } else if (cfg.controller === 'h1') {
    h1Controller = new H1CpgController(mujoco, model, data);
    h1Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      h1Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = 'h1';
  } else if (cfg.controller === 'g1') {
    g1Controller = new G1CpgController(mujoco, model, data);
    // G1 has no keyframe — set standing pose
    g1Controller.setStandingPose();
    mujoco.mj_forward(model, data);
    g1Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      g1Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = 'g1';
  } else if (cfg.controller === 'h1_2') {
    h1_2Controller = new H1_2CpgController(mujoco, model, data);
    // H1-2 has no keyframe — set standing pose via PD warm-up
    // Write home pose into qpos
    for (const [name, target] of Object.entries(h1_2Controller.homeQpos)) {
      const idx = h1_2Controller.jntIdx[name];
      if (idx !== undefined) data.qpos[idx] = target;
    }
    mujoco.mj_forward(model, data);
    h1_2Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      h1_2Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = 'h1_2';
  } else if (cfg.controller === 'factory') {
    factoryController = new FactoryController(mujoco, model, data, cfg.numRobots);
    factoryController.enabled = true;
    for (let i = 0; i < 200; i++) {
      factoryController.step();
      mujoco.mj_step(model, data);
    }
    activeController = 'factory';
  }

  if (cfg.controller !== 'factory' && cfg.controller !== 'evolve') findObstacleIndices();
  nextBall = 0;
  nextBox = 0;
  updateControllerBtn();

  // Hide evolution panel for non-evolution modes
  if (evolvePanel) evolvePanel.style.display = 'none';

  if (cfg.camera) {
    camera.position.set(...cfg.camera.pos);
    controls.target.set(...cfg.camera.target);
  }
  controls.update();

  currentScenePath = sceneKey;
  currentSceneKey = sceneKey;
  setStatus(`Ready: ${sceneKey.split('/').pop()}`);
}

function updateControllerBtn() {
  if (!controllerBtn) return;
  const labels = {
    go2: () => go2Controller?.qwopMode ? 'QWOP' : go2Controller?.enabled ? 'CPG: ON' : 'CPG: OFF',
    go2rl: () => go2RlController?.enabled ? 'RL: ON' : 'RL: OFF',
    h1: () => h1Controller?.enabled ? 'CPG: ON' : 'CPG: OFF',
    b2: () => b2Controller?.enabled ? 'CPG: ON' : 'CPG: OFF',
    g1: () => g1Controller?.enabled ? 'CPG: ON' : 'CPG: OFF',
    h1_2: () => h1_2Controller?.enabled ? 'CPG: ON' : 'CPG: OFF',
    factory: () => factoryController?.enabled ? `CPG: ON (${factoryController.numRobots}x)` : 'CPG: OFF',
    evolve: () => evolveRunner?.enabled ? `EVO: ${evolveRunner.getStatusText()}` : 'EVO: OFF',
  };
  const fn = labels[activeController];
  if (fn) {
    controllerBtn.textContent = fn();
    controllerBtn.style.display = '';
  } else {
    controllerBtn.style.display = 'none';
  }
}

function resetScene() {
  if (!model || !data) return;

  if (model.nkey > 0) {
    data.qpos.set(model.key_qpos.slice(0, model.nq));
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    if (model.key_ctrl) data.ctrl.set(model.key_ctrl.slice(0, model.nu));
    mujoco.mj_forward(model, data);
  }

  stepCounter = 0;
  nextBall = 0;
  nextBox = 0;

  if (evolveRunner) {
    evolveRunner.reset();
    evolveRunner.enabled = true;
  }
  if (go2Controller) {
    go2Controller.reset();
    go2Controller.enabled = true;
    for (let i = 0; i < 200; i++) {
      go2Controller.step();
      mujoco.mj_step(model, data);
    }
  }
  if (go2RlController) {
    go2RlController.reset();
    go2RlController.enabled = true;
    for (let i = 0; i < 200; i++) {
      go2RlController.applyPD();
      mujoco.mj_step(model, data);
    }
  }
  if (b2Controller) {
    b2Controller.reset();
    b2Controller.enabled = true;
    for (let i = 0; i < 200; i++) {
      b2Controller.step();
      mujoco.mj_step(model, data);
    }
  }
  if (h1Controller) {
    h1Controller.reset();
    h1Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      h1Controller.step();
      mujoco.mj_step(model, data);
    }
  }
  if (g1Controller) {
    g1Controller.setStandingPose();
    mujoco.mj_forward(model, data);
    g1Controller.reset();
    g1Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      g1Controller.step();
      mujoco.mj_step(model, data);
    }
  }
  if (h1_2Controller) {
    for (const [name, target] of Object.entries(h1_2Controller.homeQpos)) {
      const idx = h1_2Controller.jntIdx[name];
      if (idx !== undefined) data.qpos[idx] = target;
    }
    mujoco.mj_forward(model, data);
    h1_2Controller.reset();
    h1_2Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      h1_2Controller.step();
      mujoco.mj_step(model, data);
    }
  }
  if (factoryController) {
    factoryController.reset();
    factoryController.enabled = true;
    for (let i = 0; i < 200; i++) {
      factoryController.step();
      mujoco.mj_step(model, data);
    }
  }
  updateControllerBtn();
}

function toggleController() {
  if (activeController === 'go2' && go2Controller) {
    go2Controller.enabled = !go2Controller.enabled;
  } else if (activeController === 'go2rl' && go2RlController) {
    go2RlController.enabled = !go2RlController.enabled;
  } else if (activeController === 'h1' && h1Controller) {
    h1Controller.enabled = !h1Controller.enabled;
  } else if (activeController === 'b2' && b2Controller) {
    b2Controller.enabled = !b2Controller.enabled;
  } else if (activeController === 'g1' && g1Controller) {
    g1Controller.enabled = !g1Controller.enabled;
  } else if (activeController === 'h1_2' && h1_2Controller) {
    h1_2Controller.enabled = !h1_2Controller.enabled;
  } else if (activeController === 'factory' && factoryController) {
    factoryController.enabled = !factoryController.enabled;
  } else if (activeController === 'evolve' && evolveRunner) {
    evolveRunner.enabled = !evolveRunner.enabled;
  }
  updateControllerBtn();
}

function cycleSpeed() {
  const idx = SIM_SPEEDS.indexOf(simSpeed);
  simSpeed = SIM_SPEEDS[(idx + 1) % SIM_SPEEDS.length];
  updateSpeedBtn();
}

function updateSpeedBtn() {
  if (speedBtn) speedBtn.textContent = simSpeed + 'x';
}

// ─── Keyboard ───────────────────────────────────────────────────────
function handleInput() {
  const kbFwd = keys['KeyW'] || keys['ArrowUp'];
  const kbBack = keys['KeyS'] || keys['ArrowDown'];
  const kbLeft = keys['KeyA'] || keys['ArrowLeft'];
  const kbRight = keys['KeyD'] || keys['ArrowRight'];
  const kbRotL = keys['KeyQ'];
  const kbRotR = keys['KeyE'];

  let fwd = 0, lat = 0, turn = 0;

  // Keyboard
  if (kbFwd) fwd = 0.8;
  if (kbBack) fwd = -0.4;
  if (kbLeft) lat = 0.4;
  if (kbRight) lat = -0.4;
  if (kbRotL) turn = 0.7;
  if (kbRotR) turn = -0.7;

  // Touch joystick override
  if (Math.abs(touchY) > 0.15 || Math.abs(touchX) > 0.15) {
    fwd = touchY * 0.8;
    lat = -touchX * 0.4;
  }
  if (touchRotL) turn = 0.7;
  if (touchRotR) turn = -0.7;

  // Send commands to active controller
  const ctrl = getActiveCtrl();
  if (ctrl && ctrl.enabled && ctrl.setCommand) {
    if (!ctrl.qwopMode) ctrl.setCommand(fwd, lat, turn);
  }
  // Pass raw key state for QWOP mode
  if (ctrl && ctrl.qwopMode) ctrl.qwopKeys = keys;
}

function getActiveCtrl() {
  switch (activeController) {
    case 'go2': return go2Controller;
    case 'go2rl': return go2RlController;
    case 'h1': return h1Controller;
    case 'b2': return b2Controller;
    case 'g1': return g1Controller;
    case 'h1_2': return h1_2Controller;
    case 'factory': return factoryController;
    case 'evolve': return evolveRunner;
    default: return null;
  }
}

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

  keys[e.code] = true;

  if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
  if (e.code === 'KeyP') toggleController();
  if (e.code === 'KeyR') resetScene();
  if (e.code === 'KeyC') cameraFollow = !cameraFollow;
  if (e.code === 'KeyH' && helpOverlay) {
    helpOverlay.style.display = helpOverlay.style.display === 'none' ? '' : 'none';
  }
  if (e.code === 'KeyM') {
    const c = getActiveCtrl();
    if (c && 'qwopMode' in c) {
      c.qwopMode = !c.qwopMode;
      if (c.qwopMode) c._endTrick?.();
      setStatus(c.qwopMode ? 'QWOP Mode! Q/W=front thighs A/S=front calves I/O=rear thighs K/L=rear calves' : 'CPG Mode');
      updateControllerBtn();
    }
  }
  if (e.code === 'KeyF') spawnObstacle(Math.random() < 0.5 ? 'ball' : 'box');
  // Trick keys (Go2 CPG only)
  if (e.code === 'Digit1') { const c = getActiveCtrl(); if (c?.triggerTrick) c.triggerTrick('jump'); }
  if (e.code === 'Digit2') { const c = getActiveCtrl(); if (c?.triggerTrick) c.triggerTrick('frontflip'); }
  if (e.code === 'Digit3') { const c = getActiveCtrl(); if (c?.triggerTrick) c.triggerTrick('backflip'); }
  if (e.code === 'Digit4') { const c = getActiveCtrl(); if (c?.triggerTrick) c.triggerTrick('sideroll'); }
  if (e.code === 'BracketRight') {
    const idx = SIM_SPEEDS.indexOf(simSpeed);
    if (idx < SIM_SPEEDS.length - 1) { simSpeed = SIM_SPEEDS[idx + 1]; updateSpeedBtn(); }
  }
  if (e.code === 'BracketLeft') {
    const idx = SIM_SPEEDS.indexOf(simSpeed);
    if (idx > 0) { simSpeed = SIM_SPEEDS[idx - 1]; updateSpeedBtn(); }
  }
});

window.addEventListener('keyup', (e) => {
  keys[e.code] = false;
});

// ─── UI Events ──────────────────────────────────────────────────────
sceneSelect.addEventListener('change', async (e) => {
  try { await loadScene(e.target.value); }
  catch (err) { setStatus(`Failed: ${e.target.value}`); console.error(err); }
});

resetBtn.addEventListener('click', resetScene);
if (controllerBtn) controllerBtn.addEventListener('click', toggleController);
if (speedBtn) speedBtn.addEventListener('click', cycleSpeed);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ─── Update Loop ────────────────────────────────────────────────────
function updateBodies() {
  for (const b in bodies) {
    const body = bodies[b];
    const idx = parseInt(b);
    getPosition(data.xpos, idx, body.position);
    getQuaternion(data.xquat, idx, body.quaternion);
    body.updateWorldMatrix(false, false);
  }
}

function followCamera() {
  if (!cameraFollow || !model || !data) return;
  let rootBody = 1;
  if (activeController === 'factory' && factoryController) rootBody = factoryController.centerBodyId;
  if (activeController === 'evolve' && evolveRunner) rootBody = evolveRunner.centerBodyId;
  const x = data.xpos[rootBody * 3 + 0];
  const y = data.xpos[rootBody * 3 + 1];
  const z = data.xpos[rootBody * 3 + 2];
  controls.target.lerp(new THREE.Vector3(x, z, -y), 0.05);
}

function stepController() {
  const ctrl = getActiveCtrl();
  if (!ctrl) return;
  if (ctrl.step) ctrl.step();
}

// ─── Mobile / Pointer Controls ──────────────────────────────────────
function setupControls() {
  const joystickZone = document.getElementById('joystick-zone');
  const joystickBase = document.getElementById('joystick-base');
  const joystickThumb = document.getElementById('joystick-thumb');
  const mobilePanel = document.getElementById('mobile-panel');
  const helpOverlayEl = document.getElementById('help-overlay');

  if (!isTouchDevice) return;

  if (joystickZone) joystickZone.style.display = 'block';
  if (mobilePanel) mobilePanel.style.display = 'flex';
  if (helpOverlayEl) helpOverlayEl.style.display = 'none';

  // Movement joystick
  if (joystickZone && joystickBase && joystickThumb) {
    const baseRadius = 65, thumbHalf = 24, maxDist = 40;
    let movePid = null;

    const updateThumb = (cx, cy) => {
      const rect = joystickBase.getBoundingClientRect();
      let dx = cx - (rect.left + baseRadius);
      let dy = cy - (rect.top + baseRadius);
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxDist) { dx = dx / dist * maxDist; dy = dy / dist * maxDist; }
      joystickThumb.style.left = (baseRadius - thumbHalf + dx) + 'px';
      joystickThumb.style.top = (baseRadius - thumbHalf + dy) + 'px';
      touchX = dx / maxDist;
      touchY = -dy / maxDist;
    };

    const resetThumb = () => {
      joystickThumb.style.left = (baseRadius - thumbHalf) + 'px';
      joystickThumb.style.top = (baseRadius - thumbHalf) + 'px';
      touchX = 0; touchY = 0; movePid = null;
    };

    joystickZone.addEventListener('pointerdown', (e) => {
      e.preventDefault(); e.stopPropagation();
      movePid = e.pointerId;
      joystickZone.setPointerCapture(e.pointerId);
      updateThumb(e.clientX, e.clientY);
    });
    joystickZone.addEventListener('pointermove', (e) => {
      if (e.pointerId !== movePid) return;
      updateThumb(e.clientX, e.clientY);
    });
    joystickZone.addEventListener('pointerup', (e) => { if (e.pointerId === movePid) resetThumb(); });
    joystickZone.addEventListener('pointercancel', (e) => { if (e.pointerId === movePid) resetThumb(); });
  }

  // Buttons
  if (mobilePanel) {
    mobilePanel.querySelectorAll('[data-action]').forEach(btn => {
      const action = btn.dataset.action;
      if (action === 'rotL' || action === 'rotR') {
        btn.addEventListener('pointerdown', (e) => {
          e.preventDefault(); e.stopPropagation();
          if (action === 'rotL') touchRotL = true;
          if (action === 'rotR') touchRotR = true;
        });
        btn.addEventListener('pointerup', () => {
          if (action === 'rotL') touchRotL = false;
          if (action === 'rotR') touchRotR = false;
        });
        btn.addEventListener('pointercancel', () => { touchRotL = false; touchRotR = false; });
      }
      if (action === 'ball') {
        btn.addEventListener('pointerdown', (e) => { e.preventDefault(); e.stopPropagation(); spawnObstacle('ball'); });
      }
      if (action === 'box') {
        btn.addEventListener('pointerdown', (e) => { e.preventDefault(); e.stopPropagation(); spawnObstacle('box'); });
      }
    });
  }
}

// ─── Boot ───────────────────────────────────────────────────────────
(async () => {
  try {
    setStatus('Loading MuJoCo WASM...');
    mujoco = await load_mujoco();

    if (!mujoco.FS.analyzePath('/working').exists) {
      mujoco.FS.mkdir('/working');
    }

    await loadScene('unitree_go2/scene.xml');
    setupControls();
  } catch (e) {
    setStatus('Boot failed');
    console.error(e);
    return;
  }

  const MAX_SUBSTEPS = 160; // up to 16x speed

  async function animate() {
    if (model && data && !paused) {
      handleInput();

      const timestep = model.opt.timestep;
      const frameDt = 1.0 / 60.0;
      const nsteps = Math.min(Math.round(frameDt / timestep * simSpeed), MAX_SUBSTEPS);

      for (let s = 0; s < nsteps; s++) {
        stepController();
        mujoco.mj_step(model, data);
        stepCounter++;
      }

      // Update evolution status display
      if (activeController === 'evolve' && evolveRunner && stepCounter % 50 === 0) {
        updateControllerBtn();
      }

      updateBodies();
      followCamera();
    }

    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
