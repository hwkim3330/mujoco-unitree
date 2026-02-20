/**
 * Asset loader for MuJoCo WASM virtual filesystem.
 * Fetches scene XMLs and mesh files, writes them to /working/.
 */

const SCENE_ASSETS = {
  'unitree_go2/scene.xml': {
    files: [
      'unitree_go2/scene.xml',
      'unitree_go2/go2.xml',
      'unitree_go2/assets/base_0.obj',
      'unitree_go2/assets/base_1.obj',
      'unitree_go2/assets/base_2.obj',
      'unitree_go2/assets/base_3.obj',
      'unitree_go2/assets/base_4.obj',
      'unitree_go2/assets/hip_0.obj',
      'unitree_go2/assets/hip_1.obj',
      'unitree_go2/assets/thigh_0.obj',
      'unitree_go2/assets/thigh_1.obj',
      'unitree_go2/assets/thigh_mirror_0.obj',
      'unitree_go2/assets/thigh_mirror_1.obj',
      'unitree_go2/assets/calf_0.obj',
      'unitree_go2/assets/calf_1.obj',
      'unitree_go2/assets/calf_mirror_0.obj',
      'unitree_go2/assets/calf_mirror_1.obj',
      'unitree_go2/assets/foot.obj',
    ],
  },
  'unitree_h1/scene.xml': {
    files: [
      'unitree_h1/scene.xml',
      'unitree_h1/h1.xml',
      'unitree_h1/assets/pelvis.stl',
      'unitree_h1/assets/torso_link.stl',
      'unitree_h1/assets/logo_link.stl',
      'unitree_h1/assets/left_hip_yaw_link.stl',
      'unitree_h1/assets/left_hip_roll_link.stl',
      'unitree_h1/assets/left_hip_pitch_link.stl',
      'unitree_h1/assets/left_knee_link.stl',
      'unitree_h1/assets/left_ankle_link.stl',
      'unitree_h1/assets/right_hip_yaw_link.stl',
      'unitree_h1/assets/right_hip_roll_link.stl',
      'unitree_h1/assets/right_hip_pitch_link.stl',
      'unitree_h1/assets/right_knee_link.stl',
      'unitree_h1/assets/right_ankle_link.stl',
      'unitree_h1/assets/left_shoulder_pitch_link.stl',
      'unitree_h1/assets/left_shoulder_roll_link.stl',
      'unitree_h1/assets/left_shoulder_yaw_link.stl',
      'unitree_h1/assets/left_elbow_link.stl',
      'unitree_h1/assets/right_shoulder_pitch_link.stl',
      'unitree_h1/assets/right_shoulder_roll_link.stl',
      'unitree_h1/assets/right_shoulder_yaw_link.stl',
      'unitree_h1/assets/right_elbow_link.stl',
    ],
  },
};

const loaded = new Set();

function ensureDir(mujoco, path) {
  const parts = path.split('/').filter(Boolean);
  let current = '';
  for (const part of parts) {
    current += '/' + part;
    try {
      if (!mujoco.FS.analyzePath(current).exists) {
        mujoco.FS.mkdir(current);
      }
    } catch (e) { /* already exists */ }
  }
}

export async function loadSceneAssets(mujoco, scenePath, onStatus) {
  const manifest = SCENE_ASSETS[scenePath];
  if (!manifest) {
    console.warn('No asset manifest for:', scenePath);
    return;
  }

  const filesToLoad = manifest.files.filter(f => !loaded.has(f));
  if (filesToLoad.length === 0) return;

  onStatus?.(`Loading ${filesToLoad.length} assets...`);

  // Ensure directories exist
  const dirs = new Set();
  for (const f of filesToLoad) {
    const dir = '/working/' + f.substring(0, f.lastIndexOf('/'));
    dirs.add(dir);
  }
  for (const d of dirs) ensureDir(mujoco, d);

  // Fetch all files in parallel
  const cacheBust = `?v=${Date.now()}`;
  const results = await Promise.allSettled(
    filesToLoad.map(async (f) => {
      const url = `./assets/scenes/${f}${cacheBust}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${f}`);
      const buf = await resp.arrayBuffer();
      mujoco.FS.writeFile('/working/' + f, new Uint8Array(buf));
      loaded.add(f);
      return f;
    })
  );

  let ok = 0, fail = 0;
  for (const r of results) {
    if (r.status === 'fulfilled') ok++;
    else { fail++; console.warn('Asset load failed:', r.reason); }
  }

  onStatus?.(`Loaded ${ok}/${ok + fail} assets`);
}
