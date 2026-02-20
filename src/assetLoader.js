/**
 * Asset loader for MuJoCo WASM virtual filesystem.
 * Fetches scene XMLs and mesh files, writes them to /working/.
 */

// Shared mesh lists (scene XMLs reuse the same robot model + meshes)
const GO2_MESHES = [
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
];

const H1_MESHES = [
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
];

const B2_MESHES = [
  'unitree_b2/b2.xml',
  'unitree_b2/assets/base_link.obj',
  'unitree_b2/assets/fake_head_Link.STL',
  'unitree_b2/assets/fake_imu_link.STL',
  'unitree_b2/assets/fake_tail_link.STL',
  'unitree_b2/assets/f_dc_link.obj',
  'unitree_b2/assets/FL_calf.obj',
  'unitree_b2/assets/FL_foot.obj',
  'unitree_b2/assets/FL_hip.obj',
  'unitree_b2/assets/FL_thigh.obj',
  'unitree_b2/assets/FL_thigh_protect.obj',
  'unitree_b2/assets/f_oc_link.obj',
  'unitree_b2/assets/FR_calf.obj',
  'unitree_b2/assets/FR_foot.obj',
  'unitree_b2/assets/FR_hip.obj',
  'unitree_b2/assets/FR_thigh.obj',
  'unitree_b2/assets/FR_thigh_protect.obj',
  'unitree_b2/assets/logo_left.obj',
  'unitree_b2/assets/logo_right.obj',
  'unitree_b2/assets/r_dc_link.obj',
  'unitree_b2/assets/RL_calf.obj',
  'unitree_b2/assets/RL_foot.obj',
  'unitree_b2/assets/RL_hip.obj',
  'unitree_b2/assets/RL_thigh.obj',
  'unitree_b2/assets/RL_thigh_protect.obj',
  'unitree_b2/assets/r_oc_link.obj',
  'unitree_b2/assets/RR_calf.obj',
  'unitree_b2/assets/RR_foot.obj',
  'unitree_b2/assets/RR_hip.obj',
  'unitree_b2/assets/RR_thigh.obj',
  'unitree_b2/assets/RR_thigh_protect.obj',
  'unitree_b2/assets/unitree_ladar.obj',
];

const G1_MESHES = [
  'unitree_g1/g1.xml',
  'unitree_g1/meshes/head_link.STL',
  'unitree_g1/meshes/left_ankle_pitch_link.STL',
  'unitree_g1/meshes/left_ankle_roll_link.STL',
  'unitree_g1/meshes/left_elbow_link.STL',
  'unitree_g1/meshes/left_hand_index_0_link.STL',
  'unitree_g1/meshes/left_hand_index_1_link.STL',
  'unitree_g1/meshes/left_hand_middle_0_link.STL',
  'unitree_g1/meshes/left_hand_middle_1_link.STL',
  'unitree_g1/meshes/left_hand_palm_link.STL',
  'unitree_g1/meshes/left_hand_thumb_0_link.STL',
  'unitree_g1/meshes/left_hand_thumb_1_link.STL',
  'unitree_g1/meshes/left_hand_thumb_2_link.STL',
  'unitree_g1/meshes/left_hip_pitch_link.STL',
  'unitree_g1/meshes/left_hip_roll_link.STL',
  'unitree_g1/meshes/left_hip_yaw_link.STL',
  'unitree_g1/meshes/left_knee_link.STL',
  'unitree_g1/meshes/left_rubber_hand.STL',
  'unitree_g1/meshes/left_shoulder_pitch_link.STL',
  'unitree_g1/meshes/left_shoulder_roll_link.STL',
  'unitree_g1/meshes/left_shoulder_yaw_link.STL',
  'unitree_g1/meshes/left_wrist_pitch_link.STL',
  'unitree_g1/meshes/left_wrist_roll_link.STL',
  'unitree_g1/meshes/left_wrist_roll_rubber_hand.STL',
  'unitree_g1/meshes/left_wrist_yaw_link.STL',
  'unitree_g1/meshes/logo_link.STL',
  'unitree_g1/meshes/pelvis_contour_link.STL',
  'unitree_g1/meshes/pelvis.STL',
  'unitree_g1/meshes/right_ankle_pitch_link.STL',
  'unitree_g1/meshes/right_ankle_roll_link.STL',
  'unitree_g1/meshes/right_elbow_link.STL',
  'unitree_g1/meshes/right_hand_index_0_link.STL',
  'unitree_g1/meshes/right_hand_index_1_link.STL',
  'unitree_g1/meshes/right_hand_middle_0_link.STL',
  'unitree_g1/meshes/right_hand_middle_1_link.STL',
  'unitree_g1/meshes/right_hand_palm_link.STL',
  'unitree_g1/meshes/right_hand_thumb_0_link.STL',
  'unitree_g1/meshes/right_hand_thumb_1_link.STL',
  'unitree_g1/meshes/right_hand_thumb_2_link.STL',
  'unitree_g1/meshes/right_hip_pitch_link.STL',
  'unitree_g1/meshes/right_hip_roll_link.STL',
  'unitree_g1/meshes/right_hip_yaw_link.STL',
  'unitree_g1/meshes/right_knee_link.STL',
  'unitree_g1/meshes/right_rubber_hand.STL',
  'unitree_g1/meshes/right_shoulder_pitch_link.STL',
  'unitree_g1/meshes/right_shoulder_roll_link.STL',
  'unitree_g1/meshes/right_shoulder_yaw_link.STL',
  'unitree_g1/meshes/right_wrist_pitch_link.STL',
  'unitree_g1/meshes/right_wrist_roll_link.STL',
  'unitree_g1/meshes/right_wrist_roll_rubber_hand.STL',
  'unitree_g1/meshes/right_wrist_yaw_link.STL',
  'unitree_g1/meshes/torso_constraint_L_link.STL',
  'unitree_g1/meshes/torso_constraint_L_rod_link.STL',
  'unitree_g1/meshes/torso_constraint_R_link.STL',
  'unitree_g1/meshes/torso_constraint_R_rod_link.STL',
  'unitree_g1/meshes/torso_link.STL',
  'unitree_g1/meshes/waist_constraint_L.STL',
  'unitree_g1/meshes/waist_constraint_R.STL',
  'unitree_g1/meshes/waist_roll_link.STL',
  'unitree_g1/meshes/waist_support_link.STL',
  'unitree_g1/meshes/waist_yaw_link.STL',
];

const H1_2_MESHES = [
  'unitree_h1_2/h1_2.xml',
  'unitree_h1_2/meshes/left_ankle_A_link.STL',
  'unitree_h1_2/meshes/left_ankle_A_rod_link.STL',
  'unitree_h1_2/meshes/left_ankle_B_link.STL',
  'unitree_h1_2/meshes/left_ankle_B_rod_link.STL',
  'unitree_h1_2/meshes/left_ankle_pitch_link.STL',
  'unitree_h1_2/meshes/left_ankle_roll_link.STL',
  'unitree_h1_2/meshes/left_elbow_link.STL',
  'unitree_h1_2/meshes/left_hand_link.STL',
  'unitree_h1_2/meshes/left_hip_pitch_link.STL',
  'unitree_h1_2/meshes/left_hip_roll_link.STL',
  'unitree_h1_2/meshes/left_hip_yaw_link.STL',
  'unitree_h1_2/meshes/left_knee_link.STL',
  'unitree_h1_2/meshes/left_shoulder_pitch_link.STL',
  'unitree_h1_2/meshes/left_shoulder_roll_link.STL',
  'unitree_h1_2/meshes/left_shoulder_yaw_link.STL',
  'unitree_h1_2/meshes/left_wrist_pitch_link.STL',
  'unitree_h1_2/meshes/left_wrist_roll_link.STL',
  'unitree_h1_2/meshes/L_hand_base_link.STL',
  'unitree_h1_2/meshes/L_index_intermediate.STL',
  'unitree_h1_2/meshes/L_index_proximal.STL',
  'unitree_h1_2/meshes/link11_L.STL',
  'unitree_h1_2/meshes/link11_R.STL',
  'unitree_h1_2/meshes/link12_L.STL',
  'unitree_h1_2/meshes/link12_R.STL',
  'unitree_h1_2/meshes/link13_L.STL',
  'unitree_h1_2/meshes/link13_R.STL',
  'unitree_h1_2/meshes/link14_L.STL',
  'unitree_h1_2/meshes/link14_R.STL',
  'unitree_h1_2/meshes/link15_L.STL',
  'unitree_h1_2/meshes/link15_R.STL',
  'unitree_h1_2/meshes/link16_L.STL',
  'unitree_h1_2/meshes/link16_R.STL',
  'unitree_h1_2/meshes/link17_L.STL',
  'unitree_h1_2/meshes/link17_R.STL',
  'unitree_h1_2/meshes/link18_L.STL',
  'unitree_h1_2/meshes/link18_R.STL',
  'unitree_h1_2/meshes/link19_L.STL',
  'unitree_h1_2/meshes/link19_R.STL',
  'unitree_h1_2/meshes/link20_L.STL',
  'unitree_h1_2/meshes/link20_R.STL',
  'unitree_h1_2/meshes/link21_L.STL',
  'unitree_h1_2/meshes/link21_R.STL',
  'unitree_h1_2/meshes/link22_L.STL',
  'unitree_h1_2/meshes/link22_R.STL',
  'unitree_h1_2/meshes/L_middle_intermediate.STL',
  'unitree_h1_2/meshes/L_middle_proximal.STL',
  'unitree_h1_2/meshes/logo_link.STL',
  'unitree_h1_2/meshes/L_pinky_intermediate.STL',
  'unitree_h1_2/meshes/L_pinky_proximal.STL',
  'unitree_h1_2/meshes/L_ring_intermediate.STL',
  'unitree_h1_2/meshes/L_ring_proximal.STL',
  'unitree_h1_2/meshes/L_thumb_distal.STL',
  'unitree_h1_2/meshes/L_thumb_intermediate.STL',
  'unitree_h1_2/meshes/L_thumb_proximal_base.STL',
  'unitree_h1_2/meshes/L_thumb_proximal.STL',
  'unitree_h1_2/meshes/pelvis.STL',
  'unitree_h1_2/meshes/R_hand_base_link.STL',
  'unitree_h1_2/meshes/right_ankle_A_link.STL',
  'unitree_h1_2/meshes/right_ankle_A_rod_link.STL',
  'unitree_h1_2/meshes/right_ankle_B_link.STL',
  'unitree_h1_2/meshes/right_ankle_B_rod_link.STL',
  'unitree_h1_2/meshes/right_ankle_link.STL',
  'unitree_h1_2/meshes/right_ankle_pitch_link.STL',
  'unitree_h1_2/meshes/right_ankle_roll_link.STL',
  'unitree_h1_2/meshes/right_elbow_link.STL',
  'unitree_h1_2/meshes/right_hand_link.STL',
  'unitree_h1_2/meshes/right_hip_pitch_link.STL',
  'unitree_h1_2/meshes/right_hip_roll_link.STL',
  'unitree_h1_2/meshes/right_hip_yaw_link.STL',
  'unitree_h1_2/meshes/right_knee_link.STL',
  'unitree_h1_2/meshes/right_pitch_link.STL',
  'unitree_h1_2/meshes/right_shoulder_pitch_link.STL',
  'unitree_h1_2/meshes/right_shoulder_roll_link.STL',
  'unitree_h1_2/meshes/right_shoulder_yaw_link.STL',
  'unitree_h1_2/meshes/right_wrist_pitch_link.STL',
  'unitree_h1_2/meshes/right_wrist_roll_link.STL',
  'unitree_h1_2/meshes/R_index_intermediate.STL',
  'unitree_h1_2/meshes/R_index_proximal.STL',
  'unitree_h1_2/meshes/R_middle_intermediate.STL',
  'unitree_h1_2/meshes/R_middle_proximal.STL',
  'unitree_h1_2/meshes/R_pinky_intermediate.STL',
  'unitree_h1_2/meshes/R_pinky_proximal.STL',
  'unitree_h1_2/meshes/R_ring_intermediate.STL',
  'unitree_h1_2/meshes/R_ring_proximal.STL',
  'unitree_h1_2/meshes/R_thumb_distal.STL',
  'unitree_h1_2/meshes/R_thumb_intermediate.STL',
  'unitree_h1_2/meshes/R_thumb_proximal_base.STL',
  'unitree_h1_2/meshes/R_thumb_proximal.STL',
  'unitree_h1_2/meshes/torso_link.STL',
  'unitree_h1_2/meshes/wrist_yaw_link.STL',
];

const SCENE_ASSETS = {
  'unitree_go2/scene.xml': {
    files: ['unitree_go2/scene.xml', ...GO2_MESHES],
  },
  'unitree_go2/scene_stairs.xml': {
    files: ['unitree_go2/scene_stairs.xml', ...GO2_MESHES],
  },
  'unitree_go2/scene_rough.xml': {
    files: ['unitree_go2/scene_rough.xml', ...GO2_MESHES],
  },
  'unitree_h1/scene.xml': {
    files: ['unitree_h1/scene.xml', ...H1_MESHES],
  },
  'unitree_h1/scene_stairs.xml': {
    files: ['unitree_h1/scene_stairs.xml', ...H1_MESHES],
  },
  'unitree_b2/scene.xml': {
    files: ['unitree_b2/scene.xml', ...B2_MESHES],
  },
  'unitree_g1/scene.xml': {
    files: ['unitree_g1/scene.xml', ...G1_MESHES],
  },
  'unitree_h1_2/scene.xml': {
    files: ['unitree_h1_2/scene.xml', ...H1_2_MESHES],
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
