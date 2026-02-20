// src/main.js
import * as THREE2 from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import load_mujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";

// src/meshBuilder.js
import * as THREE from "three";
function getPosition(buffer, index, target) {
  return target.set(
    buffer[index * 3 + 0],
    buffer[index * 3 + 2],
    -buffer[index * 3 + 1]
  );
}
function getQuaternion(buffer, index, target) {
  return target.set(
    -buffer[index * 4 + 1],
    -buffer[index * 4 + 3],
    buffer[index * 4 + 2],
    -buffer[index * 4 + 0]
  );
}
function buildScene(model2) {
  const textDecoder = new TextDecoder("utf-8");
  const namesArray = new Uint8Array(model2.names);
  const mujocoRoot2 = new THREE.Group();
  mujocoRoot2.name = "MuJoCo Root";
  const bodies2 = {};
  const meshCache = {};
  for (let g = 0; g < model2.ngeom; g++) {
    if (!(model2.geom_group[g] < 3)) continue;
    const b = model2.geom_bodyid[g];
    const type = model2.geom_type[g];
    const size = [
      model2.geom_size[g * 3 + 0],
      model2.geom_size[g * 3 + 1],
      model2.geom_size[g * 3 + 2]
    ];
    if (!(b in bodies2)) {
      bodies2[b] = new THREE.Group();
      const start = model2.name_bodyadr[b];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) end++;
      bodies2[b].name = textDecoder.decode(namesArray.subarray(start, end));
      bodies2[b].bodyID = b;
    }
    let geometry;
    if (type === 0) {
      geometry = new THREE.PlaneGeometry(100, 100);
    } else if (type === 2) {
      geometry = new THREE.SphereGeometry(size[0], 20, 16);
    } else if (type === 3) {
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 12, 16);
    } else if (type === 4) {
      geometry = new THREE.SphereGeometry(1, 20, 16);
    } else if (type === 5) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 20);
    } else if (type === 6) {
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
    } else if (type === 7) {
      const meshID = model2.geom_dataid[g];
      if (meshID in meshCache) {
        geometry = meshCache[meshID];
      } else {
        geometry = buildMeshGeometry(model2, meshID);
        meshCache[meshID] = geometry;
      }
    } else {
      geometry = new THREE.SphereGeometry(Math.max(0.01, size[0] || 0.03), 10, 8);
    }
    const material = buildMaterial(model2, g);
    let mesh;
    if (type === 0) {
      mesh = new THREE.Mesh(geometry, material);
      mesh.rotateX(-Math.PI / 2);
      mesh.receiveShadow = true;
      mesh.castShadow = false;
    } else {
      mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
    mesh.bodyID = b;
    bodies2[b].add(mesh);
    getPosition(model2.geom_pos, g, mesh.position);
    if (type !== 0) {
      getQuaternion(model2.geom_quat, g, mesh.quaternion);
    }
    if (type === 4) {
      mesh.scale.set(size[0], size[2], size[1]);
    }
  }
  for (let b = 0; b < model2.nbody; b++) {
    if (!bodies2[b]) {
      bodies2[b] = new THREE.Group();
      bodies2[b].bodyID = b;
      bodies2[b].name = `body_${b}`;
    }
    if (b === 0) {
      mujocoRoot2.add(bodies2[b]);
    } else {
      (bodies2[0] || mujocoRoot2).add(bodies2[b]);
    }
  }
  return { mujocoRoot: mujocoRoot2, bodies: bodies2 };
}
function buildMeshGeometry(model2, meshID) {
  const geometry = new THREE.BufferGeometry();
  const vertStart = model2.mesh_vertadr[meshID] * 3;
  const vertCount = model2.mesh_vertnum[meshID] * 3;
  const vertexBuffer = new Float32Array(model2.mesh_vert.subarray(vertStart, vertStart + vertCount));
  for (let v = 0; v < vertexBuffer.length; v += 3) {
    const temp = vertexBuffer[v + 1];
    vertexBuffer[v + 1] = vertexBuffer[v + 2];
    vertexBuffer[v + 2] = -temp;
  }
  let normalAdr, normalNum;
  if (model2.mesh_normaladr) {
    normalAdr = model2.mesh_normaladr[meshID];
    normalNum = model2.mesh_normalnum[meshID];
  } else {
    normalAdr = model2.mesh_vertadr[meshID];
    normalNum = model2.mesh_vertnum[meshID];
  }
  const normalBuffer = new Float32Array(model2.mesh_normal.subarray(normalAdr * 3, (normalAdr + normalNum) * 3));
  for (let v = 0; v < normalBuffer.length; v += 3) {
    const temp = normalBuffer[v + 1];
    normalBuffer[v + 1] = normalBuffer[v + 2];
    normalBuffer[v + 2] = -temp;
  }
  const faceStart = model2.mesh_faceadr[meshID] * 3;
  const faceCount = model2.mesh_facenum[meshID] * 3;
  const faceBuffer = model2.mesh_face.subarray(faceStart, faceStart + faceCount);
  const numVerts = model2.mesh_vertnum[meshID];
  const swizzledNormals = new Float32Array(numVerts * 3);
  if (model2.mesh_facenormal) {
    const faceNormalBuffer = model2.mesh_facenormal.subarray(faceStart, faceStart + faceCount);
    for (let t = 0; t < faceCount / 3; t++) {
      const vi0 = faceBuffer[t * 3 + 0];
      const vi1 = faceBuffer[t * 3 + 1];
      const vi2 = faceBuffer[t * 3 + 2];
      const ni0 = faceNormalBuffer[t * 3 + 0];
      const ni1 = faceNormalBuffer[t * 3 + 1];
      const ni2 = faceNormalBuffer[t * 3 + 2];
      for (let c = 0; c < 3; c++) {
        swizzledNormals[vi0 * 3 + c] = normalBuffer[ni0 * 3 + c];
        swizzledNormals[vi1 * 3 + c] = normalBuffer[ni1 * 3 + c];
        swizzledNormals[vi2 * 3 + c] = normalBuffer[ni2 * 3 + c];
      }
    }
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(vertexBuffer, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(swizzledNormals, 3));
  geometry.setIndex(Array.from(faceBuffer));
  geometry.computeVertexNormals();
  return geometry;
}
function buildMaterial(model2, g) {
  let color = [
    model2.geom_rgba[g * 4 + 0],
    model2.geom_rgba[g * 4 + 1],
    model2.geom_rgba[g * 4 + 2],
    model2.geom_rgba[g * 4 + 3]
  ];
  let texture = null;
  if (model2.geom_matid[g] !== -1) {
    const matId = model2.geom_matid[g];
    color = [
      model2.mat_rgba[matId * 4 + 0],
      model2.mat_rgba[matId * 4 + 1],
      model2.mat_rgba[matId * 4 + 2],
      model2.mat_rgba[matId * 4 + 3]
    ];
    if (model2.mat_texid) {
      const mjNTEXROLE = 10;
      const mjTEXROLE_RGB = 1;
      let texId = -1;
      try {
        texId = model2.mat_texid[matId * mjNTEXROLE + mjTEXROLE_RGB];
      } catch (e) {
        try {
          texId = model2.mat_texid[matId];
        } catch (e2) {
        }
      }
      if (texId !== void 0 && texId !== -1 && model2.tex_data) {
        try {
          const width = model2.tex_width[texId];
          const height = model2.tex_height[texId];
          const offset = model2.tex_adr[texId];
          const channels = model2.tex_nchannel ? model2.tex_nchannel[texId] : 3;
          const texData = model2.tex_data;
          const rgbaArray = new Uint8Array(width * height * 4);
          for (let p = 0; p < width * height; p++) {
            rgbaArray[p * 4 + 0] = texData[offset + p * channels + 0];
            rgbaArray[p * 4 + 1] = channels > 1 ? texData[offset + p * channels + 1] : rgbaArray[p * 4];
            rgbaArray[p * 4 + 2] = channels > 2 ? texData[offset + p * channels + 2] : rgbaArray[p * 4];
            rgbaArray[p * 4 + 3] = channels > 3 ? texData[offset + p * channels + 3] : 255;
          }
          texture = new THREE.DataTexture(rgbaArray, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
          if (model2.mat_texrepeat) {
            texture.repeat.set(
              model2.mat_texrepeat[matId * 2 + 0],
              model2.mat_texrepeat[matId * 2 + 1]
            );
          }
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
          texture.needsUpdate = true;
        } catch (e) {
        }
      }
    }
  }
  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(color[0], color[1], color[2]),
    transparent: color[3] < 1,
    opacity: color[3],
    roughness: 0.7,
    metalness: 0.1,
    map: texture
  });
}

// src/assetLoader.js
var GO2_MESHES = [
  "unitree_go2/go2.xml",
  "unitree_go2/assets/base_0.obj",
  "unitree_go2/assets/base_1.obj",
  "unitree_go2/assets/base_2.obj",
  "unitree_go2/assets/base_3.obj",
  "unitree_go2/assets/base_4.obj",
  "unitree_go2/assets/hip_0.obj",
  "unitree_go2/assets/hip_1.obj",
  "unitree_go2/assets/thigh_0.obj",
  "unitree_go2/assets/thigh_1.obj",
  "unitree_go2/assets/thigh_mirror_0.obj",
  "unitree_go2/assets/thigh_mirror_1.obj",
  "unitree_go2/assets/calf_0.obj",
  "unitree_go2/assets/calf_1.obj",
  "unitree_go2/assets/calf_mirror_0.obj",
  "unitree_go2/assets/calf_mirror_1.obj",
  "unitree_go2/assets/foot.obj"
];
var H1_MESHES = [
  "unitree_h1/h1.xml",
  "unitree_h1/assets/pelvis.stl",
  "unitree_h1/assets/torso_link.stl",
  "unitree_h1/assets/logo_link.stl",
  "unitree_h1/assets/left_hip_yaw_link.stl",
  "unitree_h1/assets/left_hip_roll_link.stl",
  "unitree_h1/assets/left_hip_pitch_link.stl",
  "unitree_h1/assets/left_knee_link.stl",
  "unitree_h1/assets/left_ankle_link.stl",
  "unitree_h1/assets/right_hip_yaw_link.stl",
  "unitree_h1/assets/right_hip_roll_link.stl",
  "unitree_h1/assets/right_hip_pitch_link.stl",
  "unitree_h1/assets/right_knee_link.stl",
  "unitree_h1/assets/right_ankle_link.stl",
  "unitree_h1/assets/left_shoulder_pitch_link.stl",
  "unitree_h1/assets/left_shoulder_roll_link.stl",
  "unitree_h1/assets/left_shoulder_yaw_link.stl",
  "unitree_h1/assets/left_elbow_link.stl",
  "unitree_h1/assets/right_shoulder_pitch_link.stl",
  "unitree_h1/assets/right_shoulder_roll_link.stl",
  "unitree_h1/assets/right_shoulder_yaw_link.stl",
  "unitree_h1/assets/right_elbow_link.stl"
];
var B2_MESHES = [
  "unitree_b2/b2.xml",
  "unitree_b2/assets/base_link.obj",
  "unitree_b2/assets/fake_head_Link.STL",
  "unitree_b2/assets/fake_imu_link.STL",
  "unitree_b2/assets/fake_tail_link.STL",
  "unitree_b2/assets/f_dc_link.obj",
  "unitree_b2/assets/FL_calf.obj",
  "unitree_b2/assets/FL_foot.obj",
  "unitree_b2/assets/FL_hip.obj",
  "unitree_b2/assets/FL_thigh.obj",
  "unitree_b2/assets/FL_thigh_protect.obj",
  "unitree_b2/assets/f_oc_link.obj",
  "unitree_b2/assets/FR_calf.obj",
  "unitree_b2/assets/FR_foot.obj",
  "unitree_b2/assets/FR_hip.obj",
  "unitree_b2/assets/FR_thigh.obj",
  "unitree_b2/assets/FR_thigh_protect.obj",
  "unitree_b2/assets/logo_left.obj",
  "unitree_b2/assets/logo_right.obj",
  "unitree_b2/assets/r_dc_link.obj",
  "unitree_b2/assets/RL_calf.obj",
  "unitree_b2/assets/RL_foot.obj",
  "unitree_b2/assets/RL_hip.obj",
  "unitree_b2/assets/RL_thigh.obj",
  "unitree_b2/assets/RL_thigh_protect.obj",
  "unitree_b2/assets/r_oc_link.obj",
  "unitree_b2/assets/RR_calf.obj",
  "unitree_b2/assets/RR_foot.obj",
  "unitree_b2/assets/RR_hip.obj",
  "unitree_b2/assets/RR_thigh.obj",
  "unitree_b2/assets/RR_thigh_protect.obj",
  "unitree_b2/assets/unitree_ladar.obj"
];
var G1_MESHES = [
  "unitree_g1/g1.xml",
  "unitree_g1/meshes/head_link.STL",
  "unitree_g1/meshes/left_ankle_pitch_link.STL",
  "unitree_g1/meshes/left_ankle_roll_link.STL",
  "unitree_g1/meshes/left_elbow_link.STL",
  "unitree_g1/meshes/left_hand_index_0_link.STL",
  "unitree_g1/meshes/left_hand_index_1_link.STL",
  "unitree_g1/meshes/left_hand_middle_0_link.STL",
  "unitree_g1/meshes/left_hand_middle_1_link.STL",
  "unitree_g1/meshes/left_hand_palm_link.STL",
  "unitree_g1/meshes/left_hand_thumb_0_link.STL",
  "unitree_g1/meshes/left_hand_thumb_1_link.STL",
  "unitree_g1/meshes/left_hand_thumb_2_link.STL",
  "unitree_g1/meshes/left_hip_pitch_link.STL",
  "unitree_g1/meshes/left_hip_roll_link.STL",
  "unitree_g1/meshes/left_hip_yaw_link.STL",
  "unitree_g1/meshes/left_knee_link.STL",
  "unitree_g1/meshes/left_rubber_hand.STL",
  "unitree_g1/meshes/left_shoulder_pitch_link.STL",
  "unitree_g1/meshes/left_shoulder_roll_link.STL",
  "unitree_g1/meshes/left_shoulder_yaw_link.STL",
  "unitree_g1/meshes/left_wrist_pitch_link.STL",
  "unitree_g1/meshes/left_wrist_roll_link.STL",
  "unitree_g1/meshes/left_wrist_roll_rubber_hand.STL",
  "unitree_g1/meshes/left_wrist_yaw_link.STL",
  "unitree_g1/meshes/logo_link.STL",
  "unitree_g1/meshes/pelvis_contour_link.STL",
  "unitree_g1/meshes/pelvis.STL",
  "unitree_g1/meshes/right_ankle_pitch_link.STL",
  "unitree_g1/meshes/right_ankle_roll_link.STL",
  "unitree_g1/meshes/right_elbow_link.STL",
  "unitree_g1/meshes/right_hand_index_0_link.STL",
  "unitree_g1/meshes/right_hand_index_1_link.STL",
  "unitree_g1/meshes/right_hand_middle_0_link.STL",
  "unitree_g1/meshes/right_hand_middle_1_link.STL",
  "unitree_g1/meshes/right_hand_palm_link.STL",
  "unitree_g1/meshes/right_hand_thumb_0_link.STL",
  "unitree_g1/meshes/right_hand_thumb_1_link.STL",
  "unitree_g1/meshes/right_hand_thumb_2_link.STL",
  "unitree_g1/meshes/right_hip_pitch_link.STL",
  "unitree_g1/meshes/right_hip_roll_link.STL",
  "unitree_g1/meshes/right_hip_yaw_link.STL",
  "unitree_g1/meshes/right_knee_link.STL",
  "unitree_g1/meshes/right_rubber_hand.STL",
  "unitree_g1/meshes/right_shoulder_pitch_link.STL",
  "unitree_g1/meshes/right_shoulder_roll_link.STL",
  "unitree_g1/meshes/right_shoulder_yaw_link.STL",
  "unitree_g1/meshes/right_wrist_pitch_link.STL",
  "unitree_g1/meshes/right_wrist_roll_link.STL",
  "unitree_g1/meshes/right_wrist_roll_rubber_hand.STL",
  "unitree_g1/meshes/right_wrist_yaw_link.STL",
  "unitree_g1/meshes/torso_constraint_L_link.STL",
  "unitree_g1/meshes/torso_constraint_L_rod_link.STL",
  "unitree_g1/meshes/torso_constraint_R_link.STL",
  "unitree_g1/meshes/torso_constraint_R_rod_link.STL",
  "unitree_g1/meshes/torso_link.STL",
  "unitree_g1/meshes/waist_constraint_L.STL",
  "unitree_g1/meshes/waist_constraint_R.STL",
  "unitree_g1/meshes/waist_roll_link.STL",
  "unitree_g1/meshes/waist_support_link.STL",
  "unitree_g1/meshes/waist_yaw_link.STL"
];
var H1_2_MESHES = [
  "unitree_h1_2/h1_2.xml",
  "unitree_h1_2/meshes/left_ankle_A_link.STL",
  "unitree_h1_2/meshes/left_ankle_A_rod_link.STL",
  "unitree_h1_2/meshes/left_ankle_B_link.STL",
  "unitree_h1_2/meshes/left_ankle_B_rod_link.STL",
  "unitree_h1_2/meshes/left_ankle_pitch_link.STL",
  "unitree_h1_2/meshes/left_ankle_roll_link.STL",
  "unitree_h1_2/meshes/left_elbow_link.STL",
  "unitree_h1_2/meshes/left_hand_link.STL",
  "unitree_h1_2/meshes/left_hip_pitch_link.STL",
  "unitree_h1_2/meshes/left_hip_roll_link.STL",
  "unitree_h1_2/meshes/left_hip_yaw_link.STL",
  "unitree_h1_2/meshes/left_knee_link.STL",
  "unitree_h1_2/meshes/left_shoulder_pitch_link.STL",
  "unitree_h1_2/meshes/left_shoulder_roll_link.STL",
  "unitree_h1_2/meshes/left_shoulder_yaw_link.STL",
  "unitree_h1_2/meshes/left_wrist_pitch_link.STL",
  "unitree_h1_2/meshes/left_wrist_roll_link.STL",
  "unitree_h1_2/meshes/L_hand_base_link.STL",
  "unitree_h1_2/meshes/L_index_intermediate.STL",
  "unitree_h1_2/meshes/L_index_proximal.STL",
  "unitree_h1_2/meshes/link11_L.STL",
  "unitree_h1_2/meshes/link11_R.STL",
  "unitree_h1_2/meshes/link12_L.STL",
  "unitree_h1_2/meshes/link12_R.STL",
  "unitree_h1_2/meshes/link13_L.STL",
  "unitree_h1_2/meshes/link13_R.STL",
  "unitree_h1_2/meshes/link14_L.STL",
  "unitree_h1_2/meshes/link14_R.STL",
  "unitree_h1_2/meshes/link15_L.STL",
  "unitree_h1_2/meshes/link15_R.STL",
  "unitree_h1_2/meshes/link16_L.STL",
  "unitree_h1_2/meshes/link16_R.STL",
  "unitree_h1_2/meshes/link17_L.STL",
  "unitree_h1_2/meshes/link17_R.STL",
  "unitree_h1_2/meshes/link18_L.STL",
  "unitree_h1_2/meshes/link18_R.STL",
  "unitree_h1_2/meshes/link19_L.STL",
  "unitree_h1_2/meshes/link19_R.STL",
  "unitree_h1_2/meshes/link20_L.STL",
  "unitree_h1_2/meshes/link20_R.STL",
  "unitree_h1_2/meshes/link21_L.STL",
  "unitree_h1_2/meshes/link21_R.STL",
  "unitree_h1_2/meshes/link22_L.STL",
  "unitree_h1_2/meshes/link22_R.STL",
  "unitree_h1_2/meshes/L_middle_intermediate.STL",
  "unitree_h1_2/meshes/L_middle_proximal.STL",
  "unitree_h1_2/meshes/logo_link.STL",
  "unitree_h1_2/meshes/L_pinky_intermediate.STL",
  "unitree_h1_2/meshes/L_pinky_proximal.STL",
  "unitree_h1_2/meshes/L_ring_intermediate.STL",
  "unitree_h1_2/meshes/L_ring_proximal.STL",
  "unitree_h1_2/meshes/L_thumb_distal.STL",
  "unitree_h1_2/meshes/L_thumb_intermediate.STL",
  "unitree_h1_2/meshes/L_thumb_proximal_base.STL",
  "unitree_h1_2/meshes/L_thumb_proximal.STL",
  "unitree_h1_2/meshes/pelvis.STL",
  "unitree_h1_2/meshes/R_hand_base_link.STL",
  "unitree_h1_2/meshes/right_ankle_A_link.STL",
  "unitree_h1_2/meshes/right_ankle_A_rod_link.STL",
  "unitree_h1_2/meshes/right_ankle_B_link.STL",
  "unitree_h1_2/meshes/right_ankle_B_rod_link.STL",
  "unitree_h1_2/meshes/right_ankle_link.STL",
  "unitree_h1_2/meshes/right_ankle_pitch_link.STL",
  "unitree_h1_2/meshes/right_ankle_roll_link.STL",
  "unitree_h1_2/meshes/right_elbow_link.STL",
  "unitree_h1_2/meshes/right_hand_link.STL",
  "unitree_h1_2/meshes/right_hip_pitch_link.STL",
  "unitree_h1_2/meshes/right_hip_roll_link.STL",
  "unitree_h1_2/meshes/right_hip_yaw_link.STL",
  "unitree_h1_2/meshes/right_knee_link.STL",
  "unitree_h1_2/meshes/right_pitch_link.STL",
  "unitree_h1_2/meshes/right_shoulder_pitch_link.STL",
  "unitree_h1_2/meshes/right_shoulder_roll_link.STL",
  "unitree_h1_2/meshes/right_shoulder_yaw_link.STL",
  "unitree_h1_2/meshes/right_wrist_pitch_link.STL",
  "unitree_h1_2/meshes/right_wrist_roll_link.STL",
  "unitree_h1_2/meshes/R_index_intermediate.STL",
  "unitree_h1_2/meshes/R_index_proximal.STL",
  "unitree_h1_2/meshes/R_middle_intermediate.STL",
  "unitree_h1_2/meshes/R_middle_proximal.STL",
  "unitree_h1_2/meshes/R_pinky_intermediate.STL",
  "unitree_h1_2/meshes/R_pinky_proximal.STL",
  "unitree_h1_2/meshes/R_ring_intermediate.STL",
  "unitree_h1_2/meshes/R_ring_proximal.STL",
  "unitree_h1_2/meshes/R_thumb_distal.STL",
  "unitree_h1_2/meshes/R_thumb_intermediate.STL",
  "unitree_h1_2/meshes/R_thumb_proximal_base.STL",
  "unitree_h1_2/meshes/R_thumb_proximal.STL",
  "unitree_h1_2/meshes/torso_link.STL",
  "unitree_h1_2/meshes/wrist_yaw_link.STL"
];
var SCENE_ASSETS = {
  "unitree_go2/scene.xml": {
    files: ["unitree_go2/scene.xml", ...GO2_MESHES]
  },
  "unitree_go2/scene_stairs.xml": {
    files: ["unitree_go2/scene_stairs.xml", ...GO2_MESHES]
  },
  "unitree_go2/scene_rough.xml": {
    files: ["unitree_go2/scene_rough.xml", ...GO2_MESHES]
  },
  "unitree_h1/scene.xml": {
    files: ["unitree_h1/scene.xml", ...H1_MESHES]
  },
  "unitree_h1/scene_stairs.xml": {
    files: ["unitree_h1/scene_stairs.xml", ...H1_MESHES]
  },
  "unitree_b2/scene.xml": {
    files: ["unitree_b2/scene.xml", ...B2_MESHES]
  },
  "unitree_g1/scene.xml": {
    files: ["unitree_g1/scene.xml", ...G1_MESHES]
  },
  "unitree_h1_2/scene.xml": {
    files: ["unitree_h1_2/scene.xml", ...H1_2_MESHES]
  }
};
var loaded = /* @__PURE__ */ new Set();
function ensureDir(mujoco2, path) {
  const parts = path.split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current += "/" + part;
    try {
      if (!mujoco2.FS.analyzePath(current).exists) {
        mujoco2.FS.mkdir(current);
      }
    } catch (e) {
    }
  }
}
async function loadSceneAssets(mujoco2, scenePath, onStatus) {
  const manifest = SCENE_ASSETS[scenePath];
  if (!manifest) {
    console.warn("No asset manifest for:", scenePath);
    return;
  }
  const filesToLoad = manifest.files.filter((f) => !loaded.has(f));
  if (filesToLoad.length === 0) return;
  onStatus?.(`Loading ${filesToLoad.length} assets...`);
  const dirs = /* @__PURE__ */ new Set();
  for (const f of filesToLoad) {
    const dir = "/working/" + f.substring(0, f.lastIndexOf("/"));
    dirs.add(dir);
  }
  for (const d of dirs) ensureDir(mujoco2, d);
  const cacheBust = `?v=${Date.now()}`;
  const results = await Promise.allSettled(
    filesToLoad.map(async (f) => {
      const url = `./assets/scenes/${f}${cacheBust}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${f}`);
      const buf = await resp.arrayBuffer();
      mujoco2.FS.writeFile("/working/" + f, new Uint8Array(buf));
      loaded.add(f);
      return f;
    })
  );
  let ok = 0, fail = 0;
  for (const r of results) {
    if (r.status === "fulfilled") ok++;
    else {
      fail++;
      console.warn("Asset load failed:", r.reason);
    }
  }
  onStatus?.(`Loaded ${ok}/${ok + fail} assets`);
}

// src/go2CpgController.js
var Go2CpgController = class {
  constructor(mujoco2, model2, data2) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.enabled = false;
    this.simDt = model2.opt.timestep || 2e-3;
    this.frequency = 2.5;
    this.phase = 0;
    this.thighAmp = 0.25;
    this.calfAmp = 0.25;
    this.hipAmp = 0.04;
    this.hipKp = 120;
    this.hipKd = 4;
    this.thighKp = 200;
    this.thighKd = 6;
    this.calfKp = 250;
    this.calfKd = 8;
    this.balanceKp = 60;
    this.balanceKd = 5;
    this.homeHip = 0;
    this.homeThigh = 0.9;
    this.homeCalf = -1.8;
    this.legs = {
      FL: { act: [0, 1, 2], phase: 0, side: 1, prefix: "FL" },
      FR: { act: [3, 4, 5], phase: Math.PI, side: -1, prefix: "FR" },
      RL: { act: [6, 7, 8], phase: Math.PI, side: 1, prefix: "RL" },
      RR: { act: [9, 10, 11], phase: 0, side: -1, prefix: "RR" }
    };
    this.jntQpos = {};
    this.jntDof = {};
    this.findJointIndices();
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
  }
  findJointIndices() {
    const jointNames = [
      "FL_hip_joint",
      "FL_thigh_joint",
      "FL_calf_joint",
      "FR_hip_joint",
      "FR_thigh_joint",
      "FR_calf_joint",
      "RL_hip_joint",
      "RL_thigh_joint",
      "RL_calf_joint",
      "RR_hip_joint",
      "RR_thigh_joint",
      "RR_calf_joint"
    ];
    for (const name of jointNames) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0) {
          this.jntQpos[name] = this.model.jnt_qposadr[jid];
          this.jntDof[name] = this.model.jnt_dofadr[jid];
        }
      } catch (e) {
      }
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
    if (qIdx === void 0) return 0;
    const q = this.data.qpos[qIdx];
    const qdot = this.data.qvel[dIdx] || 0;
    return kp * (target - q) - kd * qdot;
  }
  step() {
    if (!this.enabled) return;
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;
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
      const isFront = name.startsWith("F");
      const turnSign = isFront ? 1 : -1;
      const thighTarget = this.homeThigh - direction * this.thighAmp * ampScale * swing + this.turnRate * 0.06 * turnSign * leg.side;
      const calfTarget = this.homeCalf - this.calfAmp * ampScale * (isSwing ? Math.sin(legPhase) : 0);
      const hipTarget = this.homeHip + leg.side * this.hipAmp * (isSwing ? 1 : -1) * ampScale + this.lateralSpeed * 0.08 * leg.side;
      const hipJoint = `${leg.prefix}_hip_joint`;
      const thighJoint = `${leg.prefix}_thigh_joint`;
      const calfJoint = `${leg.prefix}_calf_joint`;
      ctrl[leg.act[0]] = this.pdTorque(hipJoint, hipTarget, this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(thighJoint, thighTarget, this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(calfJoint, calfTarget, this.calfKp, this.calfKd);
      ctrl[leg.act[1]] += pitchCorr * 0.2;
      ctrl[leg.act[0]] += rollCorr * 0.15 * leg.side;
    }
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
};

// src/go2OnnxController.js
var Go2OnnxController = class {
  constructor(mujoco2, model2, data2) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.enabled = false;
    this.session = null;
    this.simDt = model2.opt.timestep || 2e-3;
    this.decimation = 10;
    this.stepCount = 0;
    this.Kp = 50;
    this.Kd = 1.5;
    this.actionScale = 0.25;
    this.defaultPos = new Float32Array([
      0,
      0,
      0,
      0,
      // hips
      1.1,
      1.1,
      1.1,
      1.1,
      // thighs
      -1.8,
      -1.8,
      -1.8,
      -1.8
      // calfs
    ]);
    this.defaultPosMJ = new Float32Array([
      0,
      1.1,
      -1.8,
      // FL
      0,
      1.1,
      -1.8,
      // FR
      0,
      1.1,
      -1.8,
      // RL
      0,
      1.1,
      -1.8
      // RR
    ]);
    this.MJ_TO_IL = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];
    this.IL_TO_MJ = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];
    this.lastAction = new Float32Array(12);
    this.currentTargets = new Float32Array(12);
    for (let i = 0; i < 12; i++) this.currentTargets[i] = this.defaultPos[i];
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this.jntQpos = new Int32Array(12);
    this.jntDof = new Int32Array(12);
    this.findJointIndices();
    this.footGeomIds = [-1, -1, -1, -1];
    this.floorGeomId = -1;
    this.findFootGeoms();
    this._inferring = false;
  }
  findJointIndices() {
    const jointNames = [
      "FL_hip_joint",
      "FL_thigh_joint",
      "FL_calf_joint",
      "FR_hip_joint",
      "FR_thigh_joint",
      "FR_calf_joint",
      "RL_hip_joint",
      "RL_thigh_joint",
      "RL_calf_joint",
      "RR_hip_joint",
      "RR_thigh_joint",
      "RR_calf_joint"
    ];
    for (let i = 0; i < 12; i++) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, jointNames[i]);
        if (jid >= 0) {
          this.jntQpos[i] = this.model.jnt_qposadr[jid];
          this.jntDof[i] = this.model.jnt_dofadr[jid];
        }
      } catch (e) {
      }
    }
  }
  findFootGeoms() {
    const footGeomNames = ["FL", "FR", "RL", "RR"];
    for (let i = 0; i < 4; i++) {
      try {
        const gid = this.mujoco.mj_name2id(this.model, 5, footGeomNames[i]);
        if (gid >= 0) this.footGeomIds[i] = gid;
      } catch (e) {
      }
    }
    try {
      this.floorGeomId = this.mujoco.mj_name2id(this.model, 5, "floor");
    } catch (e) {
    }
  }
  async loadModel(modelPath) {
    if (typeof ort === "undefined") {
      console.warn("ONNX Runtime Web not loaded");
      return false;
    }
    try {
      this.session = await ort.InferenceSession.create(modelPath);
      console.log("Go2 ONNX policy loaded:", modelPath);
      return true;
    } catch (e) {
      console.error("Failed to load ONNX model:", e);
      return false;
    }
  }
  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }
  /**
   * Set initial joint positions to IsaacLab defaults.
   * Call before warm-up so the robot starts in the pose the policy expects.
   */
  setInitialPose() {
    for (let i = 0; i < 12; i++) {
      this.data.qpos[this.jntQpos[i]] = this.defaultPosMJ[i];
    }
    this.data.qpos[2] = 0.35;
    for (let i = 0; i < this.model.nv; i++) this.data.qvel[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
  }
  getBodyAngVel() {
    const wx = this.data.qvel[3];
    const wy = this.data.qvel[4];
    const wz = this.data.qvel[5];
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];
    return this.rotateByInvQuat(wx, wy, wz, qw, qx, qy, qz);
  }
  getProjectedGravity() {
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];
    return this.rotateByInvQuat(0, 0, -1, qw, qx, qy, qz);
  }
  rotateByInvQuat(vx, vy, vz, qw, qx, qy, qz) {
    const iqx = -qx, iqy = -qy, iqz = -qz;
    const tx = 2 * (iqy * vz - iqz * vy);
    const ty = 2 * (iqz * vx - iqx * vz);
    const tz = 2 * (iqx * vy - iqy * vx);
    return [
      vx + qw * tx + (iqy * tz - iqz * ty),
      vy + qw * ty + (iqz * tx - iqx * tz),
      vz + qw * tz + (iqx * ty - iqy * tx)
    ];
  }
  getFootContacts() {
    const contacts = [0, 0, 0, 0];
    for (let c = 0; c < 100; c++) {
      try {
        const contact = this.data.contact.get(c);
        if (!contact) break;
        const g1 = contact.geom1;
        const g2 = contact.geom2;
        const isFloor1 = g1 === this.floorGeomId;
        const isFloor2 = g2 === this.floorGeomId;
        if (!isFloor1 && !isFloor2) continue;
        const otherGeom = isFloor1 ? g2 : g1;
        for (let f = 0; f < 4; f++) {
          if (otherGeom === this.footGeomIds[f]) contacts[f] = 1;
        }
      } catch (e) {
        break;
      }
    }
    return contacts;
  }
  buildObs() {
    const obs = new Float32Array(49);
    const angVel = this.getBodyAngVel();
    obs[0] = angVel[0];
    obs[1] = angVel[1];
    obs[2] = angVel[2];
    const grav = this.getProjectedGravity();
    obs[3] = grav[0];
    obs[4] = grav[1];
    obs[5] = grav[2];
    obs[6] = this.forwardSpeed;
    obs[7] = this.lateralSpeed;
    obs[8] = this.turnRate;
    for (let i = 0; i < 12; i++) {
      const mjIdx = this.MJ_TO_IL[i];
      const q = this.data.qpos[this.jntQpos[mjIdx]];
      obs[9 + i] = q - this.defaultPos[i];
    }
    for (let i = 0; i < 12; i++) {
      const mjIdx = this.MJ_TO_IL[i];
      obs[21 + i] = this.data.qvel[this.jntDof[mjIdx]];
    }
    const contacts = this.getFootContacts();
    obs[33] = contacts[0];
    obs[34] = contacts[1];
    obs[35] = contacts[2];
    obs[36] = contacts[3];
    for (let i = 0; i < 12; i++) {
      obs[37 + i] = this.lastAction[i];
    }
    for (let i = 0; i < 49; i++) {
      obs[i] = Math.max(-100, Math.min(100, obs[i]));
    }
    return obs;
  }
  applyPD() {
    const ctrl = this.data.ctrl;
    for (let i = 0; i < 12; i++) {
      const mjIdx = this.IL_TO_MJ[i];
      const q = this.data.qpos[this.jntQpos[mjIdx]];
      const qdot = this.data.qvel[this.jntDof[mjIdx]];
      const target = this.currentTargets[i];
      const torque = this.Kp * (target - q) - this.Kd * qdot;
      ctrl[mjIdx] = torque;
    }
    if (this.model.actuator_ctrlrange) {
      for (let i = 0; i < this.model.nu; i++) {
        const lo = this.model.actuator_ctrlrange[i * 2];
        const hi = this.model.actuator_ctrlrange[i * 2 + 1];
        ctrl[i] = Math.max(lo, Math.min(hi, ctrl[i]));
      }
    }
  }
  step() {
    if (!this.enabled) return;
    this.applyPD();
    this.stepCount++;
    if (this.stepCount % this.decimation !== 0) return;
    if (!this.session || this._inferring) return;
    this._runInference();
  }
  async _runInference() {
    this._inferring = true;
    try {
      const obs = this.buildObs();
      const input = new ort.Tensor("float32", obs, [1, 49]);
      const results = await this.session.run({ obs: input });
      const actions = results.actions.data;
      for (let i = 0; i < 12; i++) {
        const a = Math.max(-5, Math.min(5, actions[i]));
        this.lastAction[i] = a;
        this.currentTargets[i] = a * this.actionScale + this.defaultPos[i];
      }
    } catch (e) {
      console.error("ONNX inference error:", e);
    }
    this._inferring = false;
  }
  reset() {
    this.stepCount = 0;
    this.lastAction.fill(0);
    for (let i = 0; i < 12; i++) {
      this.currentTargets[i] = this.defaultPos[i];
    }
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this._inferring = false;
  }
};

// src/h1CpgController.js
var H1CpgController = class {
  constructor(mujoco2, model2, data2) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.enabled = false;
    this.simDt = model2.opt.timestep || 2e-3;
    this.frequency = 1.2;
    this.phase = 0;
    this.hipPitchAmp = 0.25;
    this.kneeAmp = 0.35;
    this.ankleAmp = 0.15;
    this.hipRollAmp = 0.03;
    this.armSwingGain = 0.3;
    this.balanceKp = 300;
    this.balanceKd = 20;
    this.homeQpos = {
      left_hip_yaw: 0,
      left_hip_roll: 0,
      left_hip_pitch: -0.4,
      left_knee: 0.8,
      left_ankle: -0.4,
      right_hip_yaw: 0,
      right_hip_roll: 0,
      right_hip_pitch: -0.4,
      right_knee: 0.8,
      right_ankle: -0.4,
      torso: 0,
      left_shoulder_pitch: 0,
      left_shoulder_roll: 0.2,
      left_shoulder_yaw: 0,
      left_elbow: -0.3,
      right_shoulder_pitch: 0,
      right_shoulder_roll: -0.2,
      right_shoulder_yaw: 0,
      right_elbow: -0.3
    };
    this.actIdx = {
      left_hip_yaw: 0,
      left_hip_roll: 1,
      left_hip_pitch: 2,
      left_knee: 3,
      left_ankle: 4,
      right_hip_yaw: 5,
      right_hip_roll: 6,
      right_hip_pitch: 7,
      right_knee: 8,
      right_ankle: 9,
      torso: 10,
      left_shoulder_pitch: 11,
      left_shoulder_roll: 12,
      left_shoulder_yaw: 13,
      left_elbow: 14,
      right_shoulder_pitch: 15,
      right_shoulder_roll: 16,
      right_shoulder_yaw: 17,
      right_elbow: 18
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
    const names = Object.keys(this.actIdx);
    for (const name of names) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0 && this.model.jnt_qposadr) {
          this.jntIdx[name] = this.model.jnt_qposadr[jid];
        }
      } catch (e) {
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
    const sinp = 2 * (qw * qy - qz * qx);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
    const roll = Math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
    return { pitch, roll };
  }
  /**
   * Compute PD torque for a joint to track a target position.
   * H1 uses torque actuators, so we compute: tau = kp*(target - q) - kd*qdot
   */
  pdTorque(jointName, target, kp, kd) {
    const idx = this.jntIdx[jointName];
    if (idx === void 0) return 0;
    let dofIdx;
    try {
      const jid = this.mujoco.mj_name2id(this.model, 3, jointName);
      dofIdx = this.model.jnt_dofadr[jid];
    } catch (e) {
      dofIdx = idx - 7 + 6;
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
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI;
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;
    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;
    const hipKp = 800, hipKd = 30;
    const kneeKp = 1200, kneeKd = 40;
    const ankleKp = 200, ankleKd = 10;
    const torsoKp = 600, torsoKd = 25;
    const armKp = 50, armKd = 3;
    const ctrl = this.data.ctrl;
    const leftSwing = Math.sin(leftPhase);
    const leftStance = Math.max(0, -Math.sin(leftPhase));
    const leftHipPitchTarget = this.homeQpos.left_hip_pitch + direction * this.hipPitchAmp * ampScale * leftSwing;
    const leftKneeTarget = this.homeQpos.left_knee + this.kneeAmp * ampScale * Math.max(0, Math.sin(leftPhase));
    const leftAnkleTarget = this.homeQpos.left_ankle - this.ankleAmp * ampScale * leftSwing;
    const leftHipRollTarget = this.homeQpos.left_hip_roll - this.hipRollAmp * leftStance + this.lateralSpeed * 0.05;
    const leftHipYawTarget = this.homeQpos.left_hip_yaw + this.turnRate * 0.05 * leftSwing;
    ctrl[this.actIdx.left_hip_yaw] = this.pdTorque("left_hip_yaw", leftHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll] = this.pdTorque("left_hip_roll", leftHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_pitch] = this.pdTorque("left_hip_pitch", leftHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_knee] = this.pdTorque("left_knee", leftKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle] = this.pdTorque("left_ankle", leftAnkleTarget, ankleKp, ankleKd);
    const rightSwing = Math.sin(rightPhase);
    const rightStance = Math.max(0, -Math.sin(rightPhase));
    const rightHipPitchTarget = this.homeQpos.right_hip_pitch + direction * this.hipPitchAmp * ampScale * rightSwing;
    const rightKneeTarget = this.homeQpos.right_knee + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rightAnkleTarget = this.homeQpos.right_ankle - this.ankleAmp * ampScale * rightSwing;
    const rightHipRollTarget = this.homeQpos.right_hip_roll + this.hipRollAmp * rightStance + this.lateralSpeed * 0.05;
    const rightHipYawTarget = this.homeQpos.right_hip_yaw + this.turnRate * 0.05 * rightSwing;
    ctrl[this.actIdx.right_hip_yaw] = this.pdTorque("right_hip_yaw", rightHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll] = this.pdTorque("right_hip_roll", rightHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_pitch] = this.pdTorque("right_hip_pitch", rightHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_knee] = this.pdTorque("right_knee", rightKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle] = this.pdTorque("right_ankle", rightAnkleTarget, ankleKp, ankleKd);
    const torsoTarget = this.homeQpos.torso + this.turnRate * 0.1;
    ctrl[this.actIdx.torso] = this.pdTorque("torso", torsoTarget, torsoKp, torsoKd);
    const pitchCorrection = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorrection = -this.balanceKp * roll - this.balanceKd * rollRate;
    ctrl[this.actIdx.left_ankle] += pitchCorrection * 0.4;
    ctrl[this.actIdx.right_ankle] += pitchCorrection * 0.4;
    ctrl[this.actIdx.left_hip_pitch] += pitchCorrection * 0.5;
    ctrl[this.actIdx.right_hip_pitch] += pitchCorrection * 0.5;
    ctrl[this.actIdx.left_hip_roll] += rollCorrection * 0.3;
    ctrl[this.actIdx.right_hip_roll] -= rollCorrection * 0.3;
    const leftArmSwing = -this.armSwingGain * direction * ampScale * leftSwing;
    const rightArmSwing = -this.armSwingGain * direction * ampScale * rightSwing;
    ctrl[this.actIdx.left_shoulder_pitch] = this.pdTorque(
      "left_shoulder_pitch",
      this.homeQpos.left_shoulder_pitch + leftArmSwing,
      armKp,
      armKd
    );
    ctrl[this.actIdx.left_shoulder_roll] = this.pdTorque(
      "left_shoulder_roll",
      this.homeQpos.left_shoulder_roll,
      armKp,
      armKd
    );
    ctrl[this.actIdx.left_shoulder_yaw] = this.pdTorque(
      "left_shoulder_yaw",
      this.homeQpos.left_shoulder_yaw,
      armKp * 0.5,
      armKd
    );
    ctrl[this.actIdx.left_elbow] = this.pdTorque(
      "left_elbow",
      this.homeQpos.left_elbow,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_pitch] = this.pdTorque(
      "right_shoulder_pitch",
      this.homeQpos.right_shoulder_pitch + rightArmSwing,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_roll] = this.pdTorque(
      "right_shoulder_roll",
      this.homeQpos.right_shoulder_roll,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_yaw] = this.pdTorque(
      "right_shoulder_yaw",
      this.homeQpos.right_shoulder_yaw,
      armKp * 0.5,
      armKd
    );
    ctrl[this.actIdx.right_elbow] = this.pdTorque(
      "right_elbow",
      this.homeQpos.right_elbow,
      armKp,
      armKd
    );
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
};

// src/b2CpgController.js
var B2CpgController = class {
  constructor(mujoco2, model2, data2) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.enabled = false;
    this.simDt = model2.opt.timestep || 2e-3;
    this.frequency = 2;
    this.phase = 0;
    this.thighAmp = 0.2;
    this.calfAmp = 0.2;
    this.hipAmp = 0.03;
    this.hipKp = 500;
    this.hipKd = 15;
    this.thighKp = 800;
    this.thighKd = 25;
    this.calfKp = 1e3;
    this.calfKd = 30;
    this.balanceKp = 200;
    this.balanceKd = 15;
    this.homeHip = 0;
    this.homeThigh = 1.28;
    this.homeCalf = -2.84;
    this.legs = {
      FR: { act: [0, 1, 2], phase: 0, side: -1, prefix: "FR" },
      FL: { act: [3, 4, 5], phase: Math.PI, side: 1, prefix: "FL" },
      RR: { act: [6, 7, 8], phase: Math.PI, side: -1, prefix: "RR" },
      RL: { act: [9, 10, 11], phase: 0, side: 1, prefix: "RL" }
    };
    this.jntQpos = {};
    this.jntDof = {};
    this.findJointIndices();
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
  }
  findJointIndices() {
    const jointNames = [
      "FR_hip_joint",
      "FR_thigh_joint",
      "FR_calf_joint",
      "FL_hip_joint",
      "FL_thigh_joint",
      "FL_calf_joint",
      "RR_hip_joint",
      "RR_thigh_joint",
      "RR_calf_joint",
      "RL_hip_joint",
      "RL_thigh_joint",
      "RL_calf_joint"
    ];
    for (const name of jointNames) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0) {
          this.jntQpos[name] = this.model.jnt_qposadr[jid];
          this.jntDof[name] = this.model.jnt_dofadr[jid];
        }
      } catch (e) {
      }
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
    if (qIdx === void 0) return 0;
    const q = this.data.qpos[qIdx];
    const qdot = this.data.qvel[dIdx] || 0;
    return kp * (target - q) - kd * qdot;
  }
  step() {
    if (!this.enabled) return;
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;
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
      const isFront = name.startsWith("F");
      const turnSign = isFront ? 1 : -1;
      const thighTarget = this.homeThigh - direction * this.thighAmp * ampScale * swing + this.turnRate * 0.06 * turnSign * leg.side;
      const calfTarget = this.homeCalf - this.calfAmp * ampScale * (isSwing ? Math.sin(legPhase) : 0);
      const hipTarget = this.homeHip + leg.side * this.hipAmp * (isSwing ? 1 : -1) * ampScale + this.lateralSpeed * 0.08 * leg.side;
      const hipJoint = `${leg.prefix}_hip_joint`;
      const thighJoint = `${leg.prefix}_thigh_joint`;
      const calfJoint = `${leg.prefix}_calf_joint`;
      ctrl[leg.act[0]] = this.pdTorque(hipJoint, hipTarget, this.hipKp, this.hipKd);
      ctrl[leg.act[1]] = this.pdTorque(thighJoint, thighTarget, this.thighKp, this.thighKd);
      ctrl[leg.act[2]] = this.pdTorque(calfJoint, calfTarget, this.calfKp, this.calfKd);
      ctrl[leg.act[1]] += pitchCorr * 0.2;
      ctrl[leg.act[0]] += rollCorr * 0.15 * leg.side;
    }
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
};

// src/g1CpgController.js
var G1CpgController = class {
  constructor(mujoco2, model2, data2) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.enabled = false;
    this.simDt = model2.opt.timestep || 2e-3;
    this.frequency = 1.4;
    this.phase = 0;
    this.hipPitchAmp = 0.2;
    this.kneeAmp = 0.3;
    this.anklePitchAmp = 0.12;
    this.ankleRollAmp = 0.02;
    this.hipRollAmp = 0.03;
    this.armSwingGain = 0.25;
    this.waistPitchGain = 0.15;
    this.waistRollGain = 0.1;
    this.balanceKp = 200;
    this.balanceKd = 15;
    this.homeQpos = {
      left_hip_pitch_joint: -0.2,
      left_hip_roll_joint: 0,
      left_hip_yaw_joint: 0,
      left_knee_joint: 0.4,
      left_ankle_pitch_joint: -0.2,
      left_ankle_roll_joint: 0,
      right_hip_pitch_joint: -0.2,
      right_hip_roll_joint: 0,
      right_hip_yaw_joint: 0,
      right_knee_joint: 0.4,
      right_ankle_pitch_joint: -0.2,
      right_ankle_roll_joint: 0,
      waist_yaw_joint: 0,
      waist_roll_joint: 0,
      waist_pitch_joint: 0,
      left_shoulder_pitch_joint: 0,
      left_shoulder_roll_joint: 0.3,
      left_shoulder_yaw_joint: 0,
      left_elbow_joint: 0.5,
      left_wrist_roll_joint: 0,
      left_wrist_pitch_joint: 0,
      left_wrist_yaw_joint: 0,
      right_shoulder_pitch_joint: 0,
      right_shoulder_roll_joint: -0.3,
      right_shoulder_yaw_joint: 0,
      right_elbow_joint: 0.5,
      right_wrist_roll_joint: 0,
      right_wrist_pitch_joint: 0,
      right_wrist_yaw_joint: 0
    };
    this.actIdx = {
      left_hip_pitch_joint: 0,
      left_hip_roll_joint: 1,
      left_hip_yaw_joint: 2,
      left_knee_joint: 3,
      left_ankle_pitch_joint: 4,
      left_ankle_roll_joint: 5,
      right_hip_pitch_joint: 6,
      right_hip_roll_joint: 7,
      right_hip_yaw_joint: 8,
      right_knee_joint: 9,
      right_ankle_pitch_joint: 10,
      right_ankle_roll_joint: 11,
      waist_yaw_joint: 12,
      waist_roll_joint: 13,
      waist_pitch_joint: 14,
      left_shoulder_pitch_joint: 15,
      left_shoulder_roll_joint: 16,
      left_shoulder_yaw_joint: 17,
      left_elbow_joint: 18,
      left_wrist_roll_joint: 19,
      left_wrist_pitch_joint: 20,
      left_wrist_yaw_joint: 21,
      right_shoulder_pitch_joint: 22,
      right_shoulder_roll_joint: 23,
      right_shoulder_yaw_joint: 24,
      right_elbow_joint: 25,
      right_wrist_roll_joint: 26,
      right_wrist_pitch_joint: 27,
      right_wrist_yaw_joint: 28
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
    const names = Object.keys(this.actIdx);
    for (const name of names) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0 && this.model.jnt_qposadr) {
          this.jntIdx[name] = this.model.jnt_qposadr[jid];
        }
      } catch (e) {
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
      if (idx !== void 0) {
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
    if (idx === void 0) return 0;
    let dofIdx;
    try {
      const jid = this.mujoco.mj_name2id(this.model, 3, jointName);
      dofIdx = this.model.jnt_dofadr[jid];
    } catch (e) {
      dofIdx = idx - 7 + 6;
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
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI;
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;
    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;
    const hipKp = 400, hipKd = 15;
    const kneeKp = 600, kneeKd = 20;
    const ankleKp = 100, ankleKd = 5;
    const waistKp = 300, waistKd = 12;
    const armKp = 30, armKd = 2;
    const ctrl = this.data.ctrl;
    const leftSwing = Math.sin(leftPhase);
    const leftStance = Math.max(0, -Math.sin(leftPhase));
    const leftHipPitchTarget = this.homeQpos.left_hip_pitch_joint + direction * this.hipPitchAmp * ampScale * leftSwing;
    const leftKneeTarget = this.homeQpos.left_knee_joint + this.kneeAmp * ampScale * Math.max(0, Math.sin(leftPhase));
    const leftAnklePitchTarget = this.homeQpos.left_ankle_pitch_joint - this.anklePitchAmp * ampScale * leftSwing;
    const leftAnkleRollTarget = this.homeQpos.left_ankle_roll_joint - this.ankleRollAmp * leftStance;
    const leftHipRollTarget = this.homeQpos.left_hip_roll_joint - this.hipRollAmp * leftStance + this.lateralSpeed * 0.05;
    const leftHipYawTarget = this.homeQpos.left_hip_yaw_joint + this.turnRate * 0.05 * leftSwing;
    ctrl[this.actIdx.left_hip_pitch_joint] = this.pdTorque("left_hip_pitch_joint", leftHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll_joint] = this.pdTorque("left_hip_roll_joint", leftHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_yaw_joint] = this.pdTorque("left_hip_yaw_joint", leftHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_knee_joint] = this.pdTorque("left_knee_joint", leftKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle_pitch_joint] = this.pdTorque("left_ankle_pitch_joint", leftAnklePitchTarget, ankleKp, ankleKd);
    ctrl[this.actIdx.left_ankle_roll_joint] = this.pdTorque("left_ankle_roll_joint", leftAnkleRollTarget, ankleKp, ankleKd);
    const rightSwing = Math.sin(rightPhase);
    const rightStance = Math.max(0, -Math.sin(rightPhase));
    const rightHipPitchTarget = this.homeQpos.right_hip_pitch_joint + direction * this.hipPitchAmp * ampScale * rightSwing;
    const rightKneeTarget = this.homeQpos.right_knee_joint + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rightAnklePitchTarget = this.homeQpos.right_ankle_pitch_joint - this.anklePitchAmp * ampScale * rightSwing;
    const rightAnkleRollTarget = this.homeQpos.right_ankle_roll_joint + this.ankleRollAmp * rightStance;
    const rightHipRollTarget = this.homeQpos.right_hip_roll_joint + this.hipRollAmp * rightStance + this.lateralSpeed * 0.05;
    const rightHipYawTarget = this.homeQpos.right_hip_yaw_joint + this.turnRate * 0.05 * rightSwing;
    ctrl[this.actIdx.right_hip_pitch_joint] = this.pdTorque("right_hip_pitch_joint", rightHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll_joint] = this.pdTorque("right_hip_roll_joint", rightHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_yaw_joint] = this.pdTorque("right_hip_yaw_joint", rightHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_knee_joint] = this.pdTorque("right_knee_joint", rightKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle_pitch_joint] = this.pdTorque("right_ankle_pitch_joint", rightAnklePitchTarget, ankleKp, ankleKd);
    ctrl[this.actIdx.right_ankle_roll_joint] = this.pdTorque("right_ankle_roll_joint", rightAnkleRollTarget, ankleKp, ankleKd);
    const waistYawTarget = this.homeQpos.waist_yaw_joint + this.turnRate * 0.1;
    const waistPitchTarget = this.homeQpos.waist_pitch_joint - this.waistPitchGain * pitch;
    const waistRollTarget = this.homeQpos.waist_roll_joint - this.waistRollGain * roll;
    ctrl[this.actIdx.waist_yaw_joint] = this.pdTorque("waist_yaw_joint", waistYawTarget, waistKp, waistKd);
    ctrl[this.actIdx.waist_roll_joint] = this.pdTorque("waist_roll_joint", waistRollTarget, waistKp, waistKd);
    ctrl[this.actIdx.waist_pitch_joint] = this.pdTorque("waist_pitch_joint", waistPitchTarget, waistKp, waistKd);
    const pitchCorrection = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorrection = -this.balanceKp * roll - this.balanceKd * rollRate;
    ctrl[this.actIdx.left_ankle_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.right_ankle_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.left_ankle_roll_joint] += rollCorrection * 0.3;
    ctrl[this.actIdx.right_ankle_roll_joint] -= rollCorrection * 0.3;
    ctrl[this.actIdx.left_hip_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.right_hip_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.left_hip_roll_joint] += rollCorrection * 0.3;
    ctrl[this.actIdx.right_hip_roll_joint] -= rollCorrection * 0.3;
    const leftArmSwing = -this.armSwingGain * direction * ampScale * leftSwing;
    const rightArmSwing = -this.armSwingGain * direction * ampScale * rightSwing;
    ctrl[this.actIdx.left_shoulder_pitch_joint] = this.pdTorque(
      "left_shoulder_pitch_joint",
      this.homeQpos.left_shoulder_pitch_joint + leftArmSwing,
      armKp,
      armKd
    );
    ctrl[this.actIdx.left_shoulder_roll_joint] = this.pdTorque(
      "left_shoulder_roll_joint",
      this.homeQpos.left_shoulder_roll_joint,
      armKp,
      armKd
    );
    ctrl[this.actIdx.left_shoulder_yaw_joint] = this.pdTorque(
      "left_shoulder_yaw_joint",
      this.homeQpos.left_shoulder_yaw_joint,
      armKp * 0.5,
      armKd
    );
    ctrl[this.actIdx.left_elbow_joint] = this.pdTorque(
      "left_elbow_joint",
      this.homeQpos.left_elbow_joint,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_pitch_joint] = this.pdTorque(
      "right_shoulder_pitch_joint",
      this.homeQpos.right_shoulder_pitch_joint + rightArmSwing,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_roll_joint] = this.pdTorque(
      "right_shoulder_roll_joint",
      this.homeQpos.right_shoulder_roll_joint,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_yaw_joint] = this.pdTorque(
      "right_shoulder_yaw_joint",
      this.homeQpos.right_shoulder_yaw_joint,
      armKp * 0.5,
      armKd
    );
    ctrl[this.actIdx.right_elbow_joint] = this.pdTorque(
      "right_elbow_joint",
      this.homeQpos.right_elbow_joint,
      armKp,
      armKd
    );
    ctrl[this.actIdx.left_wrist_roll_joint] = this.pdTorque(
      "left_wrist_roll_joint",
      this.homeQpos.left_wrist_roll_joint,
      armKp * 0.3,
      armKd * 0.5
    );
    ctrl[this.actIdx.left_wrist_pitch_joint] = this.pdTorque(
      "left_wrist_pitch_joint",
      this.homeQpos.left_wrist_pitch_joint,
      armKp * 0.3,
      armKd * 0.5
    );
    ctrl[this.actIdx.left_wrist_yaw_joint] = this.pdTorque(
      "left_wrist_yaw_joint",
      this.homeQpos.left_wrist_yaw_joint,
      armKp * 0.3,
      armKd * 0.5
    );
    ctrl[this.actIdx.right_wrist_roll_joint] = this.pdTorque(
      "right_wrist_roll_joint",
      this.homeQpos.right_wrist_roll_joint,
      armKp * 0.3,
      armKd * 0.5
    );
    ctrl[this.actIdx.right_wrist_pitch_joint] = this.pdTorque(
      "right_wrist_pitch_joint",
      this.homeQpos.right_wrist_pitch_joint,
      armKp * 0.3,
      armKd * 0.5
    );
    ctrl[this.actIdx.right_wrist_yaw_joint] = this.pdTorque(
      "right_wrist_yaw_joint",
      this.homeQpos.right_wrist_yaw_joint,
      armKp * 0.3,
      armKd * 0.5
    );
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
};

// src/h1_2CpgController.js
var H1_2CpgController = class {
  constructor(mujoco2, model2, data2) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.enabled = false;
    this.simDt = model2.opt.timestep || 2e-3;
    this.frequency = 1.2;
    this.phase = 0;
    this.hipPitchAmp = 0.25;
    this.kneeAmp = 0.35;
    this.anklePitchAmp = 0.15;
    this.ankleRollAmp = 0.04;
    this.hipRollAmp = 0.03;
    this.armSwingGain = 0.3;
    this.balanceKp = 300;
    this.balanceKd = 20;
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
      right_wrist_yaw_joint: 0
    };
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
      right_wrist_yaw_joint: 26
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
    const names = Object.keys(this.actIdx);
    for (const name of names) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0 && this.model.jnt_qposadr) {
          this.jntIdx[name] = this.model.jnt_qposadr[jid];
        }
      } catch (e) {
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
    if (idx === void 0) return 0;
    let dofIdx;
    try {
      const jid = this.mujoco.mj_name2id(this.model, 3, jointName);
      dofIdx = this.model.jnt_dofadr[jid];
    } catch (e) {
      dofIdx = idx - 7 + 6;
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
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI;
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;
    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;
    const hipKp = 800, hipKd = 30;
    const kneeKp = 1200, kneeKd = 40;
    const anklePitchKp = 200, anklePitchKd = 10;
    const ankleRollKp = 100, ankleRollKd = 5;
    const torsoKp = 600, torsoKd = 25;
    const armKp = 50, armKd = 3;
    const ctrl = this.data.ctrl;
    const leftSwing = Math.sin(leftPhase);
    const leftStance = Math.max(0, -Math.sin(leftPhase));
    const leftHipPitchTarget = this.homeQpos.left_hip_pitch_joint + direction * this.hipPitchAmp * ampScale * leftSwing;
    const leftKneeTarget = this.homeQpos.left_knee_joint + this.kneeAmp * ampScale * Math.max(0, Math.sin(leftPhase));
    const leftAnklePitchTarget = this.homeQpos.left_ankle_pitch_joint - this.anklePitchAmp * ampScale * leftSwing;
    const leftAnkleRollTarget = this.homeQpos.left_ankle_roll_joint - this.ankleRollAmp * leftStance;
    const leftHipRollTarget = this.homeQpos.left_hip_roll_joint - this.hipRollAmp * leftStance + this.lateralSpeed * 0.05;
    const leftHipYawTarget = this.homeQpos.left_hip_yaw_joint + this.turnRate * 0.05 * leftSwing;
    ctrl[this.actIdx.left_hip_yaw_joint] = this.pdTorque("left_hip_yaw_joint", leftHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_pitch_joint] = this.pdTorque("left_hip_pitch_joint", leftHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll_joint] = this.pdTorque("left_hip_roll_joint", leftHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_knee_joint] = this.pdTorque("left_knee_joint", leftKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle_pitch_joint] = this.pdTorque("left_ankle_pitch_joint", leftAnklePitchTarget, anklePitchKp, anklePitchKd);
    ctrl[this.actIdx.left_ankle_roll_joint] = this.pdTorque("left_ankle_roll_joint", leftAnkleRollTarget, ankleRollKp, ankleRollKd);
    const rightSwing = Math.sin(rightPhase);
    const rightStance = Math.max(0, -Math.sin(rightPhase));
    const rightHipPitchTarget = this.homeQpos.right_hip_pitch_joint + direction * this.hipPitchAmp * ampScale * rightSwing;
    const rightKneeTarget = this.homeQpos.right_knee_joint + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rightAnklePitchTarget = this.homeQpos.right_ankle_pitch_joint - this.anklePitchAmp * ampScale * rightSwing;
    const rightAnkleRollTarget = this.homeQpos.right_ankle_roll_joint + this.ankleRollAmp * rightStance;
    const rightHipRollTarget = this.homeQpos.right_hip_roll_joint + this.hipRollAmp * rightStance + this.lateralSpeed * 0.05;
    const rightHipYawTarget = this.homeQpos.right_hip_yaw_joint + this.turnRate * 0.05 * rightSwing;
    ctrl[this.actIdx.right_hip_yaw_joint] = this.pdTorque("right_hip_yaw_joint", rightHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_pitch_joint] = this.pdTorque("right_hip_pitch_joint", rightHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll_joint] = this.pdTorque("right_hip_roll_joint", rightHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_knee_joint] = this.pdTorque("right_knee_joint", rightKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle_pitch_joint] = this.pdTorque("right_ankle_pitch_joint", rightAnklePitchTarget, anklePitchKp, anklePitchKd);
    ctrl[this.actIdx.right_ankle_roll_joint] = this.pdTorque("right_ankle_roll_joint", rightAnkleRollTarget, ankleRollKp, ankleRollKd);
    const torsoTarget = this.homeQpos.torso_joint + this.turnRate * 0.1;
    ctrl[this.actIdx.torso_joint] = this.pdTorque("torso_joint", torsoTarget, torsoKp, torsoKd);
    const pitchCorrection = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorrection = -this.balanceKp * roll - this.balanceKd * rollRate;
    ctrl[this.actIdx.left_ankle_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.right_ankle_pitch_joint] += pitchCorrection * 0.4;
    ctrl[this.actIdx.left_ankle_roll_joint] += rollCorrection * 0.25;
    ctrl[this.actIdx.right_ankle_roll_joint] -= rollCorrection * 0.25;
    ctrl[this.actIdx.left_hip_pitch_joint] += pitchCorrection * 0.5;
    ctrl[this.actIdx.right_hip_pitch_joint] += pitchCorrection * 0.5;
    ctrl[this.actIdx.left_hip_roll_joint] += rollCorrection * 0.3;
    ctrl[this.actIdx.right_hip_roll_joint] -= rollCorrection * 0.3;
    const leftArmSwing = -this.armSwingGain * direction * ampScale * leftSwing;
    const rightArmSwing = -this.armSwingGain * direction * ampScale * rightSwing;
    ctrl[this.actIdx.left_shoulder_pitch_joint] = this.pdTorque(
      "left_shoulder_pitch_joint",
      this.homeQpos.left_shoulder_pitch_joint + leftArmSwing,
      armKp,
      armKd
    );
    ctrl[this.actIdx.left_shoulder_roll_joint] = this.pdTorque(
      "left_shoulder_roll_joint",
      this.homeQpos.left_shoulder_roll_joint,
      armKp,
      armKd
    );
    ctrl[this.actIdx.left_shoulder_yaw_joint] = this.pdTorque(
      "left_shoulder_yaw_joint",
      this.homeQpos.left_shoulder_yaw_joint,
      armKp * 0.5,
      armKd
    );
    ctrl[this.actIdx.left_elbow_joint] = this.pdTorque(
      "left_elbow_joint",
      this.homeQpos.left_elbow_joint,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_pitch_joint] = this.pdTorque(
      "right_shoulder_pitch_joint",
      this.homeQpos.right_shoulder_pitch_joint + rightArmSwing,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_roll_joint] = this.pdTorque(
      "right_shoulder_roll_joint",
      this.homeQpos.right_shoulder_roll_joint,
      armKp,
      armKd
    );
    ctrl[this.actIdx.right_shoulder_yaw_joint] = this.pdTorque(
      "right_shoulder_yaw_joint",
      this.homeQpos.right_shoulder_yaw_joint,
      armKp * 0.5,
      armKd
    );
    ctrl[this.actIdx.right_elbow_joint] = this.pdTorque(
      "right_elbow_joint",
      this.homeQpos.right_elbow_joint,
      armKp,
      armKd
    );
    const wristKp = armKp * 0.3;
    const wristKd = armKd * 0.5;
    ctrl[this.actIdx.left_wrist_roll_joint] = this.pdTorque(
      "left_wrist_roll_joint",
      this.homeQpos.left_wrist_roll_joint,
      wristKp,
      wristKd
    );
    ctrl[this.actIdx.left_wrist_pitch_joint] = this.pdTorque(
      "left_wrist_pitch_joint",
      this.homeQpos.left_wrist_pitch_joint,
      wristKp,
      wristKd
    );
    ctrl[this.actIdx.left_wrist_yaw_joint] = this.pdTorque(
      "left_wrist_yaw_joint",
      this.homeQpos.left_wrist_yaw_joint,
      wristKp,
      wristKd
    );
    ctrl[this.actIdx.right_wrist_roll_joint] = this.pdTorque(
      "right_wrist_roll_joint",
      this.homeQpos.right_wrist_roll_joint,
      wristKp,
      wristKd
    );
    ctrl[this.actIdx.right_wrist_pitch_joint] = this.pdTorque(
      "right_wrist_pitch_joint",
      this.homeQpos.right_wrist_pitch_joint,
      wristKp,
      wristKd
    );
    ctrl[this.actIdx.right_wrist_yaw_joint] = this.pdTorque(
      "right_wrist_yaw_joint",
      this.homeQpos.right_wrist_yaw_joint,
      wristKp,
      wristKd
    );
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
};

// src/factoryController.js
var SingleRobotCPG = class {
  constructor(mujoco2, model2, data2, prefix, index) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.prefix = prefix;
    this.index = index;
    this.enabled = false;
    this.simDt = model2.opt.timestep || 2e-3;
    this.frequency = 2.5;
    this.phase = Math.random() * Math.PI * 2;
    this.thighAmp = 0.25;
    this.calfAmp = 0.25;
    this.hipAmp = 0.04;
    this.hipKp = 120;
    this.hipKd = 4;
    this.thighKp = 200;
    this.thighKd = 6;
    this.calfKp = 250;
    this.calfKd = 8;
    this.balanceKp = 60;
    this.balanceKd = 5;
    this.homeHip = 0;
    this.homeThigh = 0.9;
    this.homeCalf = -1.8;
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
    this.baseQposAddr = 0;
    this.actBase = index * 12;
    this.jntQpos = {};
    this.jntDof = {};
    this.findIndices();
  }
  findIndices() {
    const p = this.prefix;
    try {
      const fjId = this.mujoco.mj_name2id(this.model, 3, `${p}freejoint`);
      if (fjId >= 0) {
        this.baseQposAddr = this.model.jnt_qposadr[fjId];
      }
    } catch (e) {
    }
    const suffixes = [
      "FL_hip_joint",
      "FL_thigh_joint",
      "FL_calf_joint",
      "FR_hip_joint",
      "FR_thigh_joint",
      "FR_calf_joint",
      "RL_hip_joint",
      "RL_thigh_joint",
      "RL_calf_joint",
      "RR_hip_joint",
      "RR_thigh_joint",
      "RR_calf_joint"
    ];
    for (const s of suffixes) {
      const name = `${p}${s}`;
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name);
        if (jid >= 0) {
          this.jntQpos[name] = this.model.jnt_qposadr[jid];
          this.jntDof[name] = this.model.jnt_dofadr[jid];
        }
      } catch (e) {
      }
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
    if (qIdx === void 0) return 0;
    const q = this.data.qpos[qIdx];
    const qdot = this.data.qvel[dIdx] || 0;
    return kp * (target - q) - kd * qdot;
  }
  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }
  step() {
    if (!this.enabled) return;
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;
    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;
    const pitchCorr = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    const rollCorr = -this.balanceKp * roll - this.balanceKd * rollRate;
    const ctrl = this.data.ctrl;
    const base = this.actBase;
    const p = this.prefix;
    const legs = [
      [0, 0, 1, "FL", true],
      [3, Math.PI, -1, "FR", true],
      [6, Math.PI, 1, "RL", false],
      [9, 0, -1, "RR", false]
    ];
    for (const [actOff, legPhaseOff, side, legName, isFront] of legs) {
      const legPhase = this.phase + legPhaseOff;
      const swing = Math.sin(legPhase);
      const isSwing = swing > 0;
      const turnSign = isFront ? 1 : -1;
      const thighTarget = this.homeThigh - direction * this.thighAmp * ampScale * swing + this.turnRate * 0.06 * turnSign * side;
      const calfTarget = this.homeCalf - this.calfAmp * ampScale * (isSwing ? Math.sin(legPhase) : 0);
      const hipTarget = this.homeHip + side * this.hipAmp * (isSwing ? 1 : -1) * ampScale + this.lateralSpeed * 0.08 * side;
      const hipJoint = `${p}${legName}_hip_joint`;
      const thighJoint = `${p}${legName}_thigh_joint`;
      const calfJoint = `${p}${legName}_calf_joint`;
      ctrl[base + actOff + 0] = this.pdTorque(hipJoint, hipTarget, this.hipKp, this.hipKd);
      ctrl[base + actOff + 1] = this.pdTorque(thighJoint, thighTarget, this.thighKp, this.thighKd);
      ctrl[base + actOff + 2] = this.pdTorque(calfJoint, calfTarget, this.calfKp, this.calfKd);
      ctrl[base + actOff + 1] += pitchCorr * 0.2;
      ctrl[base + actOff + 0] += rollCorr * 0.15 * side;
    }
    const ctrlRange = this.model.actuator_ctrlrange;
    if (ctrlRange) {
      for (let i = 0; i < 12; i++) {
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
    this.forwardSpeed = 0;
    this.lateralSpeed = 0;
    this.turnRate = 0;
  }
};
var FactoryController = class {
  constructor(mujoco2, model2, data2, numRobots) {
    this.mujoco = mujoco2;
    this.model = model2;
    this.data = data2;
    this.numRobots = numRobots;
    this.enabled = false;
    this.robots = [];
    const centerIdx = Math.floor(numRobots / 2);
    try {
      this.centerBodyId = mujoco2.mj_name2id(model2, 1, `r${centerIdx}_base`);
    } catch (e) {
      this.centerBodyId = 1;
    }
    for (let i = 0; i < numRobots; i++) {
      const robot = new SingleRobotCPG(mujoco2, model2, data2, `r${i}_`, i);
      this.robots.push(robot);
    }
  }
  setCommand(forward, lateral, turn) {
    for (const robot of this.robots) {
      robot.setCommand(forward, lateral, turn);
    }
  }
  step() {
    if (!this.enabled) return;
    for (const robot of this.robots) {
      robot.enabled = true;
      robot.step();
    }
  }
  reset() {
    for (const robot of this.robots) {
      robot.reset();
    }
  }
};

// src/factoryScene.js
function prefixNames(xml, p) {
  const names = [
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "base",
    "imu",
    "FL",
    "FR",
    "RL",
    "RR"
  ];
  for (const name of names) {
    xml = xml.replaceAll(`name="${name}"`, `name="${p}${name}"`);
    xml = xml.replaceAll(`joint="${name}"`, `joint="${p}${name}"`);
  }
  xml = xml.replace("<freejoint/>", `<freejoint name="${p}freejoint"/>`);
  return xml;
}
function generateFactoryXML(mujoco2, numRobots, spacing) {
  const go2Xml = new TextDecoder().decode(
    mujoco2.FS.readFile("/working/unitree_go2/go2.xml")
  );
  const defaultStart = go2Xml.indexOf("<default>");
  const defaultEnd = go2Xml.lastIndexOf("</default>") + "</default>".length;
  const defaultSection = go2Xml.substring(defaultStart, defaultEnd);
  const assetStart = go2Xml.indexOf("<asset>") + "<asset>".length;
  const assetEnd = go2Xml.indexOf("</asset>");
  const assetContent = go2Xml.substring(assetStart, assetEnd);
  const wbStart = go2Xml.indexOf("<worldbody>") + "<worldbody>".length;
  const wbEnd = go2Xml.indexOf("</worldbody>");
  const robotBodyXml = go2Xml.substring(wbStart, wbEnd).trim();
  const actStart = go2Xml.indexOf("<actuator>") + "<actuator>".length;
  const actEnd = go2Xml.indexOf("</actuator>");
  const robotActXml = go2Xml.substring(actStart, actEnd).trim();
  const cols = Math.ceil(Math.sqrt(numRobots));
  const rows = Math.ceil(numRobots / cols);
  let allBodies = "";
  let allActuators = "";
  const qposValues = [];
  const ctrlValues = [];
  for (let i = 0; i < numRobots; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const x = (col - (cols - 1) / 2) * spacing;
    const y = (row - (rows - 1) / 2) * spacing;
    const prefix = `r${i}_`;
    let body = prefixNames(robotBodyXml, prefix);
    body = body.replace('pos="0 0 0.445"', `pos="${x.toFixed(2)} ${y.toFixed(2)} 0.445"`);
    allBodies += "    " + body + "\n";
    allActuators += "    " + prefixNames(robotActXml, prefix) + "\n";
    qposValues.push(
      `${x.toFixed(2)} ${y.toFixed(2)} 0.27 1 0 0 0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8`
    );
    ctrlValues.push("0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8");
  }
  return `<mujoco model="go2 factory">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option cone="elliptic" impratio="100"/>

  ${defaultSection}

  <asset>
    ${assetContent}
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
${allBodies}
  </worldbody>

  <actuator>
${allActuators}
  </actuator>

  <keyframe>
    <key name="home"
      qpos="${qposValues.join(" ")}"
      ctrl="${ctrlValues.join(" ")}"/>
  </keyframe>
</mujoco>`;
}

// src/main.js
var statusEl = document.getElementById("status");
var sceneSelect = document.getElementById("scene-select");
var resetBtn = document.getElementById("btn-reset");
var controllerBtn = document.getElementById("btn-controller");
var speedBtn = document.getElementById("btn-speed");
var helpOverlay = document.getElementById("help-overlay");
var evolvePanel = document.getElementById("evolve-panel");
var evolveStatus = document.getElementById("evolve-status");
var app = document.getElementById("app");
var renderer = new THREE2.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
app.appendChild(renderer.domElement);
var scene = new THREE2.Scene();
scene.background = new THREE2.Color(1119517);
scene.add(new THREE2.HemisphereLight(16777215, 2241348, 1));
var dirLight = new THREE2.DirectionalLight(16777215, 1.2);
dirLight.position.set(3, 5, 3);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);
var camera = new THREE2.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(1.5, 1, 1.5);
var controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.3, 0);
controls.enableDamping = true;
var mujoco;
var model;
var data;
var bodies = {};
var mujocoRoot = null;
var activeController = null;
var go2Controller = null;
var go2RlController = null;
var h1Controller = null;
var b2Controller = null;
var g1Controller = null;
var h1_2Controller = null;
var factoryController = null;
var evolutionController = null;
var paused = false;
var cameraFollow = true;
var simSpeed = 1;
var SIM_SPEEDS = [0.25, 0.5, 1, 2, 4];
var keys = {};
var touchX = 0;
var touchY = 0;
var touchRotL = false;
var touchRotR = false;
var isTouchDevice = "ontouchstart" in window || navigator.maxTouchPoints > 0;
var stepCounter = 0;
var NUM_BALLS = 5;
var NUM_BOXES = 5;
var NUM_OBSTACLES = NUM_BALLS + NUM_BOXES;
var HIDE_Z = -50;
var obstacleQposBase = -1;
var obstacleQvelBase = -1;
var nextBall = 0;
var nextBox = 0;
var currentObstacleScale = 1;
var SCENES = {
  "unitree_go2/scene.xml": {
    controller: "go2",
    camera: { pos: [1.5, 1, 1.5], target: [0, 0.25, 0] }
  },
  "unitree_go2/scene.xml|rl": {
    controller: "go2rl",
    scenePath: "unitree_go2/scene.xml",
    onnxModel: "./assets/models/go2_flat_policy.onnx",
    camera: { pos: [1.5, 1, 1.5], target: [0, 0.25, 0] }
  },
  "unitree_go2/scene_stairs.xml": {
    controller: "go2",
    camera: { pos: [2, 1.5, 2], target: [1.2, 0.15, 0] }
  },
  "unitree_go2/scene_rough.xml": {
    controller: "go2",
    camera: { pos: [1.5, 1, 1.5], target: [0.5, 0.1, 0] }
  },
  "unitree_b2/scene.xml": {
    controller: "b2",
    camera: { pos: [2.5, 1.5, 2.5], target: [0, 0.35, 0] }
  },
  "unitree_h1/scene.xml": {
    controller: "h1",
    camera: { pos: [3, 2, 3], target: [0, 0.9, 0] }
  },
  "unitree_h1/scene_stairs.xml": {
    controller: "h1",
    camera: { pos: [4, 2.5, 3.5], target: [2, 0.5, 0] }
  },
  "unitree_g1/scene.xml": {
    controller: "g1",
    camera: { pos: [2.5, 1.8, 2.5], target: [0, 0.7, 0] }
  },
  "unitree_h1_2/scene.xml": {
    controller: "h1_2",
    camera: { pos: [3, 2, 3], target: [0, 0.9, 0] }
  },
  "unitree_go2/scene_factory": {
    controller: "factory",
    numRobots: 9,
    spacing: 1.5,
    camera: { pos: [4, 3, 4], target: [0, 0.25, 0] }
  }
};
var currentScenePath = "unitree_go2/scene.xml";
var currentSceneKey = "unitree_go2/scene.xml";
function generateArenaXML(sceneXml, scale) {
  let xml = sceneXml;
  const colors = [
    "0.95 0.25 0.2 1",
    "0.2 0.55 0.95 1",
    "0.2 0.85 0.35 1",
    "0.95 0.75 0.15 1",
    "0.85 0.3 0.7 1"
  ];
  let obsXml = "\n  <worldbody>\n";
  for (let i = 0; i < NUM_BALLS; i++) {
    const r = (0.015 + i * 4e-3) * scale;
    const m = (0.03 * scale * scale).toFixed(3);
    obsXml += `    <body name="obstacle_${i}" pos="0 0 ${HIDE_Z}"><freejoint name="obs_fj_${i}"/><geom type="sphere" size="${r.toFixed(4)}" rgba="${colors[i]}" mass="${m}" contype="1" conaffinity="1"/></body>
`;
  }
  for (let i = 0; i < NUM_BOXES; i++) {
    const r = (0.012 + i * 3e-3) * scale;
    const m = (0.04 * scale * scale).toFixed(3);
    const idx = NUM_BALLS + i;
    obsXml += `    <body name="obstacle_${idx}" pos="0 0 ${HIDE_Z}"><freejoint name="obs_fj_${idx}"/><geom type="box" size="${r.toFixed(4)} ${r.toFixed(4)} ${r.toFixed(4)}" rgba="${colors[i]}" mass="${m}" contype="1" conaffinity="1"/></body>
`;
  }
  obsXml += "  </worldbody>\n";
  xml = xml.replace("</mujoco>", obsXml + "</mujoco>");
  const obsQpos = Array(NUM_OBSTACLES).fill(`0 0 ${HIDE_Z} 1 0 0 0`).join(" ");
  xml = xml.replace(
    /(qpos\s*=\s*")([\s\S]*?)(")/,
    (m, pre, content, post) => pre + content.trimEnd() + " " + obsQpos + "\n    " + post
  );
  return xml;
}
function findObstacleIndices() {
  obstacleQposBase = -1;
  obstacleQvelBase = -1;
  try {
    const bodyId = mujoco.mj_name2id(model, 1, "obstacle_0");
    if (bodyId < 0) return;
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_bodyid[j] === bodyId) {
        obstacleQposBase = model.jnt_qposadr[j];
        obstacleQvelBase = model.jnt_dofadr[j];
        break;
      }
    }
  } catch (e) {
    console.warn("Could not find obstacle indices:", e);
  }
}
function spawnObstacle(type) {
  if (obstacleQposBase < 0 || !model || !data) return;
  let idx;
  if (type === "box") {
    idx = NUM_BALLS + nextBox % NUM_BOXES;
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
  if (sceneKey.includes("h1") || sceneKey.includes("g1")) return 3.5;
  if (sceneKey.includes("b2")) return 4;
  return 2.5;
}
async function loadScene(sceneKey) {
  setStatus(`Loading: ${sceneKey}`);
  const cfg = SCENES[sceneKey] || {};
  let loadPath;
  if (cfg.controller === "factory") {
    await loadSceneAssets(mujoco, "unitree_go2/scene.xml", setStatus);
    setStatus("Generating factory...");
    const factoryXml = generateFactoryXML(mujoco, cfg.numRobots, cfg.spacing);
    mujoco.FS.writeFile("/working/unitree_go2/scene_factory.xml", factoryXml);
    loadPath = "/working/unitree_go2/scene_factory.xml";
  } else {
    const scenePath = cfg.scenePath || sceneKey;
    await loadSceneAssets(mujoco, scenePath, setStatus);
    currentObstacleScale = getObstacleScale(sceneKey);
    const originalXml = new TextDecoder().decode(mujoco.FS.readFile("/working/" + scenePath));
    const arenaXml = generateArenaXML(originalXml, currentObstacleScale);
    const arenaPath = scenePath.replace(".xml", "_arena.xml");
    mujoco.FS.writeFile("/working/" + arenaPath, arenaXml);
    loadPath = "/working/" + arenaPath;
  }
  clearScene();
  if (data) {
    data.delete();
    data = null;
  }
  if (model) {
    model.delete();
    model = null;
  }
  model = mujoco.MjModel.loadFromXML(loadPath);
  data = new mujoco.MjData(model);
  console.log(`Model loaded: nq=${model.nq}, nv=${model.nv}, nu=${model.nu}, nbody=${model.nbody}`);
  if (model.nkey > 0) {
    data.qpos.set(model.key_qpos.slice(0, model.nq));
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    if (model.key_ctrl) data.ctrl.set(model.key_ctrl.slice(0, model.nu));
    mujoco.mj_forward(model, data);
  }
  const built = buildScene(model);
  mujocoRoot = built.mujocoRoot;
  bodies = built.bodies;
  scene.add(mujocoRoot);
  activeController = null;
  go2Controller = null;
  go2RlController = null;
  h1Controller = null;
  b2Controller = null;
  g1Controller = null;
  h1_2Controller = null;
  factoryController = null;
  evolutionController = null;
  stepCounter = 0;
  model.opt.iterations = 30;
  if (cfg.controller === "go2") {
    go2Controller = new Go2CpgController(mujoco, model, data);
    go2Controller.enabled = true;
    for (let i = 0; i < 200; i++) {
      go2Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = "go2";
  } else if (cfg.controller === "go2rl") {
    go2RlController = new Go2OnnxController(mujoco, model, data);
    setStatus("Loading RL policy...");
    const loaded2 = await go2RlController.loadModel(cfg.onnxModel);
    if (loaded2) {
      go2RlController.setInitialPose();
      go2RlController.enabled = true;
      for (let i = 0; i < 500; i++) {
        go2RlController.applyPD();
        mujoco.mj_step(model, data);
      }
      activeController = "go2rl";
    } else {
      setStatus("RL policy load failed, falling back to CPG");
      go2Controller = new Go2CpgController(mujoco, model, data);
      go2Controller.enabled = true;
      for (let i = 0; i < 200; i++) {
        go2Controller.step();
        mujoco.mj_step(model, data);
      }
      activeController = "go2";
    }
  } else if (cfg.controller === "b2") {
    b2Controller = new B2CpgController(mujoco, model, data);
    b2Controller.enabled = true;
    for (let i = 0; i < 200; i++) {
      b2Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = "b2";
  } else if (cfg.controller === "h1") {
    h1Controller = new H1CpgController(mujoco, model, data);
    h1Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      h1Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = "h1";
  } else if (cfg.controller === "g1") {
    g1Controller = new G1CpgController(mujoco, model, data);
    g1Controller.setStandingPose();
    mujoco.mj_forward(model, data);
    g1Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      g1Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = "g1";
  } else if (cfg.controller === "h1_2") {
    h1_2Controller = new H1_2CpgController(mujoco, model, data);
    for (const [name, target] of Object.entries(h1_2Controller.homeQpos)) {
      const idx = h1_2Controller.jntIdx[name];
      if (idx !== void 0) data.qpos[idx] = target;
    }
    mujoco.mj_forward(model, data);
    h1_2Controller.enabled = true;
    for (let i = 0; i < 500; i++) {
      h1_2Controller.step();
      mujoco.mj_step(model, data);
    }
    activeController = "h1_2";
  } else if (cfg.controller === "factory") {
    factoryController = new FactoryController(mujoco, model, data, cfg.numRobots);
    factoryController.enabled = true;
    for (let i = 0; i < 200; i++) {
      factoryController.step();
      mujoco.mj_step(model, data);
    }
    activeController = "factory";
  }
  if (cfg.controller !== "factory") findObstacleIndices();
  nextBall = 0;
  nextBox = 0;
  updateControllerBtn();
  if (evolvePanel) evolvePanel.style.display = "none";
  if (cfg.camera) {
    camera.position.set(...cfg.camera.pos);
    controls.target.set(...cfg.camera.target);
  }
  controls.update();
  currentScenePath = sceneKey;
  currentSceneKey = sceneKey;
  setStatus(`Ready: ${sceneKey.split("/").pop()}`);
}
function updateControllerBtn() {
  if (!controllerBtn) return;
  const labels = {
    go2: () => go2Controller?.enabled ? "CPG: ON" : "CPG: OFF",
    go2rl: () => go2RlController?.enabled ? "RL: ON" : "RL: OFF",
    h1: () => h1Controller?.enabled ? "CPG: ON" : "CPG: OFF",
    b2: () => b2Controller?.enabled ? "CPG: ON" : "CPG: OFF",
    g1: () => g1Controller?.enabled ? "CPG: ON" : "CPG: OFF",
    h1_2: () => h1_2Controller?.enabled ? "CPG: ON" : "CPG: OFF",
    factory: () => factoryController?.enabled ? `CPG: ON (${factoryController.numRobots}x)` : "CPG: OFF"
  };
  const fn = labels[activeController];
  if (fn) {
    controllerBtn.textContent = fn();
    controllerBtn.style.display = "";
  } else {
    controllerBtn.style.display = "none";
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
      if (idx !== void 0) data.qpos[idx] = target;
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
  if (activeController === "go2" && go2Controller) {
    go2Controller.enabled = !go2Controller.enabled;
  } else if (activeController === "go2rl" && go2RlController) {
    go2RlController.enabled = !go2RlController.enabled;
  } else if (activeController === "h1" && h1Controller) {
    h1Controller.enabled = !h1Controller.enabled;
  } else if (activeController === "b2" && b2Controller) {
    b2Controller.enabled = !b2Controller.enabled;
  } else if (activeController === "g1" && g1Controller) {
    g1Controller.enabled = !g1Controller.enabled;
  } else if (activeController === "h1_2" && h1_2Controller) {
    h1_2Controller.enabled = !h1_2Controller.enabled;
  } else if (activeController === "factory" && factoryController) {
    factoryController.enabled = !factoryController.enabled;
  }
  updateControllerBtn();
}
function cycleSpeed() {
  const idx = SIM_SPEEDS.indexOf(simSpeed);
  simSpeed = SIM_SPEEDS[(idx + 1) % SIM_SPEEDS.length];
  updateSpeedBtn();
}
function updateSpeedBtn() {
  if (speedBtn) speedBtn.textContent = simSpeed + "x";
}
function handleInput() {
  const kbFwd = keys["KeyW"] || keys["ArrowUp"];
  const kbBack = keys["KeyS"] || keys["ArrowDown"];
  const kbLeft = keys["KeyA"] || keys["ArrowLeft"];
  const kbRight = keys["KeyD"] || keys["ArrowRight"];
  const kbRotL = keys["KeyQ"];
  const kbRotR = keys["KeyE"];
  let fwd = 0, lat = 0, turn = 0;
  if (kbFwd) fwd = 0.8;
  if (kbBack) fwd = -0.4;
  if (kbLeft) lat = 0.3;
  if (kbRight) lat = -0.3;
  if (kbRotL) turn = 0.5;
  if (kbRotR) turn = -0.5;
  if (Math.abs(touchY) > 0.15 || Math.abs(touchX) > 0.15) {
    fwd = touchY * 0.8;
    lat = -touchX * 0.3;
  }
  if (touchRotL) turn = 0.5;
  if (touchRotR) turn = -0.5;
  const ctrl = getActiveCtrl();
  if (ctrl && ctrl.enabled && ctrl.setCommand) {
    ctrl.setCommand(fwd, lat, turn);
  }
}
function getActiveCtrl() {
  switch (activeController) {
    case "go2":
      return go2Controller;
    case "go2rl":
      return go2RlController;
    case "h1":
      return h1Controller;
    case "b2":
      return b2Controller;
    case "g1":
      return g1Controller;
    case "h1_2":
      return h1_2Controller;
    case "factory":
      return factoryController;
    default:
      return null;
  }
}
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  keys[e.code] = true;
  if (e.code === "Space") {
    paused = !paused;
    e.preventDefault();
  }
  if (e.code === "KeyP") toggleController();
  if (e.code === "KeyR") resetScene();
  if (e.code === "KeyC") cameraFollow = !cameraFollow;
  if (e.code === "KeyH" && helpOverlay) {
    helpOverlay.style.display = helpOverlay.style.display === "none" ? "" : "none";
  }
  if (e.code === "KeyF") spawnObstacle(Math.random() < 0.5 ? "ball" : "box");
  if (e.code === "BracketRight") {
    const idx = SIM_SPEEDS.indexOf(simSpeed);
    if (idx < SIM_SPEEDS.length - 1) {
      simSpeed = SIM_SPEEDS[idx + 1];
      updateSpeedBtn();
    }
  }
  if (e.code === "BracketLeft") {
    const idx = SIM_SPEEDS.indexOf(simSpeed);
    if (idx > 0) {
      simSpeed = SIM_SPEEDS[idx - 1];
      updateSpeedBtn();
    }
  }
});
window.addEventListener("keyup", (e) => {
  keys[e.code] = false;
});
sceneSelect.addEventListener("change", async (e) => {
  try {
    await loadScene(e.target.value);
  } catch (err) {
    setStatus(`Failed: ${e.target.value}`);
    console.error(err);
  }
});
resetBtn.addEventListener("click", resetScene);
if (controllerBtn) controllerBtn.addEventListener("click", toggleController);
if (speedBtn) speedBtn.addEventListener("click", cycleSpeed);
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
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
  const rootBody = activeController === "factory" && factoryController ? factoryController.centerBodyId : 1;
  const x = data.xpos[rootBody * 3 + 0];
  const y = data.xpos[rootBody * 3 + 1];
  const z = data.xpos[rootBody * 3 + 2];
  controls.target.lerp(new THREE2.Vector3(x, z, -y), 0.05);
}
function stepController() {
  const ctrl = getActiveCtrl();
  if (!ctrl) return;
  if (ctrl.step) ctrl.step();
}
function setupControls() {
  const joystickZone = document.getElementById("joystick-zone");
  const joystickBase = document.getElementById("joystick-base");
  const joystickThumb = document.getElementById("joystick-thumb");
  const mobilePanel = document.getElementById("mobile-panel");
  const helpOverlayEl = document.getElementById("help-overlay");
  if (!isTouchDevice) return;
  if (joystickZone) joystickZone.style.display = "block";
  if (mobilePanel) mobilePanel.style.display = "flex";
  if (helpOverlayEl) helpOverlayEl.style.display = "none";
  if (joystickZone && joystickBase && joystickThumb) {
    const baseRadius = 65, thumbHalf = 24, maxDist = 40;
    let movePid = null;
    const updateThumb = (cx, cy) => {
      const rect = joystickBase.getBoundingClientRect();
      let dx = cx - (rect.left + baseRadius);
      let dy = cy - (rect.top + baseRadius);
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxDist) {
        dx = dx / dist * maxDist;
        dy = dy / dist * maxDist;
      }
      joystickThumb.style.left = baseRadius - thumbHalf + dx + "px";
      joystickThumb.style.top = baseRadius - thumbHalf + dy + "px";
      touchX = dx / maxDist;
      touchY = -dy / maxDist;
    };
    const resetThumb = () => {
      joystickThumb.style.left = baseRadius - thumbHalf + "px";
      joystickThumb.style.top = baseRadius - thumbHalf + "px";
      touchX = 0;
      touchY = 0;
      movePid = null;
    };
    joystickZone.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      movePid = e.pointerId;
      joystickZone.setPointerCapture(e.pointerId);
      updateThumb(e.clientX, e.clientY);
    });
    joystickZone.addEventListener("pointermove", (e) => {
      if (e.pointerId !== movePid) return;
      updateThumb(e.clientX, e.clientY);
    });
    joystickZone.addEventListener("pointerup", (e) => {
      if (e.pointerId === movePid) resetThumb();
    });
    joystickZone.addEventListener("pointercancel", (e) => {
      if (e.pointerId === movePid) resetThumb();
    });
  }
  if (mobilePanel) {
    mobilePanel.querySelectorAll("[data-action]").forEach((btn) => {
      const action = btn.dataset.action;
      if (action === "rotL" || action === "rotR") {
        btn.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (action === "rotL") touchRotL = true;
          if (action === "rotR") touchRotR = true;
        });
        btn.addEventListener("pointerup", () => {
          if (action === "rotL") touchRotL = false;
          if (action === "rotR") touchRotR = false;
        });
        btn.addEventListener("pointercancel", () => {
          touchRotL = false;
          touchRotR = false;
        });
      }
      if (action === "ball") {
        btn.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          e.stopPropagation();
          spawnObstacle("ball");
        });
      }
      if (action === "box") {
        btn.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          e.stopPropagation();
          spawnObstacle("box");
        });
      }
    });
  }
}
(async () => {
  try {
    setStatus("Loading MuJoCo WASM...");
    mujoco = await load_mujoco();
    if (!mujoco.FS.analyzePath("/working").exists) {
      mujoco.FS.mkdir("/working");
    }
    await loadScene("unitree_go2/scene.xml");
    setupControls();
  } catch (e) {
    setStatus("Boot failed");
    console.error(e);
    return;
  }
  const MAX_SUBSTEPS = 40;
  async function animate() {
    if (model && data && !paused) {
      handleInput();
      const timestep = model.opt.timestep;
      const frameDt = 1 / 60;
      const nsteps = Math.min(Math.round(frameDt / timestep * simSpeed), MAX_SUBSTEPS);
      for (let s = 0; s < nsteps; s++) {
        stepController();
        mujoco.mj_step(model, data);
        stepCounter++;
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
