/**
 * Factory scene generator — creates a MuJoCo scene with multiple Go2 robots.
 * Reads go2.xml from WASM filesystem, duplicates bodies/joints/actuators
 * with prefixed names, and arranges them in a grid layout.
 */

/**
 * Replace all robot-specific names in XML with prefixed versions.
 * Handles body names, joint names, geom names, site names, and actuator joint references.
 */
function prefixNames(xml, p) {
  // All unique names in go2.xml, sorted longest-first to prevent substring issues
  const names = [
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
    'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
    'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf',
    'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
    'base', 'imu',
    'FL', 'FR', 'RL', 'RR',
  ];

  for (const name of names) {
    xml = xml.replaceAll(`name="${name}"`, `name="${p}${name}"`);
    xml = xml.replaceAll(`joint="${name}"`, `joint="${p}${name}"`);
  }

  // Name the freejoint (originally unnamed: <freejoint/>)
  xml = xml.replace('<freejoint/>', `<freejoint name="${p}freejoint"/>`);

  return xml;
}

/**
 * Generate a factory scene XML with N Go2 robots in a grid.
 * @param {object} mujoco - MuJoCo WASM instance (for FS access)
 * @param {number} numRobots - Number of robots (e.g. 4, 9, 16)
 * @param {number} spacing - Grid spacing in meters (e.g. 1.5)
 * @returns {string} Complete MuJoCo XML string
 */
export function generateFactoryXML(mujoco, numRobots, spacing) {
  const go2Xml = new TextDecoder().decode(
    mujoco.FS.readFile('/working/unitree_go2/go2.xml')
  );

  // Extract sections using indexOf (robust for nested XML tags)
  const defaultStart = go2Xml.indexOf('<default>');
  const defaultEnd = go2Xml.lastIndexOf('</default>') + '</default>'.length;
  const defaultSection = go2Xml.substring(defaultStart, defaultEnd);

  const assetStart = go2Xml.indexOf('<asset>') + '<asset>'.length;
  const assetEnd = go2Xml.indexOf('</asset>');
  const assetContent = go2Xml.substring(assetStart, assetEnd);

  const wbStart = go2Xml.indexOf('<worldbody>') + '<worldbody>'.length;
  const wbEnd = go2Xml.indexOf('</worldbody>');
  const robotBodyXml = go2Xml.substring(wbStart, wbEnd).trim();

  const actStart = go2Xml.indexOf('<actuator>') + '<actuator>'.length;
  const actEnd = go2Xml.indexOf('</actuator>');
  const robotActXml = go2Xml.substring(actStart, actEnd).trim();

  // Grid layout
  const cols = Math.ceil(Math.sqrt(numRobots));
  const rows = Math.ceil(numRobots / cols);

  let allBodies = '';
  let allActuators = '';
  const qposValues = [];
  const ctrlValues = [];

  for (let i = 0; i < numRobots; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const x = (col - (cols - 1) / 2) * spacing;
    const y = (row - (rows - 1) / 2) * spacing;
    const prefix = `r${i}_`;

    // Prefix names in body tree and update base position
    let body = prefixNames(robotBodyXml, prefix);
    body = body.replace('pos="0 0 0.445"', `pos="${x.toFixed(2)} ${y.toFixed(2)} 0.445"`);
    allBodies += '    ' + body + '\n';

    // Prefix names in actuator section
    allActuators += '    ' + prefixNames(robotActXml, prefix) + '\n';

    // Keyframe values: 3 pos + 4 quat + 12 joints = 19 qpos, 12 ctrl per robot
    qposValues.push(
      `${x.toFixed(2)} ${y.toFixed(2)} 0.27 1 0 0 0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8`
    );
    ctrlValues.push('0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8');
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
      qpos="${qposValues.join(' ')}"
      ctrl="${ctrlValues.join(' ')}"/>
  </keyframe>
</mujoco>`;
}
