/**
 * Evolution scene generator — creates a MuJoCo scene with multiple
 * copies of a humanoid robot for evolutionary learning.
 *
 * Uses regex-based name prefixing (works for any robot XML).
 */

/**
 * Prefix all entity names in robot XML (bodies, joints, geoms, sites, actuators).
 * Does NOT touch mesh=, class=, material= references (shared assets).
 */
function prefixAllNames(xml, p) {
  // Prefix name="..." and joint="..." and target="..." attributes
  xml = xml.replace(/\bname="([^"]+)"/g, `name="${p}$1"`);
  xml = xml.replace(/\bjoint="([^"]+)"/g, `joint="${p}$1"`);
  xml = xml.replace(/\btarget="([^"]+)"/g, `target="${p}$1"`);
  // Handle body1/body2 in contact excludes
  xml = xml.replace(/\bbody1="([^"]+)"/g, `body1="${p}$1"`);
  xml = xml.replace(/\bbody2="([^"]+)"/g, `body2="${p}$1"`);
  // Name unnamed freejoints
  xml = xml.replace('<freejoint/>', `<freejoint name="${p}freejoint"/>`);
  return xml;
}

/**
 * Generate a multi-robot H1 scene for evolution.
 * @param {object} mujoco - MuJoCo WASM instance
 * @param {number} numRobots - Population size (e.g. 9)
 * @param {number} spacing - Grid spacing in meters
 * @returns {string} Complete MuJoCo XML
 */
export function generateH1EvolutionXML(mujoco, numRobots, spacing) {
  const h1Xml = new TextDecoder().decode(
    mujoco.FS.readFile('/working/unitree_h1/h1.xml')
  );

  // Extract sections
  const defaultStart = h1Xml.indexOf('<default>');
  const defaultEnd = h1Xml.lastIndexOf('</default>') + '</default>'.length;
  const defaultSection = h1Xml.substring(defaultStart, defaultEnd);

  const assetStart = h1Xml.indexOf('<asset>') + '<asset>'.length;
  const assetEnd = h1Xml.indexOf('</asset>');
  const assetContent = h1Xml.substring(assetStart, assetEnd);

  // Robot body (inside worldbody, skip the light element)
  const wbStart = h1Xml.indexOf('<worldbody>') + '<worldbody>'.length;
  const wbEnd = h1Xml.indexOf('</worldbody>');
  let robotBodyXml = h1Xml.substring(wbStart, wbEnd).trim();
  // Remove any light elements (we'll add our own)
  robotBodyXml = robotBodyXml.replace(/<light[^>]*\/>/g, '');

  const actStart = h1Xml.indexOf('<actuator>') + '<actuator>'.length;
  const actEnd = h1Xml.indexOf('</actuator>');
  const robotActXml = h1Xml.substring(actStart, actEnd).trim();

  // Contact excludes
  let contactXml = '';
  const contactStart = h1Xml.indexOf('<contact>');
  const contactEnd = h1Xml.indexOf('</contact>');
  const contactContent = contactStart >= 0
    ? h1Xml.substring(contactStart + '<contact>'.length, contactEnd).trim()
    : '';

  // Grid layout
  const cols = Math.ceil(Math.sqrt(numRobots));

  let allBodies = '';
  let allActuators = '';
  let allContacts = '';
  const qposValues = [];

  // H1 home qpos: 7 (freejoint) + 19 (joints) = 26 per robot
  // left_leg: 0,0,-0.4,0.8,-0.4  right_leg: 0,0,-0.4,0.8,-0.4  torso: 0
  // left_arm: 0,0.2,0,-0.3  right_arm: 0,-0.2,0,-0.3
  const homeJoints = '0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0.2 0 -0.3 0 -0.2 0 -0.3';

  for (let i = 0; i < numRobots; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const x = (col - (cols - 1) / 2) * spacing;
    const y = (row - (cols - 1) / 2) * spacing;
    const prefix = `r${i}_`;

    // Prefix names and update pelvis position
    let body = prefixAllNames(robotBodyXml, prefix);
    body = body.replace(
      /(<body\s+name="r\d+_pelvis"\s+pos=")([^"]+)(")/,
      `$1${x.toFixed(2)} ${y.toFixed(2)} 1.06$3`
    );
    allBodies += body + '\n';

    // Prefix actuator section
    allActuators += prefixAllNames(robotActXml, prefix) + '\n';

    // Prefix contact excludes
    if (contactContent) {
      allContacts += prefixAllNames(contactContent, prefix) + '\n';
    }

    // Keyframe
    qposValues.push(`${x.toFixed(2)} ${y.toFixed(2)} 1.06 1 0 0 0 ${homeJoints}`);
  }

  return `<mujoco model="h1 evolution">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option cone="elliptic" impratio="100"/>
  <statistic meansize="0.05"/>

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

  <contact>
${allContacts}
  </contact>

  <actuator>
${allActuators}
  </actuator>

  <keyframe>
    <key name="home" qpos="${qposValues.join(' ')}"/>
  </keyframe>
</mujoco>`;
}
