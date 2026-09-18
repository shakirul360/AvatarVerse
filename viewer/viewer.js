/* Live, free-orbit avatar viewer. Ports pipeline/render.py's three.js scene setup (lights,
material, sanitizeNormals) almost line-for-line - see that file's docstring for why each piece
looks the way it does. Three things differ from the offline renderer it's ported from:
  1. camera: OrbitControls instead of a fixed eye position (participant can rotate/zoom).
  2. geometry loading: DRACOLoader decoding base.drc (connectivity+uv+frame-0 positions, once)
     and frames/frame_%04d.drc (per-frame position DELTAS - see pipeline/geometry_export.py's
     module docstring for why deltas, not absolute positions) instead of raw float32 buffers.
  3. animation: a self-driving requestAnimationFrame loop advancing by wall-clock time, instead
     of Playwright calling window.setFrame(i) once per screenshot.
*/
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { DRACOLoader } from './vendor/DRACOLoader.js';

const params = new URLSearchParams(location.search);
const bundleUrl = (params.get('bundle') || '.').replace(/\/$/, '');

const canvas = document.getElementById('c');
const loadingEl = document.getElementById('loading');

// Same fixed tripod framing as render.py (FOV, FILL, OBJ_H, distance formula) - kept identical
// so the initial view matches what every rendered reference video already shows. The eye
// direction's x/y is negated from render.py's (1.8,-1.8,1.2) to fix the avatar-faces-backward
// issue (a camera-direction choice, not a data problem) - this is the one place that fix lands.
const FOV = 45, FILL = 0.52, OBJ_H = 1.8;
const dist = OBJ_H / (2 * FILL * Math.tan(THREE.MathUtils.degToRad(FOV / 2))) * 0.92;
const eyeDir = new THREE.Vector3(-1.8, 1.8, 1.2).normalize();
const eye = eyeDir.multiplyScalar(dist);
const lightDir = new THREE.Vector3(180, -60, 90).normalize().multiplyScalar(10);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);

const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 100);
camera.position.copy(eye);
camera.up.set(0, 0, 1);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;

scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.copy(lightDir);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.35);
fill.position.set(-lightDir.x, -lightDir.y, lightDir.z);
scene.add(fill);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;

// See render.py's sanitizeNormals() docstring - a degenerate (zero-area) triangle, from
// aggressive decimation or a wildly-quantized pose, makes computeVertexNormals() emit NaN/
// Infinity, which corrupts more than just that triangle. Kept here for the same reason: real,
// possible in this data, cheap to guard against regardless of cause.
function sanitizeNormals(geometry) {
  const n = geometry.getAttribute('normal').array;
  for (let i = 0; i < n.length; i++) if (!isFinite(n[i])) n[i] = 0;
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / (h || 1);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function decodeDraco(loader, buffer) {
  return new Promise((resolve, reject) => loader.parse(buffer, resolve, reject));
}

async function fetchBuffer(path) {
  const r = await fetch(`${bundleUrl}/${path}`);
  if (!r.ok) throw new Error(`fetch ${path} failed: ${r.status}`);
  return r.arrayBuffer();
}

async function loadTexture(path) {
  const img = await new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = `${bundleUrl}/${path}`;
  });
  const tex = new THREE.Texture(img);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

async function main() {
  const meta = await (await fetch(`${bundleUrl}/meta.json`)).json();
  const nVerts = meta.n_verts;

  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath('./vendor/draco/');
  dracoLoader.preload();

  loadingEl.textContent = 'loading avatar… base mesh';
  const [baseBuf, tex] = await Promise.all([
    fetchBuffer('base.drc'),
    loadTexture('texture.jpg'),
  ]);
  const baseGeometry = await decodeDraco(dracoLoader, baseBuf);
  const basePositions = new Float32Array(baseGeometry.getAttribute('position').array);

  // One absolute-position Float32Array per frame, decoded up front (frame count is small - ~46 -
  // and each point-cloud decode is cheap; see geometry_export.py's module docstring for the
  // delta-encoding rationale). Frame 0 is the base mesh's own positions (delta-zero, no file).
  const nDeltaFrames = meta.n_delta_frames;
  const framePositions = new Array(nDeltaFrames + 1);
  framePositions[0] = basePositions;
  for (let i = 1; i <= nDeltaFrames; i++) {
    loadingEl.textContent = `loading avatar… ${i} / ${nDeltaFrames}`;
    const buf = await fetchBuffer(`frames/frame_${String(i).padStart(4, '0')}.drc`);
    const deltaGeometry = await decodeDraco(dracoLoader, buf);
    const delta = deltaGeometry.getAttribute('position').array;
    const abs = new Float32Array(nVerts * 3);
    for (let j = 0; j < nVerts * 3; j++) abs[j] = basePositions[j] + delta[j];
    framePositions[i] = abs;
    deltaGeometry.dispose();
  }
  dracoLoader.dispose();

  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(baseGeometry.index);
  geometry.setAttribute('uv', baseGeometry.getAttribute('uv'));
  geometry.setAttribute('position', new THREE.BufferAttribute(framePositions[0].slice(), 3));
  geometry.computeVertexNormals();
  sanitizeNormals(geometry);
  baseGeometry.dispose();

  const material = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85, metalness: 0.0 });
  scene.add(new THREE.Mesh(geometry, material));

  loadingEl.classList.add('hidden');
  resize();

  const nFrames = framePositions.length;
  const playbackFps = meta.playback_fps;
  let startTime = null, lastFrame = -1;
  const posAttr = geometry.getAttribute('position');

  function setFrame(i) {
    posAttr.array.set(framePositions[i]);
    posAttr.needsUpdate = true;
    geometry.computeVertexNormals();
    sanitizeNormals(geometry);
  }

  function animate(t) {
    if (startTime === null) startTime = t;
    const frameIdx = Math.floor(((t - startTime) / 1000) * playbackFps) % nFrames;
    if (frameIdx !== lastFrame) { setFrame(frameIdx); lastFrame = frameIdx; }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

main().catch(err => {
  loadingEl.textContent = `failed to load: ${err.message}`;
  console.error(err);
});
