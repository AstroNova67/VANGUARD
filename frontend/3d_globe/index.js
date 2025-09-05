import * as THREE from "three";
import { OrbitControls } from 'jsm/controls/OrbitControls.js';
import getStarfield from "./src/getStarfield.js";
import { drawThreeGeo } from "./src/threeGeoJSON.js";

const w = window.innerWidth;
const h = window.innerHeight;
const scene = new THREE.Scene();

// Fog don't use for now
// scene.fog = new THREE.FogExp2(0x000000, 0.1);

const camera = new THREE.PerspectiveCamera(75, w / h, 1, 100);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(w, h);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Sphere geometry
const geometry = new THREE.SphereGeometry(2, 64, 64);

// Load Mars texture
const textureLoader = new THREE.TextureLoader();
const marsTexture = textureLoader.load("./textures/mars_8k.jpg");

// Apply texture to material
const sphereMaterial = new THREE.MeshPhongMaterial({
  map: marsTexture,
});

// Create mesh
const sphere = new THREE.Mesh(geometry, sphereMaterial);
scene.add(sphere);

// White outline edges (optional)
const lineMat = new THREE.LineBasicMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.4,
});
const edges = new THREE.EdgesGeometry(geometry, 1);
const line = new THREE.LineSegments(edges, lineMat);
// scene.add(line); // uncomment if you want the wireframe

// Add stars
const stars = getStarfield({ numStars: 1000, fog: false });
scene.add(stars);

// Sun Light setup
const sunLight = new THREE.DirectionalLight(0xffffff, 1);
sunLight.castShadow = true;
sunLight.position.set(50, 0, 0);

sunLight.target.position.set(0, 0, 0);
scene.add(sunLight.target);

const sunPivot = new THREE.Object3D();
scene.add(sunPivot);
sunPivot.add(sunLight);

// Visible Sun Mesh
const sunGeometry = new THREE.SphereGeometry(0.5, 32, 32);
const sunMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
const sunMesh = new THREE.Mesh(sunGeometry, sunMaterial);
sunMesh.position.set(50, 0, 0);
sunPivot.add(sunMesh);

// === NEW: UI state ===
let sunRotationEnabled = true; // button toggles this
const distance = 50;           // radius of sun orbit

// Hook up button
const toggleBtn = document.getElementById("toggleSun");
toggleBtn.addEventListener("click", () => {
  sunRotationEnabled = !sunRotationEnabled;
});

// Hook up slider
const sunSlider = document.getElementById("sunAngle");
sunSlider.addEventListener("input", (e) => {
  const angle = THREE.MathUtils.degToRad(e.target.value);

  // place sun/light manually
  sunMesh.position.set(Math.cos(angle) * distance, 0, Math.sin(angle) * distance);
  sunLight.position.copy(sunMesh.position);

  // when slider is used, freeze auto-rotation
  sunRotationEnabled = false;
});

// Animate
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
  controls.update();

  if (sunRotationEnabled) {
    sunPivot.rotation.y += 0.002; // orbit speed
  }
}
animate();

// Resize
function handleWindowResize () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', handleWindowResize, false);

// Marker for clicked location
const markerGeomertry = new THREE.SphereGeometry(0.05, 16, 16);
const markerMaterial = new THREE.MeshBasicMaterial({color: 'red'});
const marker = new THREE.Mesh(markerGeomertry, markerMaterial);
scene.add(marker);
marker.visible = false; // hidden until first click

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onMouseClick(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(sphere);

  if (intersects.length > 0) {
    const point = intersects[0].point;
    const radius = sphere.geometry.parameters.radius;
    const x = point.x;
    const y = point.y;
    const z = point.z;

    const lon = Math.atan2(z, x) * (180 / Math.PI);
    const lat = Math.asin(y / radius) * (180 / Math.PI);

    // Update overlay
    const coordsDiv = document.getElementById("coords");
    coordsDiv.innerText = `Lat: ${lat.toFixed(2)}°, Lon: ${lon.toFixed(2)}°`;

    // Move marker to clicked point
    marker.position.copy(point);
    marker.visible = true;
  }
}
window.addEventListener("click", onMouseClick, false);
