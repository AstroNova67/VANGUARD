import * as THREE from "three";
import { OrbitControls } from 'jsm/controls/OrbitControls.js';
import getStarfield from "./src/getStarfield.js";
import { drawThreeGeo } from "./src/threeGeoJSON.js";

const w = window.innerWidth;
const h = window.innerHeight;
const scene = new THREE.Scene();

// Fog
scene.fog = new THREE.FogExp2(0x000000, 0.1);

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

// Add lights (needed for MeshPhongMaterial)
// const light = new THREE.DirectionalLight(0xffffff, 2);
// light.position.set(5, 5, 5);
// scene.add(light);
// scene.add(new THREE.AmbientLight(0x404040));


// Sun Light setup
const sunLight = new THREE.DirectionalLight(0xffffff, 1);
sunLight.castShadow = true;
sunLight.position.set(50, 0, 0);

// const sunHelper = new THREE.DirectionalLightHelper(sunLight, 5);
// scene.add(sunHelper);

sunLight.target.position.set(0, 0, 0);
scene.add(sunLight.target);

const sunPivot = new THREE.Object3D();
scene.add(sunPivot);
sunPivot.add(sunLight);

// const light2 = new THREE.DirectionalLight(0xffffff, 3);
// light2.position.set(-5, -5, -5);
// scene.add(light2);


// Load countries from geojson
// fetch('./geojson/ne_110m_land.json')
//   .then(response => response.text())
//   .then(text => {
//     const data = JSON.parse(text);
//     const countries = drawThreeGeo({
//       json: data,
//       radius: 2,
//       materialOptions: {
//         color: 0x80FF80,
//       },
//     });
//     scene.add(countries);
//   });

// Visible Sun Mesh
const sunGeometry = new THREE.SphereGeometry(0.5, 32, 32); // smaller than Mars
const sunMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
const sunMesh = new THREE.Mesh(sunGeometry, sunMaterial);

// put the sun mesh in the same spot as the light
sunMesh.position.set(50, 0, 0);

// attach to pivot so it orbits with the light
sunPivot.add(sunMesh);

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
  controls.update();
  sunPivot.rotation.y += 0.002; // orbit speed
}
animate();

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

// Hide maker until first click
marker.visible = false;


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

