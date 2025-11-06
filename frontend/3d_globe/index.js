import * as THREE from "three";
import { OrbitControls } from "jsm/controls/OrbitControls.js";
import getStarfield from "./src/getStarfield.js";
// GeoTIFF is loaded as global script
// --- Window & Scene Setup ---
const w = window.innerWidth;
const h = window.innerHeight;
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(75, w / h, 1, 100);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(w, h);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// --- Mars Sphere Setup ---
const marsGeometry = new THREE.SphereGeometry(2, 64, 64);
const marsTexture = new THREE.TextureLoader().load("./textures/mars_8k.jpg");
const marsMaterial = new THREE.MeshPhongMaterial({ map: marsTexture });
const marsSphere = new THREE.Mesh(marsGeometry, marsMaterial);
scene.add(marsSphere);

// Optional Wireframe
const wireframe = new THREE.LineSegments(
  new THREE.EdgesGeometry(marsGeometry),
  new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.4, transparent: true })
);
// scene.add(wireframe); // Uncomment to display wireframe

// --- Stars ---
const stars = getStarfield({ numStars: 1000, fog: false });
scene.add(stars);

// --- Sun Setup ---
const sunPivot = new THREE.Object3D();
scene.add(sunPivot);

const sunDistance = 50;
let sunRotationEnabled = true;

const sunLight = new THREE.DirectionalLight(0xffffff, 1);
sunLight.position.set(sunDistance, 0, 0);
sunLight.target.position.set(0, 0, 0);
scene.add(sunLight.target);
sunPivot.add(sunLight);

const sunMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.5, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0xffff00 })
);
sunMesh.position.set(sunDistance, 0, 0);
sunPivot.add(sunMesh);

// --- UI Controls ---
document.getElementById("toggleSun").addEventListener("click", () => {
  sunRotationEnabled = !sunRotationEnabled;
});

document.getElementById("sunAngle").addEventListener("input", (e) => {
  const angle = THREE.MathUtils.degToRad(e.target.value);
  sunMesh.position.set(Math.cos(angle) * sunDistance, 0, Math.sin(angle) * sunDistance);
  sunLight.position.copy(sunMesh.position);
  sunRotationEnabled = false;
});

// Landing suitability prediction
async function predictLandingSuitability() {
  if (!currentMarsData) {
    alert("Please click on Mars first to get data");
    return;
  }
  
  try {
    document.getElementById("landingScore").innerText = "🔄 Predicting landing suitability...";
    document.getElementById("predictLanding").disabled = true;
    
    const response = await fetch('http://localhost:5002/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(currentMarsData)
    });
    
    const result = await response.json();
    
    if (result.success) {
      const score = result.landing_score;
      let scoreColor = '#4CAF50'; // Green
      let scoreText = 'Excellent';
      
      if (score < 30) {
        scoreColor = '#f44336'; // Red
        scoreText = 'Poor';
      } else if (score < 50) {
        scoreColor = '#ff9800'; // Orange
        scoreText = 'Fair';
      } else if (score < 70) {
        scoreColor = '#ffeb3b'; // Yellow
        scoreText = 'Good';
      }
      
      document.getElementById("landingScore").innerHTML = `
        <div style="color: ${scoreColor}; font-size: 18px;">
          🚀 Landing Suitability: ${score}% (${scoreText})
        </div>
        <div style="font-size: 12px; margin-top: 5px;">
          Neural Network Predictions:<br>
          • Slope: ${result.predictions.neural_networks.slope?.toFixed(2) || 'N/A'}<br>
          • Dust: ${result.predictions.neural_networks.dust?.toFixed(2) || 'N/A'}<br>
          • Surface Temp: ${result.predictions.neural_networks.surface_temp?.toFixed(2) || 'N/A'}°C<br>
          • Thermal Inertia: ${result.predictions.neural_networks.thermal_inertia?.toFixed(2) || 'N/A'}<br>
          • Water: ${result.predictions.neural_networks.water?.toFixed(2) || 'N/A'}
        </div>
      `;
      
      console.log("Prediction result:", result);
    } else {
      document.getElementById("landingScore").innerText = `❌ Error: ${result.error}`;
    }
    
  } catch (error) {
    console.error("API call failed:", error);
    document.getElementById("landingScore").innerText = `❌ Failed to connect to API: ${error.message}`;
  } finally {
    document.getElementById("predictLanding").disabled = false;
  }
}

// Event listener for prediction button
document.getElementById("predictLanding").addEventListener("click", predictLandingSuitability);

// --- Animate Loop ---
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  if (sunRotationEnabled) sunPivot.rotation.y += 0.002;
}
animate();

// --- Handle Window Resize ---
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Marker Setup ---
const marker = new THREE.Mesh(
  new THREE.SphereGeometry(0.05, 16, 16),
  new THREE.MeshBasicMaterial({ color: "red" })
);
marker.visible = false;
scene.add(marker);

// --- Raycasting ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// --- Mars Datasets Configuration ---
const marsDatasets = {
  elevation: {
    name: "Elevation (MOLA)",
    file: "./public/data/MOLA_128ppd_topo.tif",
    unit: "m",
    description: "Mars Orbiter Laser Altimeter elevation data"
  },
  slope: {
    name: "Slope",
    file: "./public/data/mola_hrsc_blend_slope_v2.tif",
    unit: "°",
    description: "Surface slope measurements"
  },
  roughness: {
    name: "Roughness",
    file: "./public/data/mola_roughness_0.6km_numeric.tif",
    unit: "m",
    description: "Surface roughness at 0.6km scale"
  },
  albedo: {
    name: "Albedo",
    file: "./public/data/omega_albedo_r1080.tif",
    unit: "",
    description: "Surface albedo (reflectivity)"
  },
  temperature: {
    name: "Temperature",
    file: "./public/data/mars_yearly_avg_temperature_celsius.tif",
    unit: "°C",
    description: "Yearly average surface temperature"
  },
  tempRange: {
    name: "Temperature Range",
    file: "./public/data/mars_yearly_temperature_range_v1.0.tif",
    unit: "°C",
    description: "Yearly temperature variation"
  },
  crustalThickness: {
    name: "Crustal Thickness",
    file: "./public/data/mars_crustal_thickness_gmm3_rm1.tif",
    unit: "km",
    description: "Mars crustal thickness"
  },
  ferric: {
    name: "Ferric Content",
    file: "./public/data/omega_ferric_nnphs.tif",
    unit: "",
    description: "Ferric iron content"
  },
  pyroxene: {
    name: "Pyroxene",
    file: "./public/data/omega_pyroxene_bd2000.tif",
    unit: "",
    description: "Pyroxene mineral content"
  },
  basalt: {
    name: "Basalt",
    file: "./public/data/TES_Basalt_numeric.tif",
    unit: "",
    description: "Basalt abundance"
  },
  lambertAlbedo: {
    name: "Lambert Albedo",
    file: "./public/data/TES_Lambert_Albedo_numeric.tif",
    unit: "",
    description: "Lambert albedo from TES"
  }
};

let currentDataset = null;
let currentDatasetType = 'elevation';
let loadedDatasets = new Map();
let currentMarsData = null; // Store current Mars data for API calls

async function loadGeoTIFF(url) {
  try {
    console.log("Loading GeoTIFF from:", url);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    console.log("ArrayBuffer loaded, size:", arrayBuffer.byteLength);
    
    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const data = await image.readRasters({ interleave: true });
    
    
    // Handle cases where getBoundingBox() might fail due to missing affine transformation
    let bounds;
    try {
      bounds = image.getBoundingBox();
    } catch (boundsError) {
      // Silently use default bounds for Mars (assuming equirectangular projection)
      bounds = [-180, -90, 180, 90];
    }
    
    return { 
      width: image.getWidth(), 
      height: image.getHeight(), 
      data, 
      bounds 
    };
  } catch (error) {
    console.error("Error loading GeoTIFF:", error);
    // Fallback to mock data
    console.log("Falling back to mock data");
    return {
      width: 1024,
      height: 512,
      data: new Array(1024 * 512).fill(0).map(() => Math.random() * 1000 - 500),
      bounds: [-180, -90, 180, 90]
    };
  }
}

// Load a specific dataset
async function loadDataset(datasetType) {
  const dataset = marsDatasets[datasetType];
  if (!dataset) {
    console.error("Unknown dataset type:", datasetType);
    return null;
  }

  // Check if already loaded
  if (loadedDatasets.has(datasetType)) {
    return loadedDatasets.get(datasetType);
  }

  const data = await loadGeoTIFF(dataset.file);
  loadedDatasets.set(datasetType, data);
  return data;
}

// Switch to a different dataset
async function switchDataset(datasetType) {
  currentDatasetType = datasetType;
  currentDataset = await loadDataset(datasetType);
  
  // Update UI
  const dataset = marsDatasets[datasetType];
  document.getElementById("datasetInfo").innerText = `Current: ${dataset.name}`;
  document.getElementById("datasetDescription").innerText = dataset.description;
}

// Load initial dataset at startup
(async () => {
  try {
    currentDataset = await loadDataset('elevation');
    console.log("Loaded initial Mars dataset:", currentDataset.width, "x", currentDataset.height);
  } catch (error) {
    console.error("Failed to load Mars dataset:", error);
    // Show user-friendly error message
    document.getElementById("coords").innerText = "Error loading elevation data";
  }
})();

// Convert latitude/longitude to pixel coordinates
function latLonToPixel(lat, lon, width, height) {
  const x = Math.floor(((lon + 180) / 360) * width);
  const y = Math.floor(((90 - lat) / 180) * height);
  return { x, y };
}

function getValueAt(lat, lon) {
  if (!currentDataset) return null;
  const { width, height, data } = currentDataset;
  const { x, y } = latLonToPixel(lat, lon, width, height);
  
  // Check bounds to prevent array index errors
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return null;
  }
  
  const index = y * width + x;
  if (index >= data.length) {
    return null;
  }
  
  return data[index];
}

// Get value from a specific dataset at given coordinates
function getValueFromDataset(datasetType, lat, lon) {
  const dataset = loadedDatasets.get(datasetType);
  if (!dataset) return null;
  
  const { width, height, data } = dataset;
  const { x, y } = latLonToPixel(lat, lon, width, height);
  
  // Check bounds to prevent array index errors
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return null;
  }
  
  const index = y * width + x;
  if (index >= data.length) {
    return null;
  }
  
  return data[index];
}

// --- Mouse Click Handler ---
async function onMouseClick(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(marsSphere);

  if (intersects.length > 0) {
    const point = intersects[0].point;
    const radius = marsSphere.geometry.parameters.radius;
    const lon = Math.atan2(point.z, point.x) * (180 / Math.PI);
    const lat = Math.asin(point.y / radius) * (180 / Math.PI);

    // Show loading message
    document.getElementById("coords").innerText = `Loading all datasets for Lat: ${lat.toFixed(2)}°, Lon: ${lon.toFixed(2)}°...`;

    // Load all datasets and get values
    const allValues = [];
    for (const [datasetType, datasetInfo] of Object.entries(marsDatasets)) {
      try {
        // Load dataset if not already loaded
        if (!loadedDatasets.has(datasetType)) {
          await loadDataset(datasetType);
        }
        
        const value = getValueFromDataset(datasetType, lat, lon);
        allValues.push({
          name: datasetInfo.name,
          value: value,
          unit: datasetInfo.unit,
          description: datasetInfo.description
        });
      } catch (error) {
        console.warn(`Failed to load ${datasetInfo.name}:`, error);
        allValues.push({
          name: datasetInfo.name,
          value: null,
          unit: datasetInfo.unit,
          description: datasetInfo.description
        });
      }
    }

    // Format all values into a list and store data for API
    let valuesList = `Lat: ${lat.toFixed(2)}°, Lon: ${lon.toFixed(2)}°\n\n`;
    currentMarsData = {
      lat: lat,
      lon: lon,
      elevation: null,
      slope: null,
      roughness: null,
      albedo: null,
      temperature: null,
      tempRange: null,
      crustalThickness: null,
      ferric: null,
      pyroxene: null,
      basalt: null,
      lambertAlbedo: null
    };
    
    allValues.forEach(item => {
      const valueStr = item.value !== null ? `${item.value.toFixed(2)} ${item.unit}` : "N/A";
      valuesList += `• ${item.name}: ${valueStr}\n`;
      
      // Store data for API call
      const key = item.name.toLowerCase().replace(/[^a-z0-9]/g, '');
      if (key.includes('elevation')) currentMarsData.elevation = item.value;
      else if (key.includes('slope')) currentMarsData.slope = item.value;
      else if (key.includes('roughness')) currentMarsData.roughness = item.value;
      else if (key.includes('albedo') && !key.includes('lambert')) currentMarsData.albedo = item.value;
      else if (key.includes('temperature') && !key.includes('range')) currentMarsData.temperature = item.value;
      else if (key.includes('temperaturerange')) currentMarsData.tempRange = item.value;
      else if (key.includes('crustalthickness')) currentMarsData.crustalThickness = item.value;
      else if (key.includes('ferric')) currentMarsData.ferric = item.value;
      else if (key.includes('pyroxene')) currentMarsData.pyroxene = item.value;
      else if (key.includes('basalt')) currentMarsData.basalt = item.value;
      else if (key.includes('lambertalbedo')) currentMarsData.lambertAlbedo = item.value;
    });

    // Update UI
    document.getElementById("coords").innerText = valuesList;
    
    // Show prediction button
    document.getElementById("predictLanding").style.display = "block";
    document.getElementById("landingScore").innerText = "";

    // Show marker
    marker.position.copy(point);
    marker.visible = true;
  }
}

window.addEventListener("click", onMouseClick, false);

