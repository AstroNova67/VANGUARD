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

function formatPred(n, decimals = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(decimals);
}

function unitSuffix(unit) {
  if (unit === "°C") return "°C";
  if (unit === "%") return "%";
  if (unit === "°") return "°";
  return "";
}

/** @param {'slope'|'temp'|'ti'|'water'|'dust'|null} kind */
function neuralDeltaHtml(observed, pred, kind) {
  if (kind == null) return { wrapClass: "", deltaHtml: "" };
  const o = observed != null && observed !== "" ? Number(observed) : NaN;
  const p = pred != null && pred !== "" ? Number(pred) : NaN;
  if (!Number.isFinite(o) || !Number.isFinite(p)) return { wrapClass: "", deltaHtml: "" };
  const d = Math.abs(p - o);
  let level = "warn";
  if (kind === "slope") {
    if (d < 0.5) level = "good";
    else if (d < 2) level = "mid";
  } else if (kind === "temp") {
    if (d < 5) level = "good";
    else if (d < 15) level = "mid";
  } else if (kind === "ti") {
    if (d < 40) level = "good";
    else if (d < 120) level = "mid";
  } else if (kind === "water") {
    if (d < 1) level = "good";
    else if (d < 4) level = "mid";
  } else if (kind === "dust") {
    if (d < 0.12) level = "good";
    else if (d < 0.35) level = "mid";
  }
  return {
    wrapClass: ` pred-value--delta-${level}`,
    deltaHtml: ` <span class="pred-delta pred-delta--${level}" title="Neural vs raster at this pixel (illustrative)">Δ${formatPred(d)}</span>`,
  };
}

/** Which model's output feeds `landing_score` for temp/TI; slope/dust/water always neural. */
function rowScoreModel(row, overrides) {
  const o = overrides || {};
  if (row.nnKey === "slope" || row.nnKey === "dust" || row.nnKey === "water") return "nn";
  if (row.nnKey === "surface_temp") {
    const v = o.surface_temp;
    if (v && String(v).includes("xgb")) return "xgb";
    return "nn";
  }
  if (row.nnKey === "thermal_inertia") {
    const v = o.thermal_inertia;
    if (v && String(v).includes("xgb")) return "xgb";
    return "nn";
  }
  return null;
}

function scorePinHtml(which) {
  if (which === "nn") {
    return ` <span class="pred-score-pin" title="This number is what the landing score uses for this row">Neural · in score</span>`;
  }
  if (which === "xgb") {
    return ` <span class="pred-score-pin pred-score-pin--xgb" title="This number is what the landing score uses for this row">XGB · in score</span>`;
  }
  return "";
}

function buildPredRows(raw, nn, reg, overrides) {
  const rows = [
    {
      label: "Slope",
      unit: "°",
      nnKey: "slope",
      regKey: null,
      obsKey: "slope",
      deltaKind: "slope",
    },
    {
      label: "Dust",
      unit: "",
      nnKey: "dust",
      regKey: null,
      obsKey: "dustObserved",
      deltaKind: "dust",
    },
    {
      label: "Surface temp",
      unit: "°C",
      nnKey: "surface_temp",
      regKey: "surface_temp_xgb",
      obsKey: "temperature",
      deltaKind: "temp",
    },
    {
      label: "Thermal inertia",
      unit: "",
      nnKey: "thermal_inertia",
      regKey: "thermal_inertia_xgb",
      obsKey: "thermalInertia",
      deltaKind: "ti",
    },
    {
      label: "Water",
      unit: "%",
      nnKey: "water",
      regKey: null,
      obsKey: "grsWaterWt",
      deltaKind: "water",
    },
  ];

  const obsCol = rows
    .map((r) => {
      let show = "—";
      if (r.obsKey != null) {
        const v = raw[r.obsKey];
        show =
          v !== null && v !== undefined && !Number.isNaN(Number(v))
            ? `${formatPred(v)}${unitSuffix(r.unit)}`
            : "—";
      }
      return `
      <div class="pred-row">
        <span class="pred-label">${r.label}${r.unit ? ` (${r.unit})` : ""}</span>
        <span class="pred-value pred-value--obs">${show}</span>
      </div>`;
    })
    .join("");

  const nnCol = rows
    .map((r) => {
      const obsVal = r.obsKey != null ? raw[r.obsKey] : null;
      const { wrapClass, deltaHtml } = neuralDeltaHtml(obsVal, nn[r.nnKey], r.deltaKind);
      const val = `${formatPred(nn[r.nnKey])}${unitSuffix(r.unit)}`;
      const pick = rowScoreModel(r, overrides);
      const inScoreClass = pick === "nn" ? " pred-value--for-landing-score" : "";
      const pin = pick === "nn" ? scorePinHtml("nn") : "";
      return `
      <div class="pred-row">
        <span class="pred-label">${r.label}${r.unit ? ` (${r.unit})` : ""}</span>
        <span class="pred-value pred-value--nn${inScoreClass}${wrapClass}">${val}${deltaHtml}${pin}</span>
      </div>`;
    })
    .join("");

  const regCol = rows
    .map((r) => {
      const v = r.regKey != null ? reg[r.regKey] : null;
      const show =
        r.regKey != null ? `${formatPred(v)}${unitSuffix(r.unit)}` : "—";
      const pick = rowScoreModel(r, overrides);
      const inScoreClass =
        pick === "xgb" && r.regKey != null ? " pred-value--for-landing-score" : "";
      const pin =
        pick === "xgb" && r.regKey != null ? scorePinHtml("xgb") : "";
      return `
      <div class="pred-row">
        <span class="pred-label">${r.label}${r.unit ? ` (${r.unit})` : ""}</span>
        <span class="pred-value pred-value--xgb${inScoreClass}">${show}${pin}</span>
      </div>`;
    })
    .join("");

  return { obsCol, nnCol, regCol };
}

// Landing suitability prediction
async function predictLandingSuitability() {
  if (!currentMarsData) {
    document.getElementById("landingScore").innerHTML =
      '<div class="pred-panel pred-panel--error" role="alert">Click the globe first to load raster values at a point, then run prediction.</div>';
    return;
  }
  
  try {
    document.getElementById("landingScore").innerHTML =
      '<div class="pred-panel pred-panel--loading"><span class="pred-loading-spinner" aria-hidden="true"></span><span>Running prediction…</span></div>';
    document.getElementById("predictLanding").disabled = true;
    
    // Use relative URL so it works on both localhost and Render
    const apiUrl = window.location.origin + '/predict';
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(currentMarsData)
    });
    
    // Check if response is OK
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText.substring(0, 100)}`);
    }
    
    // Check content type
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      const text = await response.text();
      throw new Error(`Expected JSON but got ${contentType}. Response: ${text.substring(0, 200)}`);
    }
    
    const result = await response.json();
    
    if (result.success) {
      const score = result.landing_score;
      let scoreBand = "high";
      let scoreText = "Good";
      if (score < 30) {
        scoreBand = "low";
        scoreText = "Poor";
      } else if (score <= 50) {
        scoreBand = "mid";
        scoreText = "Fair";
      } else if (score >= 70) {
        scoreText = "Excellent";
      }
      
      const fused = result.predictions?.neural_networks || {};
      const nnOnly =
        result.predictions?.neural_networks_baseline || fused;
      const reg = result.predictions?.regression_models || {};
      const overrides = result.overrides_applied || {};
      const raw = result.raw_mars_data || {};
      const { obsCol, nnCol, regCol } = buildPredRows(raw, nnOnly, reg, overrides);

      const overrideParts = [];
      const st = overrides.surface_temp;
      const ti = overrides.thermal_inertia;
      if (st === "surface_temp_xgb") {
        overrideParts.push("surface temperature: XGBoost (used in landing score)");
      } else if (st === "surface_temp_nn") {
        overrideParts.push(
          "surface temperature: neural nets (used in landing score—closer to raster temperature than XGB, or XGB out of range)"
        );
      }
      if (ti === "thermal_inertia_xgb") {
        overrideParts.push("thermal inertia: XGBoost (used in landing score)");
      } else if (ti === "thermal_inertia_nn") {
        overrideParts.push(
          "thermal inertia: neural nets (used in landing score—closer to observed TI than XGB, or XGB out of range)"
        );
      }
      const foot =
        overrideParts.length > 0
          ? `Score blends models for temperature and thermal inertia: ${overrideParts.join("; ")}. Slope, dust, and water in the score come from the neural nets.`
          : "Score uses neural outputs for temperature and thermal inertia (XGB was not used for those this time, or regression was unavailable).";

      const footRaster =
        "Observed column = values from your GeoTIFFs (same as the JSON below). Dust uses the OMEGA ferric/dust raster. Slope / temp / TI / water use their listed products; neural targets are not identical, so treat Δ as a rough guide. Zeros on pyroxene or basalt may be real or no-data.";

      const rawJson =
        Object.keys(raw).length > 0
          ? JSON.stringify(raw, null, 2)
          : "{}";
      const rawSafe = rawJson
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      document.getElementById("landingScore").innerHTML = `
        <div class="pred-panel pred-panel--score-${scoreBand}">
          <div class="pred-score-card">
            <div class="pred-score">Landing suitability: ${score}% <span class="pred-score-note">(${scoreText})</span></div>
            <p class="pred-lead">Each row: rasters vs neural vs XGB. <strong>Neural · in score</strong> / <strong>XGB · in score</strong> marks which value the landing % actually uses for that property (temp &amp; TI can be either; slope, dust, water always neural). Δ colors on neural = rough gap vs raster.</p>
          </div>
          <div class="pred-legend" role="note" aria-label="Column legend">
            <span><span class="pred-legend-dot pred-legend-dot--obs" aria-hidden="true"></span> Raster sample</span>
            <span><span class="pred-legend-dot pred-legend-dot--nn" aria-hidden="true"></span> Neural</span>
            <span><span class="pred-legend-dot pred-legend-dot--xgb" aria-hidden="true"></span> XGBoost</span>
            <span><span class="pred-legend-dot pred-legend-dot--score" aria-hidden="true"></span> Amber box + tag = in landing score</span>
          </div>
          <div class="pred-grid-scroll">
            <div class="pred-grid">
              <div class="pred-col pred-col--obs">
                <div class="pred-col-title pred-col-title--obs">
                  <span class="pred-col-title-short">Raster</span>
                  <span class="pred-col-title-long">Observed (raster)</span>
                </div>
                ${obsCol}
              </div>
              <div class="pred-col pred-col--nn">
                <div class="pred-col-title pred-col-title--nn">
                  <span class="pred-col-title-short">Neural</span>
                  <span class="pred-col-title-long">Neural networks</span>
                </div>
                ${nnCol}
              </div>
              <div class="pred-col pred-col--xgb">
                <div class="pred-col-title pred-col-title--xgb">
                  <span class="pred-col-title-short">XGB</span>
                  <span class="pred-col-title-long">Regression (XGB)</span>
                </div>
                ${regCol}
              </div>
            </div>
          </div>
          <details class="pred-details">
            <summary>Scoring &amp; data notes</summary>
            <div class="pred-details-body">
              <p class="pred-footnote">${foot}</p>
              <p class="pred-footnote pred-footnote--raster">${footRaster}</p>
            </div>
          </details>
          <div class="pred-raw">
            <details>
              <summary>View request JSON (sent to the server)</summary>
              <pre>${rawSafe}</pre>
              <p class="pred-raw-hint">Scroll inside the box if the payload is long. This is exactly what <code>/predict</code> received.</p>
            </details>
          </div>
        </div>
      `;
      
      console.log("Prediction result:", result);
    } else {
      document.getElementById("landingScore").innerHTML = `<div class="pred-panel pred-panel--error">${result.error || "Request failed"}</div>`;
    }
    
  } catch (error) {
    console.error("API call failed:", error);
    document.getElementById("landingScore").innerHTML = `<div class="pred-panel pred-panel--error">Could not reach API: ${error.message}</div>`;
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
    description: "Mars Orbiter Laser Altimeter elevation data",
    marsDataKey: "elevation",
  },
  slope: {
    name: "Slope",
    file: "./public/data/mola_hrsc_blend_slope_v2.tif",
    unit: "°",
    description: "Surface slope measurements",
    marsDataKey: "slope",
  },
  roughness: {
    name: "Roughness",
    file: "./public/data/mola_roughness_0.6km_numeric.tif",
    unit: "m",
    description: "Surface roughness at 0.6km scale",
    marsDataKey: "roughness",
  },
  albedo: {
    name: "Albedo",
    file: "./public/data/omega_albedo_r1080.tif",
    unit: "",
    description: "Surface albedo (reflectivity)",
    marsDataKey: "albedo",
  },
  temperature: {
    name: "Temperature",
    file: "./public/data/mars_yearly_avg_temperature_celsius.tif",
    unit: "°C",
    description: "Yearly average surface temperature",
    marsDataKey: "temperature",
  },
  tempRange: {
    name: "Temperature Range",
    file: "./public/data/mars_yearly_temperature_range_v1.0.tif",
    unit: "°C",
    description: "Yearly temperature variation",
    marsDataKey: "tempRange",
  },
  crustalThickness: {
    name: "Crustal Thickness",
    file: "./public/data/mars_crustal_thickness_gmm3_rm1.tif",
    unit: "km",
    description: "Mars crustal thickness",
    marsDataKey: "crustalThickness",
  },
  ferric: {
    name: "Ferric / dust (OMEGA)",
    file: "./public/data/omega_ferric_nnphs.tif",
    unit: "",
    description:
      "OMEGA ferric/dust-related index (same raster is copied to dustObserved for the Observed column)",
    marsDataKey: "ferric",
  },
  pyroxene: {
    name: "Pyroxene",
    file: "./public/data/omega_pyroxene_bd2000.tif",
    unit: "",
    description: "Pyroxene mineral content",
    marsDataKey: "pyroxene",
  },
  basalt: {
    name: "Basalt",
    file: "./public/data/TES_Basalt_numeric.tif",
    unit: "",
    description: "Basalt abundance",
    marsDataKey: "basalt",
  },
  lambertAlbedo: {
    name: "Lambert Albedo",
    file: "./public/data/TES_Lambert_Albedo_numeric.tif",
    unit: "",
    description: "Lambert albedo from TES",
    marsDataKey: "lambertAlbedo",
  },
  thermalInertiaObs: {
    name: "Thermal inertia (TES dayside, Putzig 2007)",
    file: "./public/data/tes_dayside_ti_putzig_2007.tif",
    unit: "TIU",
    description: "TES dayside thermal inertia (Putzig et al. 2007); SI-style inertia units",
    marsDataKey: "thermalInertia",
  },
  grsWaterWt: {
    name: "GRS water equivalent (% wt)",
    file: "./public/data/mars_odyssey_grs_mons_perc_wt.tif",
    unit: "%",
    description: "Mars Odyssey GRS hydrogen / water-equivalent weight percent (MONS product)",
    marsDataKey: "grsWaterWt",
  },
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

    let nodata = null;
    try {
      if (typeof image.getGDALNoData === "function") {
        const nd = image.getGDALNoData();
        if (nd != null && nd !== "" && !Number.isNaN(Number(nd))) nodata = Number(nd);
      }
    } catch (_) {
      /* ignore */
    }
    if (nodata == null && image.fileDirectory?.GDAL_NODATA != null) {
      const s = String(image.fileDirectory.GDAL_NODATA).trim();
      if (s !== "" && !Number.isNaN(Number(s))) nodata = Number(s);
    }

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
      bounds,
      nodata,
    };
  } catch (error) {
    console.error("Error loading GeoTIFF:", error);
    // Fallback to mock data
    console.log("Falling back to mock data");
    return {
      width: 1024,
      height: 512,
      data: new Array(1024 * 512).fill(0).map(() => Math.random() * 1000 - 500),
      bounds: [-180, -90, 180, 90],
      nodata: null,
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
    const ce = document.getElementById("coords");
    ce.classList.remove("coords--empty");
    ce.innerText =
      "Could not load starting elevation layer. Check the browser console, confirm GeoTIFF paths, and refresh.";
  }
})();

// Convert latitude/longitude to pixel coordinates
function latLonToPixel(lat, lon, width, height) {
  const x = Math.floor(((lon + 180) / 360) * width);
  const y = Math.floor(((90 - lat) / 180) * height);
  return { x, y };
}

function isNoDataSample(raw, nodata) {
  if (raw == null) return true;
  if (typeof raw === "number" && Number.isNaN(raw)) return true;
  if (nodata == null || !Number.isFinite(Number(nodata))) return false;
  const nd = Number(nodata);
  const v = Number(raw);
  if (!Number.isFinite(v)) return true;
  const tol = 1e-6 * Math.max(1, Math.abs(nd));
  return v === nd || Math.abs(v - nd) <= tol;
}

/** Sample a loaded raster at lat/lon; returns null for out of bounds or GDAL no-data. */
function sampleDatasetAt(dataset, lat, lon) {
  if (!dataset) return null;
  const { width, height, data, nodata } = dataset;
  const { x, y } = latLonToPixel(lat, lon, width, height);

  if (x < 0 || x >= width || y < 0 || y >= height) return null;

  const index = y * width + x;
  if (index >= data.length) return null;

  const raw = data[index];
  if (isNoDataSample(raw, nodata)) return null;
  const v = Number(raw);
  return Number.isFinite(v) ? v : null;
}

function getValueAt(lat, lon) {
  return sampleDatasetAt(currentDataset, lat, lon);
}

// Get value from a specific dataset at given coordinates
function getValueFromDataset(datasetType, lat, lon) {
  const dataset = loadedDatasets.get(datasetType);
  return sampleDatasetAt(dataset, lat, lon);
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
    const coordsEl = document.getElementById("coords");
    coordsEl.classList.remove("coords--empty");
    coordsEl.innerText = `Loading layers…\nLat ${lat.toFixed(2)}°, Lon ${lon.toFixed(2)}°`;

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
          description: datasetInfo.description,
          marsDataKey: datasetInfo.marsDataKey,
        });
      } catch (error) {
        console.warn(`Failed to load ${datasetInfo.name}:`, error);
        allValues.push({
          name: datasetInfo.name,
          value: null,
          unit: datasetInfo.unit,
          description: datasetInfo.description,
          marsDataKey: datasetInfo.marsDataKey,
        });
      }
    }

    // Format all values into a list and store data for API (keys from marsDataKey on each layer)
    let valuesList = `Lat: ${lat.toFixed(2)}°, Lon: ${lon.toFixed(2)}°\n\n`;
    currentMarsData = { lat, lon };
    for (const d of Object.values(marsDatasets)) {
      if (d.marsDataKey) currentMarsData[d.marsDataKey] = null;
    }

    allValues.forEach((item) => {
      const valueStr =
        item.value !== null && item.value !== undefined && !Number.isNaN(Number(item.value))
          ? `${Number(item.value).toFixed(2)} ${item.unit}`
          : "N/A";
      valuesList += `• ${item.name}: ${valueStr}\n`;
      if (item.marsDataKey) {
        currentMarsData[item.marsDataKey] = item.value;
      }
    });

    currentMarsData.dustObserved = currentMarsData.ferric;

    // Update UI
    coordsEl.innerText = valuesList;

    // Show prediction button
    document.getElementById("predictLanding").style.display = "block";
    const hint = document.getElementById("predictHint");
    if (hint) hint.style.display = "block";
    document.getElementById("landingScore").innerText = "";

    // Show marker
    marker.position.copy(point);
    marker.visible = true;
  }
}

window.addEventListener("click", onMouseClick, false);

