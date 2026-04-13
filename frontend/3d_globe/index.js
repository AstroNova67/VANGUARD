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
const controlsPanel = document.getElementById("controls");
if (controlsPanel) {
  controlsPanel.insertAdjacentElement("beforebegin", renderer.domElement);
} else {
  document.body.appendChild(renderer.domElement);
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// --- ML API (Flask /predict + /health; models run on server, not in browser) ---
/** True once GET /health reports models_loaded (Keras + pipeline ready). */
let vanguardBackendModelsReady = false;

/**
 * Base URL for `/predict` and `/health`. Same origin when Flask serves this page.
 * If you use live-server, set `<meta name="vanguard-api-base" content="http://127.0.0.1:5002" />`.
 */
function getVanguardApiBase() {
  const meta = document.querySelector('meta[name="vanguard-api-base"]');
  const fromMeta = meta?.getAttribute("content")?.trim();
  if (fromMeta) {
    return fromMeta.replace(/\/$/, "");
  }
  if (window.location.protocol === "file:") {
    return "";
  }
  return window.location.origin.replace(/\/$/, "");
}

function setApiModelsStatusMessage(text) {
  const el = document.getElementById("apiModelsStatus");
  if (el) el.textContent = text;
}

function syncPredictLandingButtonWithBackend() {
  const btn = document.getElementById("predictLanding");
  if (!btn || btn.style.display === "none") return;
  btn.disabled = !vanguardBackendModelsReady;
  btn.title = vanguardBackendModelsReady
    ? "Run POST /predict (Keras + XGB on server)"
    : "Wait until the ML backend is ready (see status above).";
}

/**
 * Poll /health until models are loaded or timeout (TensorFlow startup can take minutes).
 */
async function waitForVanguardMlBackend() {
  const base = getVanguardApiBase();
  if (!base) {
    setApiModelsStatusMessage(
      "Open this app from the Flask server (e.g. http://127.0.0.1:5002) or set meta vanguard-api-base to your API URL so /predict can load Keras + XGB."
    );
    return;
  }

  const timeoutMs = 180000;
  const intervalMs = 2000;
  const t0 = Date.now();

  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(`${base}/health`);
      if (r.ok) {
        const j = await r.json();
        if (j.models_loaded === true) {
          const nn = Number(j.neural_models_loaded ?? 0);
          const reg = Number(j.regression_models_loaded ?? 0);
          vanguardBackendModelsReady = true;
          setApiModelsStatusMessage(
            `ML backend ready — ${nn}/5 neural nets loaded` +
              (reg > 0 ? `; XGB regression loaded for fusion.` : ` (XGB optional).`) +
              ` Predict uses POST /predict.`
          );
          syncPredictLandingButtonWithBackend();
          return;
        }
      }
    } catch {
      /* network or CORS */
    }
    setApiModelsStatusMessage(
      "Waiting for ML backend (loading Keras models on server)… " +
        `Trying ${base}/health every ${intervalMs / 1000}s.`
    );
    await new Promise((res) => setTimeout(res, intervalMs));
  }

  setApiModelsStatusMessage(
    `ML backend did not report ready within ${timeoutMs / 60000} min. Check Flask logs, then refresh. Base: ${base}`
  );
}

void waitForVanguardMlBackend();

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

/** Absolute + relative tolerance match for float score-source inference. */
function nearlyEqualScore(a, b, absTol, relTol = 1e-9) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  return Math.abs(a - b) <= absTol + relTol * Math.max(Math.abs(a), Math.abs(b));
}

/** Unpack /predict JSON with snake_case or camelCase keys; reject array-shaped baselines. */
function unpackPredictPayload(result) {
  const pred = result.predictions ?? result.Predictions;
  const fused = pred?.neural_networks ?? pred?.neuralNetworks ?? {};
  let nnBaseline =
    pred?.neural_networks_baseline ?? pred?.neuralNetworksBaseline ?? null;
  if (nnBaseline != null && (typeof nnBaseline !== "object" || Array.isArray(nnBaseline))) {
    nnBaseline = null;
  }
  const nnOnly = nnBaseline != null ? nnBaseline : {};
  const reg = pred?.regression_models ?? pred?.regressionModels ?? {};
  const dataSources = result.data_sources ?? result.dataSources ?? null;
  const overrides = result.overrides_applied ?? result.overridesApplied ?? {};
  return { fused, nnBaseline, nnOnly, reg, dataSources, overrides };
}

/** @param {'slope'|'temp'|'ti'|'water'|'dust'|null} kind */
function observedDeltaHtml(observed, pred, kind) {
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
    deltaHtml: ` <span class="pred-delta pred-delta--${level}" title="|model − raster| at this pixel (illustrative)">Δ${formatPred(d)}</span>`,
  };
}

/**
 * Which column gets the landing-score pin: Neural or XGB only (never observed raster).
 * Uses `data_sources` + `overrides_applied` when present; otherwise infers from fused vs NN/XGB.
 */
function rowScoreModel(row, overrides, dataSources, raw, fused, nn, reg) {
  const o = overrides || {};
  const nnRow = nn && typeof nn === "object" ? nn : {};
  const regRow = reg && typeof reg === "object" ? reg : {};

  const pickTempOrTi = (overrideKey, regKey) => {
    const v =
      overrideKey === "surface_temp"
        ? o.surface_temp ?? o.surfaceTemp
        : o.thermal_inertia ?? o.thermalInertia;
    if (v && String(v).includes("xgb")) return "xgb";
    if (v && String(v).includes("nn")) return "nn";
    const fv = Number(fused[row.nnKey]);
    const xgbv = regKey != null ? Number(regRow[regKey]) : NaN;
    const nnv = Number(nnRow[row.nnKey]);
    const absTol = row.deltaKind === "temp" ? 0.25 : row.deltaKind === "ti" ? 4 : 1e-6;
    if (
      regKey != null &&
      Number.isFinite(fv) &&
      Number.isFinite(xgbv) &&
      nearlyEqualScore(fv, xgbv, absTol) &&
      !(Number.isFinite(nnv) && nearlyEqualScore(fv, nnv, absTol))
    ) {
      return "xgb";
    }
    return "nn";
  };

  if (dataSources && typeof dataSources === "object" && Object.prototype.hasOwnProperty.call(dataSources, row.nnKey)) {
    const ds = dataSources[row.nnKey];
    if (ds === "raster") {
      if (row.nnKey === "surface_temp") return pickTempOrTi("surface_temp", "surface_temp_xgb");
      if (row.nnKey === "thermal_inertia") return pickTempOrTi("thermal_inertia", "thermal_inertia_xgb");
      return "nn";
    }
    if (ds === "ml_predicted") {
      if (row.nnKey === "surface_temp") {
        const v = o.surface_temp ?? o.surfaceTemp;
        if (v && String(v).includes("xgb")) return "xgb";
        return "nn";
      }
      if (row.nnKey === "thermal_inertia") {
        const v = o.thermal_inertia ?? o.thermalInertia;
        if (v && String(v).includes("xgb")) return "xgb";
        return "nn";
      }
      return "nn";
    }
  }

  const obsK = row.obsKey;
  const rawObs = obsK != null ? raw[obsK] : null;
  const obsNum = Number(rawObs);
  const fusedNum = Number(fused[row.nnKey]);
  const absTol =
    row.deltaKind === "temp"
      ? 0.12
      : row.deltaKind === "ti"
      ? 3
      : row.deltaKind === "slope"
      ? 0.05
      : row.deltaKind === "dust"
      ? 0.04
      : row.deltaKind === "water"
      ? 0.15
      : 1e-6;

  if (Number.isFinite(obsNum) && Number.isFinite(fusedNum) && nearlyEqualScore(obsNum, fusedNum, absTol)) {
    if (row.nnKey === "surface_temp") return pickTempOrTi("surface_temp", "surface_temp_xgb");
    if (row.nnKey === "thermal_inertia") return pickTempOrTi("thermal_inertia", "thermal_inertia_xgb");
    return "nn";
  }

  if (row.nnKey === "surface_temp") return pickTempOrTi("surface_temp", "surface_temp_xgb");
  if (row.nnKey === "thermal_inertia") return pickTempOrTi("thermal_inertia", "thermal_inertia_xgb");
  if (row.nnKey === "slope" || row.nnKey === "dust" || row.nnKey === "water") return "nn";
  return "nn";
}

function scorePinHtml(which) {
  if (which === "nn") {
    return ` <span class="pred-score-pin" title="This number is what the landing score uses for this row">Neural · in score</span>`;
  }
  if (which === "xgb") {
    return ` <span class="pred-score-pin pred-score-pin--xgb" title="This number is what the landing score uses for this row">XGB · in score</span>`;
  }
  if (which === "raster") {
    return ` <span class="pred-score-pin pred-score-pin--raster" title="Landing score uses the observed raster value for this property">Raster · in score</span>`;
  }
  return "";
}

function buildPredRows(raw, nn, reg, overrides, dataSources, fused) {
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
      const pick = rowScoreModel(r, overrides, dataSources, raw, fused, nn, reg);
      const inObs = "";
      const pinObs = "";
      return `
      <div class="pred-row">
        <span class="pred-label">${r.label}${r.unit ? ` (${r.unit})` : ""}</span>
        <span class="pred-value pred-value--obs${inObs}">${show}${pinObs}</span>
      </div>`;
    })
    .join("");

  const nnCol = rows
    .map((r) => {
      const obsVal = r.obsKey != null ? raw[r.obsKey] : null;
      const { wrapClass, deltaHtml } = observedDeltaHtml(obsVal, nn[r.nnKey], r.deltaKind);
      const val = `${formatPred(nn[r.nnKey])}${unitSuffix(r.unit)}`;
      const pick = rowScoreModel(r, overrides, dataSources, raw, fused, nn, reg);
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
      const obsVal = r.obsKey != null ? raw[r.obsKey] : null;
      const { wrapClass, deltaHtml } =
        r.regKey != null && r.deltaKind != null
          ? observedDeltaHtml(obsVal, v, r.deltaKind)
          : { wrapClass: "", deltaHtml: "" };
      const pick = rowScoreModel(r, overrides, dataSources, raw, fused, nn, reg);
      const inScoreClass =
        pick === "xgb" && r.regKey != null ? " pred-value--for-landing-score" : "";
      const pin =
        pick === "xgb" && r.regKey != null ? scorePinHtml("xgb") : "";
      return `
      <div class="pred-row">
        <span class="pred-label">${r.label}${r.unit ? ` (${r.unit})` : ""}</span>
        <span class="pred-value pred-value--xgb${inScoreClass}${wrapClass}">${show}${deltaHtml}${pin}</span>
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
    
    const base = getVanguardApiBase();
    if (!base) {
      document.getElementById("landingScore").innerHTML =
        '<div class="pred-panel pred-panel--error" role="alert">No API base URL. Serve this page from Flask or set <code>&lt;meta name="vanguard-api-base" content="http://127.0.0.1:5002" /&gt;</code> so the browser can reach <code>/predict</code> where Keras loads.</div>';
      return;
    }

    const apiUrl = `${base}/predict`;
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(currentMarsData)
    });
    
    if (response.status === 503) {
      let msg =
        "ML models are still loading on the server (503). Wait for the green status under Landing prediction, then try again.";
      try {
        const j = await response.json();
        if (j?.error) msg = `${msg} (${j.error})`;
      } catch {
        /* ignore */
      }
      throw new Error(msg);
    }

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
      
      const { fused, nnBaseline, nnOnly, reg, dataSources, overrides } = unpackPredictPayload(result);
      if (nnBaseline == null) {
        console.warn(
          "API response missing predictions.neural_networks_baseline; neural column cannot show pure Keras outputs."
        );
      }
      const raw = result.raw_mars_data ?? result.rawMarsData ?? {};
      const { obsCol, nnCol, regCol } = buildPredRows(raw, nnOnly, reg, overrides, dataSources, fused);

      const ds = dataSources || {};
      const rasterProps = Object.entries(ds)
        .filter(([, v]) => v === "raster")
        .map(([k]) => k.replace(/_/g, " "));
      const mlProps = Object.entries(ds)
        .filter(([, v]) => v === "ml_predicted")
        .map(([k]) => k.replace(/_/g, " "));

      const overrideParts = [];
      const st = overrides.surface_temp ?? overrides.surfaceTemp;
      const ti = overrides.thermal_inertia ?? overrides.thermalInertia;
      if (ds.surface_temp === "ml_predicted") {
        if (st === "surface_temp_xgb") {
          overrideParts.push("surface temperature in score: XGBoost (fusion)");
        } else if (st === "surface_temp_nn") {
          overrideParts.push("surface temperature in score: neural nets (fusion)");
        }
      }
      if (ds.thermal_inertia === "ml_predicted") {
        if (ti === "thermal_inertia_xgb") {
          overrideParts.push("thermal inertia in score: XGBoost (fusion)");
        } else if (ti === "thermal_inertia_nn") {
          overrideParts.push("thermal inertia in score: neural nets (fusion)");
        }
      }
      const footRasterFirst =
        rasterProps.length > 0
          ? `Some properties were marked raster-sourced by the API: ${rasterProps.join(", ")} (pins still target Neural / XGB columns).`
          : "Landing suitability uses Keras for slope, dust, and water; surface temperature and thermal inertia use NN vs XGB fusion. Observed rasters inform fusion where applicable.";
      const footMl = mlProps.length > 0 ? ` ML-driven inputs: ${mlProps.join(", ")}.` : "";
      const foot =
        overrideParts.length > 0
          ? `${footRasterFirst}${footMl} Where temperature or thermal inertia used ML, fusion detail: ${overrideParts.join("; ")}.`
          : `${footRasterFirst}${footMl}`;

      const footRaster =
        "Observed column = GeoTIFF samples at the click (same as the JSON below), for reference. <strong>Neural · in score</strong> / <strong>XGB · in score</strong> always marks what the landing % used. Δ compares each model column to the observed value. Zeros on pyroxene or basalt may be real or no-data.";

      const rawJson =
        Object.keys(raw).length > 0
          ? JSON.stringify(raw, null, 2)
          : "{}";
      const rawSafe = rawJson
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      const baselineWarn =
        nnBaseline == null
          ? `<p class="pred-api-warn" role="status">The API response is missing <code>predictions.neural_networks_baseline</code>, so the Neural column cannot list Keras-only outputs. Confirm you are on the current Flask backend.</p>`
          : "";

      document.getElementById("landingScore").innerHTML = `
        <div class="pred-panel pred-panel--score-${scoreBand}">
          <div class="pred-score-card">
            <div class="pred-score">Landing suitability: ${score}% <span class="pred-score-note">(${scoreText})</span></div>
            <p class="pred-lead">Each row shows observed rasters, <strong>Neural</strong> (Keras), and <strong>XGB</strong> (temp &amp; TI). The landing % uses the ML columns only: <strong>Neural · in score</strong> or <strong>XGB · in score</strong> marks which value was used (slope, dust, and water always from Keras here). <strong>Δ</strong> = |model − raster| for comparison.</p>
          </div>
          ${baselineWarn}
          <div class="pred-legend" role="note" aria-label="Column legend">
            <span><span class="pred-legend-dot pred-legend-dot--obs" aria-hidden="true"></span> Raster sample</span>
            <span><span class="pred-legend-dot pred-legend-dot--nn" aria-hidden="true"></span> Neural</span>
            <span><span class="pred-legend-dot pred-legend-dot--xgb" aria-hidden="true"></span> XGBoost</span>
            <span><span class="pred-legend-dot pred-legend-dot--score" aria-hidden="true"></span> Amber box + tag = value used in landing %</span>
          </div>
          <p class="pred-grid-scroll-hint" role="note">
            <strong>Three columns</strong> (Raster · Neural · XGB). If Neural/XGB look missing, <strong>scroll this table horizontally</strong> — the grid is wider than the sidebar.
          </p>
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

document.getElementById("predictLanding")?.addEventListener("click", predictLandingSuitability);

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

/** Cached blended globe textures per `marsDatasets` key (not including suitability). */
const globeLayerTextureCache = new Map();
/** Last non-photo value of `#globeSurfaceLayer` before suitability overlay locks the menu. */
let priorGlobeSurfaceSelect = "photo";

/**
 * @param {string} url
 * @param {{ allowFallback?: boolean }} [options] If allowFallback is false, failures throw (used for landing overlay).
 */
async function loadGeoTIFF(url, options = {}) {
  const { allowFallback = true } = options;
  try {
    const response = await fetch(url, {
      // Large rasters: avoid long-lived disk cache when the file on disk changes but the path does not.
      cache: allowFallback ? "default" : "no-cache",
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    
    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const data = await image.readRasters({ interleave: true });
    let samplesPerPixel = 1;
    try {
      if (typeof image.getSamplesPerPixel === "function") {
        const n = Number(image.getSamplesPerPixel());
        if (Number.isFinite(n) && n >= 1) samplesPerPixel = Math.floor(n);
      }
    } catch (_) {
      /* ignore */
    }

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
      samplesPerPixel,
    };
  } catch (error) {
    console.error("Error loading GeoTIFF:", error);
    if (!allowFallback) {
      throw error;
    }
    console.warn("GeoTIFF load failed; using mock raster data for this layer.");
    return {
      width: 1024,
      height: 512,
      data: new Array(1024 * 512).fill(0).map(() => Math.random() * 1000 - 500),
      bounds: [-180, -90, 180, 90],
      nodata: null,
      samplesPerPixel: 1,
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

function populateGlobeSurfaceSelect() {
  const sel = document.getElementById("globeSurfaceLayer");
  if (!sel) return;
  for (const [key, meta] of Object.entries(marsDatasets)) {
    if ([...sel.options].some((o) => o.value === key)) continue;
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = meta.name;
    sel.appendChild(opt);
  }
}
populateGlobeSurfaceSelect();

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
    currentDataset = await loadDataset("elevation");
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

/** Path to ML suitability GeoTIFF (query string for cache bust is added in {@link landingMlOverlayRequestUrl}). */
const LANDING_ML_OVERLAY_PATH = "./public/data/mars_landing_suitability_ml.tif";
const LANDING_ML_OVERLAY_NODATA_FALLBACK = -9999;
/**
 * Bump when overlay colors change, the TIFF is regenerated, or you need browsers to refetch/rebuild the overlay.
 * Appended as `?cb=` on the request URL and compared to invalidate in-memory blended textures.
 */
const LANDING_SUITABILITY_TINT_REVISION = 13;

function landingMlOverlayRequestUrl() {
  const u = new URL(LANDING_ML_OVERLAY_PATH, window.location.href);
  u.searchParams.set("cb", String(LANDING_SUITABILITY_TINT_REVISION));
  return u.href;
}

let landingMlSuitabilityBlendedTexture = null;
let landingMlSuitabilityTintRevisionBuilt = 0;
/** URL that built {@link landingMlSuitabilityBlendedTexture} (includes `cb` cache-bust; see {@link landingMlOverlayRequestUrl}). */
let landingMlSuitabilityLoadedFromUrl = null;
let landingMlOverlayLoading = false;
/** True while the ML suitability overlay is on (locks globe surface drape). */
let landingMlOverlaySessionActive = false;

function whenMarsBaseTextureReady() {
  return new Promise((resolve) => {
    const img = marsTexture.image;
    if (img && img.complete && img.naturalWidth > 0) {
      resolve();
      return;
    }
    marsTexture.addEventListener("load", () => resolve(), { once: true });
  });
}

/**
 * Piecewise RGB stops: t = suitability fraction 0 (poor) → 1 (excellent).
 * Matches the GeoTIFF (0–100, higher = better): cool blue at low %, warm red at high %
 * (heatmap-style, consistent with typical “hot = high value” maps).
 */
const LANDING_SUITABILITY_COLOR_STOPS = [
  { t: 0, rgb: [24, 52, 138] },
  { t: 0.22, rgb: [38, 118, 205] },
  { t: 0.45, rgb: [72, 188, 178] },
  { t: 0.62, rgb: [255, 208, 68] },
  { t: 0.82, rgb: [255, 118, 42] },
  { t: 1, rgb: [188, 26, 48] },
];

/** Map global suitability 0–1 (0 = poor … 1 = excellent) to RGB; used only for suitability overlay. */
function landingScoreToRgb(score01) {
  const t = Math.max(0, Math.min(1, score01));
  const s = LANDING_SUITABILITY_COLOR_STOPS;
  if (t <= s[0].t) return [...s[0].rgb];
  if (t >= s[s.length - 1].t) return [...s[s.length - 1].rgb];
  for (let i = 0; i < s.length - 1; i++) {
    if (t <= s[i + 1].t) {
      const t0 = s[i].t;
      const t1 = s[i + 1].t;
      const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
      const [r0, g0, b0] = s[i].rgb;
      const [r1, g1, b1] = s[i + 1].rgb;
      return [
        Math.round(r0 + (r1 - r0) * u),
        Math.round(g0 + (g1 - g0) * u),
        Math.round(b0 + (b1 - b0) * u),
      ];
    }
  }
  return [...s[s.length - 1].rgb];
}

/** HSL blue→red ramp for non-suitability draped layers (legend + globe tint). */
function genericLayerNormToRgb(x) {
  const t = Math.max(0, Math.min(1, x));
  const h = (1 - t) * (240 / 360);
  const s = 0.78;
  const l = 0.42 + t * 0.22;
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hue2rgb = (p0, q0, t0) => {
    let tt = t0;
    if (tt < 0) tt += 1;
    if (tt > 1) tt -= 1;
    if (tt < 1 / 6) return p0 + (q0 - p0) * 6 * tt;
    if (tt < 1 / 2) return q0;
    if (tt < 2 / 3) return p0 + (q0 - p0) * (2 / 3 - tt) * 6;
    return p0;
  };
  const r = hue2rgb(p, q, h + 1 / 3);
  const g = hue2rgb(p, q, h);
  const b = hue2rgb(p, q, h - 1 / 3);
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/** CSS `linear-gradient` for the sidebar tint key (must match `landingScoreToRgb`). */
function landingSuitabilityLegendGradientCss() {
  const parts = LANDING_SUITABILITY_COLOR_STOPS.map(
    ({ t, rgb }) => `rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${(t * 100).toFixed(1)}%`
  );
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

function unwrapRasterData(dataset) {
  let d = dataset.data;
  if (Array.isArray(d) && d[0] && ArrayBuffer.isView(d[0])) {
    d = d[0];
  }
  return d;
}

/**
 * Flat buffer + samples-per-pixel for interleaved GeoTIFF rasters (used by suitability overlay + band stats).
 * @returns {{ sdata: ArrayBufferView; samplesPerPixel: number; w: number; h: number }}
 */
function resolveRasterFlatLayout(dataset) {
  const w = dataset.width;
  const h = dataset.height;
  let sdata = dataset.data;
  if (Array.isArray(sdata) && sdata[0] && ArrayBuffer.isView(sdata[0])) {
    sdata = sdata[0];
  }
  let samplesPerPixel = dataset.samplesPerPixel ?? 1;
  if (
    ArrayBuffer.isView(sdata) &&
    sdata.length >= w * h &&
    sdata.length % (w * h) === 0
  ) {
    const inferred = Math.floor(sdata.length / (w * h));
    if (Number.isFinite(inferred) && inferred >= 1) samplesPerPixel = inferred;
  }
  return { sdata, samplesPerPixel, w, h };
}

/** Strided min/max of band 0 (or the only band) for interleaved multi-band buffers. */
function estimateRasterBand0MinMax(dataset, stride = 41) {
  const { sdata, samplesPerPixel, w, h } = resolveRasterFlatLayout(dataset);
  const nodata = dataset.nodata;
  const spp = samplesPerPixel;
  if (!ArrayBuffer.isView(sdata) || sdata.length < w * h * spp) {
    return { minV: 0, maxV: 1 };
  }
  let minV = Infinity;
  let maxV = -Infinity;
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const raw = spp <= 1 ? sdata[y * w + x] : sdata[(y * w + x) * spp];
      if (isNoDataSample(raw, nodata)) continue;
      const v = Number(raw);
      if (!Number.isFinite(v)) continue;
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  }
  if (!Number.isFinite(minV)) {
    minV = 0;
    maxV = 1;
  } else if (minV === maxV) {
    maxV = minV + 1e-6;
  }
  return { minV, maxV };
}

function estimateRasterMinMax(dataset, stride = 41) {
  const w = dataset.width;
  const h = dataset.height;
  const data = unwrapRasterData(dataset);
  const nodata = dataset.nodata;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const raw = data[y * w + x];
      if (isNoDataSample(raw, nodata)) continue;
      const v = Number(raw);
      if (!Number.isFinite(v)) continue;
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  }
  if (!Number.isFinite(minV)) {
    minV = 0;
    maxV = 1;
  } else if (minV === maxV) {
    maxV = minV + 1e-6;
  }
  return { minV, maxV };
}

/** Normalized value t in [0,1] → RGB for globe tint (must match legend gradient). */
function layerNormToRgb(layerKey, t) {
  const x = Math.max(0, Math.min(1, t));
  if (layerKey === "elevation" || layerKey === "roughness") {
    return [
      Math.round(26 + 215 * x),
      Math.round(20 + 175 * x),
      Math.round(14 + 125 * x),
    ];
  }
  if (layerKey === "crustalThickness") {
    return [
      Math.round(32 + 140 * x),
      Math.round(24 + 110 * x),
      Math.round(70 + 150 * x),
    ];
  }
  return genericLayerNormToRgb(x);
}

function rgbToCss(rgb) {
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

function getLegendGradientCss(layerKey) {
  const a = layerNormToRgb(layerKey, 0);
  const m = layerNormToRgb(layerKey, 0.5);
  const b = layerNormToRgb(layerKey, 1);
  return `linear-gradient(90deg, ${rgbToCss(a)} 0%, ${rgbToCss(m)} 50%, ${rgbToCss(b)} 100%)`;
}

function setGlobeRasterLegendVisible(show, layerKey, minV, maxV) {
  const leg = document.getElementById("globeRasterLegend");
  if (!leg) return;
  if (!show || !layerKey) {
    leg.classList.remove("is-visible");
    leg.setAttribute("aria-hidden", "true");
    leg.innerHTML = "";
    return;
  }
  const info = marsDatasets[layerKey];
  const suf = info.unit ? ` ${info.unit}` : "";
  const fmt = (v) => (Number.isFinite(v) ? Number(v).toFixed(2) : "—") + suf;
  const gradient = getLegendGradientCss(layerKey);
  leg.innerHTML = `
    <p class="landing-overlay-legend__title">${info.name}</p>
    <p class="globe-raster-legend__keyline"><code>${info.marsDataKey}</code> — ${info.description}</p>
    <div class="landing-overlay-legend__bar" role="img" aria-label="Color scale for ${info.name}" style="background:${gradient}"></div>
    <div class="landing-overlay-legend__ticks">
      <span>${fmt(minV)}<br /><span class="landing-overlay-legend__sub">low</span></span>
      <span style="text-align:center">${fmt(minV + (maxV - minV) * 0.5)}<br /><span class="landing-overlay-legend__sub">mid</span></span>
      <span>${fmt(maxV)}<br /><span class="landing-overlay-legend__sub">high</span></span>
    </div>
    <p class="landing-overlay-legend__note">
      Scale uses min/max from a strided sample of the raster (not full precision). No-data pixels keep the base photo.
      Tint is blended ~55% with <code>textures/mars_8k.jpg</code>.
    </p>
  `;
  leg.classList.add("is-visible");
  leg.setAttribute("aria-hidden", "false");
}

async function buildGlobeRasterTexture(dataset, layerKey, maxDim) {
  const sw = dataset.width;
  const sh = dataset.height;
  const sdata = unwrapRasterData(dataset);
  if (!ArrayBuffer.isView(sdata) || sdata.length < sw * sh) {
    throw new Error(`Unexpected raster layout for ${layerKey}`);
  }
  const nodata = dataset.nodata;
  const { minV, maxV } = estimateRasterMinMax(dataset, 41);
  const span = maxV - minV || 1;

  const scale = maxDim / Math.max(sw, sh);
  const tw = Math.max(2, Math.floor(sw * scale));
  const th = Math.max(2, Math.floor(sh * scale));

  await whenMarsBaseTextureReady();
  const canvas = document.createElement("canvas");
  canvas.width = tw;
  canvas.height = th;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(marsTexture.image, 0, 0, tw, th);
  const img = ctx.getImageData(0, 0, tw, th);
  const px = img.data;
  const blend = 0.55;

  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      const su = ((tx + 0.5) / tw) * sw;
      const sv = ((ty + 0.5) / th) * sh;
      const raw = sampleBilinearScalar(sdata, sw, sh, su, sv);
      const idx = (ty * tw + tx) * 4;
      if (raw == null || !Number.isFinite(raw) || isNoDataSample(raw, nodata)) continue;
      const norm = (Number(raw) - minV) / span;
      const [cr, cg, cb] = layerNormToRgb(layerKey, norm);
      px[idx] = Math.round(px[idx] * (1 - blend) + cr * blend);
      px[idx + 1] = Math.round(px[idx + 1] * (1 - blend) + cg * blend);
      px[idx + 2] = Math.round(px[idx + 2] * (1 - blend) + cb * blend);
    }
  }
  ctx.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  if (THREE.SRGBColorSpace != null) {
    tex.colorSpace = THREE.SRGBColorSpace;
  }
  tex.needsUpdate = true;
  return { texture: tex, minV, maxV };
}

async function applyGlobeSurfaceLayerSelect() {
  if (getLandingMlOverlayMode() !== "off") return;

  const sel = document.getElementById("globeSurfaceLayer");
  const statusEl = document.getElementById("globeSurfaceStatus");
  const val = sel?.value ?? "photo";

  if (val !== "photo") {
    priorGlobeSurfaceSelect = val;
  }

  if (val === "photo") {
    marsMaterial.map = marsTexture;
    marsMaterial.needsUpdate = true;
    if (statusEl) statusEl.textContent = "";
    setGlobeRasterLegendVisible(false);
    return;
  }

  if (statusEl) statusEl.textContent = "Building globe texture…";
  try {
    const ds = await loadDataset(val);
    if (!globeLayerTextureCache.has(val)) {
      const maxDim = Math.min(4096, renderer.capabilities.maxTextureSize);
      const built = await buildGlobeRasterTexture(ds, val, maxDim);
      globeLayerTextureCache.set(val, built);
    }
    const { texture, minV, maxV } = globeLayerTextureCache.get(val);
    marsMaterial.map = texture;
    marsMaterial.needsUpdate = true;
    setGlobeRasterLegendVisible(true, val, minV, maxV);
    if (statusEl) statusEl.textContent = marsDatasets[val]?.name ?? "";
  } catch (err) {
    console.error("Globe surface layer:", err);
    if (sel) sel.value = "photo";
    marsMaterial.map = marsTexture;
    marsMaterial.needsUpdate = true;
    if (statusEl) statusEl.textContent = "Could not load this layer (see console).";
    setGlobeRasterLegendVisible(false);
  }
}

function sampleBilinearScalar(data, w, h, fx, fy) {
  if (fx < 0 || fy < 0 || fx > w - 1 || fy > h - 1) return null;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const at = (x, y) => Number(data[y * w + x]);
  const v00 = at(x0, y0);
  const v10 = at(x1, y0);
  const v01 = at(x0, y1);
  const v11 = at(x1, y1);
  if (![v00, v10, v01, v11].every((v) => Number.isFinite(v))) return null;
  return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11;
}

/** Bilinear sample of one band in an interleaved multi-sample raster (pixel order: … bands …). */
function sampleBilinearScalarInterleaved(data, w, h, fx, fy, samplesPerPixel, band) {
  const spp = samplesPerPixel;
  if (spp < 1 || band < 0 || band >= spp) return null;
  if (fx < 0 || fy < 0 || fx > w - 1 || fy > h - 1) return null;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const at = (x, y) => Number(data[(y * w + x) * spp + band]);
  const v00 = at(x0, y0);
  const v10 = at(x1, y0);
  const v01 = at(x0, y1);
  const v11 = at(x1, y1);
  if (![v00, v10, v01, v11].every((v) => Number.isFinite(v))) return null;
  return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11;
}

/**
 * Build a blended equirectangular texture: base mars_8k underlay + suitability-style colormap.
 * @param {{ colormap?: "absolutePercent" | "minMaxScore"; onProgress?: (t01: number) => void }} [options]
 *   `absolutePercent` — values treated as 0–100% (ML suitability GeoTIFF).
 *   `minMaxScore` — band 0 linearly stretched to the color ramp (non-percent input layers).
 *   `onProgress` — optional 0→1 callback while tinting (yields between row chunks so the tab stays responsive).
 * @returns {Promise<{ texture: import("three").CanvasTexture; scoreScale: "absolutePercent" | "minMaxScore"; minV?: number; maxV?: number }>}
 */
async function buildBlendedLandingSuitabilityTexture(dataset, maxDim, options = {}) {
  const colormap = options.colormap ?? "absolutePercent";
  const { sdata, samplesPerPixel, w: sw, h: sh } = resolveRasterFlatLayout(dataset);
  if (!ArrayBuffer.isView(sdata) || sdata.length < sw * sh * samplesPerPixel) {
    throw new Error(
      `Unexpected suitability raster layout: expected ~${sw * sh * samplesPerPixel} samples, got ${sdata?.length ?? "none"}`
    );
  }
  let minV = 0;
  let maxV = 100;
  let span = 100;
  if (colormap === "minMaxScore") {
    const mm = estimateRasterBand0MinMax(dataset, 41);
    minV = mm.minV;
    maxV = mm.maxV;
    span = maxV - minV || 1e-6;
  }
  const nodata =
    dataset.nodata != null && Number.isFinite(Number(dataset.nodata))
      ? Number(dataset.nodata)
      : LANDING_ML_OVERLAY_NODATA_FALLBACK;

  const scale = maxDim / Math.max(sw, sh);
  const tw = Math.max(2, Math.floor(sw * scale));
  const th = Math.max(2, Math.floor(sh * scale));

  await whenMarsBaseTextureReady();
  const canvas = document.createElement("canvas");
  canvas.width = tw;
  canvas.height = th;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const baseImg = marsTexture.image;
  ctx.drawImage(baseImg, 0, 0, tw, th);
  const img = ctx.getImageData(0, 0, tw, th);
  const px = img.data;
  const blend = 0.55;
  const onProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  const rowChunk = 64;

  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      const su = ((tx + 0.5) / tw) * sw;
      const sv = ((ty + 0.5) / th) * sh;
      const raw =
        samplesPerPixel <= 1
          ? sampleBilinearScalar(sdata, sw, sh, su, sv)
          : sampleBilinearScalarInterleaved(sdata, sw, sh, su, sv, samplesPerPixel, 0);
      const idx = (ty * tw + tx) * 4;
      if (raw == null || !Number.isFinite(raw) || isNoDataSample(raw, nodata)) continue;
      let t01;
      if (colormap === "minMaxScore") {
        t01 = (Number(raw) - minV) / span;
        t01 = Math.max(0, Math.min(1, t01));
      } else {
        const score = Math.max(0, Math.min(100, Number(raw)));
        t01 = score / 100;
      }
      const [cr, cg, cb] = landingScoreToRgb(t01);
      px[idx] = Math.round(px[idx] * (1 - blend) + cr * blend);
      px[idx + 1] = Math.round(px[idx + 1] * (1 - blend) + cg * blend);
      px[idx + 2] = Math.round(px[idx + 2] * (1 - blend) + cb * blend);
    }
    if (onProgress && (ty + 1) % rowChunk === 0) {
      onProgress((ty + 1) / th);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }
  onProgress?.(1);
  ctx.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  if (THREE.SRGBColorSpace != null) {
    tex.colorSpace = THREE.SRGBColorSpace;
  }
  tex.needsUpdate = true;
  return {
    texture: tex,
    scoreScale: colormap,
    ...(colormap === "minMaxScore" ? { minV, maxV } : {}),
  };
}

/**
 * @param {false | "mlSuitability"} mode
 * @param {string} [overlayBasename] filename shown in legend
 */
function setLandingOverlayLegend(mode, overlayBasename) {
  const leg = document.getElementById("landingOverlayLegend");
  if (!leg) return;
  const visible = mode === "mlSuitability";
  leg.classList.toggle("is-visible", visible);
  leg.setAttribute("aria-hidden", visible ? "false" : "true");
  if (!visible) {
    leg.innerHTML = "";
    return;
  }
  const fn = overlayBasename || "mars_landing_suitability_ml.tif";
  leg.innerHTML = `
      <p class="landing-overlay-legend__title"><code>${fn}</code> — ML global suitability</p>
      <p class="landing-overlay-legend__subtitle">0–100% landing suitability (heatmap ramp).</p>
      <div class="landing-overlay-legend__bar" role="img" aria-label="Color scale from low to high" style="background:${landingSuitabilityLegendGradientCss()}"></div>
      <div class="landing-overlay-legend__ticks">
        <span>0%<br /><span class="landing-overlay-legend__sub">Poor</span></span>
        <span style="text-align: center">50%<br /><span class="landing-overlay-legend__sub">Mid</span></span>
        <span>100%<br /><span class="landing-overlay-legend__sub">High</span></span>
      </div>
      <p class="landing-overlay-legend__note">
        <code>mars_landing_suitability_ml.tif</code> (0–100%). No-data (−9999) not tinted.
      </p>`;
}

/** @returns {"off" | "mlSuitability"} */
function getLandingMlOverlayMode() {
  const el = document.getElementById("landingMlOverlayToggle");
  return el?.checked ? "mlSuitability" : "off";
}

// --- ML suitability overlay (checkbox; mars_landing_suitability_ml.tif only) ---
document.getElementById("landingMlOverlayToggle")?.addEventListener("change", async () => {
  const toggleEl = document.getElementById("landingMlOverlayToggle");
  const mode = getLandingMlOverlayMode();
  const statusEl = document.getElementById("landingOverlayStatus");
  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  if (mode === "off") {
    if (landingMlOverlaySessionActive) {
      landingMlOverlaySessionActive = false;
      const gSel = document.getElementById("globeSurfaceLayer");
      if (gSel) {
        gSel.disabled = false;
        gSel.value = priorGlobeSurfaceSelect;
      }
    }
    const gs = document.getElementById("globeSurfaceStatus");
    if (gs) gs.textContent = "";
    setLandingOverlayLegend(false);
    void applyGlobeSurfaceLayerSelect();
    return;
  }

  if (mode !== "off" && landingMlOverlayLoading) {
    return;
  }

  const maxDim = Math.min(4096, renderer.capabilities.maxTextureSize);

  try {
    landingMlOverlayLoading = true;
    if (toggleEl) toggleEl.disabled = true;

    const gSel = document.getElementById("globeSurfaceLayer");
    if (!landingMlOverlaySessionActive && gSel) {
      priorGlobeSurfaceSelect = gSel.value;
      gSel.value = "photo";
      gSel.disabled = true;
      landingMlOverlaySessionActive = true;
    }
    const gsStat = document.getElementById("globeSurfaceStatus");
    if (gsStat) gsStat.textContent = "";
    setGlobeRasterLegendVisible(false);

    if (mode === "mlSuitability") {
      const url = landingMlOverlayRequestUrl();
      const basename = url.split("/").pop() || url;
      setStatus(
        "Downloading mars_landing_suitability_ml.tif (~265 MB uncompressed; first load can take 1–3 min)…"
      );
      const needRebuild =
        !landingMlSuitabilityBlendedTexture ||
        landingMlSuitabilityTintRevisionBuilt !== LANDING_SUITABILITY_TINT_REVISION ||
        landingMlSuitabilityLoadedFromUrl !== url;
      if (needRebuild) {
        if (landingMlSuitabilityBlendedTexture) {
          landingMlSuitabilityBlendedTexture.dispose();
          landingMlSuitabilityBlendedTexture = null;
        }
        setStatus("Decoding GeoTIFF (large raster)…");
        const dataset = await loadGeoTIFF(url, { allowFallback: false });
        setStatus("Tinting heatmap onto Mars (30–120 s; tab stays responsive)…");
        const built = await buildBlendedLandingSuitabilityTexture(dataset, maxDim, {
          colormap: "absolutePercent",
          onProgress: (t) => {
            setStatus(`Tinting heatmap… ${Math.round(Math.min(1, Math.max(0, t)) * 100)}%`);
          },
        });
        landingMlSuitabilityBlendedTexture = built.texture;
        landingMlSuitabilityTintRevisionBuilt = LANDING_SUITABILITY_TINT_REVISION;
        landingMlSuitabilityLoadedFromUrl = url;
      }
      marsMaterial.map = landingMlSuitabilityBlendedTexture;
      marsMaterial.needsUpdate = true;
      setStatus(`ML suitability (${basename}, 0–100%). Uncheck to turn off.`);
      setLandingOverlayLegend("mlSuitability", basename);
    }
  } catch (err) {
    console.error("ML suitability overlay:", err);
    if (toggleEl) toggleEl.checked = false;
    landingMlOverlaySessionActive = false;
    const gSel = document.getElementById("globeSurfaceLayer");
    if (gSel) {
      gSel.disabled = false;
      gSel.value = priorGlobeSurfaceSelect;
    }
    void applyGlobeSurfaceLayerSelect();
    setLandingOverlayLegend(false);
    setStatus(
      "Could not load mars_landing_suitability_ml.tif — check frontend/3d_globe/public/data/ and the browser console."
    );
  } finally {
    landingMlOverlayLoading = false;
    if (toggleEl) toggleEl.disabled = false;
  }
});

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

/** Matches globe click convention: lon = atan2(z,x), lat = asin(y/r). */
function marsSurfacePointFromLatLon(latDeg, lonDeg) {
  const r = marsSphere.geometry.parameters.radius;
  const latR = THREE.MathUtils.degToRad(latDeg);
  const lonR = THREE.MathUtils.degToRad(lonDeg);
  return new THREE.Vector3(
    r * Math.cos(latR) * Math.cos(lonR),
    r * Math.sin(latR),
    r * Math.cos(latR) * Math.sin(lonR)
  );
}

/**
 * Place the camera outside the globe along the surface normal so the point faces the viewer.
 * @param {{ tight?: boolean }} [options] — `tight`: closer framing for small craters / landers (still global view).
 */
function frameCameraOnMarsLatLon(lat, lon, options = {}) {
  const r = marsSphere.geometry.parameters.radius;
  const outward = marsSurfacePointFromLatLon(lat, lon).clone().normalize();
  const mult = options.tight ? 2.02 : 2.85;
  const distance = r * mult;
  camera.position.copy(outward.multiplyScalar(distance));
  controls.target.set(0, 0, 0);
  controls.update();
}

function parseManualLatLon(latStr, lonStr) {
  const lat = Number(String(latStr).trim().replace(",", "."));
  const lon = Number(String(lonStr).trim().replace(",", "."));
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return { error: "Enter numeric latitude and longitude." };
  }
  if (lat < -90 || lat > 90) {
    return { error: "Latitude must be between −90° and 90°." };
  }
  let lonN = lon;
  while (lonN < -180) lonN += 360;
  while (lonN > 180) lonN -= 360;
  return { lat, lon: lonN };
}

async function populateMarsReadingsAtLatLon(lat, lon, markerPoint) {
  const coordsEl = document.getElementById("coords");
  coordsEl.classList.remove("coords--empty");
  coordsEl.innerText = `Loading layers…\nLat ${lat.toFixed(2)}°, Lon ${lon.toFixed(2)}°`;

  const allValues = [];
  for (const [datasetType, datasetInfo] of Object.entries(marsDatasets)) {
    try {
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

  let valuesList = "";
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

  coordsEl.innerText = valuesList;

  const predBtn = document.getElementById("predictLanding");
  predBtn.style.display = "block";
  syncPredictLandingButtonWithBackend();
  const hint = document.getElementById("predictHint");
  if (hint) hint.style.display = "block";
  document.getElementById("landingScore").innerText = "";

  const pt = markerPoint ?? marsSurfacePointFromLatLon(lat, lon);
  marker.position.copy(pt);
  marker.visible = true;

  const manualLatEl = document.getElementById("manualLat");
  const manualLonEl = document.getElementById("manualLon");
  if (manualLatEl) manualLatEl.value = lat.toFixed(4);
  if (manualLonEl) manualLonEl.value = lon.toFixed(4);
}

async function onApplyManualLatLon() {
  const fb = document.getElementById("manualCoordsFeedback");
  const latStr = document.getElementById("manualLat")?.value ?? "";
  const lonStr = document.getElementById("manualLon")?.value ?? "";
  const parsed = parseManualLatLon(latStr, lonStr);
  if (parsed.error) {
    if (fb) fb.textContent = parsed.error;
    return;
  }
  if (fb) fb.textContent = "";
  frameCameraOnMarsLatLon(parsed.lat, parsed.lon);
  await populateMarsReadingsAtLatLon(parsed.lat, parsed.lon, null);
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
    const fb = document.getElementById("manualCoordsFeedback");
    if (fb) fb.textContent = "";
    await populateMarsReadingsAtLatLon(lat, lon, point);
  }
}

window.addEventListener("click", onMouseClick, false);

document.getElementById("applyManualLatLon")?.addEventListener("click", () => {
  void onApplyManualLatLon();
});
for (const id of ["manualLat", "manualLon"]) {
  document.getElementById(id)?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void onApplyManualLatLon();
    }
  });
}

document.getElementById("globeSurfaceLayer")?.addEventListener("change", () => {
  void applyGlobeSurfaceLayerSelect();
});

