# V.A.N.G.U.A.R.D
**Visual & Analytical Navigation for Geospatial Understanding And Rover Deployment**

<img src="assets/vanguard-demo.png" width="1100" alt="VANGUARD Mars globe with landing prediction panel: observed rasters vs neural vs XGB, landing suitability score, and score-source highlights" />

## Overview

Our goal is to leverage Martian geospatial data to build machine learning models that predict key surface and environmental attributes of Mars. These models aim to support the identification of interesting landing-site candidates for future missions, based on scientific and engineering-style criteria.

The **3D globe** loads **bundled GeoTIFFs** under `frontend/3d_globe/public/data/`, samples them at the clicked lat/lon, and sends those values to `POST /predict`.

**Live pipeline**

| Stage | Where | What happens |
|--------|--------|----------------|
| **Client** | `frontend/3d_globe/index.js` | Click → lat/lon → each registered GeoTIFF is sampled at that point → JSON body (`elevation`, `slope`, `temperature`, `thermalInertia`, `ferric`, …). |
| **Inference inputs** | `backend/scoring.py` → `map_mars_data_to_features` | Turns that JSON into scaled feature vectors per model (same file also runs inverse transforms on raw NN outputs). |
| **Models** | `backend/app.py` | Loads saved Keras + XGB checkpoints from `saved_models/`; runs all heads for transparency; **raster-first** per scored property (see `data_sources` in the response). |
| **Landing %** | `backend/scoring.py` → `LandingSuitabilityScorer` | Combines the five property values (observed where valid, else ML) with configurable weights (research defaults in `LANDING_SCORING_SOURCES.md`; override via UI or `scoring_weights` in `/predict` and `/agent/chat`). |

**Trained ML weights:** Checkpoints under `saved_models/` come from the `backend/*_predictor.py` training scripts and are unchanged by the landing-score rubric. **`map_mars_data_to_features`** (`scoring.py`) still maps JSON into model inputs; the surface-temperature NN/XGB **input** slot 3 uses **`thermalInertia`** (TES dayside TI), matching training. **Gap-fill:** when a scored property has **no valid raster sample** in the request, `backend/app.py` uses ML for that property. For **surface temperature** and **thermal inertia** only, if the corresponding raster is missing, **`_fuse_*_for_score`** in `app.py` keeps the prior **NN vs XGB** choice among model outputs (no raster substitution in that branch). **`data_sources`** in the JSON states, per property, whether the value that entered the rubric was **`raster`** or **`ml_predicted`**.

**Publication / provenance:** `data_sources` answers “why use ML when you have data?”—when a layer is valid at the click, the score uses that observation; ML appears only for gap-fill (or for NN/XGB disambiguation when those two rasters are absent).

## Objectives

- Collect and process Mars surface data  
- Engineer features that are relevant to mission planning  
- Train and evaluate models that predict surface attributes  
- Generalize predictions across diverse Martian terrain types  
- Identify and rank promising candidate sites for future exploration

## Features

### Machine Learning Models
- **Dust Prediction**: Predicts dust coverage on Martian surface
- **Slope Analysis**: Estimates terrain slope for landing safety
- **Surface Temperature**: Predicts surface temperature variations
- **Thermal Inertia**: Analyzes thermal properties of surface materials
- **Water Content**: Estimates water equivalent hydrogen (WEH) percentage

### 3D Visualization
- **Interactive 3D globe** (Three.js): orbit controls, optional sun lighting
- **Per-click GeoTIFF sampling**: each registered layer is read at the clicked point (elevation, slope, temperature, etc.); values appear as a text list in the side panel
- **Landing prediction panel**: after **Predict landing suitability**, shows **landing %**, a **three-column** table (**Observed (raster)** vs **Neural networks** vs **Regression (XGBoost)**), **Raster / Neural / XGB “in score”** tags driven by **`data_sources`**, footnotes for raster-first + ML gap-fill, optional **Δ** badges, and an expandable **raw JSON** payload

### Web Interface (what the app actually shows)
- **Point-and-click** on the globe → numeric **raster-derived** properties for that location
- **One-button** call to `POST /predict` (same origin as the page when you use the Flask server below)
- **No separate charting dashboard** in the globe UI (no built-in plots); exploration is tabular / text plus the 3D view

**Observed column:** Raster values in the prediction table mirror fields in the JSON body to `/predict`. When those fields are valid, **the same numbers can feed the landing % directly** (see **`data_sources`**). **Dust (scored from raster)** uses the **OMEGA ferric/dust** index (`ferric` in JSON; same as **`dustObserved`** in the table). The scorer’s dust band was tuned for NN outputs (~0.6–0.7); using raw ferric for the dust term is intentional for “observed first” behavior but may sit outside that normalization band—treat as a known limitation unless you add calibration. **Slope**, **yearly-average surface temperature** (`temperature`), **TES dayside TI** (`thermalInertia`), and **GRS water** (`grsWaterWt`) follow the same raster-first rule when finite and present.

## 🚀 Quick Start

This project uses `uv` for dependency management. If you don't have `uv` installed, install it first:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running the API

```bash
# Method 1: Using the script
./start_api.sh

# Method 2: Using uv directly
uv run python backend/app.py
```

The API will start on `http://localhost:5002` (or port 5000 if 5002 is unavailable).

**Mars assistant (optional):** copy `.env.example` to `.env` at the repo root and set `OPENAI_API_KEY`. With the Flask app running, open the globe UI — a **Mars Assistant** panel appears on the **right** (minimize with **−**, reopen via the **Assistant** tab). It calls `POST /agent/chat` and automatically includes your current lat/lon when you have loaded a point on the globe. Ask it to **show Gale crater** or **switch the globe to slope** — it returns `ui_actions` that move the camera, load rasters, run prediction, and change the surface layer (safe Markdown in chat, not raw HTML). Server-side trace: `VANGUARD_AGENT_TRACE=1` logs each tool call and UI action to stderr.

```bash
# Interactive CLI (alternative to the in-app panel)
uv run python backend/agent.py
```

Optional: for detailed server logs and per-request HTTP lines, run with `VANGUARD_VERBOSE=1` (default is a short, readable startup + one line per `/predict`).

**Recommended way to use the globe + API together:** after `./start_api.sh` (or `uv run python backend/app.py`), open **`http://127.0.0.1:5002`** (or the port printed in the terminal—**5000** if 5002 is taken). Flask serves `frontend/3d_globe/` and `POST /predict` on the **same origin**, which matches the client’s `fetch` to `/predict`.

### Running without `uv` (pip)

If you prefer not to use `uv`, you can install Python dependencies via `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backend/app.py
```

### Running Individual Predictors

```bash
# Test all models
uv run python backend/scoring.py

# Run individual predictors
uv run python backend/dust_predictor.py
uv run python backend/slope_predictor.py
uv run python backend/surface_temp_predictor.py
uv run python backend/thermal_inertia_predictor.py
uv run python backend/water_predictor.py
```

## Installation

### Prerequisites

- **Python 3.11+**
- **uv** - Fast Python package installer (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Node.js** - For frontend development

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd VANGUARD
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```
   This will automatically:
   - Create a virtual environment
   - Install all dependencies from `pyproject.toml`
   - Set up the project environment

3. **Start the API server**
   ```bash
   ./start_api.sh
   # Or manually:
   uv run python backend/app.py
   ```

### Frontend setup

**Default (no extra steps):** the backend serves the globe; see **Running the API** above.

**Optional — `live-server` only for frontend-only hacking:** the page calls **`/predict` on the same host as the page**. If you open the app on port **8080** without a proxy, `/predict` will not hit the Flask API unless you change the fetch URL or add a dev proxy. For a working predict button, prefer the **Flask URL**.

```bash
cd frontend/3d_globe
npm install
npx live-server --port=8080   # optional; wire API separately if you need predictions
```

### Demo: GeoTIFF layers sampled on click

These files (under `frontend/3d_globe/public/data/`) are registered in `frontend/3d_globe/index.js` and sampled at the clicked location:

| Layer | File (repo) |
|--------|-------------|
| Elevation (MOLA) | `MOLA_128ppd_topo.tif` |
| Slope | `mola_hrsc_blend_slope_v2.tif` |
| Roughness | `mola_roughness_0.6km_numeric.tif` |
| Albedo | `omega_albedo_r1080.tif` |
| Temperature (yearly average °C) | `mars_yearly_avg_temperature_celsius.tif` |
| Temperature range | `mars_yearly_temperature_range_v1.0.tif` |
| Crustal thickness | `mars_crustal_thickness_gmm3_rm1.tif` |
| Ferric / dust (OMEGA) | `omega_ferric_nnphs.tif` (also exposed as `dustObserved` in the API payload) |
| Pyroxene | `omega_pyroxene_bd2000.tif` |
| Basalt | `TES_Basalt_numeric.tif` |
| Lambert albedo | `TES_Lambert_Albedo_numeric.tif` |
| Thermal inertia (TES dayside, Putzig 2007) | `tes_dayside_ti_putzig_2007.tif` |
| GRS water equivalent (% wt) | `mars_odyssey_grs_mons_perc_wt.tif` |

GeoTIFF **no-data** values are respected when GDAL metadata is present so missing pixels are less likely to show as false `0.00`.

## Managing Dependencies with UV

### Basic Commands

```bash
# Install all dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --group dev package-name

# Remove a dependency
uv remove package-name

# Update all dependencies
uv sync --upgrade
```

### Development Tools

```bash
# Run code formatting
uv run black backend/

# Run linting
uv run flake8 backend/

# Run type checking
uv run mypy backend/

# Run tests
uv run pytest
```

## Project Structure

```
VANGUARD/
├── backend/                    # Python backend with ML models
│   ├── datasets/              # Mars surface data
│   ├── saved_models/          # Trained ML models
│   ├── plots/                 # Data visualizations
│   ├── dust_predictor.py      # Dust prediction model
│   ├── slope_predictor.py      # Slope prediction model
│   ├── surface_temp_predictor.py  # Surface temperature model
│   ├── thermal_inertia_predictor.py  # Thermal inertia model
│   ├── water_predictor.py     # Water content prediction model
│   ├── scoring.py             # Landing suitability scoring system
│   ├── batch_global_landing_suitability.py  # Global suitability GeoTIFF (raster-first)
│   └── app.py                 # Flask API server
├── frontend/
│   ├── 3d_globe/              # Interactive 3D Mars visualization
├── pyproject.toml             # Project configuration and dependencies
├── uv.lock                    # Locked dependency versions
└── start_api.sh              # API startup script
```

## Usage

### API Endpoints

The Flask API provides the following endpoints:

- `POST /predict` - Get landing suitability prediction (includes all property predictions)
- `POST /agent/chat` - Chat with the VANGUARD Mars assistant (`{"message": "..."}`); uses OpenAI + tools (site lookup, README, landing analysis at lat/lon). Requires `OPENAI_API_KEY` in repo-root `.env`.
- `GET /health` - Check API server health and model loading status
- `GET /models` - Get information about loaded ML models

### Landing Suitability Prediction

The main endpoint always runs the five neural heads and **XGBoost** for surface temperature and thermal inertia (when model files load) so the response can show model-vs-observed comparisons. It then builds the **five values that enter `LandingSuitabilityScorer`**: for each property, **use the raster sample when valid**, otherwise **ML** (for surface temperature and thermal inertia only, ML means the existing **NN vs XGB fusion** when those rasters are missing). The response includes **`data_sources`** (`raster` vs `ml_predicted`) per property.

**Request:**
```json
{
  "lat": 15.23,
  "lon": -45.67,
  "elevation": 1234.56,
  "slope": 2.34,
  "roughness": 0.89,
  "albedo": 0.15,
  "temperature": -45.23,
  "tempRange": 12.45,
  "crustalThickness": 45.67,
  "ferric": 0.23,
  "pyroxene": 0.12,
  "basalt": 0.45,
  "lambertAlbedo": 0.18,
  "thermalInertia": 412.0,
  "grsWaterWt": 2.1,
  "dustObserved": 0.23
}
```

Optional / UI-driven fields: `thermalInertia`, `temperature`, `slope`, `ferric`, `grsWaterWt`, etc., are echoed in `raw_mars_data`. **Landing score** uses each of the five rubric inputs from **raster-first** logic: valid **`temperature`** → scored surface temp is observed °C; valid **`thermalInertia`** → scored TI is observed; same for **`slope`**, **`ferric`** (dust term), **`grsWaterWt`** (water term).

For **surface-temperature** NN/XGB **inputs**, the five **features** are built in `map_mars_data_to_features` (`scoring.py`). The **third** feature is **dayside thermal inertia**, from **`thermalInertia`** in JSON. The **`temperature`** field is **not** that third input; it is the yearly average used for **scoring when valid**, and still feeds other model heads as today. **`grsWaterWt`** is not a neural-net input in `map_mars_data_to_features`. **`dustObserved`** duplicates **`ferric`** for display only.

**API-only callers:** Include **`thermalInertia`** whenever possible so NN/XGB **inputs** match training (the globe does this automatically). Raster-first scoring still benefits from complete JSON so each property can resolve to `raster` in **`data_sources`**.

**Response (illustrative numbers):**

- `predictions.neural_networks` — **final property bundle used for `landing_score`** (same key name as before; values may be observed or ML).
- `predictions.neural_networks_baseline` — **neural nets only** (always model outputs).
- `predictions.regression_models` — raw XGB outputs (`surface_temp_xgb`, `thermal_inertia_xgb`).
- `data_sources` — per property (`slope`, `dust`, `surface_temp`, `thermal_inertia`, `water`): **`raster`** or **`ml_predicted`** for what entered the rubric.
- `overrides_applied` — when temp/TI use **ML**, which head won (**`surface_temp_xgb`** / **`surface_temp_nn`**, etc.); omitted when that property used a raster for scoring.
- `raw_mars_data` — **echo of the full request JSON** the server received (not truncated in real responses).

```json
{
  "success": true,
  "landing_score": 78.5,
  "predictions": {
    "neural_networks": {
      "slope": 2.34,
      "dust": 0.23,
      "surface_temp": -45.23,
      "thermal_inertia": 412.0,
      "water": 3.5
    },
    "neural_networks_baseline": {
      "slope": 2.1,
      "dust": 0.15,
      "surface_temp": -41.0,
      "thermal_inertia": 320.0,
      "water": 3.5
    },
    "regression_models": {
      "surface_temp_xgb": -44.8,
      "thermal_inertia_xgb": 445.2
    }
  },
  "data_sources": {
    "slope": "raster",
    "dust": "raster",
    "surface_temp": "raster",
    "thermal_inertia": "raster",
    "water": "ml_predicted"
  },
  "overrides_applied": {},
  "raw_mars_data": {
    "lat": 15.23,
    "lon": -45.67,
    "elevation": 1234.56,
    "slope": 2.34,
    "roughness": 0.89,
    "albedo": 0.15,
    "temperature": -45.23,
    "tempRange": 12.45,
    "crustalThickness": 45.67,
    "ferric": 0.23,
    "pyroxene": 0.12,
    "basalt": 0.45,
    "lambertAlbedo": 0.18,
    "thermalInertia": 412.0,
    "grsWaterWt": 2.1,
    "dustObserved": 0.23
  }
}
```

**Raster-first and fusion:** `predictions.neural_networks` is what `LandingSuitabilityScorer` receives. If **`temperature`** is a valid observation, that value is used for the **surface temperature** term and **`data_sources.surface_temp`** is **`raster`**; **NN vs XGB fusion for that property runs only when `temperature` is missing or non-finite** (same pattern for **`thermalInertia`** / **`thermal_inertia`**). In the ML-only branches, `_fuse_*_for_score` in `app.py` is unchanged (XGB vs NN among model outputs). **`overrides_applied`** records NN vs XGB for temp/TI only when **`data_sources`** for that property is **`ml_predicted`**.

**Global batch GeoTIFF** (`backend/batch_global_landing_suitability.py`): Reads the 13-band stack **`mars_global_input_stack_32ppd.tif`** (`--input`; never modified). By default writes only **`mars_landing_suitability_ml.tif`** (`--output`). Pass **`--with-hybrid-coverage`** to also write hybrid + **`mars_ml_coverage.tif`** (see `--hybrid-output` / `--coverage-output`). Regenerate ML map: `uv run python backend/batch_global_landing_suitability.py`.

### Landing Score Interpretation

- **90-100%**: Excellent landing site
- **70-89%**: Good landing site
- **50-69%**: Fair landing site
- **30-49%**: Poor landing site
- **0-29%**: Very poor landing site

The scoring system is implemented in `backend/scoring.py` (`LandingSuitabilityScorer`) and is based on NASA/JPL engineering constraints. See `LANDING_SCORING_SOURCES.md` for detailed source citations.

### Example Usage

```python
import requests

# Example API call
response = requests.post('http://localhost:5002/predict', json={
    "lat": 15.23,
    "lon": -45.67,
    "elevation": 1234.56,
    "slope": 2.34,
    "roughness": 0.89,
    "albedo": 0.15,
    "temperature": -45.23,
    "tempRange": 12.45,
    "crustalThickness": 45.67,
    "ferric": 0.23,
    "pyroxene": 0.12,
    "basalt": 0.45,
    "lambertAlbedo": 0.18,
    "thermalInertia": 412.0,
    "grsWaterWt": 2.1,
    "dustObserved": 0.23,
})

result = response.json()
print(f"Landing Score: {result['landing_score']}%")
print(f"Data sources: {result.get('data_sources')}")
print(f"Predictions: {result['predictions']}")
```

## Data sources

- **JMARS and mission products** — Training data was extracted from [JMARS](https://jmars.asu.edu/); the demo uses bundled GeoTIFFs derived from NASA mission data products (files in `frontend/3d_globe/public/data/`; provenance per layer in the demo GeoTIFF table above).
- **Globe texture** — Mars imagery for the 3D sphere (e.g. `frontend/3d_globe/textures/`); separate from the scientific rasters used for sampling and API inputs.

## Dependencies

### Backend
- TensorFlow 2.16.2 - Machine learning framework (Linux) / tensorflow-macos (macOS)
- scikit-learn 1.7.1 - Machine learning library
- XGBoost 3.0.4 - Gradient boosting framework
- Flask 2.3.3 - Web framework
- NumPy 1.26.4 - Numerical computing
- Pandas 2.3.2 - Data analysis
- Matplotlib 3.10.6 - Plotting

All dependencies are managed through `pyproject.toml` and installed via `uv`.

### Frontend
- Three.js 0.179.1 - 3D graphics library

## Landing Site Scoring System

The landing suitability scoring system uses an expert system based on NASA/JPL engineering constraints from actual Mars mission landing site selection processes. The scoring criteria include:

- **Slope (30%)**: Critical for rover stability at touchdown
- **Dust (20%)**: Avoid dust-dominated surfaces for safe landing
- **Surface Temperature (20%)**: Thermal management constraint
- **Thermal Inertia (20%)**: Indicates surface stability and load-bearing capacity
- **Water (10%)**: Scientific interest (secondary to engineering safety)

For detailed source citations and justification, see [LANDING_SCORING_SOURCES.md](LANDING_SCORING_SOURCES.md). The API uses `LandingSuitabilityScorer` in `backend/scoring.py` to compute the `landing_score`.

## Contributors

This project is developed by:

- **Eshaan Khare** - Project Lead, System Architecture, and ML Model Development (Slope, Surface Temperature, and Thermal Inertia models)
- **Arv Jain** - Water and Dust Prediction Models

## Deployment

The stack is set up for **local or self-hosted** use: run the Flask app (`./start_api.sh` or `uv run python backend/app.py`), then open the URL it prints so the served globe and `POST /predict` share the same origin. There is no separate production deploy manifest in this repository; adapt hosting (WSGI, TLS, static assets) to your environment if you expose it beyond localhost.

## License

See [license.md](license.md) for detailed licensing information including third-party contributions and dependencies.

## Acknowledgments

- **JMARS Team**: For providing Mars surface data
- **Three.js Community**: For 3D visualization capabilities
- **Open Source ML Libraries**: TensorFlow, scikit-learn, XGBoost
- **Jared Dominguez**: Original 3D globe implementation
- **Natural Earth Data**: Geographic data sources
- **NASA/JPL**: For landing site selection criteria and engineering constraints
