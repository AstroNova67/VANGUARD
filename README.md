# V.A.N.G.U.A.R.D
**Visual & Analytical Navigation for Geospatial Understanding And Rover Deployment**

## Overview

Our goal is to leverage martian data from JMARS to build machine learning models that predict key surface and environmental attributes of Mars. These models aim to support the identification of ideal landing sites for future missions, based on scientific and engineering criteria.

By extracting and analyzing datasets such as **elevation**, **albedo**, **slope**, **thermal inertia**, and **surface roughness** from JMARS, we can train models that learn patterns associated with terrain suitability. These predictions can then guide the selection of potential landing zones that balance safety, scientific value, and accessibility.

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
- **Interactive 3D Globe**: Rotate and zoom Mars using orbit controls
- **Surface Feature Visualization**: Displays Mars surface data layers
- **Real-time Data Integration**: Connects ML predictions to visual interface

### Web Interface
- **Point-and-Click Analysis**: Select locations on Mars for analysis
- **Module Descriptions**: Detailed explanations of each prediction model
- **Data Visualization**: Charts and graphs of Mars surface properties

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

### Frontend Setup

1. **Navigate to 3D globe directory**
   ```bash
   cd frontend/3d_globe
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Start local server**
   ```bash
   npx live-server --port=8080
   ```

4. **Open in browser**
   Navigate to `http://localhost:8080` to view the 3D Mars visualization.

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
│   └── app.py                 # Flask API server
├── frontend/
│   ├── 3d_globe/              # Interactive 3D Mars visualization
│   └── website/               # Web interface
├── pyproject.toml             # Project configuration and dependencies
├── uv.lock                    # Locked dependency versions
└── start_api.sh              # API startup script
```

## Usage

### API Endpoints

The Flask API provides the following endpoints:

- `POST /predict` - Get landing suitability prediction (includes all property predictions)
- `GET /health` - Check API server health and model loading status
- `GET /models` - Get information about loaded ML models

### Landing Suitability Prediction

The main endpoint combines predictions from all models and calculates a landing suitability score:

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
  "lambertAlbedo": 0.18
}
```

**Response:**
```json
{
  "success": true,
  "landing_score": 78.5,
  "predictions": {
    "neural_networks": {
      "slope": 2.1,
      "dust": 0.15,
      "surface_temp": -42.3,
      "thermal_inertia": 450.2,
      "water": 0.05
    }
  }
}
```

### Landing Score Interpretation

- **90-100%**: Excellent landing site
- **70-89%**: Good landing site
- **50-69%**: Fair landing site
- **30-49%**: Poor landing site
- **0-29%**: Very poor landing site

The scoring system is based on NASA/JPL engineering constraints. See `LANDING_SCORING_SOURCES.md` for detailed source citations.

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
    "lambertAlbedo": 0.18
})

result = response.json()
print(f"Landing Score: {result['landing_score']}%")
print(f"Predictions: {result['predictions']}")
```

## Data Sources

- **JMARS**: Mars data visualization and analysis platform (https://jmars.asu.edu/)
- **Mars datasets**: Elevation, albedo, slope, thermal inertia, and surface roughness data
- **Natural Earth Data**: For 3D globe visualization (https://www.naturalearthdata.com/)

## Dependencies

### Backend
- TensorFlow 2.16.2 - Machine learning framework
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

For detailed source citations and justification, see [LANDING_SCORING_SOURCES.md](LANDING_SCORING_SOURCES.md).

## Contributors

This project is developed by:

- **Eshaan Khare** - Project Lead, System Architecture, and ML Model Development (Slope, Surface Temperature, and Thermal Inertia models)
- **Arv Jain** - Water and Dust Prediction Models


## License

See [license.md](license.md) for detailed licensing information including third-party contributions and dependencies.

## Acknowledgments

- **JMARS Team**: For providing Mars surface data
- **Three.js Community**: For 3D visualization capabilities
- **Open Source ML Libraries**: TensorFlow, scikit-learn, XGBoost
- **Jared Dominguez**: Original 3D globe implementation
- **Natural Earth Data**: Geographic data sources
- **NASA/JPL**: For landing site selection criteria and engineering constraints
