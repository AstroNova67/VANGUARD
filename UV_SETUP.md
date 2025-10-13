# VANGUARD Mars Landing System - UV Setup Guide

## 🚀 Quick Start

Your project is now set up with `uv`! Here's how to use it:

### Running the API
```bash
# Method 1: Using the script
./start_api.sh

# Method 2: Using uv directly
uv run python backend/app.py
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

### Managing Dependencies
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

## 📁 Project Structure
- `pyproject.toml` - Project configuration and dependencies
- `backend/` - All your Python code and models
- `frontend/` - Web interface and 3D globe
- `start_api.sh` - Convenient script to start the API

## 🔄 Migration Complete
- ✅ Migrated from `requirements.txt` to `pyproject.toml`
- ✅ All dependencies installed with `uv`
- ✅ Virtual environment managed by `uv`
- ✅ Development tools configured

You can now delete your old `venv-api/` directory if you want - `uv` manages everything now!
