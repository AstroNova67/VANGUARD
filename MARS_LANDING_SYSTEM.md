# Mars Landing Suitability Prediction System

A complete pipeline that combines Mars surface data visualization with machine learning predictions for landing site suitability.

## 🌍 System Overview

This system provides:
- **Interactive 3D Mars Globe** with real surface data from multiple datasets
- **Machine Learning Predictions** using trained neural networks and regression models
- **Landing Suitability Scoring** based on multiple Mars surface properties
- **Real-time API** for seamless frontend-backend communication

## 🚀 Quick Start

### 1. Start the Backend API Server

```bash
cd backend
./start_api.sh
```

This will:
- Create a Python virtual environment
- Install all required dependencies
- Load your trained ML models
- Start the Flask API server on `http://localhost:5000`

### 2. Open the Frontend

Open `frontend/3d_globe/index.html` in your web browser.

### 3. Use the System

1. **Click anywhere on Mars** to see all surface data values
2. **Click "Predict Landing Suitability"** to get ML predictions
3. **View the landing score** with detailed predictions

## 📊 Available Mars Datasets

The system loads 11 different Mars datasets:

- **Elevation (MOLA)** - Surface elevation data
- **Slope** - Surface slope measurements
- **Roughness** - Surface roughness at 0.6km scale
- **Albedo** - Surface reflectivity
- **Temperature** - Yearly average surface temperature
- **Temperature Range** - Yearly temperature variation
- **Crustal Thickness** - Mars crustal thickness
- **Ferric Content** - Ferric iron content
- **Pyroxene** - Pyroxene mineral content
- **Basalt** - Basalt abundance
- **Lambert Albedo** - Lambert albedo from TES

## 🤖 Machine Learning Models

The system uses your trained models:

### Neural Networks
- **Slope Predictor** - Predicts surface slope
- **Dust Predictor** - Predicts dust content
- **Surface Temperature Predictor** - Predicts surface temperature
- **Thermal Inertia Predictor** - Predicts thermal inertia
- **Water Predictor** - Predicts water content

### Regression Models
- **Random Forest** - Surface temperature and thermal inertia
- **Support Vector Regression** - Thermal inertia
- **XGBoost** - Surface temperature and thermal inertia

## 🎯 Landing Suitability Scoring

The landing score is calculated using weighted factors:

- **Slope (30%)** - Lower slopes are better for landing
- **Dust (20%)** - Less dust is preferable
- **Surface Temperature (20%)** - Moderate temperatures are ideal
- **Thermal Inertia (20%)** - Higher thermal inertia indicates stable surface
- **Water (10%)** - Water presence affects landing suitability

**Score Interpretation:**
- **90-100%**: Excellent landing site
- **70-89%**: Good landing site
- **50-69%**: Fair landing site
- **30-49%**: Poor landing site
- **0-29%**: Very poor landing site

## 🔧 API Endpoints

### POST /predict
Send Mars surface data and get landing suitability prediction.

**Request Body:**
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
    },
    "regression_models": {
      "surface_temp_rf": -41.8,
      "thermal_inertia_rf": 445.1,
      "thermal_inertia_svr": 448.3
    }
  }
}
```

### GET /health
Check API server health and model loading status.

### GET /models
Get information about loaded ML models.

## 🛠️ Technical Details

### Frontend
- **Three.js** for 3D Mars visualization
- **GeoTIFF.js** for loading Mars surface data
- **Vanilla JavaScript** for API communication

### Backend
- **Flask** web framework
- **TensorFlow** for neural network predictions
- **Scikit-learn** for regression models
- **NumPy** for numerical computations

### Data Flow
1. User clicks on Mars surface
2. Frontend loads all GeoTIFF datasets
3. Mars data is sent to backend API
4. Backend maps data to ML model features
5. Neural networks and regression models make predictions
6. Landing suitability score is calculated
7. Results are returned to frontend
8. Frontend displays predictions and score

## 🔍 Troubleshooting

### API Server Issues
- Ensure all ML models are in the correct directories
- Check that all dependencies are installed
- Verify Python virtual environment is activated

### Frontend Issues
- Ensure API server is running on `http://localhost:5000`
- Check browser console for JavaScript errors
- Verify GeoTIFF files are accessible

### Model Loading Issues
- Check model file paths in `app.py`
- Ensure model files are not corrupted
- Verify TensorFlow version compatibility

## 📁 File Structure

```
VANGUARD/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── scoring.py            # Landing suitability scoring
│   ├── start_api.sh         # API startup script
│   ├── api_requirements.txt  # Python dependencies
│   └── saved_models/        # Trained ML models
└── frontend/
    └── 3d_globe/
        ├── index.html       # Main frontend
        ├── index.js         # Frontend JavaScript
        └── public/data/     # Mars GeoTIFF datasets
```

## 🎉 Features

- ✅ Real Mars surface data visualization
- ✅ Multiple dataset support (11 different properties)
- ✅ Machine learning predictions
- ✅ Landing suitability scoring
- ✅ Interactive 3D Mars globe
- ✅ Real-time API communication
- ✅ Comprehensive error handling
- ✅ Responsive user interface

## 🔮 Future Enhancements

- Add more Mars datasets
- Implement data visualization overlays
- Add historical landing site markers
- Create landing site comparison tools
- Add export functionality for predictions
- Implement caching for better performance
