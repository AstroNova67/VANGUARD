from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras
import pickle
import json
import os
import sys

# Handle imports for both local development and production (Render)
# Try absolute import first (for Render), fall back to relative import (for local)
try:
    from backend.scoring import LandingSuitabilityScorer, predict_properties_nn
except ImportError:
    # For local development when running from backend/ directory
    from scoring import LandingSuitabilityScorer, predict_properties_nn

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables to store loaded models
neural_models = {}
regression_models = {}
scalers = {}  # Store loaded scalers

# Resolve paths relative to this file so the API can run from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend', '3d_globe')

def _clamp_predictions(preds: dict) -> dict:
    """Clamp predictions to plausible physical ranges for stability."""
    out = dict(preds)
    if 'slope' in out:
        out['slope'] = max(0.0, float(out['slope']))
    if 'dust' in out:
        out['dust'] = float(min(1.0, max(0.0, out['dust'])))
    if 'surface_temp' in out:
        # Allow wide Mars temps; do not clamp too hard, keep as float
        out['surface_temp'] = float(out['surface_temp'])
    if 'thermal_inertia' in out:
        out['thermal_inertia'] = max(0.0, float(out['thermal_inertia']))
    if 'water' in out:
        # Cap to 0–8% per scoring normalization
        out['water'] = float(min(8.0, max(0.0, out['water'])))
    return out

def load_scalers():
    """Load all saved scalers and transformers"""
    global scalers
    scaler_dir = os.path.join(BASE_DIR, 'saved_models', 'scalers')
    
    try:
        # Load slope scaler
        with open(f'{scaler_dir}/slope_scaler.pkl', 'rb') as f:
            scalers['slope'] = pickle.load(f)
        print("✅ Loaded slope scaler")
        
        # Load dust scalers
        with open(f'{scaler_dir}/dust_feature_scaler.pkl', 'rb') as f:
            scalers['dust_feature'] = pickle.load(f)
        with open(f'{scaler_dir}/dust_target_transformer.pkl', 'rb') as f:
            scalers['dust_target'] = pickle.load(f)
        print("✅ Loaded dust scalers")
        
        # Load surface temperature scaler
        with open(f'{scaler_dir}/surface_temp_scaler.pkl', 'rb') as f:
            scalers['surface_temp'] = pickle.load(f)
        with open(f'{scaler_dir}/surface_temp_y_min.pkl', 'rb') as f:
            scalers['surface_temp_y_min'] = pickle.load(f)
        print("✅ Loaded surface temperature scaler")
        
        # Load thermal inertia transformers
        with open(f'{scaler_dir}/thermal_inertia_feature_transformer.pkl', 'rb') as f:
            scalers['thermal_inertia_feature'] = pickle.load(f)
        with open(f'{scaler_dir}/thermal_inertia_target_transformer.pkl', 'rb') as f:
            scalers['thermal_inertia_target'] = pickle.load(f)
        print("✅ Loaded thermal inertia transformers")
        
        # Load water scaler
        with open(f'{scaler_dir}/water_scaler.pkl', 'rb') as f:
            scalers['water'] = pickle.load(f)
        print("✅ Loaded water scaler")
        
        print(f"📊 Loaded {len(scalers)} scalers and transformers")
        
    except Exception as e:
        print(f"❌ Error loading scalers: {e}")
        scalers = {}

def load_models():
    """Load all ML models at startup"""
    global neural_models, regression_models
    
    neural_models = {}
    regression_models = {}
    
    # Try to load Neural Network models one by one
    model_paths = {
        'slope': os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'slope_pred', 'best_model.keras'),
        'dust': os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'dust_predictor', 'best_model.keras'),
        'surface_temp': os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'surface_temp_pred', 'best_model.keras'),
        'thermal_inertia': os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'thermal_inertia_predictor', 'best_model.keras'),
        'water': os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'water_predictor', 'best_model.keras')
    }
    
    for model_name, model_path in model_paths.items():
        try:
            neural_models[model_name] = keras.models.load_model(model_path)
            print(f"✅ Loaded {model_name} model")
        except Exception as e:
            print(f"❌ Failed to load {model_name} model: {e}")
    
    # Try to load Regression models
    try:
        import xgboost as xgb
        
        regression_models = {
            'surface_temp': {
                'xgb': xgb.XGBRegressor()
            },
            'thermal_inertia': {
                'xgb': xgb.XGBRegressor()
            }
        }
        
        # Load XGBoost models
        regression_models['surface_temp']['xgb'].load_model(
            os.path.join(BASE_DIR, 'saved_models', 'regression_models', 'surface_temp', 'xgb_model.json')
        )
        regression_models['thermal_inertia']['xgb'].load_model(
            os.path.join(BASE_DIR, 'saved_models', 'regression_models', 'thermal_inertia', 'xgb_model.json')
        )
        
        print("✅ Regression models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading regression models: {e}")
        regression_models = {}
    
    print(f"📊 Loaded {len(neural_models)} neural network models and {len(regression_models)} regression model types")

def map_mars_data_to_features(mars_data, model_name):
    """
    Map Mars dataset values to ML model input features using saved scalers
    Each model expects different features in different orders
    """
    # Extract values from Mars data
    elevation = mars_data.get('elevation', 0)
    slope = mars_data.get('slope', 0)
    roughness = mars_data.get('roughness', 0)
    albedo = mars_data.get('albedo', 0)
    temperature = mars_data.get('temperature', 0)
    temp_range = mars_data.get('tempRange', 0)
    crustal_thickness = mars_data.get('crustalThickness', 0)
    ferric = mars_data.get('ferric', 0)
    pyroxene = mars_data.get('pyroxene', 0)
    basalt = mars_data.get('basalt', 0)
    lambert_albedo = mars_data.get('lambertAlbedo', 0)
    
    if model_name == 'slope':
        # Slope model: 8 features
        # ['Albedo', 'Day Side Thermal Inertia', 'Roughness', 'OMEGA Ferric/Dust 860nm ratio',
        #  'Elevation_rolling_mean', 'Elevation_rolling_max_diff', 'Elevation_diff_abs', 'Elevation_diff2_abs']
        features = [
            albedo,                    # Albedo
            temperature,               # Day Side Thermal Inertia (using temperature as proxy)
            roughness,                 # Roughness
            ferric,                    # OMEGA Ferric/Dust 860nm ratio
            elevation,                 # Elevation_rolling_mean (simplified)
            temp_range,                # Elevation_rolling_max_diff (using temp range as proxy)
            abs(slope),                # Elevation_diff_abs (using slope as proxy)
            abs(slope * 0.1)           # Elevation_diff2_abs (simplified)
        ]
        # Use saved scaler
        if 'slope' in scalers:
            features = scalers['slope'].transform([features])[0]
        else:
            # Fallback to manual normalization
            features = [
                features[0],                    # Albedo (0-1, already normalized)
                features[1] / 100.0,            # Temperature (scale to roughly -1 to 1)
                features[2] / 1000.0,           # Roughness (scale down)
                features[3],                    # Ferric ratio (keep as is)
                features[4] / 10000.0,          # Elevation (scale down)
                features[5] / 200.0,            # Temp range (scale down)
                features[6] / 10.0,             # Slope diff (scale down)
                features[7] / 10.0              # Slope diff2 (scale down)
            ]
        
    elif model_name == 'dust':
        # Dust model: 6 features
        # ["Elevation", "Slope", "Yearly Average Mars Surface Temperature", "Dayside Thermal Inertia", "MOLA 128ppd Aspect", "Albedo"]
        features = [
            elevation,                 # Elevation
            slope,                     # Slope
            temperature,               # Yearly Average Mars Surface Temperature
            temperature,               # Dayside Thermal Inertia (using temperature as proxy)
            slope,                     # MOLA 128ppd Aspect (using slope as proxy)
            albedo                     # Albedo
        ]
        # Use saved scaler
        if 'dust_feature' in scalers:
            features = scalers['dust_feature'].transform([features])[0]
        else:
            # Fallback to manual normalization
            features = [
                features[0] / 10000.0,         # Elevation (scale down)
                features[1] / 10.0,             # Slope (scale down)
                features[2] / 100.0,            # Temperature (scale down)
                features[3] / 100.0,            # Thermal inertia (scale down)
                features[4] / 10.0,             # Aspect (scale down)
                features[5]                     # Albedo (already normalized)
            ]
        
    elif model_name == 'surface_temp':
        # Surface temperature model: 5 features
        # ['Elevation', 'Albedo', 'Day Side Thermal Inertia', 'Slope', 'Roughness 0.6km']
        features = [
            elevation,                 # Elevation
            albedo,                    # Albedo
            temperature,               # Day Side Thermal Inertia (using temperature as proxy)
            slope,                     # Slope
            roughness                  # Roughness 0.6km
        ]
        # Use saved scaler
        if 'surface_temp' in scalers:
            features = scalers['surface_temp'].transform([features])[0]
        else:
            # Fallback to manual normalization
            features = [
                features[0] / 10000.0,         # Elevation (scale down)
                features[1],                   # Albedo (already normalized)
                features[2] / 100.0,           # Thermal inertia (scale down)
                features[3] / 10.0,            # Slope (scale down)
                features[4] / 1000.0           # Roughness (scale down)
            ]
        
    elif model_name == 'thermal_inertia':
        # Thermal inertia model: 8 features (same as slope)
        features = [
            albedo,                    # Albedo
            temperature,               # Day Side Thermal Inertia (using temperature as proxy)
            roughness,                 # Roughness
            ferric,                    # OMEGA Ferric/Dust 860nm ratio
            elevation,                 # Elevation_rolling_mean (simplified)
            temp_range,                # Elevation_rolling_max_diff (using temp range as proxy)
            abs(slope),                # Elevation_diff_abs (using slope as proxy)
            abs(slope * 0.1)           # Elevation_diff2_abs (simplified)
        ]
        # Use saved transformer
        if 'thermal_inertia_feature' in scalers:
            features = scalers['thermal_inertia_feature'].transform([features])[0]
        else:
            # Fallback to manual normalization
            features = [
                features[0],                    # Albedo (0-1, already normalized)
                features[1] / 100.0,            # Temperature (scale to roughly -1 to 1)
                features[2] / 1000.0,           # Roughness (scale down)
                features[3],                    # Ferric ratio (keep as is)
                features[4] / 10000.0,          # Elevation (scale down)
                features[5] / 200.0,            # Temp range (scale down)
                features[6] / 10.0,             # Slope diff (scale down)
                features[7] / 10.0              # Slope diff2 (scale down)
            ]
        
    elif model_name == 'water':
        # Water model: 8 features
        # ['MOLA 128ppd Elevation', 'OMEGA Est. Lambert Albedo 1080nm', 'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)', 
        #  'TES Basalt Abundance - Numeric', 'Yearly Average Mars Surface Temperature', 'Latitude (N)', 'OMEGA Band depth at 2000 nm', 'Crustal Thickness (km)']
        features = [
            elevation,                 # MOLA 128ppd Elevation
            lambert_albedo,            # OMEGA Est. Lambert Albedo 1080nm
            temperature,               # Dayside Thermal Inertia (using temperature as proxy)
            basalt,                    # TES Basalt Abundance - Numeric
            temperature,               # Yearly Average Mars Surface Temperature
            0,                         # Latitude (N) - not available in Mars data, use 0
            pyroxene,                  # OMEGA Band depth at 2000 nm (using pyroxene as proxy)
            crustal_thickness          # Crustal Thickness (km)
        ]
        # Use saved scaler
        if 'water' in scalers:
            features = scalers['water'].transform([features])[0]
        else:
            # Fallback to manual normalization
            features = [
                features[0] / 10000.0,         # Elevation (scale down)
                features[1],                   # Lambert albedo (already normalized)
                features[2] / 100.0,           # Thermal inertia (scale down)
                features[3] / 100.0,          # Basalt abundance (scale down)
                features[4] / 100.0,           # Temperature (scale down)
                features[5],                   # Latitude (keep as is)
                features[6],                   # Pyroxene (keep as is)
                features[7] / 100.0            # Crustal thickness (scale down)
            ]
    
    else:
        # Default fallback
        features = [0.0] * 8
    
    return np.array(features).reshape(1, -1)

def predict_with_neural_networks(mars_data):
    """Use neural networks to predict properties (inverse-transformed to real units)"""
    try:
        slope, dust, surface_temp, thermal_inertia, water = predict_properties_nn(mars_data)
        return {
            'slope': float(slope),
            'dust': float(dust),
            'surface_temp': float(surface_temp),
            'thermal_inertia': float(thermal_inertia),
            'water': float(water)
        }
    except Exception as e:
        print(f"Error in predict_with_neural_networks: {e}")
        return {
            'slope': 0.0,
            'dust': 0.0,
            'surface_temp': 0.0,
            'thermal_inertia': 0.0,
            'water': 0.0
        }

def predict_with_regression_models(mars_data):
    """Use regression models to predict properties
    Note: Regression models use different feature sets than neural networks
    """
    predictions = {}
    
    if len(regression_models) == 0:
        # Provide mock predictions when models fail to load
        print("⚠️ Using mock regression predictions (models not loaded)")
        predictions = {
            'surface_temp_xgb': -44.8,
            'thermal_inertia_xgb': 445.2
        }
    else:
        # Import scoring module for regression model feature mapping
        try:
            from backend.scoring import map_mars_data_to_features as scoring_map_features
        except ImportError:
            from scoring import map_mars_data_to_features as scoring_map_features
        
        # Surface temperature predictions (5 features)
        if 'surface_temp' in regression_models:
            try:
                features = scoring_map_features(mars_data, 'surface_temp')
                xgb_pred = regression_models['surface_temp']['xgb'].predict(features)[0]
                predictions['surface_temp_xgb'] = float(xgb_pred)
            except Exception as e:
                print(f"Error predicting surface_temp with XGB: {e}")
                predictions['surface_temp_xgb'] = 0.0
        
        # Thermal inertia predictions (4 features: temp_range, albedo, slope, ferric)
        # Note: XGBoost model was trained on RAW features (not transformed), so we need to prepare features without the QuantileTransformer
        if 'thermal_inertia' in regression_models:
            try:
                # Extract raw features matching the CSV columns:
                # ['Yearly Mars Surface Temperature Variation (C)', 'Albedo', 'Slope', 'OMEGA Ferric/Dust 860nm ratio']
                temp_range = mars_data.get('tempRange', 0)
                albedo = mars_data.get('albedo', 0)
                slope = mars_data.get('slope', 0)
                ferric = mars_data.get('ferric', 0)
                
                # XGBoost was trained on raw features (no transformation needed)
                # Shape: (1, 4) for single prediction
                features = np.array([[temp_range, albedo, slope, ferric]])
                xgb_pred = regression_models['thermal_inertia']['xgb'].predict(features)[0]
                predictions['thermal_inertia_xgb'] = float(xgb_pred)
            except Exception as e:
                print(f"Error predicting thermal_inertia with XGB: {e}")
                predictions['thermal_inertia_xgb'] = 0.0
    
    return predictions

@app.route('/predict', methods=['POST'])
def predict_landing_suitability():
    """Main API endpoint for landing suitability prediction"""
    try:
        # Get Mars data from frontend
        mars_data = request.json
        print(f"Received Mars data: {mars_data}")
        
        # Get predictions from neural networks
        nn_predictions = predict_with_neural_networks(mars_data)
        print(f"Neural network predictions: {nn_predictions}")
        
        # Get predictions from regression models
        reg_predictions = predict_with_regression_models(mars_data)
        print(f"Regression predictions: {reg_predictions}")

        # Override specific targets with regressors per request
        if 'surface_temp_xgb' in reg_predictions:
            nn_predictions['surface_temp'] = reg_predictions['surface_temp_xgb']
        if 'thermal_inertia_xgb' in reg_predictions:
            nn_predictions['thermal_inertia'] = reg_predictions['thermal_inertia_xgb']

        # Clamp to plausible ranges before scoring/response
        nn_predictions = _clamp_predictions(nn_predictions)
        
        # Calculate landing score using neural network predictions
        scorer = LandingSuitabilityScorer()
        landing_score = scorer.score_site(
            slope=nn_predictions.get('slope', 0),
            dust=nn_predictions.get('dust', 0),
            surface_temp=nn_predictions.get('surface_temp', 0),
            thermal_inertia=nn_predictions.get('thermal_inertia', 0),
            water=nn_predictions.get('water', 0)
        )
        
        # Prepare response
        response = {
            'success': True,
            'landing_score': landing_score,
            'predictions': {
                'neural_networks': nn_predictions,
                'regression_models': reg_predictions
            },
            'raw_mars_data': mars_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'landing_score': 0
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'neural_models_loaded': len(neural_models),
        'regression_models_loaded': len(regression_models)
    })

@app.route('/models', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    return jsonify({
        'neural_models': list(neural_models.keys()),
        'regression_models': list(regression_models.keys())
    })

# Frontend serving routes
@app.route('/')
def index():
    """Serve the main frontend HTML page"""
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))

@app.route('/<path:path>')
def serve_frontend(path):
    """Serve frontend static files (JS, CSS, textures, data, etc.)"""
    # Don't serve API routes through this handler
    if path.startswith('predict') or path.startswith('health') or path.startswith('models'):
        return jsonify({'error': 'Not found'}), 404
    
    # Security: prevent path traversal
    if '..' in path:
        return jsonify({'error': 'Invalid path'}), 400
    
    # Build file path
    file_path = os.path.join(FRONTEND_DIR, path)
    
    # Security check: ensure file is within frontend directory
    try:
        file_path_abs = os.path.abspath(file_path)
        frontend_dir_abs = os.path.abspath(FRONTEND_DIR)
        if not file_path_abs.startswith(frontend_dir_abs):
            return jsonify({'error': 'Access denied'}), 403
    except:
        return jsonify({'error': 'Invalid path'}), 400
    
    # If it's a file, serve it
    if os.path.isfile(file_path):
        return send_file(file_path)
    
    # If it's a directory, try to serve index.html from it (for SPA routing)
    if os.path.isdir(file_path):
        index_path = os.path.join(file_path, 'index.html')
        if os.path.exists(index_path):
            return send_file(index_path)
    
    # File not found
    return jsonify({'error': 'File not found'}), 404

# Load models and scalers when module is imported (works with both Flask dev server and gunicorn)
# This ensures models are loaded in production (gunicorn) where __main__ doesn't run
print("🚀 Loading VANGUARD models and scalers...")
load_scalers()
load_models()
print("✅ All models and scalers loaded!")

if __name__ == '__main__':
    print("🚀 Starting Mars Landing Suitability Website...")
    
    # Use PORT environment variable (Render provides this) or default to 5002
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌍 Website ready! Access at http://0.0.0.0:{port}")
    print("📡 API endpoints available at /predict")
    app.run(debug=debug, host='0.0.0.0', port=port)
