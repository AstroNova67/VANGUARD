from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras
import pickle
import json
import os
import sys
import mimetypes

# Optimize TensorFlow memory usage for production
# Limit GPU memory growth (if GPU available)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# Set TensorFlow to use less memory
tf.config.experimental.enable_op_determinism()
# Disable eager execution optimizations that use extra memory
tf.config.run_functions_eagerly(False)

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
models_loaded = False  # Flag to track if models are loaded (for lazy loading)

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
        print("📥 Received prediction request")
        
        # Check if models are loaded
        if not models_loaded:
            print("⚠️ Models not loaded yet")
            return jsonify({
                'success': False,
                'error': 'Models are still loading. Please try again in a moment.',
                'landing_score': 0
            }), 503
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON',
                'landing_score': 0
            }), 400
        
        # Get Mars data from frontend
        mars_data = request.get_json()
        if mars_data is None:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON in request body',
                'landing_score': 0
            }), 400
        
        print(f"Received Mars data: {mars_data}")
        
        # Get predictions from neural networks
        nn_predictions = predict_with_neural_networks(mars_data)
        print(f"Neural network predictions: {nn_predictions}")
        
        # Get predictions from regression models
        reg_predictions = predict_with_regression_models(mars_data)
        print(f"Regression predictions: {reg_predictions}")

        # Override specific targets with regressors only when values are plausible.
        # (Some regressor outputs can be out-of-distribution / negative and get clamped to 0.)
        overrides_applied = {}

        surface_temp_xgb = reg_predictions.get('surface_temp_xgb', None)
        if surface_temp_xgb is not None and np.isfinite(surface_temp_xgb):
            # Mars surface temperatures are typically well below 0°C; keep a generous bound.
            if -200.0 <= float(surface_temp_xgb) <= 50.0:
                nn_predictions['surface_temp'] = float(surface_temp_xgb)
                overrides_applied['surface_temp'] = 'surface_temp_xgb'

        thermal_inertia_xgb = reg_predictions.get('thermal_inertia_xgb', None)
        if thermal_inertia_xgb is not None and np.isfinite(thermal_inertia_xgb):
            # Thermal inertia should not be negative; scoring expects ~100–400.
            if 50.0 <= float(thermal_inertia_xgb) <= 2000.0:
                nn_predictions['thermal_inertia'] = float(thermal_inertia_xgb)
                overrides_applied['thermal_inertia'] = 'thermal_inertia_xgb'

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
            'overrides_applied': overrides_applied,
            'raw_mars_data': mars_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {e}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'landing_score': 0
        }), 500

# Global error handler to ensure all errors return JSON
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_trace = traceback.format_exc()
    print(f"Unhandled exception: {e}")
    print(f"Traceback: {error_trace}")
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
        'regression_models_loaded': len(regression_models),
        'models_loaded': models_loaded
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
    except Exception as e:
        print(f"Error checking file path security: {e}")
        return jsonify({'error': 'Invalid path'}), 400
    
    # If it's a file, serve it
    if os.path.isfile(file_path):
        # Set correct MIME type for TIF files
        mimetype = None
        if file_path.lower().endswith(('.tif', '.tiff')):
            mimetype = 'image/tiff'
        elif file_path.lower().endswith('.js'):
            mimetype = 'application/javascript'
        elif file_path.lower().endswith('.json'):
            mimetype = 'application/json'
        else:
            mimetype, _ = mimetypes.guess_type(file_path)
        
        print(f"Serving file: {path} (mimetype: {mimetype})")
        return send_file(file_path, mimetype=mimetype)
    
    # If it's a directory, try to serve index.html from it (for SPA routing)
    if os.path.isdir(file_path):
        index_path = os.path.join(file_path, 'index.html')
        if os.path.exists(index_path):
            return send_file(index_path)
    
    # File not found - log for debugging
    print(f"File not found: {path} (resolved to: {file_path})")
    return jsonify({'error': 'File not found', 'path': path}), 404

# Load models and scalers when module is imported (works with both Flask dev server and gunicorn)
# This ensures models are loaded in production (gunicorn) where __main__ doesn't run
# Note: This may take 30-60 seconds on Render free tier, but it ensures models are ready
print("🚀 Loading VANGUARD models and scalers...")
try:
    load_scalers()
    load_models()
    models_loaded = True
    print("✅ All models and scalers loaded!")
except Exception as e:
    print(f"❌ Error loading models at startup: {e}")
    models_loaded = False

if __name__ == '__main__':
    print("🚀 Starting Mars Landing Suitability Website...")
    
    # Use PORT environment variable (Render provides this) or default to 5002
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌍 Website ready! Access at http://0.0.0.0:{port}")
    print("📡 API endpoints available at /predict")
    app.run(debug=debug, host='0.0.0.0', port=port)
