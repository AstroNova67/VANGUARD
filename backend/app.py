import logging
import mimetypes
import os
import pickle
import sys
import warnings

# Pickled scalers may be one sklearn micro-version off from the installed wheel; suppress noisy UI.
try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Quieter TensorFlow C++ logs (must be set before importing tensorflow).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras


def _verbose() -> bool:
    return os.environ.get("VANGUARD_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def _log_debug(msg: str) -> None:
    if _verbose():
        print(msg, flush=True)


def _log_info(msg: str) -> None:
    print(msg, flush=True)


def _log_error(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

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

# Default: quiet Werkzeug (set VANGUARD_VERBOSE=1 for request logs).
if not _verbose():
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

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


def _finite_number(x) -> bool:
    if x is None:
        return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f))


def _fuse_surface_temp_for_score(nn_val, xgb_val, mars_data: dict) -> tuple:
    """
    Choose NN vs XGB for landing score.
    When raster `temperature` is present in the payload, pick whichever prediction is
    closest to it (among physically plausible candidates). Otherwise prefer XGB when
    in range, matching the previous API behavior.
    Returns (value_for_score, overrides_applied_key_or_None).
    """
    obs = mars_data.get("temperature")
    nn_ok = _finite_number(nn_val)
    xgb_ok = _finite_number(xgb_val) and -200.0 <= float(xgb_val) <= 50.0

    if _finite_number(obs):
        obs_f = float(obs)
        candidates = []
        if nn_ok:
            candidates.append(("nn", float(nn_val), abs(float(nn_val) - obs_f)))
        if xgb_ok:
            candidates.append(("xgb", float(xgb_val), abs(float(xgb_val) - obs_f)))
        if not candidates:
            return (float(nn_val) if nn_ok else 0.0, None)
        # Min error; tie-break prefers XGB
        candidates.sort(key=lambda t: (t[2], 0 if t[0] == "xgb" else 1))
        winner, val, _ = candidates[0]
        src = "surface_temp_xgb" if winner == "xgb" else "surface_temp_nn"
        return (val, src)

    if xgb_ok:
        return (float(xgb_val), "surface_temp_xgb")
    return (float(nn_val) if nn_ok else 0.0, None)


def _fuse_thermal_inertia_for_score(nn_val, xgb_val, mars_data: dict) -> tuple:
    """
    Same pattern as surface temp. Uses optional `thermalInertia` or `thermal_inertia`
    in the request when present (not sent by the current globe UI); otherwise XGB when
    in [50, 2000], else neural.
    """
    obs = mars_data.get("thermalInertia")
    if obs is None:
        obs = mars_data.get("thermal_inertia")
    nn_ok = _finite_number(nn_val)
    xgb_ok = _finite_number(xgb_val) and 50.0 <= float(xgb_val) <= 2000.0

    if _finite_number(obs):
        obs_f = float(obs)
        candidates = []
        if nn_ok:
            candidates.append(("nn", float(nn_val), abs(float(nn_val) - obs_f)))
        if xgb_ok:
            candidates.append(("xgb", float(xgb_val), abs(float(xgb_val) - obs_f)))
        if not candidates:
            return (float(nn_val) if nn_ok else 0.0, None)
        candidates.sort(key=lambda t: (t[2], 0 if t[0] == "xgb" else 1))
        src = "thermal_inertia_xgb" if candidates[0][0] == "xgb" else "thermal_inertia_nn"
        return (candidates[0][1], src)

    if xgb_ok:
        return (float(xgb_val), "thermal_inertia_xgb")
    return (float(nn_val) if nn_ok else 0.0, None)


def load_scalers():
    """Load all saved scalers and transformers"""
    global scalers
    scaler_dir = os.path.join(BASE_DIR, 'saved_models', 'scalers')
    
    try:
        # Load slope scaler
        with open(f'{scaler_dir}/slope_scaler.pkl', 'rb') as f:
            scalers['slope'] = pickle.load(f)
        _log_debug("  scaler: slope")
        
        # Load dust scalers
        with open(f'{scaler_dir}/dust_feature_scaler.pkl', 'rb') as f:
            scalers['dust_feature'] = pickle.load(f)
        with open(f'{scaler_dir}/dust_target_transformer.pkl', 'rb') as f:
            scalers['dust_target'] = pickle.load(f)
        _log_debug("  scaler: dust (feature + target)")
        
        # Load surface temperature scaler
        with open(f'{scaler_dir}/surface_temp_scaler.pkl', 'rb') as f:
            scalers['surface_temp'] = pickle.load(f)
        with open(f'{scaler_dir}/surface_temp_y_min.pkl', 'rb') as f:
            scalers['surface_temp_y_min'] = pickle.load(f)
        _log_debug("  scaler: surface_temp")
        
        # Load thermal inertia transformers
        with open(f'{scaler_dir}/thermal_inertia_feature_transformer.pkl', 'rb') as f:
            scalers['thermal_inertia_feature'] = pickle.load(f)
        with open(f'{scaler_dir}/thermal_inertia_target_transformer.pkl', 'rb') as f:
            scalers['thermal_inertia_target'] = pickle.load(f)
        _log_debug("  scaler: thermal_inertia")
        
        # Load water scaler
        with open(f'{scaler_dir}/water_scaler.pkl', 'rb') as f:
            scalers['water'] = pickle.load(f)
        _log_debug("  scaler: water")
        
        _log_info(f"  Scalers loaded ({len(scalers)} items)")
        
    except Exception as e:
        _log_error(f"Error loading scalers: {e}")
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
            _log_debug(f"  neural net: {model_name}")
        except Exception as e:
            _log_error(f"Failed to load neural net '{model_name}': {e}")
    
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
        
        _log_debug("  regression: surface_temp + thermal_inertia (XGB)")
    except Exception as e:
        _log_error(f"Error loading regression models: {e}")
        regression_models = {}
    
    reg_ok = bool(regression_models)
    _log_info(
        f"  Neural nets: {len(neural_models)}/5 loaded"
        + ("; regression: ok" if reg_ok else "; regression: unavailable")
    )

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
        _log_error(f"Error in predict_with_neural_networks: {e}")
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
        _log_debug("Using mock regression predictions (models not loaded)")
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
                _log_error(f"Error predicting surface_temp with XGB: {e}")
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
                _log_error(f"Error predicting thermal_inertia with XGB: {e}")
                predictions['thermal_inertia_xgb'] = 0.0
    
    return predictions

@app.route('/predict', methods=['POST'])
def predict_landing_suitability():
    """Main API endpoint for landing suitability prediction"""
    try:
        # Check if models are loaded
        if not models_loaded:
            _log_error("Models not loaded yet (503)")
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
        
        # Pure neural-network outputs (clamped for display; no XGB override).
        nn_baseline = _clamp_predictions(predict_with_neural_networks(mars_data))

        # Fused predictions: start from NN, optionally replace temp / TI with XGB when plausible.
        fused_predictions = dict(nn_baseline)
        reg_predictions = predict_with_regression_models(mars_data)

        overrides_applied = {}

        surface_temp_xgb = reg_predictions.get("surface_temp_xgb", None)
        st_val, st_src = _fuse_surface_temp_for_score(
            nn_baseline.get("surface_temp"),
            surface_temp_xgb,
            mars_data,
        )
        fused_predictions["surface_temp"] = st_val
        if st_src is not None:
            overrides_applied["surface_temp"] = st_src

        thermal_inertia_xgb = reg_predictions.get("thermal_inertia_xgb", None)
        ti_val, ti_src = _fuse_thermal_inertia_for_score(
            nn_baseline.get("thermal_inertia"),
            thermal_inertia_xgb,
            mars_data,
        )
        fused_predictions["thermal_inertia"] = ti_val
        if ti_src is not None:
            overrides_applied["thermal_inertia"] = ti_src

        fused_predictions = _clamp_predictions(fused_predictions)
        
        # Landing score: LandingSuitabilityScorer matches LANDING_SCORING_SOURCES.md
        # (weights 30/20/20/20/10, normalization ranges, slope/dust inverted).
        # Inputs are the final fused + clamped property predictions used in the API response.
        scorer = LandingSuitabilityScorer()
        landing_score = scorer.score_site(
            slope=fused_predictions.get('slope', 0),
            dust=fused_predictions.get('dust', 0),
            surface_temp=fused_predictions.get('surface_temp', 0),
            thermal_inertia=fused_predictions.get('thermal_inertia', 0),
            water=fused_predictions.get('water', 0)
        )

        lat = mars_data.get("lat")
        lon = mars_data.get("lon")
        loc = ""
        if lat is not None and lon is not None:
            loc = f" lat={lat} lon={lon}"
        _log_info(f"POST /predict → landing_score={landing_score}%{loc}")
        _log_debug(f"  fused: {fused_predictions}")
        _log_debug(f"  regression: {reg_predictions} overrides={overrides_applied}")
        
        # Prepare response
        response = {
            'success': True,
            'landing_score': landing_score,
            'predictions': {
                # Fused values used for landing_score (XGB may override temp / thermal inertia).
                'neural_networks': fused_predictions,
                # Neural nets only (before XGB override), for side-by-side UI.
                'neural_networks_baseline': nn_baseline,
                'regression_models': reg_predictions
            },
            'overrides_applied': overrides_applied,
            'raw_mars_data': mars_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        _log_error(f"Error in prediction: {e}\n{error_trace}")
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
    _log_error(f"Unhandled exception: {e}\n{error_trace}")
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
        _log_error(f"Error checking file path security: {e}")
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
        
        _log_debug(f"Serving static: {path}")
        return send_file(file_path, mimetype=mimetype)
    
    # If it's a directory, try to serve index.html from it (for SPA routing)
    if os.path.isdir(file_path):
        index_path = os.path.join(file_path, 'index.html')
        if os.path.exists(index_path):
            return send_file(index_path)
    
    # File not found - log for debugging
    _log_debug(f"File not found: {path}")
    return jsonify({'error': 'File not found', 'path': path}), 404

# Load models and scalers when module is imported (works with both Flask dev server and gunicorn)
# This ensures models are loaded in production (gunicorn) where __main__ doesn't run
_log_info("VANGUARD — loading models (this may take a minute)…")
try:
    load_scalers()
    load_models()
    models_loaded = True
    _log_info("VANGUARD — models ready.")
except Exception as e:
    _log_error(f"VANGUARD — failed to load models: {e}")
    models_loaded = False

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV') != 'production'
    _log_info("")
    _log_info("  Open in browser:  http://127.0.0.1:" + str(port))
    _log_info("  Predict endpoint: POST http://127.0.0.1:" + str(port) + "/predict")
    if _verbose():
        _log_info("  (VANGUARD_VERBOSE=1 — debug logging on)")
    else:
        _log_info("  Tip: set VANGUARD_VERBOSE=1 for detailed logs and HTTP access logs.")
    _log_info("  Press Ctrl+C to stop.\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
