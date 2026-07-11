import asyncio
import json
import logging
import mimetypes
import os
import pickle
import sys
import traceback
import warnings

from dotenv import load_dotenv

# Pickled scalers may be one sklearn micro-version off from the installed wheel; suppress noisy UI.
try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Quieter TensorFlow C++ logs (must be set before importing tensorflow).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras

# Suppress noisy Python-side TF warnings (e.g. tf.function retracing on repeated predict shapes).
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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
except Exception:
    pass

# Set TensorFlow to use less memory
tf.config.experimental.enable_op_determinism()
# Disable eager execution optimizations that use extra memory
tf.config.run_functions_eagerly(False)

# Handle imports for both local development and production (Render)
# Try absolute import first (for Render), fall back to relative import (for local)

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

load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

try:
    from backend.landing_predict import compute_landing_prediction
except ImportError:
    from landing_predict import compute_landing_prediction

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
    
    # Keras-style terminal bar (same widget as training); load_model itself has no native progress.
    _paths = list(model_paths.items())
    _prog = tf.keras.utils.Progbar(len(_paths), unit_name="model")
    for i, (model_name, model_path) in enumerate(_paths):
        try:
            neural_models[model_name] = keras.models.load_model(model_path)
            _log_debug(f"  Keras neural net loaded: {model_name} ({model_path})")
        except Exception as e:
            _log_error(f"Failed to load neural net '{model_name}': {e}")
        _prog.update(i + 1)
    
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
        
        _log_info("  XGBoost regression loaded: surface_temp + thermal_inertia")
    except Exception as e:
        _log_error(f"Error loading regression models: {e}")
        regression_models = {}
    
    reg_ok = bool(regression_models)
    _log_info(
        f"  Neural nets: {len(neural_models)}/5 loaded"
        + ("; regression: ok" if reg_ok else "; regression: unavailable")
    )

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
        
        body = request.get_json()
        if body is None:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON in request body',
                'landing_score': 0
            }), 400

        try:
            from backend.landing_predict import split_predict_payload
        except ImportError:
            from landing_predict import split_predict_payload

        try:
            mars_data, scoring_weights = split_predict_payload(body)
        except ValueError as ve:
            return jsonify({
                'success': False,
                'error': str(ve),
                'landing_score': 0,
            }), 400

        response = compute_landing_prediction(
            mars_data,
            models_loaded=models_loaded,
            regression_models=regression_models,
            scoring_weights=scoring_weights,
            log_error=_log_error,
            log_debug=_log_debug,
        )
        if not response.get("success"):
            status = 503 if "loading" in str(response.get("error", "")).lower() else 500
            return jsonify(response), status

        fused = (response.get("predictions") or {}).get("neural_networks") or {}
        overrides_applied = response.get("overrides_applied") or {}
        data_sources = response.get("data_sources") or {}
        landing_score = response.get("landing_score")
        lat = mars_data.get("lat")
        lon = mars_data.get("lon")
        loc = ""
        if lat is not None and lon is not None:
            loc = f" lat={lat} lon={lon}"
        st_src_log = overrides_applied.get("surface_temp") or data_sources.get("surface_temp")
        ti_src_log = overrides_applied.get("thermal_inertia") or data_sources.get(
            "thermal_inertia"
        )
        _log_info(
            f"POST /predict → landing_score={landing_score}%{loc} | "
            f"score_inputs slope={fused.get('slope')} dust={fused.get('dust')} "
            f"surface_temp={fused.get('surface_temp')} (src={st_src_log}) "
            f"thermal_inertia={fused.get('thermal_inertia')} (src={ti_src_log}) "
            f"water={fused.get('water')}"
        )
        _log_debug(f"  fused: {fused}")
        _log_debug(f"  data_sources: {data_sources}")
        _log_debug(f"  overrides={overrides_applied}")

        return jsonify(response)
        
    except Exception as e:
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
    error_trace = traceback.format_exc()
    _log_error(f"Unhandled exception: {e}\n{error_trace}")
    return jsonify({
        'success': False,
        'error': str(e),
        'landing_score': 0
    }), 500

@app.route('/scoring/weights', methods=['GET'])
def get_scoring_weights_defaults():
    """Research-based default landing suitability weights (fractions and percents)."""
    try:
        from backend.scoring import (
            DEFAULT_SCORING_WEIGHTS,
            SCORING_WEIGHT_KEYS,
            scoring_weights_as_percent,
        )
    except ImportError:
        from scoring import (
            DEFAULT_SCORING_WEIGHTS,
            SCORING_WEIGHT_KEYS,
            scoring_weights_as_percent,
        )
    return jsonify({
        'success': True,
        'keys': list(SCORING_WEIGHT_KEYS),
        'weights': DEFAULT_SCORING_WEIGHTS,
        'weights_percent': scoring_weights_as_percent(DEFAULT_SCORING_WEIGHTS),
        'description': (
            'Customize via POST /predict or /agent/chat using scoring_weights. '
            'Values may be fractions (0–1) or percents (0–100); they are renormalized to sum to 1.'
        ),
    })


@app.route('/agent/chat', methods=['POST'])
def agent_chat():
    """Chat with the VANGUARD Mars assistant (OpenAI Agents SDK)."""
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
    body = request.get_json() or {}
    message = body.get('message') or body.get('input')
    if not message or not str(message).strip():
        return jsonify({'success': False, 'error': 'message is required'}), 400
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({
            'success': False,
            'error': 'OPENAI_API_KEY is not set. Add it to .env at the repo root.',
        }), 503
    try:
        try:
            from backend.agent import run_agent_turn
        except ImportError:
            from agent import run_agent_turn
        scoring_weights = body.get('scoring_weights') or body.get('scoringWeights')
        out = asyncio.run(
            run_agent_turn(str(message).strip(), scoring_weights=scoring_weights)
        )
        reply = out.get('reply', '')
        if not isinstance(reply, str):
            reply = json.dumps(reply, ensure_ascii=False) if isinstance(reply, (dict, list)) else str(reply)
        ui_actions = out.get('ui_actions') or []
        if _verbose():
            _log_debug(
                f"  /agent/chat reply_chars={len(reply)} ui_actions={len(ui_actions)}"
            )
        structured = out.get('structured')
        payload = {
            'success': True,
            'reply': reply,
            'ui_actions': ui_actions,
        }
        if structured is not None:
            payload['structured'] = structured
        return jsonify(payload)
    except Exception as e:
        error_trace = traceback.format_exc()
        _log_error(f"Error in /agent/chat: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 500


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
    return send_file(
        os.path.join(FRONTEND_DIR, 'index.html'),
        max_age=0,
        conditional=False,
    )

@app.route('/<path:path>')
def serve_frontend(path):
    """Serve frontend static files (JS, CSS, textures, data, etc.)"""
    # Don't serve API routes through this static handler (allow e.g. agent-chat.js)
    if path in ('predict', 'health', 'models', 'scoring/weights') or path.startswith('agent/chat'):
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
        # Avoid stale module/HTML/rasters during local iteration on the globe UI.
        max_age = 0 if path.lower().endswith((".html", ".js", ".css", ".tif", ".tiff")) else None
        return send_file(
            file_path,
            mimetype=mimetype,
            max_age=max_age,
            conditional=False if max_age == 0 else True,
        )
    
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
    _log_info("  Agent chat:       POST http://127.0.0.1:" + str(port) + "/agent/chat")
    if _verbose():
        _log_info("  (VANGUARD_VERBOSE=1 — debug logging on)")
    else:
        _log_info("  Tip: set VANGUARD_VERBOSE=1 for detailed logs and HTTP access logs.")
    _log_info("  Press Ctrl+C to stop.\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
