from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras
import pickle
import json
import os
from scoring import LandingSuitabilityScorer, predict_properties_nn

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables to store loaded models
neural_models = {}
regression_models = {}

def load_models():
    """Load all ML models at startup"""
    global neural_models, regression_models
    
    neural_models = {}
    regression_models = {}
    
    # Try to load Neural Network models one by one
    model_paths = {
        'slope': 'saved_models/neural_nets/slope_pred/best_model.keras',
        'dust': 'saved_models/neural_nets/dust_predictor/best_model.keras',
        'surface_temp': 'saved_models/neural_nets/surface_temp_pred/best_model.keras',
        'thermal_inertia': 'saved_models/neural_nets/thermal_inertia_predictor/best_model.keras',
        'water': 'saved_models/neural_nets/water_predictor/best_model.keras'
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
        regression_models['surface_temp']['xgb'].load_model('saved_models/regression_models/surface_temp/xgb_model.json')
        regression_models['thermal_inertia']['xgb'].load_model('saved_models/regression_models/thermal_inertia/xgb_model.json')
        
        print("✅ Regression models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading regression models: {e}")
        regression_models = {}
    
    print(f"📊 Loaded {len(neural_models)} neural network models and {len(regression_models)} regression model types")

def map_mars_data_to_features(mars_data, model_name):
    """
    Map Mars dataset values to ML model input features
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
        # Apply normalization
        normalized_features = [
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
        # Apply normalization
        normalized_features = [
            features[0] / 10000.0,          # Elevation (scale down)
            features[1] / 10.0,             # Slope (scale down)
            features[2] / 100.0,            # Temperature (scale down)
            features[3] / 100.0,            # Thermal Inertia (scale down)
            features[4] / 10.0,             # Aspect (scale down)
            features[5]                     # Albedo (already normalized)
        ]
        
    elif model_name == 'surface_temp':
        # Surface temp model: 5 features
        # ['Elevation', 'Albedo', 'Day Side Thermal Inertia', 'Slope', 'Roughness 0.6km']
        features = [
            elevation,                 # Elevation
            albedo,                    # Albedo
            temperature,               # Day Side Thermal Inertia (using temperature as proxy)
            slope,                     # Slope
            roughness                   # Roughness 0.6km
        ]
        # Apply normalization
        normalized_features = [
            features[0] / 10000.0,          # Elevation (scale down)
            features[1],                     # Albedo (already normalized)
            features[2] / 100.0,            # Thermal Inertia (scale down)
            features[3] / 10.0,             # Slope (scale down)
            features[4] / 1000.0             # Roughness (scale down)
        ]
        
    elif model_name == 'thermal_inertia':
        # Thermal inertia model: 4 features
        # ['Yearly Mars Surface Temperature Variation (C)', 'Albedo', 'Slope', 'OMEGA Ferric/Dust 860nm ratio']
        features = [
            temp_range,                # Yearly Mars Surface Temperature Variation (C)
            albedo,                    # Albedo
            slope,                     # Slope
            ferric                     # OMEGA Ferric/Dust 860nm ratio
        ]
        # Apply normalization
        normalized_features = [
            features[0] / 200.0,            # Temp variation (scale down)
            features[1],                     # Albedo (already normalized)
            features[2] / 10.0,             # Slope (scale down)
            features[3]                     # Ferric ratio (keep as is)
        ]
        
    elif model_name == 'water':
        # Water model: 8 features (same as slope for now)
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
        # Apply normalization
        normalized_features = [
            features[0],                    # Albedo (0-1, already normalized)
            features[1] / 100.0,            # Temperature (scale to roughly -1 to 1)
            features[2] / 1000.0,           # Roughness (scale down)
            features[3],                    # Ferric ratio (keep as is)
            features[4] / 10000.0,          # Elevation (scale down)
            features[5] / 200.0,            # Temp range (scale down)
            features[6] / 10.0,             # Slope diff (scale down)
            features[7] / 10.0              # Slope diff2 (scale down)
        ]
    
    else:
        # Default fallback
        normalized_features = [0.0] * 8
    
    return np.array(normalized_features).reshape(1, -1)

def predict_with_neural_networks(mars_data):
    """Use neural networks to predict properties"""
    predictions = {}
    
    if len(neural_models) == 0:
        # Provide mock predictions when models fail to load
        print("⚠️ Using mock predictions (models not loaded)")
        predictions = {
            'slope': 5.2,
            'dust': 0.15,
            'surface_temp': -45.3,
            'thermal_inertia': 450.7,
            'water': 0.08
        }
    else:
        for model_name, model in neural_models.items():
            try:
                # Get model-specific features
                features = map_mars_data_to_features(mars_data, model_name)
                pred = model.predict(features, verbose=0)[0][0]
                predictions[model_name] = float(pred)
            except Exception as e:
                print(f"Error predicting {model_name}: {e}")
                predictions[model_name] = 0.0
    
    return predictions

def predict_with_regression_models(mars_data):
    """Use regression models to predict properties"""
    predictions = {}
    
    if len(regression_models) == 0:
        # Provide mock predictions when models fail to load
        print("⚠️ Using mock regression predictions (models not loaded)")
        predictions = {
            'surface_temp_xgb': -44.8,
            'thermal_inertia_xgb': 445.2
        }
    else:
        # Surface temperature predictions
        if 'surface_temp' in regression_models:
            try:
                features = map_mars_data_to_features(mars_data, 'surface_temp')
                xgb_pred = regression_models['surface_temp']['xgb'].predict(features)[0]
                predictions['surface_temp_xgb'] = float(xgb_pred)
            except Exception as e:
                print(f"Error predicting surface_temp with XGB: {e}")
                predictions['surface_temp_xgb'] = 0.0
        
        # Thermal inertia predictions
        if 'thermal_inertia' in regression_models:
            try:
                features = map_mars_data_to_features(mars_data, 'thermal_inertia')
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

if __name__ == '__main__':
    print("🚀 Starting Mars Landing Suitability API...")
    load_models()
    print("🌍 API ready! Access at http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)
