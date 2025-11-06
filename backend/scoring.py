import numpy as np
import pickle
import os
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import tensorflow as tf


# Global variable to store loaded scalers
scalers = {}

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_scalers():
    """Load all saved scalers and transformers"""
    global scalers
    scaler_dir = os.path.join(BASE_DIR, 'saved_models', 'scalers')
    
    try:
        # Load slope scaler
        with open(f'{scaler_dir}/slope_scaler.pkl', 'rb') as f:
            scalers['slope'] = pickle.load(f)
        
        # Load dust scalers
        with open(f'{scaler_dir}/dust_feature_scaler.pkl', 'rb') as f:
            scalers['dust_feature'] = pickle.load(f)
        with open(f'{scaler_dir}/dust_target_transformer.pkl', 'rb') as f:
            scalers['dust_target'] = pickle.load(f)
        
        # Load surface temperature scaler
        with open(f'{scaler_dir}/surface_temp_scaler.pkl', 'rb') as f:
            scalers['surface_temp'] = pickle.load(f)
        with open(f'{scaler_dir}/surface_temp_y_min.pkl', 'rb') as f:
            scalers['surface_temp_y_min'] = pickle.load(f)
        
        # Load thermal inertia transformers
        with open(f'{scaler_dir}/thermal_inertia_feature_transformer.pkl', 'rb') as f:
            scalers['thermal_inertia_feature'] = pickle.load(f)
        with open(f'{scaler_dir}/thermal_inertia_target_transformer.pkl', 'rb') as f:
            scalers['thermal_inertia_target'] = pickle.load(f)
        
        # Load water scaler
        with open(f'{scaler_dir}/water_scaler.pkl', 'rb') as f:
            scalers['water'] = pickle.load(f)
        
        print(f"✅ Loaded {len(scalers)} scalers and transformers")
        
    except Exception as e:
        print(f"❌ Error loading scalers: {e}")
        scalers = {}

class LandingSuitabilityScorer:
    """
    Expert system for scoring Mars landing site suitability.
    
    Scoring criteria and weights are based on NASA/JPL engineering constraints and
    scientific objectives from official Mars mission landing site selection processes.
    
    Primary Sources:
    - Golombek et al. (2012). "Selection of the Mars Science Laboratory landing site."
    - Golombek et al. (2003). "Selection of the Mars Exploration Rover landing sites."
    - NASA/JPL-Caltech (2018). "Mars in a Minute: How Do You Choose a Landing Site?"
    - NASA. "Mars Landing Site Selection: A Crew Perspective."
    
    Weight Distribution:
    - Slope (30%): Critical for rover stability at touchdown (<30° constraint)
    - Dust (20%): Avoid dust-dominated surfaces for safe landing and roving
    - Surface Temperature (20%): Thermal management constraint (±30° latitude)
    - Thermal Inertia (20%): Indicates surface stability and load-bearing capacity
    - Water (10%): Scientific interest (secondary to engineering safety)
    
    For detailed source citations, excerpts, and justification, see:
    backend/LANDING_SCORING_SOURCES.md
    """
    def __init__(self, weights=None):
        default_weights = {
            "slope": 0.3,
            "dust": 0.2,
            "surface_temp": 0.2,
            "thermal_inertia": 0.2,
            "water": 0.1
        }
        self.weights = weights if weights else default_weights

    def normalize(self, value, min_val, max_val, invert=False):
        score = (value - min_val) / (max_val - min_val)
        score = np.clip(score, 0, 1)
        if invert:
            score = 1 - score
        return score

    def score_site(self, slope, dust, surface_temp, thermal_inertia, water):
        """
        Score a landing site based on predicted surface properties.
        
        Scoring ranges based on ML model predictions and NASA engineering constraints.
        See LANDING_SCORING_SOURCES.md for detailed source citations.
        """
        # Slope: <30° constraint for rover stability (Golombek et al., 2012)
        # Range 0-5° selected for discrimination within safe zone
        slope_score = self.normalize(slope, 0, 5, invert=True)  # ML gives 0.7-4.8°, so use 0-5°
        
        # Dust: Avoid dust-dominated surfaces (multiple NASA sources)
        # Lower dust = better surface stability and load-bearing capacity
        dust_score = self.normalize(dust, 0.6, 0.7, invert=True)  # ML gives 0.64-0.70, so use 0.6-0.7
        
        # Temperature: Thermal management constraint (±30° latitude)
        # Warmer temperatures better for instrument operation and power efficiency
        temp_score = self.normalize(surface_temp, -90, -40, invert=False)  # ML gives -40 to -90°C, so use -90 to -40°C
        
        # Thermal Inertia: Higher = more stable, rocky surface with better load-bearing
        # Indicates surface trafficability and stability
        inertia_score = self.normalize(thermal_inertia, 100, 400, invert=False)  # ML gives 100-400, so use 100-400
        
        # Water: Scientific interest (secondary to engineering safety)
        # Higher water content indicates scientific value but not safety-critical
        water_score = self.normalize(water, 1, 8, invert=False)  # ML gives 1-8%, so use 1-8%

        final_score = (
                slope_score * self.weights["slope"] +
                dust_score * self.weights["dust"] +
                temp_score * self.weights["surface_temp"] +
                inertia_score * self.weights["thermal_inertia"] +
                water_score * self.weights["water"]
        )

        return round(final_score * 100, 2)


# Example: Using Neural Network predictions
def map_mars_data_to_features(mars_data, model_name):
    """Map Mars data to model-specific features with proper scaling using saved scalers"""
    # Load scalers if not already loaded
    if not scalers:
        load_scalers()
    
    # Extract raw data
    albedo = mars_data.get('albedo', 0.2)
    temperature = mars_data.get('temperature', -30.0)
    roughness = mars_data.get('roughness', 50.0)
    ferric = mars_data.get('ferric', 0.5)
    elevation = mars_data.get('elevation', 1000.0)
    temp_range = mars_data.get('tempRange', 50.0)
    slope = mars_data.get('slope', 2.0)
    
    if model_name == 'slope':
        features = [albedo, temperature, roughness, ferric, elevation, temp_range, abs(slope), abs(slope * 0.1)]
        if 'slope' in scalers:
            normalized_features = scalers['slope'].transform([features])
        else:
            # Fallback to manual scaling if scaler not available
            normalized_features = [[features[0], features[1] / 100.0, features[2] / 1000.0, features[3], features[4] / 10000.0, features[5] / 200.0, features[6] / 10.0, features[7] / 10.0]]
    elif model_name == 'dust':
        features = [elevation, slope, temperature, temperature, slope, albedo]
        if 'dust_feature' in scalers:
            normalized_features = scalers['dust_feature'].transform([features])
        else:
            # Fallback to manual scaling if scaler not available
            normalized_features = [[features[0] / 10000.0, features[1] / 10.0, features[2] / 100.0, features[3] / 100.0, features[4] / 10.0, features[5]]]
    elif model_name == 'surface_temp':
        # Surface temp model: 5 features - ['Elevation', 'Albedo', 'Day Side Thermal Inertia', 'Slope', 'Roughness 0.6km']
        features = [elevation, albedo, temperature, slope, roughness]
        if 'surface_temp' in scalers:
            normalized_features = scalers['surface_temp'].transform([features])
        else:
            # Fallback to manual scaling if scaler not available
            normalized_features = [[features[0] / 10000.0, features[1], features[2] / 100.0, features[3] / 10.0, features[4] / 1000.0]]
    elif model_name == 'thermal_inertia':
        # Thermal inertia model: 4 features - ['Yearly Mars Surface Temperature Variation (C)', 'Albedo', 'Slope', 'OMEGA Ferric/Dust 860nm ratio']
        features = [temp_range, albedo, slope, ferric]
        if 'thermal_inertia_feature' in scalers:
            normalized_features = scalers['thermal_inertia_feature'].transform([features])
        else:
            # Fallback to manual scaling if scaler not available
            normalized_features = [[features[0] / 200.0, features[1], features[2] / 10.0, features[3]]]
    elif model_name == 'water':
        features = [albedo, temperature, roughness, ferric, elevation, temp_range, abs(slope), abs(slope * 0.1)]
        if 'water' in scalers:
            normalized_features = scalers['water'].transform([features])
        else:
            # Fallback to manual scaling if scaler not available
            normalized_features = [[features[0], features[1] / 100.0, features[2] / 1000.0, features[3], features[4] / 10000.0, features[5] / 200.0, features[6] / 10.0, features[7] / 10.0]]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return np.array(normalized_features)

def inverse_transform_predictions(slope_pred, dust_pred, temp_pred, TI_pred, water_pred):
    """
    Convert raw ML predictions back to realistic Mars values using exact inverse transformations
    """
    # Slope: log1p transform -> expm1 inverse
    slope_real = max(0, np.expm1(slope_pred))
    
    # Dust: QuantileTransformer inverse
    if 'dust_target' in scalers:
        dust_real = scalers['dust_target'].inverse_transform([[dust_pred]])[0][0]
    else:
        # Fallback approximation
        dust_real = max(0, min(1, 0.5 + dust_pred * 0.2))
    
    # Surface Temperature: log1p + shift inverse
    if 'surface_temp_y_min' in scalers:
        y_min = scalers['surface_temp_y_min']
        temp_real = np.expm1(temp_pred) + y_min - 1
    else:
        # Fallback approximation
        temp_real = np.expm1(temp_pred) - 100 - 1
    
    # Thermal Inertia: QuantileTransformer inverse
    if 'thermal_inertia_target' in scalers:
        TI_real = scalers['thermal_inertia_target'].inverse_transform([[TI_pred]])[0][0]
    else:
        # Fallback approximation
        TI_real = max(100, min(1200, 400 + TI_pred * 200))
    
    # Water: log1p transform -> expm1 inverse
    water_real = max(0, np.expm1(water_pred))
    
    return slope_real, dust_real, temp_real, TI_real, water_real

def predict_properties_nn(mars_data):
    """
    mars_data: dict with Mars surface data
    returns: predicted property values using model-specific features with proper inverse transforms
    """
    # Load scalers if not already loaded
    if not scalers:
        load_scalers()
    try:
        # Get raw model predictions
        slope_pred_raw = tf.keras.models.load_model(os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'slope_pred', 'best_model.keras')).predict(
            map_mars_data_to_features(mars_data, 'slope'))[0][0]
        dust_pred_raw = tf.keras.models.load_model(os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'dust_predictor', 'best_model.keras')).predict(
            map_mars_data_to_features(mars_data, 'dust'))[0][0]
        temp_pred_raw = tf.keras.models.load_model(os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'surface_temp_pred', 'best_model.keras')).predict(
            map_mars_data_to_features(mars_data, 'surface_temp'))[0][0]
        TI_pred_raw = tf.keras.models.load_model(os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'thermal_inertia_predictor', 'best_model.keras')).predict(
            map_mars_data_to_features(mars_data, 'thermal_inertia'))[0][0]
        water_pred_raw = tf.keras.models.load_model(os.path.join(BASE_DIR, 'saved_models', 'neural_nets', 'water_predictor', 'best_model.keras')).predict(
            map_mars_data_to_features(mars_data, 'water'))[0][0]
        
        # Apply inverse transformations
        slope_pred, dust_pred, temp_pred, TI_pred, water_pred = inverse_transform_predictions(
            slope_pred_raw, dust_pred_raw, temp_pred_raw, TI_pred_raw, water_pred_raw
        )
        
        return slope_pred, dust_pred, temp_pred, TI_pred, water_pred
    except Exception as e:
        print(f"Error in predict_properties_nn: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0


# Example usage (commented out to prevent execution on import)
# nn_models = {"slope": slope_nn, "dust": dust_nn, "temp": temp_nn, ...}
# features_for_point = [/* your input features at the clicked location */]

# Predict values from neural networks
# pred_slope, pred_dust, pred_temp, pred_TI, pred_water = predict_properties_nn(features_for_point)

# Compute landing score
# scorer = LandingSuitabilityScorer()
# landing_score = scorer.score_site(pred_slope, pred_dust, pred_temp, pred_TI, pred_water)
# print(f"Predicted Landing Score (NN-based): {landing_score}%")

