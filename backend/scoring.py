import logging
import numpy as np
import pickle
import os
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Global variable to store loaded scalers
scalers = {}

# Neural nets loaded once (used by predict_properties_nn and batch raster jobs)
_nn_models = None

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Raster-first scoring: which JSON/stack fields count as ground truth per scored property ---
DATA_SOURCE_RASTER = "raster"
DATA_SOURCE_ML_PREDICTED = "ml_predicted"

# Keys tried in order (first valid value wins). Must match globe `marsDatasets` / batch stack.
MARS_DATA_RASTER_KEY_PRIORITY = {
    "slope": ("slope",),
    "dust": ("ferric",),  # OMEGA ferric/dust index (same source as UI dustObserved)
    "surface_temp": ("temperature",),
    "thermal_inertia": ("thermalInertia", "thermal_inertia"),
    "water": ("grsWaterWt",),
}


def raster_observation_valid_scalar(value) -> bool:
    """True if a single payload/stack value is finite and not missing."""
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f))


def mars_raster_observation_valid(mars_data: dict, property_name: str) -> bool:
    """Whether `mars_data` carries a usable raster observation for this scored property."""
    keys = MARS_DATA_RASTER_KEY_PRIORITY[property_name]
    for k in keys:
        if raster_observation_valid_scalar(mars_data.get(k)):
            return True
    return False


def mars_raster_value_for_property(mars_data: dict, property_name: str):
    """First valid numeric raster value for `property_name`, or None."""
    for k in MARS_DATA_RASTER_KEY_PRIORITY[property_name]:
        v = mars_data.get(k)
        if raster_observation_valid_scalar(v):
            return float(v)
    return None


def raster_band_valid(arr: np.ndarray, nodata) -> np.ndarray:
    """
    Element-wise mask: finite and not equal to GDAL nodata (when defined).
    Aligns with batch `_per_band_nodata_mask` single-band logic.
    """
    arr = np.asarray(arr, dtype=np.float64)
    ok = np.isfinite(arr)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        ok &= arr != float(nodata)
    return ok


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
    LANDING_SCORING_SOURCES.md (repo root)
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

    def score_site(
        self,
        slope,
        dust,
        surface_temp,
        thermal_inertia,
        water,
        property_sources=None,
    ):
        """
        Score a landing site based on predicted surface properties.

        Scoring ranges based on ML model predictions and NASA engineering constraints.
        See LANDING_SCORING_SOURCES.md for detailed source citations.

        property_sources: optional dict mapping each property name to \"raster\" or
        \"ml_predicted\" (API `data_sources`). Ignored for arithmetic; accepted for
        transparency and future extensions.
        """
        _ = property_sources  # echoed at API layer; rubric unchanged
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

    def score_site_arrays(self, slope, dust, surface_temp, thermal_inertia, water):
        """
        Same rubric as score_site, vectorized over numpy arrays (element-wise).
        Returns float32 percent in [0, 100], shape broadcast from inputs.
        """
        slope = np.asarray(slope, dtype=np.float64)
        dust = np.asarray(dust, dtype=np.float64)
        surface_temp = np.asarray(surface_temp, dtype=np.float64)
        thermal_inertia = np.asarray(thermal_inertia, dtype=np.float64)
        water = np.asarray(water, dtype=np.float64)

        slope_score = self.normalize(slope, 0, 5, invert=True)
        dust_score = self.normalize(dust, 0.6, 0.7, invert=True)
        temp_score = self.normalize(surface_temp, -90, -40, invert=False)
        inertia_score = self.normalize(thermal_inertia, 100, 400, invert=False)
        water_score = self.normalize(water, 1, 8, invert=False)

        final_score = (
            slope_score * self.weights["slope"]
            + dust_score * self.weights["dust"]
            + temp_score * self.weights["surface_temp"]
            + inertia_score * self.weights["thermal_inertia"]
            + water_score * self.weights["water"]
        )
        return np.round(np.clip(final_score, 0, 1) * 100, 2).astype(np.float32)


def get_nn_models():
    """Load all five Keras NNs once and cache them."""
    global _nn_models
    if _nn_models is not None:
        return _nn_models
    _nn_models = {}
    paths = {
        "slope": os.path.join(BASE_DIR, "saved_models", "neural_nets", "slope_pred", "best_model.keras"),
        "dust": os.path.join(BASE_DIR, "saved_models", "neural_nets", "dust_predictor", "best_model.keras"),
        "surface_temp": os.path.join(BASE_DIR, "saved_models", "neural_nets", "surface_temp_pred", "best_model.keras"),
        "thermal_inertia": os.path.join(BASE_DIR, "saved_models", "neural_nets", "thermal_inertia_predictor", "best_model.keras"),
        "water": os.path.join(BASE_DIR, "saved_models", "neural_nets", "water_predictor", "best_model.keras"),
    }
    for name, path in paths.items():
        _nn_models[name] = tf.keras.models.load_model(path)
    return _nn_models


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
    # Surface-temp NN/XGB were trained with Day Side Thermal Inertia in column 3 (see surface_temp_predictor.py).
    # Must match that quantity at inference — not yearly average °C (`temperature`).
    ti_surface_temp = mars_data.get("thermalInertia")
    if ti_surface_temp is None:
        ti_surface_temp = mars_data.get("thermal_inertia")
    if ti_surface_temp is None:
        ti_surface_temp = 300.0
    else:
        ti_surface_temp = float(ti_surface_temp)
    
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
        features = [elevation, albedo, ti_surface_temp, slope, roughness]
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


def inverse_transform_predictions_batch(slope_pred, dust_pred, temp_pred, TI_pred, water_pred):
    """
    Vectorized version of inverse_transform_predictions for 1-D arrays of equal length.
    """
    if not scalers:
        load_scalers()
    slope_pred = np.asarray(slope_pred, dtype=np.float64)
    dust_pred = np.asarray(dust_pred, dtype=np.float64)
    temp_pred = np.asarray(temp_pred, dtype=np.float64)
    TI_pred = np.asarray(TI_pred, dtype=np.float64)
    water_pred = np.asarray(water_pred, dtype=np.float64)

    slope_real = np.maximum(0, np.expm1(slope_pred))

    if "dust_target" in scalers:
        dust_real = scalers["dust_target"].inverse_transform(dust_pred.reshape(-1, 1)).ravel()
    else:
        dust_real = np.clip(0.5 + dust_pred * 0.2, 0.0, 1.0)

    if "surface_temp_y_min" in scalers:
        y_min = float(scalers["surface_temp_y_min"])
        temp_real = np.expm1(temp_pred) + y_min - 1
    else:
        temp_real = np.expm1(temp_pred) - 100 - 1

    if "thermal_inertia_target" in scalers:
        TI_real = scalers["thermal_inertia_target"].inverse_transform(TI_pred.reshape(-1, 1)).ravel()
    else:
        TI_real = np.clip(400 + TI_pred * 200, 100, 1200)

    water_real = np.maximum(0, np.expm1(water_pred))

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
        models = get_nn_models()
        # Get raw model predictions
        slope_pred_raw = models["slope"].predict(
            map_mars_data_to_features(mars_data, "slope"), verbose=0
        )[0][0]
        dust_pred_raw = models["dust"].predict(
            map_mars_data_to_features(mars_data, "dust"), verbose=0
        )[0][0]
        temp_pred_raw = models["surface_temp"].predict(
            map_mars_data_to_features(mars_data, "surface_temp"), verbose=0
        )[0][0]
        TI_pred_raw = models["thermal_inertia"].predict(
            map_mars_data_to_features(mars_data, "thermal_inertia"), verbose=0
        )[0][0]
        water_pred_raw = models["water"].predict(
            map_mars_data_to_features(mars_data, "water"), verbose=0
        )[0][0]
        
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

