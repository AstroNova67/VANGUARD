import numpy as np
import tensorflow as tf


class LandingSuitabilityScorer:
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
        slope_score = self.normalize(slope, 0, 30, invert=True)
        dust_score = self.normalize(dust, 0, 1, invert=True)
        temp_score = self.normalize(surface_temp, -100, 20, invert=False)
        inertia_score = self.normalize(thermal_inertia, 100, 1200, invert=False)
        water_score = self.normalize(water, 0, 1, invert=False)

        final_score = (
                slope_score * self.weights["slope"] +
                dust_score * self.weights["dust"] +
                temp_score * self.weights["surface_temp"] +
                inertia_score * self.weights["thermal_inertia"] +
                water_score * self.weights["water"]
        )

        return round(final_score * 100, 2)


# Example: Using Neural Network predictions
# Assume nn_model_* are your trained Keras/PyTorch models
def predict_properties_nn(features):
    """
    features: list or array of input features for the NN models
    nn_models: dict with keys 'slope', 'dust', 'temp', 'TI', 'water'
               and values = trained neural networks
    returns: predicted property values
    """
    slope_pred = \
    tf.keras.models.load_model('saved_models/neural_nets/slope_pred/best_model.keras').predict(np.array([features]))[0][
        0]
    dust_pred = tf.keras.models.load_model('saved_models/neural_nets/dust_predictor/best_model.keras').predict(
        np.array([features]))[0][0]
    temp_pred = tf.keras.models.load_model('saved_models/neural_nets/surface_temp_pred/best_model.keras').predict(
        np.array([features]))[0][0]
    TI_pred = tf.keras.models.load_model('saved_models/neural_nets/thermal_inertia_predictor/best_model.keras').predict(
        np.array([features]))[0][0]
    water_pred = tf.keras.models.load_model('saved_models/neural_nets/water_predictor/best_model.keras').predict(
        np.array([features]))[0][0]

    return slope_pred, dust_pred, temp_pred, TI_pred, water_pred


# Example usage
# nn_models = {"slope": slope_nn, "dust": dust_nn, "temp": temp_nn, ...}
# features_for_point = [/* your input features at the clicked location */]

# Predict values from neural networks
pred_slope, pred_dust, pred_temp, pred_TI, pred_water = predict_properties_nn()

# Compute landing score
scorer = LandingSuitabilityScorer()
landing_score = scorer.score_site(pred_slope, pred_dust, pred_temp, pred_TI, pred_water)
print(f"Predicted Landing Score (NN-based): {landing_score}%")
