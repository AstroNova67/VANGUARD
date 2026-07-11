"""Shared landing-suitability prediction logic (API + agent)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    from backend.scoring import (
        DATA_SOURCE_ML_PREDICTED,
        DEFAULT_SCORING_WEIGHTS,
        LandingSuitabilityScorer,
        parse_scoring_weights,
        predict_properties_nn,
        scoring_weights_as_percent,
        map_mars_data_to_features as scoring_map_features,
    )
except ImportError:
    from scoring import (
        DATA_SOURCE_ML_PREDICTED,
        DEFAULT_SCORING_WEIGHTS,
        LandingSuitabilityScorer,
        parse_scoring_weights,
        predict_properties_nn,
        scoring_weights_as_percent,
        map_mars_data_to_features as scoring_map_features,
    )

LogFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def finite_number(x: Any) -> bool:
    if x is None:
        return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f))


def clamp_predictions(preds: dict) -> dict:
    out = dict(preds)
    if "slope" in out:
        out["slope"] = max(0.0, float(out["slope"]))
    if "dust" in out:
        out["dust"] = float(min(1.0, max(0.0, out["dust"])))
    if "surface_temp" in out:
        out["surface_temp"] = float(out["surface_temp"])
    if "thermal_inertia" in out:
        out["thermal_inertia"] = max(0.0, float(out["thermal_inertia"]))
    if "water" in out:
        out["water"] = float(min(8.0, max(0.0, out["water"])))
    return out


def fuse_surface_temp_for_score(nn_val, xgb_val, mars_data: dict) -> tuple[float, str | None]:
    obs = mars_data.get("temperature")
    nn_ok = finite_number(nn_val)
    xgb_ok = finite_number(xgb_val) and -200.0 <= float(xgb_val) <= 50.0

    if finite_number(obs):
        obs_f = float(obs)
        candidates = []
        if nn_ok:
            candidates.append(("nn", float(nn_val), abs(float(nn_val) - obs_f)))
        if xgb_ok:
            candidates.append(("xgb", float(xgb_val), abs(float(xgb_val) - obs_f)))
        if not candidates:
            return (float(nn_val), "surface_temp_nn") if nn_ok else (0.0, None)
        candidates.sort(key=lambda t: (t[2], 0 if t[0] == "xgb" else 1))
        winner, val, _ = candidates[0]
        src = "surface_temp_xgb" if winner == "xgb" else "surface_temp_nn"
        return (val, src)

    if xgb_ok:
        return (float(xgb_val), "surface_temp_xgb")
    if nn_ok:
        return (float(nn_val), "surface_temp_nn")
    return (0.0, None)


def fuse_thermal_inertia_for_score(nn_val, xgb_val, mars_data: dict) -> tuple[float, str | None]:
    obs = mars_data.get("thermalInertia")
    if obs is None:
        obs = mars_data.get("thermal_inertia")
    nn_ok = finite_number(nn_val)
    xgb_ok = finite_number(xgb_val) and 50.0 <= float(xgb_val) <= 2000.0

    if finite_number(obs):
        obs_f = float(obs)
        candidates = []
        if nn_ok:
            candidates.append(("nn", float(nn_val), abs(float(nn_val) - obs_f)))
        if xgb_ok:
            candidates.append(("xgb", float(xgb_val), abs(float(xgb_val) - obs_f)))
        if not candidates:
            return (float(nn_val), "thermal_inertia_nn") if nn_ok else (0.0, None)
        candidates.sort(key=lambda t: (t[2], 0 if t[0] == "xgb" else 1))
        src = "thermal_inertia_xgb" if candidates[0][0] == "xgb" else "thermal_inertia_nn"
        return (candidates[0][1], src)

    if xgb_ok:
        return (float(xgb_val), "thermal_inertia_xgb")
    if nn_ok:
        return (float(nn_val), "thermal_inertia_nn")
    return (0.0, None)


def predict_with_neural_networks(mars_data: dict, *, log_error: LogFn = _noop) -> dict:
    try:
        slope, dust, surface_temp, thermal_inertia, water = predict_properties_nn(mars_data)
        return {
            "slope": float(slope),
            "dust": float(dust),
            "surface_temp": float(surface_temp),
            "thermal_inertia": float(thermal_inertia),
            "water": float(water),
        }
    except Exception as e:
        log_error(f"Error in predict_with_neural_networks: {e}")
        return {
            "slope": 0.0,
            "dust": 0.0,
            "surface_temp": 0.0,
            "thermal_inertia": 0.0,
            "water": 0.0,
        }


def predict_with_regression_models(
    mars_data: dict,
    regression_models: dict,
    *,
    log_debug: LogFn = _noop,
    log_error: LogFn = _noop,
) -> dict:
    predictions: dict = {}
    if len(regression_models) == 0:
        log_debug("Using mock regression predictions (models not loaded)")
        return {"surface_temp_xgb": -44.8, "thermal_inertia_xgb": 445.2}

    if "surface_temp" in regression_models:
        try:
            features = scoring_map_features(mars_data, "surface_temp")
            xgb_pred = regression_models["surface_temp"]["xgb"].predict(features)[0]
            predictions["surface_temp_xgb"] = float(xgb_pred)
        except Exception as e:
            log_error(f"Error predicting surface_temp with XGB: {e}")
            predictions["surface_temp_xgb"] = 0.0

    if "thermal_inertia" in regression_models:
        try:
            temp_range = mars_data.get("tempRange", 0)
            albedo = mars_data.get("albedo", 0)
            slope = mars_data.get("slope", 0)
            ferric = mars_data.get("ferric", 0)
            features = np.array([[temp_range, albedo, slope, ferric]])
            xgb_pred = regression_models["thermal_inertia"]["xgb"].predict(features)[0]
            predictions["thermal_inertia_xgb"] = float(xgb_pred)
        except Exception as e:
            log_error(f"Error predicting thermal_inertia with XGB: {e}")
            predictions["thermal_inertia_xgb"] = 0.0

    return predictions


def score_band_label(landing_score: float) -> str:
    if landing_score >= 90:
        return "Excellent landing site"
    if landing_score >= 70:
        return "Good landing site"
    if landing_score >= 50:
        return "Fair landing site"
    if landing_score >= 30:
        return "Poor landing site"
    return "Very poor landing site"


def split_predict_payload(body: dict) -> tuple[dict, dict[str, float] | None]:
    """
    Separate Mars observation fields from optional custom scoring_weights on POST /predict.
    """
    payload = dict(body)
    raw_weights = payload.pop("scoring_weights", None)
    if raw_weights is None:
        raw_weights = payload.pop("scoringWeights", None)
    weights = parse_scoring_weights(raw_weights) if raw_weights is not None else None
    return payload, weights


def compute_landing_prediction(
    mars_data: dict,
    *,
    models_loaded: bool,
    regression_models: dict,
    scoring_weights: dict[str, float] | None = None,
    log_error: LogFn = _noop,
    log_debug: LogFn = _noop,
) -> dict[str, Any]:
    """Same JSON shape as POST /predict (success, landing_score, predictions, …)."""
    if not models_loaded:
        return {
            "success": False,
            "error": "Models are still loading. Please try again in a moment.",
            "landing_score": 0,
        }

    nn_baseline = clamp_predictions(predict_with_neural_networks(mars_data, log_error=log_error))
    reg_predictions = predict_with_regression_models(
        mars_data, regression_models, log_debug=log_debug, log_error=log_error
    )

    overrides_applied: dict[str, str] = {}
    data_sources: dict[str, str] = {}
    fused_predictions: dict[str, float] = {}

    fused_predictions["slope"] = nn_baseline["slope"]
    data_sources["slope"] = DATA_SOURCE_ML_PREDICTED

    fused_predictions["dust"] = nn_baseline["dust"]
    data_sources["dust"] = DATA_SOURCE_ML_PREDICTED

    st_val, st_src = fuse_surface_temp_for_score(
        nn_baseline.get("surface_temp"),
        reg_predictions.get("surface_temp_xgb"),
        mars_data,
    )
    if st_src:
        fused_predictions["surface_temp"] = st_val
        overrides_applied["surface_temp"] = st_src
    elif finite_number(nn_baseline.get("surface_temp")):
        fused_predictions["surface_temp"] = float(nn_baseline["surface_temp"])
        overrides_applied["surface_temp"] = "surface_temp_nn"
    elif finite_number(st_val):
        fused_predictions["surface_temp"] = st_val
        overrides_applied["surface_temp"] = "surface_temp_nn"
    else:
        fused_predictions["surface_temp"] = 0.0
        overrides_applied["surface_temp"] = "surface_temp_nn"
    data_sources["surface_temp"] = DATA_SOURCE_ML_PREDICTED

    ti_val, ti_src = fuse_thermal_inertia_for_score(
        nn_baseline.get("thermal_inertia"),
        reg_predictions.get("thermal_inertia_xgb"),
        mars_data,
    )
    if ti_src:
        fused_predictions["thermal_inertia"] = ti_val
        overrides_applied["thermal_inertia"] = ti_src
    elif finite_number(nn_baseline.get("thermal_inertia")):
        fused_predictions["thermal_inertia"] = float(nn_baseline["thermal_inertia"])
        overrides_applied["thermal_inertia"] = "thermal_inertia_nn"
    elif finite_number(ti_val):
        fused_predictions["thermal_inertia"] = ti_val
        overrides_applied["thermal_inertia"] = "thermal_inertia_nn"
    else:
        fused_predictions["thermal_inertia"] = 0.0
        overrides_applied["thermal_inertia"] = "thermal_inertia_nn"
    data_sources["thermal_inertia"] = DATA_SOURCE_ML_PREDICTED

    fused_predictions["water"] = nn_baseline["water"]
    data_sources["water"] = DATA_SOURCE_ML_PREDICTED

    fused_predictions = clamp_predictions(fused_predictions)

    weights_used = parse_scoring_weights(scoring_weights)
    scorer = LandingSuitabilityScorer(weights_used)
    landing_score, score_breakdown = scorer.score_site_breakdown(
        slope=fused_predictions.get("slope", 0),
        dust=fused_predictions.get("dust", 0),
        surface_temp=fused_predictions.get("surface_temp", 0),
        thermal_inertia=fused_predictions.get("thermal_inertia", 0),
        water=fused_predictions.get("water", 0),
    )

    return {
        "success": True,
        "landing_score": landing_score,
        "score_interpretation": score_band_label(landing_score),
        "scoring_weights": weights_used,
        "scoring_weights_percent": scoring_weights_as_percent(weights_used),
        "scoring_weights_default": DEFAULT_SCORING_WEIGHTS,
        "score_breakdown": score_breakdown,
        "predictions": {
            "neural_networks": fused_predictions,
            "neural_networks_baseline": nn_baseline,
            "regression_models": reg_predictions,
        },
        "data_sources": data_sources,
        "overrides_applied": overrides_applied,
        "raw_mars_data": mars_data,
    }
