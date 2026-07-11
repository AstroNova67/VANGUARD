#!/usr/bin/env python3
"""Sample ferric and thermal-inertia bands; report percentiles for scoring range calibration."""

from __future__ import annotations

import os
import sys

import numpy as np
import rasterio

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
BACKEND = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, BACKEND)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from scoring import (  # noqa: E402
    get_nn_models,
    inverse_transform_predictions_batch,
    load_scalers,
    raster_band_valid,
    scalers,
)

STACK_PATH = os.path.join(
    REPO_ROOT, "frontend", "3d_globe", "public", "data", "mars_global_input_stack_32ppd.tif"
)
N_SAMPLE = 100_000
SEED = 42

# Band indices (1-based) from stack_mars_layers.LAYER_BASENAMES
BAND_ELEV = 1
BAND_SLOPE = 2
BAND_ROUGH = 3
BAND_ALBEDO = 4
BAND_TEMP = 5
BAND_TEMP_RANGE = 6
BAND_FERRIC = 8
BAND_TI = 12

PERCENTILES = (2, 5, 10, 25, 50, 75, 90, 95, 98)


def _pct_report(values: np.ndarray, label: str) -> dict[str, float]:
    out = {f"P{p}": float(np.percentile(values, p)) for p in PERCENTILES}
    print(f"\n{label} (n={len(values):,})")
    for p in PERCENTILES:
        print(f"  P{p:2d}: {out[f'P{p}']:.6f}")
    return out


def _sample_valid_pixels(stack_path: str, band: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    with rasterio.open(stack_path) as ds:
        arr = ds.read(band).astype(np.float64)
        nodata = ds.nodata
    valid = raster_band_valid(arr, nodata)
    idx = np.flatnonzero(valid.ravel())
    if len(idx) < n:
        raise RuntimeError(f"Only {len(idx)} valid pixels in band {band}, need {n}")
    pick = rng.choice(idx, size=n, replace=False)
    rows, cols = np.unravel_index(pick, arr.shape)
    return pick, arr[rows, cols]


def _stack_features_at_indices(stack_path: str, indices: np.ndarray) -> dict[str, np.ndarray]:
    with rasterio.open(stack_path) as ds:
        h, w = ds.height, ds.width
        rows, cols = np.unravel_index(indices, (h, w))
    bands = {
        "elevation": BAND_ELEV,
        "slope": BAND_SLOPE,
        "roughness": BAND_ROUGH,
        "albedo": BAND_ALBEDO,
        "temperature": BAND_TEMP,
        "tempRange": BAND_TEMP_RANGE,
        "ferric": BAND_FERRIC,
        "thermalInertia": BAND_TI,
    }
    out: dict[str, np.ndarray] = {}
    with rasterio.open(stack_path) as ds:
        for key, b in bands.items():
            arr = ds.read(b).astype(np.float64)
            out[key] = arr[rows, cols]
    return out


def _predict_dust_batch(feats: dict[str, np.ndarray]) -> np.ndarray:
    load_scalers()
    models = get_nn_models()
    n = len(feats["elevation"])
    mat = np.column_stack(
        [
            feats["elevation"],
            feats["slope"],
            feats["temperature"],
            feats["temperature"],
            feats["slope"],
            feats["albedo"],
        ]
    )
    if "dust_feature" not in scalers:
        raise RuntimeError("dust_feature scaler not loaded")
    x = scalers["dust_feature"].transform(mat)
    raw = models["dust"].predict(x, verbose=0, batch_size=8192).ravel()
    _, dust, _, _, _ = inverse_transform_predictions_batch(
        np.zeros(n), raw, np.zeros(n), np.zeros(n), np.zeros(n)
    )
    return dust


def _predict_ti_batch(feats: dict[str, np.ndarray]) -> np.ndarray:
    load_scalers()
    models = get_nn_models()
    n = len(feats["elevation"])
    mat = np.column_stack(
        [
            feats["tempRange"],
            feats["albedo"],
            feats["slope"],
            feats["ferric"],
        ]
    )
    if "thermal_inertia_feature" not in scalers:
        raise RuntimeError("thermal_inertia_feature scaler not loaded")
    x = scalers["thermal_inertia_feature"].transform(mat)
    raw = models["thermal_inertia"].predict(x, verbose=0, batch_size=8192).ravel()
    _, _, _, ti, _ = inverse_transform_predictions_batch(
        np.zeros(n), np.zeros(n), np.zeros(n), raw, np.zeros(n)
    )
    return ti


def main() -> None:
    print(f"Stack: {STACK_PATH}")
    if not os.path.isfile(STACK_PATH):
        sys.exit(f"Missing stack: {STACK_PATH}")

    # --- Fix 1: Ferric / dust ---
    ferric_idx, ferric_obs = _sample_valid_pixels(STACK_PATH, BAND_FERRIC, N_SAMPLE, SEED)
    ferric_pct = _pct_report(ferric_obs, "Observed ferric (band 8)")

    feats = _stack_features_at_indices(STACK_PATH, ferric_idx)
    print("\nRunning dust NN on same pixels …")
    dust_ml = _predict_dust_batch(feats)
    dust_pct = _pct_report(dust_ml, "ML predicted dust (same pixels)")

    # --- Fix 2: Thermal inertia ---
    ti_idx, ti_obs = _sample_valid_pixels(STACK_PATH, BAND_TI, N_SAMPLE, SEED + 1)
    ti_pct = _pct_report(ti_obs, "Observed thermal inertia (band 12)")

    feats_ti = _stack_features_at_indices(STACK_PATH, ti_idx)
    print("\nRunning thermal inertia NN on same pixels …")
    ti_ml = _predict_ti_batch(feats_ti)
    ti_ml_pct = _pct_report(ti_ml, "ML predicted thermal inertia (same pixels)")

    print("\n--- Proposed scoring ranges (P5–P95 of observed) ---")
    print(f"Dust:  min={ferric_pct['P5']:.6f}  max={ferric_pct['P95']:.6f}  (was 0.6–0.7)")
    print(f"TI:    min={ti_pct['P5']:.6f}  max={ti_pct['P95']:.6f}  (was 100–400)")


if __name__ == "__main__":
    main()
