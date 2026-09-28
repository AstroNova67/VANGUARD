#!/usr/bin/env python3
"""
Apply trained gap imputers → frontend/3d_globe/public/data_gap_ml_filled/ (Arm C).

Copies data_gap_unfilled/, replaces gap pixels in the five target bands with ML
predictions, then applies the same gaussian feather used by IDW fill.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gap_imputer_lib import (  # noqa: E402
    FEATURE_NAMES,
    FEATHER_PX,
    GAUSS_SIGMA,
    ML_FILLED,
    MODEL_DIR,
    TARGET_BANDS,
    UNFILLED,
    covariate_valid_mask,
    feature_matrix,
    gaussian_blend,
    load_covariate_stack,
    null_mask,
    read_band,
    write_geotiff,
)

BATCH = 50_000


def predict_gaps(
    model,
    cov: dict[str, np.ndarray],
    gap_rows: np.ndarray,
    gap_cols: np.ndarray,
) -> np.ndarray:
    out = np.empty(gap_rows.size, dtype=np.float64)
    for start in range(0, gap_rows.size, BATCH):
        end = min(start + BATCH, gap_rows.size)
        X = feature_matrix(cov, gap_rows[start:end], gap_cols[start:end])
        out[start:end] = model.predict(X)
    return out


def fill_band(band: str, fname: str, cov: dict[str, np.ndarray], cov_ok: np.ndarray) -> dict:
    t0 = time.perf_counter()
    bundle = joblib.load(MODEL_DIR / f"{band}.joblib")
    model = bundle["model"]
    feats = bundle.get("feature_names") or list(FEATURE_NAMES)
    if list(feats) != list(FEATURE_NAMES):
        raise RuntimeError(f"{band}: feature mismatch {feats} vs {FEATURE_NAMES}")

    data, nodata = read_band(UNFILLED / fname)
    bad = null_mask(data, nodata)
    n_bad = int(bad.sum())
    if n_bad == 0:
        write_geotiff(ML_FILLED / fname, data.astype(np.float32))
        return {
            "band": band,
            "filename": fname,
            "filled": 0,
            "skipped_no_covariates": 0,
            "remaining_zeros": 0,
            "pct": 0.0,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    # Only predict where covariates are valid; leave others for spatial fallback later
    predict_mask = bad & cov_ok
    skip_mask = bad & ~cov_ok
    rows, cols = np.where(predict_mask)
    filled = data.astype(np.float64, copy=True)
    n_pred = 0
    if rows.size:
        preds = predict_gaps(model, cov, rows, cols)
        # Avoid exact zeros so null_mask does not re-flag filled pixels
        preds = np.where(np.abs(preds) < 1e-6, np.sign(preds) * 1e-6 + 1e-6, preds)
        filled[rows, cols] = preds
        n_pred = int(rows.size)

    # Spatial nearest-valid fallback for gaps without covariates
    n_skip = int(skip_mask.sum())
    if n_skip:
        from scipy.ndimage import distance_transform_edt

        valid = ~null_mask(filled)
        if valid.any():
            # edt input: non-zero = foreground; distance to nearest 0 (valid)
            _, (ir, ic) = distance_transform_edt(~valid, return_indices=True)
            sr, sc = np.where(skip_mask)
            filled[sr, sc] = filled[ir[sr, sc], ic[sr, sc]]
            filled[sr, sc] = np.where(
                np.abs(filled[sr, sc]) < 1e-6, 1e-6, filled[sr, sc]
            )

    filled = gaussian_blend(filled.astype(np.float32), bad, sigma=GAUSS_SIGMA, feather_px=FEATHER_PX)
    filled = filled.copy()
    still_bad = null_mask(filled)
    if still_bad.any():
        filled[still_bad] = 1e-6

    write_geotiff(ML_FILLED / fname, filled.astype(np.float32))
    remaining = int(null_mask(filled).sum())
    return {
        "band": band,
        "filename": fname,
        "filled": n_pred,
        "skipped_no_covariates": n_skip,
        "remaining_zeros": remaining,
        "pct": round(100.0 * n_bad / data.size, 3),
        "shape": list(data.shape),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    if not UNFILLED.is_dir():
        print(f"ERROR: missing {UNFILLED}", file=sys.stderr)
        return 1
    if not (MODEL_DIR / "manifest.json").is_file():
        print(f"ERROR: missing models — run scripts/train_gap_imputers.py first", file=sys.stderr)
        return 1

    for band in TARGET_BANDS:
        if not (MODEL_DIR / f"{band}.joblib").is_file():
            print(f"ERROR: missing {MODEL_DIR / f'{band}.joblib'}", file=sys.stderr)
            return 1

    if ML_FILLED.exists():
        shutil.rmtree(ML_FILLED)
    shutil.copytree(UNFILLED, ML_FILLED)

    ref, _ = read_band(UNFILLED / TARGET_BANDS["ferric"])
    h, w = ref.shape
    print(f"Loading covariates on {w}×{h}…", flush=True)
    cov = load_covariate_stack(UNFILLED, h, w)
    cov_ok = covariate_valid_mask(cov)

    fill_report = []
    for band, fname in TARGET_BANDS.items():
        print(f"ML-filling {band} → {fname}…", flush=True)
        rec = fill_band(band, fname, cov, cov_ok)
        fill_report.append(rec)
        print(
            f"  predicted {rec['filled']:,}  fallback {rec['skipped_no_covariates']:,}  "
            f"remain_zero={rec['remaining_zeros']}  {rec['seconds']}s",
            flush=True,
        )

    meta = {
        "method": "ml_hist_gradient_boosting",
        "source": str(UNFILLED),
        "output": str(ML_FILLED),
        "model_dir": str(MODEL_DIR),
        "features": list(FEATURE_NAMES),
        "params": {"GAUSS_SIGMA": GAUSS_SIGMA, "FEATHER_PX": FEATHER_PX},
        "filled_layers": fill_report,
        "note": (
            "Copied data_gap_unfilled/, then ML-imputed the five gap bands "
            "(ferric, albedo, roughness, tempRange, elevation) with HGBR + gaussian feather."
        ),
    }
    (ML_FILLED / "fill_manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {ML_FILLED / 'fill_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
