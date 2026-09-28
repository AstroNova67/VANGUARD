#!/usr/bin/env python3
"""
Train per-band HistGradientBoosting imputers for Mars Geoscience gap layers (Arm C).

Reads covariates + targets from frontend/3d_globe/public/data_gap_unfilled/.
Writes models to backend/saved_models/gap_imputers/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gap_imputer_lib import (  # noqa: E402
    FEATURE_NAMES,
    HOLDOUT_FRAC,
    MAX_TRAIN_SAMPLES,
    MODEL_DIR,
    RANDOM_SEED,
    TARGET_BANDS,
    UNFILLED,
    covariate_valid_mask,
    feature_matrix,
    load_covariate_stack,
    null_mask,
    read_band,
)


def train_one_band(
    band: str,
    fname: str,
    cov: dict[str, np.ndarray],
    cov_ok: np.ndarray,
) -> dict:
    path = UNFILLED / fname
    data, nodata = read_band(path)
    bad = null_mask(data, nodata)
    train_mask = (~bad) & cov_ok
    rows, cols = np.where(train_mask)
    n = rows.size
    if n < 500:
        raise RuntimeError(f"{band}: too few valid training pixels ({n})")

    rng = np.random.default_rng(RANDOM_SEED)
    if n > MAX_TRAIN_SAMPLES:
        pick = rng.choice(n, size=MAX_TRAIN_SAMPLES, replace=False)
        rows, cols = rows[pick], cols[pick]
        n = rows.size

    X = feature_matrix(cov, rows, cols)
    y = data[rows, cols].astype(np.float64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=HOLDOUT_FRAC, random_state=RANDOM_SEED
    )

    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    pred = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    y_std = float(np.std(y_te))

    out_path = MODEL_DIR / f"{band}.joblib"
    joblib.dump(
        {
            "model": model,
            "band": band,
            "filename": fname,
            "feature_names": list(FEATURE_NAMES),
        },
        out_path,
    )

    return {
        "band": band,
        "filename": fname,
        "n_valid_pixels": int((~bad).sum()),
        "n_gap_pixels": int(bad.sum()),
        "gap_pct": round(100.0 * float(bad.mean()), 3),
        "n_train_pool": int(train_mask.sum()),
        "n_fit": int(len(y_tr)),
        "n_holdout": int(len(y_te)),
        "holdout_mae": round(mae, 5),
        "holdout_rmse": round(rmse, 5),
        "holdout_y_std": round(y_std, 5),
        "train_seconds": round(train_s, 2),
        "model_path": str(out_path.relative_to(MODEL_DIR.parent.parent.parent)),
    }


def main() -> int:
    if not UNFILLED.is_dir():
        print(f"ERROR: missing {UNFILLED}", file=sys.stderr)
        return 1

    # Reference grid from ferric (same MG grid as other fill targets)
    ref, _ = read_band(UNFILLED / TARGET_BANDS["ferric"])
    h, w = ref.shape
    print(f"Grid {w}×{h} from {UNFILLED}", flush=True)

    print("Loading covariates…", flush=True)
    cov = load_covariate_stack(UNFILLED, h, w)
    cov_ok = covariate_valid_mask(cov)
    print(f"  covariate-ok pixels: {int(cov_ok.sum()):,} / {cov_ok.size:,}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for band, fname in TARGET_BANDS.items():
        print(f"Training {band} ({fname})…", flush=True)
        rec = train_one_band(band, fname, cov, cov_ok)
        reports.append(rec)
        print(
            f"  gap={rec['gap_pct']}%  holdout MAE={rec['holdout_mae']}  "
            f"RMSE={rec['holdout_rmse']} (y_std={rec['holdout_y_std']})  "
            f"{rec['train_seconds']}s",
            flush=True,
        )

    manifest = {
        "method": "HistGradientBoostingRegressor",
        "features": list(FEATURE_NAMES),
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "holdout_frac": HOLDOUT_FRAC,
        "random_seed": RANDOM_SEED,
        "source_dir": str(UNFILLED),
        "bands": reports,
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MODEL_DIR / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
