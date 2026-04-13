#!/usr/bin/env python3
"""
Raster batch job: 13-band input GeoTIFF → ML-only suitability (default), optionally hybrid + coverage.

Band order must match `frontend/3d_globe/index.js` `marsDatasets` / `scripts/stack_mars_layers.py`
(`STACK_LAYER_NAMES`). Default paths live under `frontend/3d_globe/public/data/`.

**Default run** (no flags beyond defaults) writes **only** ``mars_landing_suitability_ml.tif``.
The stack GeoTIFF (``--input``, default ``mars_global_input_stack_32ppd.tif``) is **read only** — it is never
modified or replaced by this job; it holds the per-band inputs the models consume.

Pass ``--with-hybrid-coverage`` to also write hybrid + coverage rasters. Hybrid defaults to
``mars_landing_suitability_hybrid.tif`` (use ``--hybrid-output`` for another path).

Auxiliary stack NoData uses `map_mars_data_to_features` defaults. Primary bands (indices
1, 4, 7, 11, 12): NoData (-9999) on all outputs only where all five are simultaneously invalid.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.windows import Window

# -----------------------------------------------------------------------------
# Path setup (run from repo root or backend/)
# -----------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DEFAULT_GLOBE_DATA_DIR = os.path.join(
    PROJECT_ROOT, "frontend", "3d_globe", "public", "data"
)
DEFAULT_STACK_PATH = os.path.join(
    DEFAULT_GLOBE_DATA_DIR, "mars_global_input_stack_32ppd.tif"
)
DEFAULT_ML_OUTPUT_PATH = os.path.join(
    DEFAULT_GLOBE_DATA_DIR, "mars_landing_suitability_ml.tif"
)
# Never default to mars_landing_suitability.tif — that name is often a manually expensive hybrid product.
DEFAULT_HYBRID_SUITABILITY_PATH = os.path.join(
    DEFAULT_GLOBE_DATA_DIR, "mars_landing_suitability_hybrid.tif"
)
DEFAULT_COVERAGE_PATH = os.path.join(
    DEFAULT_GLOBE_DATA_DIR, "mars_ml_coverage.tif"
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import xgboost as xgb  # noqa: E402

from scoring import (  # noqa: E402
    BASE_DIR,
    LandingSuitabilityScorer,
    get_nn_models,
    inverse_transform_predictions_batch,
    load_scalers,
    raster_band_valid,
    scalers,
)

# Band order = globe `marsDatasets` key order (1-based GDAL band index = index + 1)
STACK_LAYER_NAMES = (
    "elevation",
    "slope",
    "roughness",
    "albedo",
    "temperature",
    "tempRange",
    "crustalThickness",
    "ferric",
    "pyroxene",
    "basalt",
    "lambertAlbedo",
    "thermalInertia",
    "grsWaterWt",
)

OUTPUT_NODATA = -9999.0


def _finite_number(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def _clamp_pred_arrays(preds: dict) -> dict:
    out = {k: np.asarray(v, dtype=np.float64) for k, v in preds.items()}
    if "slope" in out:
        out["slope"] = np.maximum(out["slope"], 0.0)
    if "dust" in out:
        out["dust"] = np.clip(out["dust"], 0.0, 1.0)
    if "surface_temp" in out:
        out["surface_temp"] = out["surface_temp"]
    if "thermal_inertia" in out:
        out["thermal_inertia"] = np.maximum(out["thermal_inertia"], 0.0)
    if "water" in out:
        out["water"] = np.clip(out["water"], 0.0, 8.0)
    return out


def _fuse_surface_temp_vectorized(nn_val, xgb_val, obs):
    """Same decision rule as `app._fuse_surface_temp_for_score`, vectorized."""
    nn_val = np.asarray(nn_val, dtype=np.float64)
    xgb_val = np.asarray(xgb_val, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    nn_ok = _finite_number(nn_val)
    xgb_ok = _finite_number(xgb_val) & (xgb_val >= -200.0) & (xgb_val <= 50.0)
    obs_ok = _finite_number(obs)

    e_nn = np.where(nn_ok, np.abs(nn_val - obs), np.inf)
    e_xgb = np.where(xgb_ok, np.abs(xgb_val - obs), np.inf)

    out = np.zeros_like(nn_val, dtype=np.float64)
    m_no_obs = ~obs_ok
    out[m_no_obs & xgb_ok] = xgb_val[m_no_obs & xgb_ok]
    out[m_no_obs & ~xgb_ok & nn_ok] = nn_val[m_no_obs & ~xgb_ok & nn_ok]
    out[m_no_obs & ~xgb_ok & ~nn_ok] = 0.0

    use_xgb = obs_ok & np.isfinite(e_xgb) & (e_xgb <= e_nn)
    use_nn = obs_ok & ~use_xgb & np.isfinite(e_nn)
    use_zero = obs_ok & ~use_xgb & ~use_nn
    out[use_xgb] = xgb_val[use_xgb]
    out[use_nn] = nn_val[use_nn]
    out[use_zero] = 0.0
    return out


def _fuse_thermal_inertia_vectorized(nn_val, xgb_val, obs):
    """Same decision rule as `app._fuse_thermal_inertia_for_score`, vectorized."""
    nn_val = np.asarray(nn_val, dtype=np.float64)
    xgb_val = np.asarray(xgb_val, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    nn_ok = _finite_number(nn_val)
    xgb_ok = _finite_number(xgb_val) & (xgb_val >= 50.0) & (xgb_val <= 2000.0)
    obs_ok = _finite_number(obs)

    e_nn = np.where(nn_ok, np.abs(nn_val - obs), np.inf)
    e_xgb = np.where(xgb_ok, np.abs(xgb_val - obs), np.inf)

    out = np.zeros_like(nn_val, dtype=np.float64)
    m_no_obs = ~obs_ok
    out[m_no_obs & xgb_ok] = xgb_val[m_no_obs & xgb_ok]
    out[m_no_obs & ~xgb_ok & nn_ok] = nn_val[m_no_obs & ~xgb_ok & nn_ok]
    out[m_no_obs & ~xgb_ok & ~nn_ok] = 0.0

    use_xgb = obs_ok & np.isfinite(e_xgb) & (e_xgb <= e_nn)
    use_nn = obs_ok & ~use_xgb & np.isfinite(e_nn)
    use_zero = obs_ok & ~use_xgb & ~use_nn
    out[use_xgb] = xgb_val[use_xgb]
    out[use_nn] = nn_val[use_nn]
    out[use_zero] = 0.0
    return out


def _predict_keras_batches(model, x: np.ndarray, subbatch: int) -> np.ndarray:
    out = []
    n = x.shape[0]
    for i in range(0, n, subbatch):
        chunk = x[i : i + subbatch]
        out.append(model.predict(chunk, verbose=0))
    y = np.concatenate(out, axis=0)
    return y.reshape(-1)


def _build_nn_feature_matrices(
    elev, slope, rough, alb, temp, tr, ferr, ti
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrices matching `map_mars_data_to_features` for each NN head."""
    slope_m = np.column_stack(
        [
            alb,
            temp,
            rough,
            ferr,
            elev,
            tr,
            np.abs(slope),
            np.abs(slope * 0.1),
        ]
    )
    dust_m = np.column_stack([elev, slope, temp, temp, slope, alb])
    st_m = np.column_stack([elev, alb, ti, slope, rough])
    ti_m = np.column_stack([tr, alb, slope, ferr])
    water_m = slope_m

    x_slope = scalers["slope"].transform(slope_m)
    x_dust = scalers["dust_feature"].transform(dust_m)
    x_st = scalers["surface_temp"].transform(st_m)
    x_ti = scalers["thermal_inertia_feature"].transform(ti_m)
    x_water = scalers["water"].transform(water_m)
    return x_slope, x_dust, x_st, x_ti, x_water


def _load_xgb_models():
    st_path = os.path.join(
        BASE_DIR, "saved_models", "regression_models", "surface_temp", "xgb_model.json"
    )
    ti_path = os.path.join(
        BASE_DIR, "saved_models", "regression_models", "thermal_inertia", "xgb_model.json"
    )
    xgb_st = xgb.XGBRegressor()
    xgb_st.load_model(st_path)
    xgb_ti = xgb.XGBRegressor()
    xgb_ti.load_model(ti_path)
    return xgb_st, xgb_ti


def _xgb_surface_temp_features(elev, alb, ti, slope, rough):
    """Same transformed features as `map_mars_data_to_features(..., 'surface_temp')`."""
    st_m = np.column_stack([elev, alb, ti, slope, rough])
    return scalers["surface_temp"].transform(st_m)


def _xgb_ti_raw_features(tr, alb, slope, ferr):
    return np.column_stack([tr, alb, slope, ferr])


def _fill_band_for_inference(src, data: np.ndarray, band_idx: int, default: float) -> np.ndarray:
    """Finite + not-nodata → raw value; else default (matches missing keys in map_mars_data_to_features)."""
    arr = data[band_idx].astype(np.float64, copy=False)
    ok = raster_band_valid(arr, src.nodatavals[band_idx])
    return np.where(ok, arr, default)


def run(args: argparse.Namespace) -> None:
    load_scalers()
    if len(scalers) < 8:
        raise SystemExit("Scalers failed to load; check backend/saved_models/scalers.")

    inp_abs = os.path.abspath(args.input)
    for label, path in (
        ("--output", args.output),
        ("--hybrid-output", args.hybrid_output),
        ("--coverage-output", args.coverage_output),
    ):
        if os.path.abspath(path) == inp_abs:
            raise SystemExit(
                f"{label} must not be the same file as --input (stack). "
                "The stack is read-only model input; pick a different output path."
            )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.with_hybrid_coverage:
        cov_dir = os.path.dirname(os.path.abspath(args.coverage_output))
        if cov_dir:
            os.makedirs(cov_dir, exist_ok=True)
        hybrid_dir = os.path.dirname(os.path.abspath(args.hybrid_output))
        if hybrid_dir:
            os.makedirs(hybrid_dir, exist_ok=True)

    print("Loading Keras models (once)…", flush=True)
    nn = get_nn_models()
    print("Loading XGBoost fusion models…", flush=True)
    xgb_st, xgb_ti = _load_xgb_models()
    scorer = LandingSuitabilityScorer()
    print("Models ready. Opening stack and output rasters…", flush=True)

    pixels_scored = 0
    pixels_all_five_raster = 0
    pixels_ml_fallback = 0
    pixels_fully_nodata = 0

    with rasterio.open(args.input) as src:
        if src.count != len(STACK_LAYER_NAMES):
            raise SystemExit(
                f"Expected {len(STACK_LAYER_NAMES)} bands ({STACK_LAYER_NAMES}), "
                f"got count={src.count}."
            )

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            nodata=OUTPUT_NODATA,
        )
        if args.compress:
            profile["compress"] = args.compress
        if args.tiled:
            profile["tiled"] = True
            profile["blockxsize"] = int(args.blockxsize)
            profile["blockysize"] = int(args.blockysize)

        height, width = src.height, src.width
        total_pixels = int(height) * int(width)
        bs = int(args.block_size)
        ncols = (width + bs - 1) // bs
        nrows = (height + bs - 1) // bs
        total_blocks = nrows * ncols
        block_id = 0
        t0 = time.perf_counter()
        print(
            f"Raster pass: {width}×{height} px, {total_blocks} windows "
            f"(block_size={bs}). ETA prints after each window finishes "
            "(first line may take several minutes).",
            flush=True,
        )

        with ExitStack() as stack_ctx:
            dst_ml = stack_ctx.enter_context(rasterio.open(args.output, "w", **profile))
            dst_hybrid = None
            dst_cov = None
            if args.with_hybrid_coverage:
                dst_hybrid = stack_ctx.enter_context(
                    rasterio.open(args.hybrid_output, "w", **profile)
                )
                dst_cov = stack_ctx.enter_context(
                    rasterio.open(args.coverage_output, "w", **profile)
                )

            for row in range(0, height, bs):
                win_h = min(bs, height - row)
                for col in range(0, width, bs):
                    win_w = min(bs, width - col)
                    window = Window(col, row, win_w, win_h)
                    data = src.read(window=window).astype(np.float64, copy=False)

                    elev = data[0]
                    slope = data[1]
                    rough = data[2]
                    alb = data[3]
                    temp = data[4]
                    tr = data[5]
                    ferr = data[7]
                    ti_obs = data[11]
                    grs = data[12]

                    v_slope = raster_band_valid(slope, src.nodatavals[1])
                    v_dust = raster_band_valid(ferr, src.nodatavals[7])
                    v_temp = raster_band_valid(temp, src.nodatavals[4])
                    v_ti = raster_band_valid(ti_obs, src.nodatavals[11])
                    v_wat = raster_band_valid(grs, src.nodatavals[12])

                    eligible = v_slope | v_dust | v_temp | v_ti | v_wat
                    all_five_raster = v_slope & v_dust & v_temp & v_ti & v_wat

                    out_block_ml = np.full((1, win_h, win_w), OUTPUT_NODATA, dtype=np.float32)
                    out_block_hybrid = np.full((1, win_h, win_w), OUTPUT_NODATA, dtype=np.float32)
                    cov_block = np.full((1, win_h, win_w), OUTPUT_NODATA, dtype=np.float32)

                    flat_el = eligible.ravel()
                    n_el = int(np.count_nonzero(flat_el))
                    pixels_fully_nodata += int(np.count_nonzero(~eligible))

                    if n_el > 0:
                        pixels_scored += n_el
                        pixels_all_five_raster += int(np.count_nonzero(all_five_raster))
                        pixels_ml_fallback += int(np.count_nonzero(eligible & ~all_five_raster))

                        e_inf = _fill_band_for_inference(src, data, 0, 1000.0)
                        s_inf = _fill_band_for_inference(src, data, 1, 2.0)
                        r_inf = _fill_band_for_inference(src, data, 2, 50.0)
                        a_inf = _fill_band_for_inference(src, data, 3, 0.2)
                        t_inf = _fill_band_for_inference(src, data, 4, -30.0)
                        tr_inf = _fill_band_for_inference(src, data, 5, 50.0)
                        fe_inf = _fill_band_for_inference(src, data, 7, 0.5)
                        ti_inf = _fill_band_for_inference(src, data, 11, 300.0)

                        e = e_inf.ravel()[flat_el]
                        s_nn = s_inf.ravel()[flat_el]
                        r = r_inf.ravel()[flat_el]
                        a = a_inf.ravel()[flat_el]
                        t_nn = t_inf.ravel()[flat_el]
                        trv = tr_inf.ravel()[flat_el]
                        fe_nn = fe_inf.ravel()[flat_el]
                        ti_nn_in = ti_inf.ravel()[flat_el]

                        s_raw = slope.ravel()[flat_el]
                        t_raw = temp.ravel()[flat_el]
                        fe_raw = ferr.ravel()[flat_el]
                        ti_raw = ti_obs.ravel()[flat_el]
                        g_raw = grs.ravel()[flat_el]

                        v_s = v_slope.ravel()[flat_el]
                        v_d = v_dust.ravel()[flat_el]
                        v_t = v_temp.ravel()[flat_el]
                        v_i = v_ti.ravel()[flat_el]
                        v_w = v_wat.ravel()[flat_el]

                        x_sl, x_du, x_st, x_ti_nn, x_wa = _build_nn_feature_matrices(
                            e, s_nn, r, a, t_nn, trv, fe_nn, ti_nn_in
                        )
                        sb = int(args.inference_subbatch)

                        raw_sl = _predict_keras_batches(nn["slope"], x_sl, sb)
                        raw_du = _predict_keras_batches(nn["dust"], x_du, sb)
                        raw_st = _predict_keras_batches(nn["surface_temp"], x_st, sb)
                        raw_ti = _predict_keras_batches(nn["thermal_inertia"], x_ti_nn, sb)
                        raw_wa = _predict_keras_batches(nn["water"], x_wa, sb)

                        sl, du, st_nn, ti_nn, wa = inverse_transform_predictions_batch(
                            raw_sl, raw_du, raw_st, raw_ti, raw_wa
                        )

                        x_st_xgb = _xgb_surface_temp_features(e, a, ti_nn_in, s_nn, r)
                        st_xgb = np.asarray(xgb_st.predict(x_st_xgb), dtype=np.float64).reshape(-1)

                        x_ti_xgb = _xgb_ti_raw_features(trv, a, s_nn, fe_nn)
                        ti_xgb = np.asarray(xgb_ti.predict(x_ti_xgb), dtype=np.float64).reshape(-1)

                        st_fused = _fuse_surface_temp_vectorized(st_nn, st_xgb, t_raw)
                        ti_fused = _fuse_thermal_inertia_vectorized(ti_nn, ti_xgb, ti_raw)

                        ml_only = _clamp_pred_arrays(
                            {
                                "slope": sl,
                                "dust": du,
                                "surface_temp": st_fused,
                                "thermal_inertia": ti_fused,
                                "water": wa,
                            }
                        )
                        ml_scores = scorer.score_site_arrays(
                            ml_only["slope"],
                            ml_only["dust"],
                            ml_only["surface_temp"],
                            ml_only["thermal_inertia"],
                            ml_only["water"],
                        )

                        bad_ml = ~np.isfinite(ml_scores)
                        ml_scores[bad_ml] = OUTPUT_NODATA

                        flat_ml = out_block_ml[0].ravel()
                        flat_ml[flat_el] = ml_scores.astype(np.float32, copy=False)
                        out_block_ml[0] = flat_ml.reshape(win_h, win_w)

                        if args.with_hybrid_coverage:
                            sl_use = np.where(v_s, s_raw, sl)
                            du_use = np.where(v_d, fe_raw, du)
                            st_use = np.where(v_t, t_raw, st_fused)
                            ti_use = np.where(v_i, ti_raw, ti_fused)
                            wa_use = np.where(v_w, g_raw, wa)

                            fused = _clamp_pred_arrays(
                                {
                                    "slope": sl_use,
                                    "dust": du_use,
                                    "surface_temp": st_use,
                                    "thermal_inertia": ti_use,
                                    "water": wa_use,
                                }
                            )

                            hybrid_scores = scorer.score_site_arrays(
                                fused["slope"],
                                fused["dust"],
                                fused["surface_temp"],
                                fused["thermal_inertia"],
                                fused["water"],
                            )

                            bad_hyb = ~np.isfinite(hybrid_scores)
                            hybrid_scores[bad_hyb] = OUTPUT_NODATA

                            cov_1d = np.where(v_s & v_d & v_t & v_i & v_w, 0.0, 1.0).astype(
                                np.float32
                            )
                            cov_1d[bad_hyb] = OUTPUT_NODATA

                            flat_hyb = out_block_hybrid[0].ravel()
                            flat_cov = cov_block[0].ravel()
                            flat_hyb[flat_el] = hybrid_scores.astype(np.float32, copy=False)
                            flat_cov[flat_el] = cov_1d
                            out_block_hybrid[0] = flat_hyb.reshape(win_h, win_w)
                            cov_block[0] = flat_cov.reshape(win_h, win_w)

                    dst_ml.write(out_block_ml, window=window)
                    if args.with_hybrid_coverage:
                        dst_hybrid.write(out_block_hybrid, window=window)
                        dst_cov.write(cov_block, window=window)

                    block_id += 1
                    elapsed = time.perf_counter() - t0
                    pct = 100.0 * block_id / total_blocks
                    rate = block_id / max(elapsed, 1e-9)
                    eta_s = (total_blocks - block_id) / max(rate, 1e-9)
                    print(
                        f"  [{block_id}/{total_blocks}] {pct:5.1f}%  "
                        f"window row={row} col={col}  "
                        f"elapsed={elapsed/60:.1f}m  ETA remaining ~{eta_s/60:.1f}m",
                        flush=True,
                    )

    print(f"Done. Wrote {args.output} (ML suitability 0–100%).", flush=True)
    if args.with_hybrid_coverage:
        print(f"       {args.hybrid_output} (raster-first hybrid)", flush=True)
        print(f"       {args.coverage_output} (coverage)", flush=True)
    else:
        print("       (Hybrid + coverage skipped; pass --with-hybrid-coverage to write them.)", flush=True)
    print("", flush=True)
    print("Summary (global map):", flush=True)
    print(f"  Total pixels:              {total_pixels:,}", flush=True)
    print(f"  Pixels scored:             {pixels_scored:,}", flush=True)
    print(f"  All five from raster:      {pixels_all_five_raster:,}", flush=True)
    print(f"  At least one ML fallback:  {pixels_ml_fallback:,}", flush=True)
    print(f"  Fully NoData (5 primaries): {pixels_fully_nodata:,}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Global landing suitability GeoTIFF from stacked inputs.")
    p.add_argument(
        "--input",
        default=DEFAULT_STACK_PATH,
        help=(
            f"13-band stacked GeoTIFF — model inputs only, never modified (default: {DEFAULT_STACK_PATH})"
        ),
    )
    p.add_argument(
        "--output",
        default=DEFAULT_ML_OUTPUT_PATH,
        help=f"ML suitability float32 GeoTIFF (default: {DEFAULT_ML_OUTPUT_PATH})",
    )
    p.add_argument(
        "--hybrid-output",
        default=DEFAULT_HYBRID_SUITABILITY_PATH,
        help=(
            f"Raster-first hybrid GeoTIFF (default: {DEFAULT_HYBRID_SUITABILITY_PATH}; "
            "used only with --with-hybrid-coverage)"
        ),
    )
    p.add_argument(
        "--coverage-output",
        default=DEFAULT_COVERAGE_PATH,
        help=f"ML coverage float32 GeoTIFF (default: {DEFAULT_COVERAGE_PATH}; used with --with-hybrid-coverage)",
    )
    p.add_argument(
        "--with-hybrid-coverage",
        action="store_true",
        help="Also write hybrid suitability + coverage rasters (default: off; only ML --output is written).",
    )
    p.add_argument("--block-size", type=int, default=1024, help="Processing window size in pixels.")
    p.add_argument(
        "--inference-subbatch",
        type=int,
        default=32768,
        help="Rows per Keras/XGB predict call within a window.",
    )
    p.add_argument("--compress", default="", help="Optional GDAL compression, e.g. LZW or DEFLATE.")
    p.add_argument("--tiled", action="store_true", help="Write tiled output (recommended for huge rasters).")
    p.add_argument("--blockxsize", type=int, default=512)
    p.add_argument("--blockysize", type=int, default=512)
    args = p.parse_args()
    if args.compress == "":
        args.compress = None
    run(args)


if __name__ == "__main__":
    main()
