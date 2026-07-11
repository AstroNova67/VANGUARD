#!/usr/bin/env python3
"""
Artifact audit for mars_landing_suitability_ml.tif and mars_global_input_stack_32ppd.tif.

Progress → stderr. Final report → stdout as JSON (pipe to a file).

Example::

    cd /path/to/VANGUARD
    uv run python scripts/audit_mars_suitability_artifacts.py 2>audit_progress.log | tee audit_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
import xgboost as xgb

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
BACKEND = os.path.join(REPO_ROOT, "backend")
DATA = os.path.join(REPO_ROOT, "frontend", "3d_globe", "public", "data")

sys.path.insert(0, BACKEND)
sys.path.insert(0, _SCRIPT_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from stack_mars_layers import LAYER_BASENAMES, resolve_layer_path  # noqa: E402
from scoring import (  # noqa: E402
    BASE_DIR,
    get_nn_models,
    inverse_transform_predictions_batch,
    load_scalers,
    raster_band_valid,
    scalers,
)

STACK_PATH = os.path.join(DATA, "mars_global_input_stack_32ppd.tif")
SUIT_PATH = os.path.join(DATA, "mars_landing_suitability_ml.tif")
TARGET_PPD = 32.0
OUTPUT_NODATA = -9999.0
RNG = np.random.default_rng(42)

STACK_LAYER_NAMES = (
    "elevation", "slope", "roughness", "albedo", "temperature", "tempRange",
    "crustalThickness", "ferric", "pyroxene", "basalt", "lambertAlbedo",
    "thermalInertia", "grsWaterWt",
)

NORM_RANGES = {
    "slope": (0, 5, True),
    "dust": (0.6, 0.7, True),
    "surface_temp": (-90, -40, False),
    "thermal_inertia": (100, 400, False),
    "water": (1, 8, False),
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def ppd(w: int) -> float:
    return w / 360.0


def clamp_stats(values: np.ndarray, lo: float, hi: float, invert: bool) -> dict:
    outside = (values < lo) | (values > hi)
    raw = (values - lo) / (hi - lo)
    if invert:
        normed = 1.0 - raw
        clamped_0 = values > hi
        clamped_1 = values < lo
    else:
        normed = raw
        clamped_0 = values < lo
        clamped_1 = values > hi
    return {
        "pct_outside": round(100.0 * float(outside.mean()), 3),
        "pct_clamped_0": round(100.0 * float(clamped_0.mean()), 3),
        "pct_clamped_1": round(100.0 * float(clamped_1.mean()), 3),
        "flag_gt_10pct_clamped": bool(
            float(clamped_0.mean() + clamped_1.mean()) * 100.0 > 10.0
        ),
    }


def blockiness_score(band: np.ndarray, mask: np.ndarray, max_pixels: int = 400_000) -> dict:
    valid = mask.copy()
    h, w = band.shape
    if int(valid.sum()) < 1000:
        return {
            "mean_diff_1px": None,
            "mean_diff_4px": None,
            "blockiness_ratio": None,
            "blocky": False,
        }
    ys, xs = np.where(valid)
    if ys.size > max_pixels:
        idx = RNG.choice(ys.size, max_pixels, replace=False)
        ys, xs = ys[idx], xs[idx]
    keep = (xs >= 4) & (xs < w - 4) & (ys >= 4) & (ys < h - 4)
    ys, xs = ys[keep], xs[keep]
    if ys.size == 0:
        return {
            "mean_diff_1px": None,
            "mean_diff_4px": None,
            "blockiness_ratio": None,
            "blocky": False,
        }
    v = band[ys, xs].astype(np.float64)
    d1 = (np.abs(band[ys, xs + 1] - v) + np.abs(band[ys + 1, xs] - v)) / 2.0
    d4 = (np.abs(band[ys, xs + 4] - v) + np.abs(band[ys + 4, xs] - v)) / 2.0
    m1 = float(d1.mean())
    m4 = float(d4.mean())
    ratio = m1 / m4 if m4 > 1e-12 else None
    return {
        "mean_diff_1px": round(m1, 6),
        "mean_diff_4px": round(m4, 6),
        "blockiness_ratio": round(ratio, 4) if ratio is not None else None,
        "blocky": bool(ratio is not None and ratio > 0.85),
    }


def lonlat_to_pixel(lon: float, lat: float, width: int, height: int) -> tuple[int, int]:
    x = int(((lon + 180.0) / 360.0) * width)
    y = int(((90.0 - lat) / 180.0) * height)
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _build_nn_feature_matrices(elev, slope, rough, alb, temp, tr, ferr, ti):
    slope_m = np.column_stack([alb, temp, rough, ferr, elev, tr, np.abs(slope), np.abs(slope * 0.1)])
    dust_m = np.column_stack([elev, slope, temp, temp, slope, alb])
    st_m = np.column_stack([elev, alb, ti, slope, rough])
    ti_m = np.column_stack([tr, alb, slope, ferr])
    water_m = slope_m
    return (
        scalers["slope"].transform(slope_m),
        scalers["dust_feature"].transform(dust_m),
        scalers["surface_temp"].transform(st_m),
        scalers["thermal_inertia_feature"].transform(ti_m),
        scalers["water"].transform(water_m),
    )


def _fuse_surface_temp_vectorized(nn_val, xgb_val, obs):
    nn_val = np.asarray(nn_val, dtype=np.float64)
    xgb_val = np.asarray(xgb_val, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    nn_ok = np.isfinite(nn_val)
    xgb_ok = np.isfinite(xgb_val) & (xgb_val >= -200.0) & (xgb_val <= 50.0)
    obs_ok = np.isfinite(obs)
    e_nn = np.where(nn_ok, np.abs(nn_val - obs), np.inf)
    e_xgb = np.where(xgb_ok, np.abs(xgb_val - obs), np.inf)
    out = np.zeros_like(nn_val, dtype=np.float64)
    m_no_obs = ~obs_ok
    out[m_no_obs & xgb_ok] = xgb_val[m_no_obs & xgb_ok]
    out[m_no_obs & ~xgb_ok & nn_ok] = nn_val[m_no_obs & ~xgb_ok & nn_ok]
    use_xgb = obs_ok & np.isfinite(e_xgb) & (e_xgb <= e_nn)
    use_nn = obs_ok & ~use_xgb & np.isfinite(e_nn)
    out[use_xgb] = xgb_val[use_xgb]
    out[use_nn] = nn_val[use_nn]
    return out


def _fuse_thermal_inertia_vectorized(nn_val, xgb_val, obs):
    nn_val = np.asarray(nn_val, dtype=np.float64)
    xgb_val = np.asarray(xgb_val, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    nn_ok = np.isfinite(nn_val)
    xgb_ok = np.isfinite(xgb_val) & (xgb_val >= 50.0) & (xgb_val <= 2000.0)
    obs_ok = np.isfinite(obs)
    e_nn = np.where(nn_ok, np.abs(nn_val - obs), np.inf)
    e_xgb = np.where(xgb_ok, np.abs(xgb_val - obs), np.inf)
    out = np.zeros_like(nn_val, dtype=np.float64)
    m_no_obs = ~obs_ok
    out[m_no_obs & xgb_ok] = xgb_val[m_no_obs & xgb_ok]
    out[m_no_obs & ~xgb_ok & nn_ok] = nn_val[m_no_obs & ~xgb_ok & nn_ok]
    use_xgb = obs_ok & np.isfinite(e_xgb) & (e_xgb <= e_nn)
    use_nn = obs_ok & ~use_xgb & np.isfinite(e_nn)
    out[use_xgb] = xgb_val[use_xgb]
    out[use_nn] = nn_val[use_nn]
    return out


def _json_default(o):
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def run_audit(n_sample: int, block_size: int) -> dict:
    t0 = time.time()
    results: dict = {}

    log("Task 1: upsampling / blockiness …")
    task1 = []
    with rasterio.open(STACK_PATH) as stk:
        sw, sh = stk.width, stk.height
    for i, basename in enumerate(LAYER_BASENAMES):
        src_path = resolve_layer_path(DATA, basename)
        with rasterio.open(src_path) as src:
            ow, oh = src.width, src.height
        native = ppd(ow)
        factor = TARGET_PPD / native
        row = {
            "band": i + 1,
            "name": STACK_LAYER_NAMES[i],
            "basename": basename,
            "source_path": src_path,
            "original_dims": f"{ow}×{oh}",
            "native_ppd": round(native, 2),
            "factor_to_32ppd": round(factor, 2),
            "direction": "downsample" if factor < 1 else ("none" if abs(factor - 1) < 0.01 else "upsample"),
            "resampling": "bilinear (align_mars_input_layers.py default)",
            "note": (
                "Source already 32 PPD on disk — alignment may have been done in-place."
                if abs(factor - 1.0) < 0.01
                else None
            ),
        }
        task1.append(row)

    with rasterio.open(STACK_PATH) as stk:
        h, w = stk.height, stk.width
        win = rasterio.windows.Window(w // 4, h // 4, w // 2, h // 2)
        for row in task1:
            bi = row["band"]
            arr = stk.read(bi, window=win).astype(np.float32)
            mask = raster_band_valid(arr, stk.nodatavals[bi - 1])
            row.update(blockiness_score(arr, mask))
    results["task1"] = task1

    log(f"Reservoir sampling {n_sample:,} eligible pixels …")
    load_scalers()
    nn = get_nn_models()
    xgb_st = xgb.XGBRegressor()
    xgb_st.load_model(
        os.path.join(BASE_DIR, "saved_models", "regression_models", "surface_temp", "xgb_model.json")
    )
    xgb_ti = xgb.XGBRegressor()
    xgb_ti.load_model(
        os.path.join(BASE_DIR, "saved_models", "regression_models", "thermal_inertia", "xgb_model.json")
    )

    sample_coords: list[tuple[int, int]] = []
    seen = 0
    with rasterio.open(STACK_PATH) as src:
        height, width = src.height, src.width
        for row0 in range(0, height, block_size):
            rh = min(block_size, height - row0)
            for col0 in range(0, width, block_size):
                cw = min(block_size, width - col0)
                data = src.read(window=rasterio.windows.Window(col0, row0, cw, rh))
                elig = (
                    raster_band_valid(data[1], src.nodatavals[1])
                    | raster_band_valid(data[4], src.nodatavals[4])
                    | raster_band_valid(data[7], src.nodatavals[7])
                    | raster_band_valid(data[11], src.nodatavals[11])
                    | raster_band_valid(data[12], src.nodatavals[12])
                )
                ys, xs = np.where(elig)
                for y, x in zip(ys, xs, strict=False):
                    seen += 1
                    gy, gx = row0 + int(y), col0 + int(x)
                    if len(sample_coords) < n_sample:
                        sample_coords.append((gy, gx))
                    else:
                        j = int(RNG.integers(0, seen))
                        if j < n_sample:
                            sample_coords[j] = (gy, gx)
    log(f"  eligible population: {seen:,}; sample: {len(sample_coords):,}")

    band_vals = {k: np.zeros(n_sample, dtype=np.float64) for k in STACK_LAYER_NAMES}
    band_valid = {k: np.zeros(n_sample, dtype=bool) for k in STACK_LAYER_NAMES}
    lats = np.zeros(n_sample, dtype=np.float64)
    lons = np.zeros(n_sample, dtype=np.float64)
    with rasterio.open(STACK_PATH) as src:
        height, width = src.height, src.width
        for si, (y, x) in enumerate(sample_coords):
            lats[si] = 90.0 - (y + 0.5) * 180.0 / height
            lons[si] = -180.0 + (x + 0.5) * 360.0 / width
            for bi, name in enumerate(STACK_LAYER_NAMES):
                v = float(src.read(bi + 1, window=rasterio.windows.Window(x, y, 1, 1))[0, 0])
                band_vals[name][si] = v
                band_valid[name][si] = bool(
                    raster_band_valid(np.array([v]), src.nodatavals[bi])[0]
                )

    log("Tasks 2–3: ML inference on sample …")

    def col(name: str, default: float) -> np.ndarray:
        arr = band_vals[name].copy()
        arr[~band_valid[name]] = default
        return arr

    e = col("elevation", 1000.0)
    s = col("slope", 2.0)
    r = col("roughness", 50.0)
    a = col("albedo", 0.2)
    t = col("temperature", -30.0)
    tr = col("tempRange", 50.0)
    fe = col("ferric", 0.5)
    ti = col("thermalInertia", 300.0)

    x_sl, x_du, x_st, x_ti, x_wa = _build_nn_feature_matrices(e, s, r, a, t, tr, fe, ti)
    raw_sl = nn["slope"].predict(x_sl, verbose=0).reshape(-1)
    raw_du = nn["dust"].predict(x_du, verbose=0).reshape(-1)
    raw_st = nn["surface_temp"].predict(x_st, verbose=0).reshape(-1)
    raw_ti = nn["thermal_inertia"].predict(x_ti, verbose=0).reshape(-1)
    raw_wa = nn["water"].predict(x_wa, verbose=0).reshape(-1)
    sl, du, st_nn, ti_nn, wa = inverse_transform_predictions_batch(
        raw_sl, raw_du, raw_st, raw_ti, raw_wa
    )
    st_xgb = np.asarray(
        xgb_st.predict(scalers["surface_temp"].transform(np.column_stack([e, a, ti, s, r]))),
        dtype=np.float64,
    ).reshape(-1)
    ti_xgb = np.asarray(
        xgb_ti.predict(np.column_stack([tr, a, s, fe])),
        dtype=np.float64,
    ).reshape(-1)
    st_fused = _fuse_surface_temp_vectorized(st_nn, st_xgb, band_vals["temperature"])
    ti_fused = _fuse_thermal_inertia_vectorized(ti_nn, ti_xgb, band_vals["thermalInertia"])
    ml = {
        "slope": np.maximum(sl, 0.0),
        "dust": np.clip(du, 0.0, 1.0),
        "surface_temp": st_fused,
        "thermal_inertia": np.maximum(ti_fused, 0.0),
        "water": np.clip(wa, 0.0, 8.0),
    }

    task2 = {prop: clamp_stats(ml[prop], *NORM_RANGES[prop]) for prop in NORM_RANGES}
    results["task2"] = task2

    obs_map = {
        "slope": ("slope", band_valid["slope"]),
        "dust": ("ferric", band_valid["ferric"]),
        "surface_temp": ("temperature", band_valid["temperature"]),
        "thermal_inertia": ("thermalInertia", band_valid["thermalInertia"]),
        "water": ("grsWaterWt", band_valid["grsWaterWt"]),
    }
    task3 = {}
    tile_delta = np.zeros((18, 36), dtype=np.float64)
    tile_count = np.zeros((18, 36), dtype=np.int64)
    for prop in NORM_RANGES:
        obs_name, mask = obs_map[prop]
        if int(mask.sum()) < 100:
            task3[prop] = {"note": "insufficient valid observations"}
            continue
        pred = ml[prop][mask]
        obs = band_vals[obs_name][mask]
        delta = np.abs(pred - obs)
        std = float(delta.std())
        thresh = 3.0 * std if std > 0 else np.inf
        task3[prop] = {
            "n_valid": int(mask.sum()),
            "mean_delta": round(float(delta.mean()), 4),
            "std_delta": round(std, 4),
            "p95_delta": round(float(np.percentile(delta, 95)), 4),
            "max_delta": round(float(delta.max()), 4),
            "anomalies_3sigma": int((delta > thresh).sum()),
            "pct_anomalies_3sigma": round(100.0 * float((delta > thresh).mean()), 3),
            "top5_hotspot_tiles": [],
        }
        lat_m = lats[mask]
        lon_m = lons[mask]
        for lat, lon, d in zip(lat_m, lon_m, delta, strict=False):
            ti = min(17, max(0, int((lat + 90) // 10)))
            li = min(35, max(0, int((lon + 180) // 10)))
            tile_delta[ti, li] += d
            tile_count[ti, li] += 1
        mean_tile = np.where(tile_count > 0, tile_delta / tile_count, 0)
        flat_idx = np.argsort(mean_tile.ravel())[::-1]
        hot = []
        for rank, idx in enumerate(flat_idx[:5], 1):
            ti, li = divmod(int(idx), 36)
            if tile_count[ti, li] == 0:
                continue
            lat_min = -90 + 10 * ti
            lon_min = -180 + 10 * li
            hot.append({
                "rank": rank,
                "lon_min": lon_min,
                "lon_max": lon_min + 10,
                "lat_min": lat_min,
                "lat_max": lat_min + 10,
                "mean_delta": round(float(mean_tile[ti, li]), 4),
                "n": int(tile_count[ti, li]),
            })
        task3[prop]["top5_hotspot_tiles"] = hot
        tile_delta.fill(0)
        tile_count.fill(0)
    results["task3"] = task3

    log("Task 4: MOLA elevation seams …")
    seam_reports = []
    max_jump = 0.0
    with rasterio.open(STACK_PATH) as src:
        elev = src.read(1).astype(np.float32)
        h, w = src.height, src.width
        valid = raster_band_valid(elev, src.nodatavals[0])
        for lon in (0, 90, 180, -90):
            x = lonlat_to_pixel(lon, 0, w, h)[0]
            if 1 <= x < w - 1:
                left = elev[:, x - 1]
                right = elev[:, x + 1]
                m = valid[:, x - 1] & valid[:, x + 1]
                if m.any():
                    j = np.abs(right[m] - left[m])
                    mj = float(j.max())
                    max_jump = max(max_jump, mj)
                    seam_reports.append({
                        "type": "lon_cross_seam",
                        "lon": lon,
                        "col": x,
                        "max_jump_m": round(mj, 2),
                        "mean_jump_m": round(float(j.mean()), 2),
                        "p95_jump_m": round(float(np.percentile(j, 95)), 2),
                    })
        for lat in (44, -44, 88, -88):
            y = lonlat_to_pixel(0, lat, w, h)[1]
            if 1 <= y < h - 1:
                top = elev[y - 1, :]
                bot = elev[y + 1, :]
                m = valid[y - 1, :] & valid[y + 1, :]
                if m.any():
                    j = np.abs(bot[m] - top[m])
                    mj = float(j.max())
                    max_jump = max(max_jump, mj)
                    seam_reports.append({
                        "type": "lat_cross_seam",
                        "lat": lat,
                        "row": y,
                        "max_jump_m": round(mj, 2),
                        "mean_jump_m": round(float(j.mean()), 2),
                        "p95_jump_m": round(float(np.percentile(j, 95)), 2),
                    })
    results["task4"] = {
        "seam_reports": seam_reports,
        "max_elevation_jump_m": round(max_jump, 2),
        "discontinuities_present": bool(max_jump > 200.0),
    }

    log("Task 5: NoData coverage …")
    total_px = 0
    valid_suit = 0
    lat_bins = np.zeros(18, dtype=np.int64)
    lat_valid = np.zeros(18, dtype=np.int64)
    band_nodata = {name: 0 for name in STACK_LAYER_NAMES}
    with rasterio.open(STACK_PATH) as src, rasterio.open(SUIT_PATH) as suit:
        h, w = src.height, src.width
        total_px = h * w
        for row0 in range(0, h, block_size):
            rh = min(block_size, h - row0)
            for col0 in range(0, w, block_size):
                cw = min(block_size, w - col0)
                win = rasterio.windows.Window(col0, row0, cw, rh)
                data = src.read(window=win)
                scores = suit.read(1, window=win).astype(np.float32)
                suit_ok = np.isfinite(scores) & (scores != OUTPUT_NODATA)
                valid_suit += int(suit_ok.sum())
                for bi, name in enumerate(STACK_LAYER_NAMES):
                    band_nodata[name] += int(
                        (~raster_band_valid(data[bi], src.nodatavals[bi])).sum()
                    )
                for local_y in range(rh):
                    gy = row0 + local_y
                    lat = 90.0 - (gy + 0.5) * 180.0 / h
                    bi = min(17, max(0, int((lat + 90) // 10)))
                    lat_bins[bi] += cw
                    lat_valid[bi] += int(suit_ok[local_y, :].sum())

    lat_nd = []
    for i in range(18):
        lat_min = -90 + 10 * i
        vp = 100.0 * lat_valid[i] / lat_bins[i] if lat_bins[i] else 0.0
        lat_nd.append({
            "lat_band": f"{lat_min}° to {lat_min + 10}°",
            "valid_pct": round(vp, 2),
            "nodata_pct": round(100.0 - vp, 2),
        })
    lat_nd.sort(key=lambda x: x["nodata_pct"], reverse=True)
    band_nd_pct = {
        k: round(100.0 * v / total_px, 4) for k, v in band_nodata.items()
    }
    results["task5"] = {
        "total_pixels": total_px,
        "valid_pixels": valid_suit,
        "nodata_pixels": total_px - valid_suit,
        "valid_pct": round(100.0 * valid_suit / total_px, 4),
        "band_nodata_pct": band_nd_pct,
        "top_latitude_nodata_bands": lat_nd[:5],
        "largest_nodata_contributor": max(band_nd_pct, key=band_nd_pct.get),
    }

    log("Task 6: score distribution …")
    all_scores = []
    eq_scores, mid_scores, pol_scores = [], [], []
    with rasterio.open(SUIT_PATH) as suit:
        h, w = suit.height, suit.width
        for row0 in range(0, h, block_size):
            rh = min(block_size, h - row0)
            for col0 in range(0, w, block_size):
                cw = min(block_size, w - col0)
                arr = suit.read(
                    1, window=rasterio.windows.Window(col0, row0, cw, rh)
                ).astype(np.float32)
                ok = np.isfinite(arr) & (arr != OUTPUT_NODATA)
                if not ok.any():
                    continue
                v = arr[ok]
                if v.size > 50_000:
                    v = RNG.choice(v, 50_000, replace=False)
                all_scores.append(v)
                for local_y in range(rh):
                    gy = row0 + local_y
                    lat_abs = abs(90.0 - (gy + 0.5) * 180.0 / h)
                    row_v = arr[local_y, ok[local_y]]
                    if row_v.size == 0:
                        continue
                    if lat_abs <= 30:
                        eq_scores.append(row_v)
                    elif lat_abs <= 60:
                        mid_scores.append(row_v)
                    else:
                        pol_scores.append(row_v)

    scores = np.concatenate(all_scores)
    hist, edges = np.histogram(scores, bins=10, range=(0, 100))
    hist_pct = (100.0 * hist / hist.sum()).round(2)
    eq = np.concatenate(eq_scores) if eq_scores else np.array([])
    mid = np.concatenate(mid_scores) if mid_scores else np.array([])
    pol = np.concatenate(pol_scores) if pol_scores else np.array([])
    skew = float(scores.mean() - np.median(scores))
    peaks = [
        i
        for i in range(1, 9)
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > hist.mean()
    ]
    results["task6"] = {
        "n_scores_sampled": int(scores.size),
        "histogram_bins": [
            {
                "range": f"{edges[i]:.0f}-{edges[i + 1]:.0f}",
                "count": int(hist[i]),
                "pct": float(hist_pct[i]),
            }
            for i in range(10)
        ],
        "mean": round(float(scores.mean()), 3),
        "std": round(float(scores.std()), 3),
        "median": round(float(np.median(scores)), 3),
        "skew_mean_minus_median": round(skew, 3),
        "distribution_shape": (
            "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "approximately symmetric")
        ),
        "multimodal_peak_bins": peaks,
        "latitudinal": {
            "equatorial_abs_lat_le_30": {
                "mean": round(float(eq.mean()), 3) if eq.size else None,
                "std": round(float(eq.std()), 3) if eq.size else None,
                "median": round(float(np.median(eq)), 3) if eq.size else None,
            },
            "mid_30_to_60": {
                "mean": round(float(mid.mean()), 3) if mid.size else None,
                "std": round(float(mid.std()), 3) if mid.size else None,
                "median": round(float(np.median(mid)), 3) if mid.size else None,
            },
            "polar_60_to_90": {
                "mean": round(float(pol.mean()), 3) if pol.size else None,
                "std": round(float(pol.std()), 3) if pol.size else None,
                "median": round(float(np.median(pol)), 3) if pol.size else None,
            },
        },
    }

    results["elapsed_s"] = round(time.time() - t0, 1)
    results["paths"] = {"stack": STACK_PATH, "suitability": SUIT_PATH}
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Audit mars_landing_suitability_ml.tif artifacts.")
    p.add_argument("--sample-size", type=int, default=50_000)
    p.add_argument("--block-size", type=int, default=1024)
    args = p.parse_args()

    for path in (STACK_PATH, SUIT_PATH):
        if not os.path.isfile(path):
            log(f"error: missing {path}")
            sys.exit(1)

    report = run_audit(args.sample_size, args.block_size)
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
