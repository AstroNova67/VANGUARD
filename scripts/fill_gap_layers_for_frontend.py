#!/usr/bin/env python3
"""
Fill instrument-gap zeros in Mars Geoscience layers; write a frontend-ready data folder.

Outputs to: frontend/3d_globe/public/data_gap_filled/
Score comparison (separate): scripts/compare_gap_fill_scores.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
SRC = Path.home() / "Downloads" / "Mars Geoscience"
OUT = REPO / "frontend" / "3d_globe" / "public" / "data_gap_filled"
OLD_PUBLIC = REPO / "frontend" / "3d_globe" / "public" / "data"


# Mars Geoscience name → frontend filename
FILL_LAYERS = {
    "omega_ferric_nnphs.tif": "omega_ferric_nnphs.tif",
    "omega_albedo_r1080.tif": "omega_albedo_r1080.tif",
    "mola_roughness_0.6km_numeric.tif": "mola_roughness_0.6km_numeric.tif",
    "mars_yearly_temperature_range_v1.0.tif": "mars_yearly_temperature_range_v1.0.tif",
    "mola_hrsc_blend_topo_v2.tif": "MOLA_128ppd_topo.tif",
}

# Copy as-is (no gap fill)
COPY_LAYERS = {
    "mola_hrsc_blend_slope_v2.tif": "mola_hrsc_blend_slope_v2.tif",
    "mars_yearly_avg_temperature_celsius.tif": "mars_yearly_avg_temperature_celsius.tif",
    "mars_crustal_thickness_gmm3_rm1.tif": "mars_crustal_thickness_gmm3_rm1.tif",
    "mars_odyssey_grs_mons_perc_wt.tif": "mars_odyssey_grs_mons_perc_wt.tif",
    "TES_Basalt_numeric.tif": "TES_Basalt_numeric.tif",
    "omega_pyroxene_bd2000.tif": "omega_pyroxene_bd2000.tif",
}

# Still needed by frontend; take from existing public/data
KEEP_FROM_PUBLIC = (
    "TES_Lambert_Albedo_numeric.tif",
    "tes_dayside_ti_putzig_2007.tif",
    "mars_landing_suitability_ml.tif",
)

K_NEIGHBORS = 24
GAUSS_SIGMA = 6.0
FEATHER_PX = 20


def mars_transform(width: int, height: int):
    return from_bounds(-180, -90, 180, 90, width, height)


def null_mask(data: np.ndarray, nodata=None) -> np.ndarray:
    bad = (data == 0) | ~np.isfinite(data)
    if nodata is not None:
        try:
            nd = float(nodata)
            if np.isfinite(nd):
                bad |= data == nd
        except (TypeError, ValueError):
            pass
    return bad


def idw_fill(data: np.ndarray, bad: np.ndarray, k_neighbors: int = 24) -> np.ndarray:
    out = data.astype(np.float64, copy=True)
    if not bad.any():
        return out.astype(data.dtype)
    valid = ~bad
    fringe = valid & binary_dilation(bad, iterations=3)
    rs, cs = np.where(fringe if fringe.any() else valid)
    if len(rs) == 0:
        raise ValueError("no valid pixels to interpolate from")
    tree = cKDTree(np.column_stack([rs, cs]).astype(np.float64))
    values = out[rs, cs]
    qr, qc = np.where(bad)
    k = min(k_neighbors, len(values))
    batch = 50_000
    for start in range(0, len(qr), batch):
        end = min(start + batch, len(qr))
        q = np.column_stack([qr[start:end], qc[start:end]]).astype(np.float64)
        dists, idxs = tree.query(q, k=k, workers=-1)
        if dists.ndim == 1:
            dists, idxs = dists[:, None], idxs[:, None]
        w = 1.0 / np.maximum(dists, 1e-6) ** 2
        out[qr[start:end], qc[start:end]] = (w * values[idxs]).sum(axis=1) / w.sum(axis=1)
    return out.astype(data.dtype)


def gaussian_blend(data: np.ndarray, bad: np.ndarray, sigma: float = 6.0, feather_px: float = 20) -> np.ndarray:
    smoothed = gaussian_filter(data.astype(np.float64), sigma=sigma, mode="nearest")
    dist_out = distance_transform_edt(~bad)
    alpha = np.zeros(data.shape, dtype=np.float64)
    alpha[bad] = 1.0
    edge = (~bad) & (dist_out <= feather_px)
    if feather_px > 0:
        alpha[edge] = 1.0 - dist_out[edge] / feather_px
    out = alpha * smoothed + (1.0 - alpha) * data.astype(np.float64)
    return out.astype(data.dtype)


def write_geotiff(path: Path, data: np.ndarray, nodata=None) -> None:
    h, w = data.shape
    transform = mars_transform(w, h)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": data.dtype.name if data.dtype.name != "float64" else "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    arr = data.astype(profile["dtype"])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def fill_and_write(src_name: str, dst_name: str) -> dict:
    src_path = SRC / src_name
    dst_path = OUT / dst_name
    t0 = time.perf_counter()
    with rasterio.open(src_path) as src:
        raw = src.read(1).astype(np.float32)
        nodata = src.nodata
    bad = null_mask(raw, nodata)
    n_bad = int(bad.sum())
    if n_bad == 0:
        write_geotiff(dst_path, raw, nodata=None)
        return {
            "src": src_name,
            "dst": dst_name,
            "filled": 0,
            "pct": 0.0,
            "seconds": time.perf_counter() - t0,
        }
    filled = idw_fill(raw, bad, k_neighbors=K_NEIGHBORS)
    filled = gaussian_blend(filled, bad, sigma=GAUSS_SIGMA, feather_px=FEATHER_PX)
    filled = filled.copy()
    filled[bad] = np.where(filled[bad] == 0, 1e-6, filled[bad])
    write_geotiff(dst_path, filled.astype(np.float32), nodata=None)
    remaining = int(null_mask(filled).sum())
    return {
        "src": src_name,
        "dst": dst_name,
        "filled": n_bad,
        "remaining_zeros": remaining,
        "pct": 100.0 * n_bad / raw.size,
        "shape": list(raw.shape),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def copy_as_is(src_name: str, dst_name: str) -> None:
    with rasterio.open(SRC / src_name) as src:
        data = src.read(1)
    write_geotiff(OUT / dst_name, data)


def main() -> int:
    if not SRC.is_dir():
        print(f"ERROR: missing source folder {SRC}", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(f"Source: {SRC}")
    print(f"Output: {OUT}")
    fill_report = []
    for src_name, dst_name in FILL_LAYERS.items():
        print(f"Filling {src_name} -> {dst_name} …", flush=True)
        rec = fill_and_write(src_name, dst_name)
        fill_report.append(rec)
        print(
            f"  filled {rec['filled']:,} ({rec['pct']:.2f}%) in {rec['seconds']}s",
            flush=True,
        )

    for src_name, dst_name in COPY_LAYERS.items():
        print(f"Copy+georef {src_name} -> {dst_name}", flush=True)
        copy_as_is(src_name, dst_name)

    for name in KEEP_FROM_PUBLIC:
        src = OLD_PUBLIC / name
        if src.is_file():
            print(f"Keep from public/data: {name}", flush=True)
            shutil.copy2(src, OUT / name)
        else:
            print(f"WARNING: missing {src}", flush=True)

    meta = {
        "source": str(SRC),
        "output": str(OUT),
        "fill_method": "idw_gaussian",
        "params": {
            "K_NEIGHBORS": K_NEIGHBORS,
            "GAUSS_SIGMA": GAUSS_SIGMA,
            "FEATHER_PX": FEATHER_PX,
        },
        "filled_layers": fill_report,
        "copied_unfilled": list(COPY_LAYERS.values()),
        "kept_from_public": [n for n in KEEP_FROM_PUBLIC if (OUT / n).exists()],
    }
    (OUT / "fill_manifest.json").write_text(json.dumps(meta, indent=2))
    print("Wrote fill_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
