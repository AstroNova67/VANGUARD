#!/usr/bin/env python3
"""
Smooth MOLA 128 PPD full-res elevation at PDS tile seams.

Reads ``scripts/derived_layers/MOLA_128ppd_topo_fullres.tif``, applies a
Gaussian blend (width 32 px, σ = half-width/2) centered on tile boundaries at
0°/90°/180°/270° longitude and ±44°/±88° latitude, and writes
``MOLA_128ppd_topo_fullres_smoothed.tif`` without overwriting the original.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter1d

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_SCRIPT_DIR, "derived_layers", "MOLA_128ppd_topo_fullres.tif")
DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "derived_layers", "MOLA_128ppd_topo_fullres_smoothed.tif")

PPD = 128
HALF_WIDTH = 16  # 32-pixel kernel centered on seam


def _lat_row(lat_deg: float, height: int) -> int:
    """Row index for north-up equirectangular grid (90°N → row 0)."""
    return int(round((90.0 - lat_deg) * PPD))


def _lon_col(lon_deg: float, width: int) -> int:
    """Column index for −180…180° longitude."""
    lon = ((lon_deg + 180.0) % 360.0) - 180.0
    return int(round((lon + 180.0) * PPD)) % width


def _smooth_horizontal_seam(data: np.ndarray, row: int, half_width: int, sigma: float) -> None:
    h, _ = data.shape
    r0 = max(0, row - half_width)
    r1 = min(h, row + half_width)
    if r1 - r0 < 2:
        return
    strip = data[r0:r1, :].astype(np.float64, copy=True)
    smoothed = gaussian_filter1d(strip, sigma=sigma, axis=0, mode="nearest")
    data[r0:r1, :] = smoothed.astype(data.dtype, copy=False)


def _smooth_vertical_seam(data: np.ndarray, col: int, half_width: int, sigma: float) -> None:
    _, w = data.shape
    c0 = max(0, col - half_width)
    c1 = min(w, col + half_width)
    if c1 - c0 < 2:
        return
    strip = data[:, c0:c1].astype(np.float64, copy=True)
    smoothed = gaussian_filter1d(strip, sigma=sigma, axis=1, mode="nearest")
    data[:, c0:c1] = smoothed.astype(data.dtype, copy=False)


def _max_seam_jump(data: np.ndarray, row: int) -> float:
    """Max |Δelevation| across the seam row vs its immediate neighbors."""
    if row <= 0 or row >= data.shape[0] - 1:
        return 0.0
    above = data[row - 1, :].astype(np.float64)
    seam = data[row, :].astype(np.float64)
    below = data[row + 1, :].astype(np.float64)
    return float(max(np.nanmax(np.abs(seam - above)), np.nanmax(np.abs(below - seam))))


def main() -> None:
    p = argparse.ArgumentParser(description="Gaussian-blend MOLA full-res tile seams.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--half-width", type=int, default=HALF_WIDTH)
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"error: missing input {args.input}", file=sys.stderr)
        sys.exit(1)

    sigma = args.half_width / 2.0
    lat_seams = (88.0, 44.0, -44.0, -88.0)
    lon_seams = (0.0, 90.0, 180.0, 270.0)

    print(f"Reading {args.input} …", flush=True)
    with rasterio.open(args.input) as src:
        profile = src.profile.copy()
        data = src.read(1).astype(np.float32)
        h, w = data.shape

    lat_rows = [_lat_row(lat, h) for lat in lat_seams]
    lon_cols = [_lon_col(lon, w) for lon in lon_seams]
    # 180° seam also appears at column 0 when mosaic wraps.
    if 0 not in lon_cols:
        lon_cols.append(0)

    row_88n = _lat_row(88.0, h)
    row_88s = _lat_row(-88.0, h)
    jump_before_88n = _max_seam_jump(data, row_88n)
    jump_before_88s = _max_seam_jump(data, row_88s)
    print(f"±88° seam max jump BEFORE — 88°N row {row_88n}: {jump_before_88n:.1f} m; "
          f"88°S row {row_88s}: {jump_before_88s:.1f} m")

    print(f"Smoothing {len(lat_rows)} latitude + {len(lon_cols)} longitude seams "
          f"(half_width={args.half_width}, σ={sigma}) …", flush=True)
    for row in lat_rows:
        _smooth_horizontal_seam(data, row, args.half_width, sigma)
    for col in lon_cols:
        _smooth_vertical_seam(data, col, args.half_width, sigma)

    jump_after_88n = _max_seam_jump(data, row_88n)
    jump_after_88s = _max_seam_jump(data, row_88s)
    print(f"±88° seam max jump AFTER  — 88°N row {row_88n}: {jump_after_88n:.1f} m; "
          f"88°S row {row_88s}: {jump_after_88s:.1f} m")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Writing {args.output} …", flush=True)
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(data, 1)

    print("Done.")


if __name__ == "__main__":
    main()
