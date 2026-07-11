#!/usr/bin/env python3
"""
Derive Mars surface slope (degrees) from MOLA 128 PPD elevation, downsample, replace
the globe slope layer, and print a comparison report.

Uses rasterio + numpy only (no GDAL CLI).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, rowcol, xy
from rasterio.warp import reproject

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_ELEV = REPO_ROOT / "frontend" / "3d_globe" / "public" / "data" / "MOLA_128ppd_topo.tif"
DEFAULT_DERIVED = _SCRIPT_DIR / "derived_layers"
DEFAULT_PUBLIC_SLOPE = (
    REPO_ROOT / "frontend" / "3d_globe" / "public" / "data" / "mola_hrsc_blend_slope_v2.tif"
)
BACKUP_NAME = "mola_hrsc_blend_slope_v2_original_backup.tif"

MARS_R_M = 3_396_190.0
MARS_PROJ = "+proj=longlat +R=3396190 +no_defs"
WEST, SOUTH, EAST, NORTH = -180.0, -90.0, 180.0, 90.0
DEG2RAD = np.pi / 180.0

# Validation sites (lon °E, positive east)
JEZERO = (18.4, 77.7)
OLYMPUS = (18.6, 226.2)  # 226.2°E → −133.8° in −180…180

PPD_TARGETS = (
    (128, 46080, 23040, "mola_slope_128ppd.tif"),
    (32, 11520, 5760, "mola_slope_32ppd.tif"),
    (20, 7200, 3600, "mola_slope_20ppd.tif"),
    (8, 2880, 1440, "mola_slope_8ppd.tif"),
)


def _lon_east_to_180(lon_e: float) -> float:
    lon = float(lon_e)
    if lon > 180.0:
        lon -= 360.0
    return lon


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _ppd(width: int) -> float:
    return width / 360.0


def _sample_geotiff(path: Path, lat: float, lon_e: float) -> float | None:
    lon = _lon_east_to_180(lon_e)
    with rasterio.open(path) as src:
        if src.crs is None or src.transform.is_identity:
            x = int(((lon + 180.0) / 360.0) * src.width)
            y = int(((90.0 - lat) / 180.0) * src.height)
            x = max(0, min(src.width - 1, x))
            y = max(0, min(src.height - 1, y))
        else:
            r, c = rowcol(src.transform, lon, lat)
            x, y = int(c), int(r)
            x = max(0, min(src.width - 1, x))
            y = max(0, min(src.height - 1, y))
        val = float(src.read(1, window=rasterio.windows.Window(x, y, 1, 1))[0, 0])
        if src.nodata is not None and val == src.nodata:
            return None
        if not np.isfinite(val):
            return None
        return val


def _elev_valid(z: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(z)
    if nodata is not None and np.isfinite(nodata):
        valid &= z != nodata
    return valid


def _compute_slope_block(
    elev: np.ndarray,
    lat_deg: np.ndarray,
    pixel_w_deg: float,
    pixel_h_deg: float,
    nodata: float | None,
) -> np.ndarray:
    """
    Central-difference slope (degrees) on a padded block elev shape (h+2, w+2).
    lat_deg length h (interior rows).
    """
    h, w = elev.shape[0] - 2, elev.shape[1] - 2
    out = np.full((h, w), np.nan, dtype=np.float32)
    valid = _elev_valid(elev, nodata)
    # Require 3×3 neighborhood valid for central differences
    core = valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    core &= valid[:-2, 1:-1] & valid[2:, 1:-1]

    lat_rad = np.deg2rad(lat_deg)
    mx = (MARS_R_M * np.cos(lat_rad) * pixel_w_deg * DEG2RAD).astype(np.float64)
    my = float(MARS_R_M * pixel_h_deg * DEG2RAD)

    dz_dx = (elev[1:-1, 2:].astype(np.float64) - elev[1:-1, :-2].astype(np.float64)) / (
        2.0 * mx[:, None]
    )
    dz_dy = (elev[2:, 1:-1].astype(np.float64) - elev[:-2, 1:-1].astype(np.float64)) / (2.0 * my)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    out[core] = slope[core].astype(np.float32)
    return out


def derive_slope_128ppd(
    elev_path: Path,
    out_path: Path,
    *,
    chunk_rows: int = 256,
) -> None:
    print(f"\n=== Derive slope (128 PPD) ===")
    print(f"  input:  {elev_path}")
    print(f"  output: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(elev_path) as src:
        if src.width != 46080 or src.height != 23040:
            print(
                f"  warning: expected 46080×23040, got {src.width}×{src.height}",
                file=sys.stderr,
            )
        transform = src.transform
        crs = src.crs or CRS.from_proj4(MARS_PROJ)
        nodata = src.nodata
        pixel_w_deg = abs(transform.a)
        pixel_h_deg = abs(transform.e)

        profile = {
            "driver": "GTiff",
            "width": src.width,
            "height": src.height,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": np.nan,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        n_rows = src.height
        n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
        t0 = time.time()

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.set_band_description(1, "Slope from MOLA 128 ppd elevation (degrees)")

            for chunk_i, row0 in enumerate(range(0, n_rows, chunk_rows)):
                row1 = min(row0 + chunk_rows, n_rows)
                # Pad one row above/below for central differences
                read_row0 = max(0, row0 - 1)
                read_row1 = min(n_rows, row1 + 1)
                win = rasterio.windows.Window(0, read_row0, src.width, read_row1 - read_row0)
                block = src.read(1, window=win).astype(np.float32)

                # Longitude padding for dateline: wrap east-west
                if block.shape[1] >= 2:
                    padded = np.empty(
                        (block.shape[0], block.shape[1] + 2), dtype=np.float32
                    )
                    padded[:, 1:-1] = block
                    padded[:, 0] = block[:, -1]
                    padded[:, -1] = block[:, 0]
                else:
                    padded = block

                out_h = row1 - row0
                pad_row_start = row0 - read_row0
                need = out_h + 2
                interior = padded[pad_row_start : pad_row_start + need, :]
                if interior.shape[0] < need:
                    pad_n = need - interior.shape[0]
                    interior = np.vstack(
                        [interior, np.repeat(interior[-1:], pad_n, axis=0)]
                    )

                n_out = interior.shape[0] - 2
                global_rows = np.arange(row0, row0 + n_out) + 0.5
                lats = NORTH - global_rows * pixel_h_deg

                slope_block = _compute_slope_block(
                    interior, lats, pixel_w_deg, pixel_h_deg, nodata
                )

                write_win = rasterio.windows.Window(0, row0, src.width, n_out)
                dst.write(slope_block, 1, window=write_win)

                done = row1
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (n_rows - done) / rate if rate > 0 else 0.0
                pct = 100.0 * done / n_rows
                print(
                    f"  chunk {chunk_i + 1}/{n_chunks}: rows {row0}–{row1 - 1} "
                    f"({pct:5.1f}%) — {remaining / 60.0:.1f} min remaining",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"  finished in {elapsed / 60.0:.1f} min ({_human_bytes(out_path.stat().st_size)})")


def downsample_slope(
    src_path: Path,
    dst_path: Path,
    width: int,
    height: int,
) -> None:
    print(f"  downsample → {dst_path.name} ({width}×{height}, {width / 360.0:.0f} PPD)")
    dst_transform = from_bounds(WEST, SOUTH, EAST, NORTH, width, height)
    crs = CRS.from_proj4(MARS_PROJ)

    with rasterio.open(src_path) as src:
        src_arr = src.read(1, masked=True).astype(np.float32)
        src_data = np.where(src_arr.mask, np.nan, src_arr.data).astype(np.float32)
        dst_data = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dst_data, 1)
        dst.set_band_description(1, f"Slope from MOLA ({width / 360.0:.0f} PPD, degrees)")


def downsample_all(derived_dir: Path, src_128: Path) -> None:
    print("\n=== Downsample slope ===")
    for ppd, width, height, name in PPD_TARGETS[1:]:
        downsample_slope(src_128, derived_dir / name, width, height)


def backup_and_replace(derived_dir: Path, public_slope: Path) -> None:
    print("\n=== Replace public slope layer ===")
    src_32 = derived_dir / "mola_slope_32ppd.tif"
    backup = derived_dir / BACKUP_NAME
    if not src_32.is_file():
        raise FileNotFoundError(src_32)

    if public_slope.is_file() and not backup.is_file():
        shutil.copy2(public_slope, backup)
        print(f"  backed up → {backup}")
    elif backup.is_file():
        print(f"  backup already exists: {backup}")

    shutil.copy2(src_32, public_slope)
    print(f"  installed {public_slope.name} ({_human_bytes(public_slope.stat().st_size)})")


def verify_pipeline(public_slope: Path) -> None:
    print("\n=== Pipeline verification ===")
    with rasterio.open(public_slope) as src:
        w, h = src.width, src.height
        crs_ok = src.crs is not None and "3396190" in (src.crs.to_proj4() or "")
        exp_t = from_bounds(WEST, SOUTH, EAST, NORTH, w, h)
        t_ok = src.transform.almost_equals(exp_t, precision=1e-9)
        print(f"  dimensions: {w}×{h} ({_ppd(w):.1f} PPD)")
        print(f"  CRS Mars R=3396190: {crs_ok}")
        print(f"  global equirectangular transform: {t_ok}")
        if w != 11520 or h != 5760:
            print(
                "  note: align_mars_input_layers.py default grid is 11520×5760 (32 PPD)",
                file=sys.stderr,
            )

    jezero = _sample_geotiff(public_slope, *JEZERO)
    print(f"  Jezero slope: {jezero}°" if jezero is not None else "  Jezero: no data")
    if jezero is None or not (0.0 <= jezero <= 5.0):
        print("  WARNING: Jezero slope outside expected 0–5°", file=sys.stderr)
    else:
        print("  Jezero: OK (0–5°)")


def _score_band(score: float) -> int:
    """Decile band 0–9 for suitability percent 0–100."""
    return min(9, max(0, int(score // 10)))


def comparison_report(
    derived_dir: Path,
    public_slope: Path,
    *,
    n_samples: int = 10_000,
    seed: int = 42,
) -> None:
    print("\n=== Comparison report ===")
    backup = derived_dir / BACKUP_NAME
    old_path = backup if backup.is_file() else public_slope
    new_path = public_slope

    with rasterio.open(old_path) as old, rasterio.open(new_path) as new:
        old_w, old_h = old.width, old.height
        new_w, new_h = new.width, new.height
        print(f"  Old ({old_path.name}): {old_w}×{old_h} (~{_ppd(old_w):.1f} PPD)")
        print(f"  New ({new_path.name}): {new_w}×{new_h} (~{_ppd(new_w):.1f} PPD)")

    for label, lat, lon_e in (("Jezero", *JEZERO), ("Olympus", *OLYMPUS)):
        print(f"\n  {label} ({lat}°N, {lon_e}°E):")
        for tag, path in (("old", old_path), ("new", new_path)):
            v = _sample_geotiff(path, lat, lon_e)
            print(f"    {tag}: {v if v is not None else 'nodata'}°")

    # Score-band comparison via LandingSuitabilityScorer
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from backend.scoring import LandingSuitabilityScorer

    scorer = LandingSuitabilityScorer()
    rng = np.random.default_rng(seed)
    # Fixed mid-range non-slope inputs
    dust, st, ti, water = 0.65, -60.0, 250.0, 4.0

    changed = 0
    valid = 0
    for _ in range(n_samples):
        lat = float(rng.uniform(-60.0, 60.0))
        lon_e = float(rng.uniform(0.0, 360.0))
        old_s = _sample_geotiff(old_path, lat, lon_e)
        new_s = _sample_geotiff(new_path, lat, lon_e)
        if old_s is None or new_s is None:
            continue
        valid += 1
        s_old = scorer.score_site(old_s, dust, st, ti, water)
        s_new = scorer.score_site(new_s, dust, st, ti, water)
        if _score_band(s_old) != _score_band(s_new):
            changed += 1

    pct = 100.0 * changed / valid if valid else 0.0
    print(
        f"\n  Suitability score decile change (n={valid} valid of {n_samples} samples, "
        f"other inputs fixed): {changed} ({pct:.1f}%)"
    )


def _max_slope_near(path: Path, lat: float, lon_e: float, delta_deg: float = 0.5) -> float | None:
    """Peak slope within a lat/lon box (for caldera rims wider than one pixel)."""
    lon = _lon_east_to_180(lon_e)
    with rasterio.open(path) as src:
        r0, c0 = rowcol(src.transform, lon - delta_deg, lat + delta_deg)
        r1, c1 = rowcol(src.transform, lon + delta_deg, lat - delta_deg)
        row0, row1 = sorted((int(r0), int(r1)))
        col0, col1 = sorted((int(c0), int(c1)))
        row0 = max(0, row0)
        col0 = max(0, col0)
        row1 = min(src.height, row1 + 1)
        col1 = min(src.width, col1 + 1)
        if row1 <= row0 or col1 <= col0:
            return None
        block = src.read(
            1, window=rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        )
        if block.size == 0:
            return None
        peak = float(np.nanmax(block))
        return peak if np.isfinite(peak) else None


def validate_derived(path: Path, label: str) -> None:
    print(f"\n=== Validate {label} ===")
    j = _sample_geotiff(path, *JEZERO)
    o_pt = _sample_geotiff(path, *OLYMPUS)
    o_max = _max_slope_near(path, *OLYMPUS, delta_deg=0.5)
    print(f"  Jezero (point): {j}° (expect ~0.5–2°)")
    print(f"  Olympus (point): {o_pt}°; max within ±0.5°: {o_max}° (expect rim >10°)")
    if j is not None and not (0.2 <= j <= 2.5):
        print("  WARNING: Jezero outside ~0.5–2°", file=sys.stderr)
    if o_max is not None and o_max <= 10.0:
        print("  WARNING: no rim slopes >10° in Olympus neighborhood", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--elevation", type=Path, default=DEFAULT_ELEV)
    p.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED)
    p.add_argument("--public-slope", type=Path, default=DEFAULT_PUBLIC_SLOPE)
    p.add_argument("--chunk-rows", type=int, default=256)
    p.add_argument("--skip-derive", action="store_true")
    p.add_argument("--skip-downsample", action="store_true")
    p.add_argument("--skip-replace", action="store_true")
    p.add_argument("--skip-report", action="store_true")
    args = p.parse_args()

    derived = args.derived_dir.resolve()
    elev = args.elevation.resolve()
    out_128 = derived / "mola_slope_128ppd.tif"

    if not elev.is_file():
        raise SystemExit(f"Elevation not found: {elev}")

    if not args.skip_derive:
        derive_slope_128ppd(elev, out_128, chunk_rows=args.chunk_rows)
        validate_derived(out_128, "128 PPD slope")

    if not out_128.is_file():
        raise SystemExit(f"Missing {out_128}; run without --skip-derive")

    if not args.skip_downsample:
        downsample_all(derived, out_128)
        validate_derived(derived / "mola_slope_32ppd.tif", "32 PPD slope")

    if not args.skip_replace:
        backup_and_replace(derived, args.public_slope.resolve())
        verify_pipeline(args.public_slope.resolve())

    if not args.skip_report:
        comparison_report(derived, args.public_slope.resolve())

    print("\nDone.")


if __name__ == "__main__":
    main()
