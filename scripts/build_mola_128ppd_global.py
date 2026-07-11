#!/usr/bin/env python3
"""
Download MOLA MEGDR 128 PPD topography tiles (megt*hb) from NASA PDS,
convert to georeferenced GeoTIFFs, mosaic to a global Mars GeoTIFF, verify, and
optionally install under frontend/3d_globe/public/data/.

Uses rasterio + numpy only (no GDAL CLI).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_bounds

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent

PDS_MEG128_URL = (
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg128"
)
MARS_CRS = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
GLOBAL_WEST, GLOBAL_SOUTH, GLOBAL_EAST, GLOBAL_NORTH = -180.0, -90.0, 180.0, 90.0
GLOBAL_WIDTH = 360 * 128  # 46080
GLOBAL_HEIGHT = 180 * 128  # 23040
PPD = 128.0

DEFAULT_RAW_DIR = _SCRIPT_DIR / "mola_128ppd_raw"
DEFAULT_GLOBAL_NAME = "MOLA_128ppd_global.tif"
DEFAULT_PUBLIC_NAME = "MOLA_128ppd_topo.tif"
PUBLIC_DEST = REPO_ROOT / "frontend" / "3d_globe" / "public" / "data" / DEFAULT_PUBLIC_NAME

# 16 tiles: 4 latitude bands × 4 longitude quadrants (non-polar MEG128 set)
TILE_STEMS = [
    f"megt{lat}{hem}{lon}hb"
    for lat, hem in [("88", "n"), ("44", "n"), ("44", "s"), ("00", "n")]
    for lon in ("000", "090", "180", "270")
]

_LABEL_PAIR = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.+)$", re.MULTILINE)


def _parse_pds_label(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="ascii", errors="replace")
    out: dict[str, str] = {}
    for m in _LABEL_PAIR.finditer(text):
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith('"'):
            end = raw.find('"', 1)
            val = raw[1:end] if end > 0 else raw.strip('"')
        else:
            val = raw.split("<")[0].strip()
        out[key] = val
    return out


def _float_val(meta: dict[str, str], key: str) -> float:
    return float(meta[key])


def _int_val(meta: dict[str, str], key: str) -> int:
    return int(float(meta[key]))


def _lon_bounds_180(west: float, east: float) -> tuple[float, float]:
    """Convert PDS 0–360°E quadrant bounds to −180…180 with west < east."""
    west = west % 360.0
    east = east % 360.0
    if east <= 180.0:
        w = west if west <= 180.0 else west - 360.0
        return w, east
    if west >= 180.0:
        return west - 360.0, east - 360.0
    raise ValueError(f"Longitude quadrant crosses 180° meridian: {west}–{east}")


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def check_disk_space(raw_dir: Path, min_gb: float = 3.0) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(raw_dir)
    avail_gb = usage.free / (1024**3)
    print(f"Disk: {_human_bytes(usage.free)} free on {raw_dir.parent} (need ~{min_gb:.1f} GB)")
    if avail_gb < min_gb:
        raise SystemExit(f"Insufficient disk space: {avail_gb:.2f} GB free, need ~{min_gb} GB")


def download_file(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name} ({_human_bytes(dest.stat().st_size)})")
        return

    print(f"  downloading {dest.name} …")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total > 0 and done % (10 << 20) < len(chunk):
                    pct = 100.0 * done / total
                    print(f"    … {pct:5.1f}% ({_human_bytes(done)} / {_human_bytes(total)})")

    elapsed = time.time() - t0
    print(f"    done {dest.name} ({_human_bytes(dest.stat().st_size)}) in {elapsed:.1f}s")


def download_tiles(raw_dir: Path, base_url: str) -> None:
    print(f"\n=== Download {len(TILE_STEMS)} MEGDR tiles → {raw_dir} ===")
    for stem in TILE_STEMS:
        for ext in (".lbl", ".img"):
            name = stem + ext
            download_file(f"{base_url}/{name}", raw_dir / name)


def read_tile_array(img_path: Path, meta: dict[str, str]) -> np.ndarray:
    lines = _int_val(meta, "LINES")
    samples = _int_val(meta, "LINE_SAMPLES")
    bits = _int_val(meta, "SAMPLE_BITS")
    record_bytes = _int_val(meta, "RECORD_BYTES")
    file_records = _int_val(meta, "FILE_RECORDS")

    expected = lines * samples * (bits // 8)
    file_bytes = record_bytes * file_records
    if file_bytes != expected:
        print(
            f"  note: {img_path.name} FILE_RECORDS×RECORD_BYTES={file_bytes} "
            f"vs LINES×SAMPLES×2={expected}"
        )

    dtype = ">i2" if bits == 16 else None
    if dtype is None:
        raise ValueError(f"Unsupported SAMPLE_BITS={bits} in {img_path}")

    # Detached .IMG is raw binary (no PDS header); optional ^IMAGE OFFSET not used here.
    data = np.fromfile(img_path, dtype=dtype)
    if data.size != lines * samples:
        raise ValueError(
            f"{img_path}: expected {lines * samples} samples, got {data.size}"
        )
    return data.reshape(lines, samples)


def tile_geotiff_path(raw_dir: Path, stem: str) -> Path:
    return raw_dir / f"{stem}_georef.tif"


def convert_tile(raw_dir: Path, stem: str, *, force: bool = False) -> Path:
    lbl_path = raw_dir / f"{stem}.lbl"
    img_path = raw_dir / f"{stem}.img"
    out_path = tile_geotiff_path(raw_dir, stem)

    meta = _parse_pds_label(lbl_path)
    arr = read_tile_array(img_path, meta)

    lat_s = _float_val(meta, "MINIMUM_LATITUDE")
    lat_n = _float_val(meta, "MAXIMUM_LATITUDE")
    lon_w = _float_val(meta, "WESTERNMOST_LONGITUDE")
    lon_e = _float_val(meta, "EASTERNMOST_LONGITUDE")
    lon_w, lon_e = _lon_bounds_180(lon_w, lon_e)

    height, width = arr.shape
    transform = from_bounds(lon_w, lat_s, lon_e, lat_n, width, height)

    if out_path.is_file() and not force:
        print(f"  skip (exists) {out_path.name}")
        return out_path

    print(
        f"  write {out_path.name} {width}×{height} "
        f"lon [{lon_w:.1f}, {lon_e:.1f}] lat [{lat_s:.1f}, {lat_n:.1f}]"
    )
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int16",
        "crs": MARS_CRS,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.int16), 1)
        dst.set_band_description(1, "MOLA MEGDR topography (m, areoid)")
    return out_path


def mosaic_global(tile_paths: list[Path], out_path: Path) -> None:
    print(f"\n=== Mosaic → {out_path.name} ({GLOBAL_WIDTH}×{GLOBAL_HEIGHT}) ===")
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(
            datasets,
            bounds=(GLOBAL_WEST, GLOBAL_SOUTH, GLOBAL_EAST, GLOBAL_NORTH),
            res=(1.0 / PPD, 1.0 / PPD),
            method="first",
        )
    finally:
        for ds in datasets:
            ds.close()

    if mosaic.shape[1] != GLOBAL_HEIGHT or mosaic.shape[2] != GLOBAL_WIDTH:
        raise ValueError(
            f"Mosaic shape {mosaic.shape[2]}×{mosaic.shape[1]} != "
            f"expected {GLOBAL_WIDTH}×{GLOBAL_HEIGHT}"
        )

    profile = {
        "driver": "GTiff",
        "height": GLOBAL_HEIGHT,
        "width": GLOBAL_WIDTH,
        "count": 1,
        "dtype": "int16",
        "crs": MARS_CRS,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0].astype(np.int16), 1)
        dst.set_band_description(1, "MOLA MEGDR 128 ppd topography (m, areoid)")
    print(f"  wrote {out_path} ({_human_bytes(out_path.stat().st_size)})")


def verify_global(path: Path) -> None:
    print(f"\n=== Verify {path.name} ===")
    with rasterio.open(path) as src:
        print(f"  size: {src.width} × {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  transform: {src.transform}")
        b = src.bounds
        print(f"  bounds: W={b.left:.4f} S={b.bottom:.4f} E={b.right:.4f} N={b.top:.4f}")

        crs_ok = src.crs and src.crs.to_proj4() == MARS_CRS.to_proj4()
        if not (src.width == GLOBAL_WIDTH and src.height == GLOBAL_HEIGHT and crs_ok):
            raise SystemExit("Verification failed: dimensions or CRS mismatch")

        exp_transform = from_bounds(
            GLOBAL_WEST, GLOBAL_SOUTH, GLOBAL_EAST, GLOBAL_NORTH,
            GLOBAL_WIDTH, GLOBAL_HEIGHT,
        )
        if not src.transform.almost_equals(exp_transform, precision=1e-9):
            print(f"  expected transform: {exp_transform}")
            raise SystemExit("Verification failed: geotransform mismatch")

        # Jezero Crater ~18.4°N, 77.7°E, elevation ~−2500 m
        lat, lon = 18.4, 77.7
        x = int(((lon + 180.0) / 360.0) * src.width)
        y = int(((90.0 - lat) / 180.0) * src.height)
        val = float(src.read(1, window=rasterio.windows.Window(x, y, 1, 1))[0, 0])
        print(f"  Jezero ({lat}°N, {lon}°E) pixel ({x}, {y}): {val:.1f} m")
        if not (-3200 < val < -1800):
            print("  WARNING: Jezero elevation outside expected ~−2500 m (±700 m)")


def install_public(global_path: Path, dest: Path) -> None:
    print(f"\n=== Install {dest.relative_to(REPO_ROOT)} ===")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(global_path, dest)
    print(f"  copied ({_human_bytes(dest.stat().st_size)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--pds-url", default=PDS_MEG128_URL)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument(
        "--force-convert",
        action="store_true",
        help="Re-write per-tile *_georef.tif even if they already exist",
    )
    p.add_argument("--min-disk-gb", type=float, default=3.0)
    args = p.parse_args()

    raw_dir = args.raw_dir.resolve()
    global_path = raw_dir / DEFAULT_GLOBAL_NAME

    check_disk_space(raw_dir, args.min_disk_gb)

    if not args.skip_download:
        download_tiles(raw_dir, args.pds_url.rstrip("/"))

    print("\n=== Convert tiles to GeoTIFF ===")
    tile_tifs: list[Path] = []
    for stem in TILE_STEMS:
        print(f"--- {stem} ---")
        tile_tifs.append(convert_tile(raw_dir, stem, force=args.force_convert))

    mosaic_global(tile_tifs, global_path)
    verify_global(global_path)

    if not args.skip_install:
        install_public(global_path, PUBLIC_DEST)

    print("\nDone.")


if __name__ == "__main__":
    main()
