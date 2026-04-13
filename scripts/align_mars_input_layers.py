#!/usr/bin/env python3
"""
Reproject and resample the 13 globe input GeoTIFFs to one common global
equirectangular grid (default 11520×5760, −180…180° lon, −90…90° lat).

JMARS exports often have no CRS and an identity geotransform. For those, this
script assumes each raster covers the full planet in that lon/lat range and
builds a synthetic `from_bounds` transform for the source before warping to
the destination grid. If a raster is not actually global (e.g. non–2:1
aspect ratio), that assumption stretches the geographic extent across the
image — see project docs.

Does not modify originals; writes aligned copies under ``scripts/aligned_layers/``
using the same basenames as ``frontend/3d_globe/index.js`` / ``stack_mars_layers.py``.

Example::

    python scripts/align_mars_input_layers.py
    python scripts/stack_mars_layers.py --data-dir scripts/aligned_layers \\
        --output frontend/3d_globe/public/data/mars_global_input_stack_32ppd.tif
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

# Same directory as this file → import shared layer list
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from stack_mars_layers import LAYER_BASENAMES  # noqa: E402

DEFAULT_MARS_PROJ = "+proj=longlat +R=3396190 +no_defs"


def _mars_crs(proj_string: str) -> CRS:
    return CRS.from_proj4(proj_string)


def _is_identity_geotransform(transform: Affine) -> bool:
    return transform == Affine.identity()


def _resolve_source_georef(
    src: rasterio.DatasetReader,
    mars_crs: CRS,
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[Affine, CRS]:
    """
    If the file has no placement on the globe (identity transform), assume
    global equirectangular coverage over [west..east] x [south..north].
    Otherwise use the file's transform; if CRS is missing, assume mars_crs.
    """
    if _is_identity_geotransform(src.transform):
        src_transform = from_bounds(west, south, east, north, src.width, src.height)
        return src_transform, mars_crs
    src_crs = src.crs if src.crs is not None else mars_crs
    return src.transform, src_crs


def _build_dst_profile(
    width: int,
    height: int,
    west: float,
    south: float,
    east: float,
    north: float,
    crs: CRS,
    nodata: float | None,
) -> tuple[dict, Affine]:
    transform = from_bounds(west, south, east, north, width, height)
    profile: dict = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    return profile, transform


def _align_one(
    src_path: str,
    dst_path: str,
    dst_profile: dict,
    dst_transform: Affine,
    dst_crs: CRS,
    mars_crs: CRS,
    west: float,
    south: float,
    east: float,
    north: float,
    resampling: Resampling,
) -> None:
    with rasterio.open(src_path) as src:
        src_transform, src_crs = _resolve_source_georef(
            src, mars_crs, west, south, east, north
        )
        src_nodata = src.nodata
        arr = src.read(1, masked=True).astype(np.float32)
        data = arr.filled(np.nan).astype(np.float32)

    height = int(dst_profile["height"])
    width = int(dst_profile["width"])
    destination = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
        source=data,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=None,
    )

    out_profile = dst_profile.copy()
    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(destination, 1)


def _verify_aligned_outputs(
    output_dir: str,
    basenames: tuple[str, ...],
    expected_width: int,
    expected_height: int,
    expected_transform: Affine,
    expected_crs: CRS,
) -> None:
    rows: list[tuple[str, int, int, str, str]] = []
    ref_meta: dict | None = None
    for name in basenames:
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            print(f"error: expected output missing: {path}", file=sys.stderr)
            sys.exit(1)
        with rasterio.open(path) as ds:
            w, h = ds.width, ds.height
            t, c = ds.transform, ds.crs
            nodata = ds.nodata
            rows.append((name, w, h, str(nodata), str(np.dtype(ds.dtypes[0]).name)))
            meta = {"width": w, "height": h, "transform": t, "crs": c}
            if ref_meta is None:
                ref_meta = meta
            else:
                if (
                    meta["width"] != ref_meta["width"]
                    or meta["height"] != ref_meta["height"]
                    or meta["transform"] != ref_meta["transform"]
                    or meta["crs"] != ref_meta["crs"]
                ):
                    print(
                        "error: aligned outputs do not share identical grid metadata.\n"
                        f"  reference ({basenames[0]}): {ref_meta}\n"
                        f"  mismatch ({name}): {meta}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    if ref_meta is None:
        print("error: no outputs to verify", file=sys.stderr)
        sys.exit(1)

    def _proj4(crs: CRS) -> str:
        return " ".join(crs.to_proj4().split())

    crs_ok = _proj4(ref_meta["crs"]) == _proj4(expected_crs)
    if (
        ref_meta["width"] != expected_width
        or ref_meta["height"] != expected_height
        or ref_meta["transform"] != expected_transform
        or not crs_ok
    ):
        print(
            "error: outputs do not match requested reference grid.\n"
            f"  expected: {expected_width}x{expected_height}, transform={expected_transform}, crs_proj4={_proj4(expected_crs)!r}\n"
            f"  got: {ref_meta['width']}x{ref_meta['height']}, transform={ref_meta['transform']}, crs_proj4={_proj4(ref_meta['crs'])!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nVerification: all aligned rasters share the same width, height, transform, and CRS.")
    print(f"  size: {expected_width} x {expected_height}")
    print(f"  transform: {expected_transform}")
    print(f"  crs: {expected_crs}")
    print("\nPer-file summary:")
    print(f"{'filename':<42} {'W':>6} {'H':>6} {'dtype':>10} {'nodata':>12}")
    for name, w, h, nd, dt in rows:
        print(f"{name:<42} {w:>6} {h:>6} {dt:>10} {nd:>12}")


def main() -> None:
    repo_root = os.path.dirname(_SCRIPT_DIR)
    default_source = os.path.join(repo_root, "frontend", "3d_globe", "public", "data")
    default_out = os.path.join(repo_root, "scripts", "aligned_layers")

    p = argparse.ArgumentParser(
        description="Align 13 Mars GeoTIFFs to a common global equirectangular grid."
    )
    p.add_argument("--source-dir", default=default_source, help="Directory with raw JMARS GeoTIFFs")
    p.add_argument("--output-dir", default=default_out, help="Directory for aligned outputs (created if needed)")
    p.add_argument("--width", type=int, default=11520, help="Output width (default 32 PPD * 360°)")
    p.add_argument("--height", type=int, default=5760, help="Output height (default 32 PPD * 180°)")
    p.add_argument("--west", type=float, default=-180.0)
    p.add_argument("--south", type=float, default=-90.0)
    p.add_argument("--east", type=float, default=180.0)
    p.add_argument("--north", type=float, default=90.0)
    p.add_argument(
        "--crs-proj",
        default=DEFAULT_MARS_PROJ,
        help="PROJ string for Mars geographic lon/lat (destination and synthetic sources)",
    )
    p.add_argument(
        "--resampling",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help="Warp resampling (default bilinear for continuous science rasters)",
    )
    p.add_argument(
        "--dst-nodata",
        type=float,
        default=None,
        help="Optional NoData value for output GeoTIFFs (default: unset / None)",
    )
    p.add_argument(
        "--compress",
        default=None,
        help="GTiff compression, e.g. LZW or DEFLATE (optional)",
    )
    args = p.parse_args()

    resampling = Resampling.bilinear if args.resampling == "bilinear" else Resampling.nearest
    mars_crs = _mars_crs(args.crs_proj)
    dst_profile, dst_transform = _build_dst_profile(
        args.width,
        args.height,
        args.west,
        args.south,
        args.east,
        args.north,
        mars_crs,
        args.dst_nodata,
    )
    if args.compress:
        dst_profile["compress"] = args.compress

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reference grid: {args.width}x{args.height}  bounds "
          f"({args.west}, {args.south}, {args.east}, {args.north})")
    print(f"CRS: {args.crs_proj}")
    print(f"Output directory: {args.output_dir}\n")

    for i, basename in enumerate(LAYER_BASENAMES, start=1):
        src_path = os.path.join(args.source_dir, basename)
        dst_path = os.path.join(args.output_dir, basename)
        if not os.path.isfile(src_path):
            print(f"error: missing source: {src_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[{i:2}/{len(LAYER_BASENAMES)}] {basename} …", flush=True)
        _align_one(
            src_path,
            dst_path,
            dst_profile,
            dst_transform,
            mars_crs,
            mars_crs,
            args.west,
            args.south,
            args.east,
            args.north,
            resampling,
        )

    _verify_aligned_outputs(
        args.output_dir,
        LAYER_BASENAMES,
        args.width,
        args.height,
        dst_transform,
        mars_crs,
    )
    print("\nDone. Stack with:\n"
          f"  python scripts/stack_mars_layers.py --data-dir {args.output_dir} "
          f"--output <path/to/stack.tif>")


if __name__ == "__main__":
    main()
