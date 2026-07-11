#!/usr/bin/env python3
"""
Attach global equirectangular georef to unreferenced JMARS GeoTIFFs (metadata + pixel copy).

Writes copies under scripts/georef_layers/ — never modifies frontend/3d_globe/public/data originals.
Skips mars_global_input_stack_32ppd.tif, mars_landing_suitability_ml.tif (already georeferenced).
Skips MOLA_128ppd_topo.tif when height is not ~width/2 (likely partial export).
"""

from __future__ import annotations

import argparse
import os
import sys

import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_bounds

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_SOURCE = os.path.join(REPO_ROOT, "frontend", "3d_globe", "public", "data")
DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "georef_layers")

MARS_PROJ = "+proj=longlat +R=3396190 +no_defs"
WEST, SOUTH, EAST, NORTH = -180.0, -90.0, 180.0, 90.0

SKIP_ALWAYS = frozenset(
    {
        "mars_global_input_stack_32ppd.tif",
        "mars_landing_suitability_ml.tif",
    }
)
MOLA_BASE = "MOLA_128ppd_topo.tif"
ASPECT_TOLERANCE = 0.05  # height within 5% of width/2 => treat as global 2:1


def _is_unreferenced(src: rasterio.DatasetReader) -> bool:
    return src.crs is None or src.transform == Affine.identity()


def _likely_partial_export(width: int, height: int) -> bool:
    expected_h = width / 2.0
    if expected_h <= 0:
        return True
    return abs(height - expected_h) / expected_h > ASPECT_TOLERANCE


def attach_one(src_path: str, dst_path: str, *, dry_run: bool) -> str:
    """Return status: written | skipped_georef | skipped_partial | skipped_list."""
    basename = os.path.basename(src_path)
    if basename in SKIP_ALWAYS:
        return "skipped_list"
    with rasterio.open(src_path) as src:
        if not _is_unreferenced(src):
            return "skipped_georef"
        if basename == MOLA_BASE and _likely_partial_export(src.width, src.height):
            return "skipped_partial"
        transform = from_bounds(WEST, SOUTH, EAST, NORTH, src.width, src.height)
        crs = CRS.from_proj4(MARS_PROJ)
        if dry_run:
            print(f"  would write {dst_path} ({src.width}x{src.height})")
            return "written"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        profile = src.profile.copy()
        profile.update(crs=crs, transform=transform)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
                if src.descriptions and src.descriptions[i - 1]:
                    dst.set_band_description(i, src.descriptions[i - 1])
        return "written"


def main() -> None:
    p = argparse.ArgumentParser(description="Copy JMARS GeoTIFFs with Mars global georef metadata.")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"error: source dir not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    counts = {"written": 0, "skipped_georef": 0, "skipped_partial": 0, "skipped_list": 0}
    partial_notes: list[str] = []

    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}\n")

    for name in sorted(f for f in os.listdir(args.source_dir) if f.lower().endswith((".tif", ".tiff"))):
        src_path = os.path.join(args.source_dir, name)
        dst_path = os.path.join(args.output_dir, name)
        if name == MOLA_BASE:
            with rasterio.open(src_path) as src:
                w, h = src.width, src.height
                if _likely_partial_export(w, h):
                    partial_notes.append(
                        f"FLAG: {name} is {w}x{h} (expected ~{w}x{int(w/2)} for global equirectangular). "
                        "Georef not attached — confirm JMARS extent before tagging as global."
                    )
        status = attach_one(src_path, dst_path, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        if status == "written":
            print(f"  OK  {name}")
        elif status == "skipped_partial":
            print(f"  SKIP (partial export?)  {name}")
        elif status == "skipped_list":
            print(f"  SKIP (already georef stack/output)  {name}")
        elif status == "skipped_georef":
            print(f"  SKIP (already has georef)  {name}")

    print("\nSummary:", counts)
    for note in partial_notes:
        print(note)


if __name__ == "__main__":
    main()
