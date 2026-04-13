#!/usr/bin/env python3
"""
Stack the 13 globe input GeoTIFFs into one multiband file (same band order as
frontend/3d_globe/index.js marsDatasets).

When every file shares the same width, height, CRS, and affine — including the
common case of unreferenced exports (no CRS, identity transform) — bands are
written in order as a single GeoTIFF.

`rio warp --like` cannot be used if the reference has no geotransform; this script
only copies pixels and metadata from the first raster for the output profile.

If transforms or CRS differ between files but dimensions match (e.g. mixed
Y-axis convention), use --pixel-stack-only to stack by pixel index anyway
(at your own risk for misalignment).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rasterio

# Same order as frontend/3d_globe/index.js marsDatasets (must stay in sync).
LAYER_BASENAMES: tuple[str, ...] = (
    "MOLA_128ppd_topo.tif",
    "mola_hrsc_blend_slope_v2.tif",
    "mola_roughness_0.6km_numeric.tif",
    "omega_albedo_r1080.tif",
    "mars_yearly_avg_temperature_celsius.tif",
    "mars_yearly_temperature_range_v1.0.tif",
    "mars_crustal_thickness_gmm3_rm1.tif",
    "omega_ferric_nnphs.tif",
    "omega_pyroxene_bd2000.tif",
    "TES_Basalt_numeric.tif",
    "TES_Lambert_Albedo_numeric.tif",
    "tes_dayside_ti_putzig_2007.tif",
    "mars_odyssey_grs_mons_perc_wt.tif",
)


def _layer_paths(data_dir: str) -> list[str]:
    return [os.path.join(data_dir, n) for n in LAYER_BASENAMES]


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    default_data = os.path.join(repo_root, "frontend", "3d_globe", "public", "data")

    p = argparse.ArgumentParser(description="Stack 13 Mars layer GeoTIFFs for batch_global_landing_suitability.py")
    p.add_argument("--data-dir", default=default_data, help="Directory containing the 13 input GeoTIFFs")
    p.add_argument(
        "--output",
        default=os.path.join(default_data, "mars_global_input_stack_32ppd.tif"),
        help="Output multiband GeoTIFF path",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64", "preserve"),
        help="Array dtype for output (preserve = each band keeps its native dtype)",
    )
    p.add_argument(
        "--pixel-stack-only",
        action="store_true",
        help="Only require matching width/height; ignore CRS/transform differences between inputs.",
    )
    args = p.parse_args()

    paths = _layer_paths(args.data_dir)
    for path in paths:
        if not os.path.isfile(path):
            print(f"error: missing file: {path}", file=sys.stderr)
            sys.exit(1)

    metas: list[tuple[rasterio.crs.CRS | None, object]] = []
    wh: list[tuple[int, int]] = []
    for path in paths:
        with rasterio.open(path) as src:
            wh.append((src.width, src.height))
            metas.append((src.crs, src.transform))

    w0, h0 = wh[0]
    if any((w, h) != (w0, h0) for w, h in wh[1:]):
        print(
            f"error: all rasters must be {w0}x{h0} pixels; got mismatched sizes.",
            file=sys.stderr,
        )
        sys.exit(1)

    ref_meta = metas[0]
    geo_mismatch = any(m != ref_meta for m in metas[1:])
    if geo_mismatch and not args.pixel_stack_only:
        print(
            "error: CRS and/or transform differ between some inputs (common with mixed "
            'north-up / south-up GeoTIFFs). Either fix georeferencing in the source files, '
            "or if you are sure rows/columns already align pixel-for-pixel, rerun with:\n"
            "  --pixel-stack-only",
            file=sys.stderr,
        )
        sys.exit(1)
    if geo_mismatch and args.pixel_stack_only:
        print(
            "warning: stacking by pixels only; CRS/transform were not identical across inputs.",
            file=sys.stderr,
        )

    bands: list[np.ndarray] = []
    for path in paths:
        with rasterio.open(path) as src:
            arr = src.read(1)
            if args.dtype == "preserve":
                bands.append(arr)
            else:
                bands.append(arr.astype(np.dtype(args.dtype), copy=False))

    if args.dtype == "preserve":
        out_dtype = bands[0].dtype
        for i, b in enumerate(bands[1:], start=2):
            if b.dtype != out_dtype:
                print(
                    f"error: --dtype preserve but band 1 is {out_dtype} and band {i} is {b.dtype}",
                    file=sys.stderr,
                )
                sys.exit(1)
        stacked = np.stack(bands, axis=0)
    else:
        stacked = np.stack(bands, axis=0)
        out_dtype = stacked.dtype

    with rasterio.open(paths[0]) as ref:
        profile = ref.profile.copy()
        profile.update(count=len(paths), dtype=np.dtype(out_dtype).name)

    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(stacked)
        for i, path in enumerate(paths, start=1):
            dst.set_band_description(i, os.path.basename(path))

    print(f"Wrote {args.output} ({len(paths)} bands, {w0}x{h0}, dtype={out_dtype})")


if __name__ == "__main__":
    main()
