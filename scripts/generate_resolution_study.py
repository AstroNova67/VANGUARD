#!/usr/bin/env python3
"""
Resolution study: downsample 32 PPD input stack → 4 / 8 / 32 PPD, run ML suitability batch, compare.

Uses bilinear resampling for stack downsampling (continuous science bands).
Invokes the same scoring path as backend/batch_global_landing_suitability.py.

Example::

    uv run python scripts/generate_resolution_study.py
    uv run python scripts/generate_resolution_study.py --skip-batch  # stacks + report only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

DEFAULT_STACK = os.path.join(
    REPO_ROOT, "frontend", "3d_globe", "public", "data", "mars_global_input_stack_32ppd.tif"
)
DEFAULT_STUDY_DIR = os.path.join(_SCRIPT_DIR, "resolution_study")

MARS_PROJ = "+proj=longlat +R=3396190 +no_defs"
WEST, SOUTH, EAST, NORTH = -180.0, -90.0, 180.0, 90.0

RESOLUTIONS: tuple[tuple[str, int, int], ...] = (
    ("4ppd", 1440, 720),
    ("8ppd", 2880, 1440),
    ("32ppd", 11520, 5760),
)

# Score bands for comparison (percent 0–100)
SCORE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Very Poor", 0.0, 20.0),
    ("Poor", 20.0, 30.0),
    ("Fair", 30.0, 50.0),
    ("Good", 50.0, 70.0),
    ("Excellent", 70.0, 100.0),
)

OUTPUT_NODATA = -9999.0


def _band_label(score: float) -> str:
    for name, lo, hi in SCORE_BANDS:
        if hi >= 100.0:
            if score >= lo:
                return name
        elif lo <= score < hi:
            return name
    return "Very Poor"


def stack_path_for(study_dir: str, label: str) -> str:
    return os.path.join(study_dir, "stacks", f"mars_global_input_stack_{label}.tif")


def suitability_path_for(study_dir: str, label: str) -> str:
    return os.path.join(study_dir, "outputs", f"mars_suitability_{label}.tif")


def _dst_profile(width: int, height: int, ref_profile: dict) -> dict:
    crs = CRS.from_proj4(MARS_PROJ)
    transform = from_bounds(WEST, SOUTH, EAST, NORTH, width, height)
    profile = ref_profile.copy()
    profile.update(
        width=width,
        height=height,
        crs=crs,
        transform=transform,
        tiled=ref_profile.get("tiled", False),
    )
    return profile


def downsample_stack(
    src_path: str,
    dst_path: str,
    width: int,
    height: int,
    *,
    on_progress: callable | None = None,
) -> None:
    """Bilinear warp all bands from src stack to dst grid."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with rasterio.open(src_path) as src:
        dst_profile = _dst_profile(width, height, src.profile)
        dst_profile["count"] = src.count
        if src_path == dst_path and src.width == width and src.height == height:
            return
        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            for b in range(1, src.count + 1):
                if src.descriptions and src.descriptions[b - 1]:
                    dst.set_band_description(b, src.descriptions[b - 1])
                source = src.read(b).astype(np.float32)
                dest = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=source,
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst.transform,
                    dst_crs=dst.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodatavals[b - 1] if src.nodatavals else None,
                    dst_nodata=None,
                )
                dst.write(dest, b)
                if on_progress:
                    on_progress(b, src.count)
    print(f"  Wrote stack {dst_path} ({width}x{height})", flush=True)


def run_batch_on_stack(stack_path: str, output_path: str, *, block_size: int) -> None:
    from batch_global_landing_suitability import run  # noqa: WPS433

    args = argparse.Namespace(
        input=stack_path,
        output=output_path,
        hybrid_output=os.path.join(os.path.dirname(output_path), "_unused_hybrid.tif"),
        coverage_output=os.path.join(os.path.dirname(output_path), "_unused_coverage.tif"),
        with_hybrid_coverage=False,
        block_size=block_size,
        inference_subbatch=32768,
        compress=None,
        tiled=False,
        blockxsize=512,
        blockysize=512,
    )
    run(args)


def _read_valid_scores(path: str) -> tuple[np.ndarray, rasterio.Affine, int, int, np.ndarray]:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata if ds.nodata is not None else OUTPUT_NODATA
        valid = np.isfinite(arr) & (arr != nodata) & (arr >= 0) & (arr <= 100)
        return arr, ds.transform, ds.width, ds.height, valid


def score_distribution(scores: np.ndarray, valid: np.ndarray) -> dict:
    v = scores[valid]
    if v.size == 0:
        return {"count": 0}
    pct = np.percentile(v, [5, 25, 50, 75, 95])
    return {
        "count": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p5": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
    }


def band_change_matrix(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    """Count pixels where band label differs between a and b."""
    changes: dict[str, int] = {}
    rows, cols = np.where(valid)
    for r, c in zip(rows, cols, strict=False):
        la = _band_label(float(a[r, c]))
        lb = _band_label(float(b[r, c]))
        if la != lb:
            key = f"{la} → {lb}"
            changes[key] = changes.get(key, 0) + 1
    return changes


def top_diff_regions(
    scores_lo: np.ndarray,
    scores_hi: np.ndarray,
    valid: np.ndarray,
    transform: rasterio.Affine,
    width: int,
    height: int,
    *,
    n_tiles_lon: int = 36,
    n_tiles_lat: int = 18,
    top_k: int = 20,
) -> list[dict]:
    """Tile mean |diff| between low and high resolution; return top bounding boxes."""
    diff = np.abs(scores_hi.astype(np.float64) - scores_lo.astype(np.float64))
    diff[~valid] = np.nan
    tile_w = width // n_tiles_lon
    tile_h = height // n_tiles_lat
    boxes: list[dict] = []
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    for ty in range(n_tiles_lat):
        for tx in range(n_tiles_lon):
            row0 = ty * tile_h
            row1 = (ty + 1) * tile_h if ty < n_tiles_lat - 1 else height
            col0 = tx * tile_w
            col1 = (tx + 1) * tile_w if tx < n_tiles_lon - 1 else width
            block = diff[row0:row1, col0:col1]
            if not np.any(np.isfinite(block)):
                continue
            mean_diff = float(np.nanmean(block))
            # geographic bounds from transform (north-up)
            lon_w = WEST + col0 * pixel_size_x
            lon_e = WEST + col1 * pixel_size_x
            lat_n = NORTH - row0 * pixel_size_y
            lat_s = NORTH - row1 * pixel_size_y
            boxes.append(
                {
                    "mean_abs_diff": mean_diff,
                    "lon_min": lon_w,
                    "lon_max": lon_e,
                    "lat_min": lat_s,
                    "lat_max": lat_n,
                    "tile": f"tx={tx} ty={ty}",
                }
            )
    boxes.sort(key=lambda x: x["mean_abs_diff"], reverse=True)
    return boxes[:top_k]


def _resample_scores_to_grid(
    path: str,
    width: int,
    height: int,
    transform: rasterio.Affine,
    crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear resample suitability scores onto the reference 32 PPD grid."""
    with rasterio.open(path) as src:
        source = src.read(1).astype(np.float32)
        dest = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata if src.nodata is not None else OUTPUT_NODATA,
            dst_nodata=np.nan,
        )
    nodata = OUTPUT_NODATA
    valid = np.isfinite(dest) & (dest != nodata) & (dest >= 0) & (dest <= 100)
    return dest, valid


def write_comparison_report(
    study_dir: str,
    paths: dict[str, str],
    report_path: str,
) -> None:
    native: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_transform = None
    ref_crs = None
    ref_w = ref_h = 0
    for label, path in paths.items():
        arr, t, w, h, valid = _read_valid_scores(path)
        native[label] = (arr, valid)
        if label == "32ppd":
            ref_transform, ref_w, ref_h = t, w, h
    assert ref_transform is not None
    with rasterio.open(paths["32ppd"]) as ref_ds:
        ref_crs = ref_ds.crs

    # Align 4/8 PPD scores onto 32 PPD grid for pairwise comparison
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {"32ppd": native["32ppd"]}
    for label in ("4ppd", "8ppd"):
        aligned[label] = _resample_scores_to_grid(
            paths[label], ref_w, ref_h, ref_transform, ref_crs
        )

    common_valid = aligned["32ppd"][1].copy()
    for label in ("4ppd", "8ppd"):
        common_valid &= aligned[label][1]

    lines: list[str] = [
        "# Mars landing suitability — resolution study report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Score distribution (native grid, all valid pixels per file)",
        "",
        "| Resolution | Count | Mean | Std | Min | Max | P5 | P25 | P50 | P75 | P95 |",
        "|------------|------:|-----:|----:|----:|----:|---:|----:|----:|----:|----:|",
    ]
    for label in ("4ppd", "8ppd", "32ppd"):
        stats = score_distribution(native[label][0], native[label][1])
        if stats.get("count", 0) == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {stats['count']:,} | {stats['mean']:.2f} | {stats['std']:.2f} | "
            f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['p5']:.2f} | {stats['p25']:.2f} | "
            f"{stats['p50']:.2f} | {stats['p75']:.2f} | {stats['p95']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Score distribution (32 PPD grid, common valid mask)",
            "",
            "4 and 8 PPD scores bilinearly resampled to the 32 PPD grid before masking.",
            "",
            "| Resolution | Count | Mean | Std | Min | Max | P5 | P25 | P50 | P75 | P95 |",
            "|------------|------:|-----:|----:|----:|----:|---:|----:|----:|----:|----:|",
        ]
    )
    for label in ("4ppd", "8ppd", "32ppd"):
        stats = score_distribution(aligned[label][0], common_valid)
        if stats.get("count", 0) == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {stats['count']:,} | {stats['mean']:.2f} | {stats['std']:.2f} | "
            f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['p5']:.2f} | {stats['p25']:.2f} | "
            f"{stats['p50']:.2f} | {stats['p75']:.2f} | {stats['p95']:.2f} |"
        )

    lines.extend(["", "## Score band changes (pairwise, common valid mask)", ""])
    pairs = (("4ppd", "8ppd"), ("8ppd", "32ppd"), ("4ppd", "32ppd"))
    for a, b in pairs:
        same = int(
            np.sum(
                common_valid
                & np.isclose(aligned[a][0], aligned[b][0], rtol=0, atol=0.05)
            )
        )
        total = int(np.sum(common_valid))
        band_changes = band_change_matrix(aligned[a][0], aligned[b][0], common_valid)
        changed = sum(band_changes.values())
        lines.append(f"### {a} vs {b}")
        lines.append(f"- Identical score (exact): {same:,} / {total:,} ({100*same/max(total,1):.2f}%)")
        lines.append(f"- Band label changes: {changed:,} ({100*changed/max(total,1):.2f}%)")
        if band_changes:
            lines.append("")
            lines.append("| Transition | Pixel count |")
            lines.append("|------------|------------:|")
            for k, v in sorted(band_changes.items(), key=lambda x: -x[1])[:15]:
                lines.append(f"| {k} | {v:,} |")
        lines.append("")

    lines.extend(
        [
            "## Largest |score₄ − score₃₂| regions (10°×10° tiles, top 20)",
            "",
            "| Rank | Mean |Δscore| | Lon min | Lon max | Lat min | Lat max |",
            "|-----:|-------------:|--------:|--------:|--------:|--------:|",
        ]
    )
    top = top_diff_regions(
        aligned["4ppd"][0],
        aligned["32ppd"][0],
        common_valid,
        ref_transform,
        ref_w,
        ref_h,
    )
    for i, box in enumerate(top, 1):
        lines.append(
            f"| {i} | {box['mean_abs_diff']:.3f} | {box['lon_min']:.1f} | {box['lon_max']:.1f} | "
            f"{box['lat_min']:.1f} | {box['lat_max']:.1f} |"
        )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote report {report_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Resolution study: downsample stack, batch score, compare.")
    p.add_argument("--input-stack", default=DEFAULT_STACK, help="32 PPD source stack")
    p.add_argument("--study-dir", default=DEFAULT_STUDY_DIR)
    p.add_argument("--block-size", type=int, default=1024, help="Batch window size")
    p.add_argument("--skip-downsample", action="store_true")
    p.add_argument("--skip-batch", action="store_true", help="Only downsample + report (stacks must exist)")
    p.add_argument("--skip-report", action="store_true")
    p.add_argument(
        "--reuse-existing-32ppd",
        default=os.path.join(
            REPO_ROOT, "frontend", "3d_globe", "public", "data", "mars_landing_suitability_ml.tif"
        ),
        help="If set and file exists, copy to mars_suitability_32ppd.tif instead of re-running batch",
    )
    args = p.parse_args()

    study_dir = os.path.abspath(args.study_dir)
    stacks_dir = os.path.join(study_dir, "stacks")
    outputs_dir = os.path.join(study_dir, "outputs")
    os.makedirs(stacks_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    if not os.path.isfile(args.input_stack):
        print(f"error: input stack not found: {args.input_stack}", file=sys.stderr)
        sys.exit(1)

    t_total = time.perf_counter()
    stack_paths: dict[str, str] = {}

    # --- Downsample stacks ---
    if not args.skip_downsample:
        print("=== Downsample input stacks (bilinear) ===", flush=True)
        n_res = len(RESOLUTIONS)
        for ri, (label, w, h) in enumerate(RESOLUTIONS):
            dst = os.path.join(stacks_dir, f"mars_global_input_stack_{label}.tif")
            stack_paths[label] = dst
            t0 = time.perf_counter()
            print(f"[{ri+1}/{n_res}] {label} ({w}x{h}) …", flush=True)
            if label == "32ppd":
                # Use canonical stack in public/data (already 32 PPD); copy if study path differs
                if os.path.abspath(args.input_stack) == os.path.abspath(dst):
                    stack_paths[label] = args.input_stack
                    print(f"  Using existing stack {args.input_stack}", flush=True)
                else:
                    downsample_stack(args.input_stack, dst, w, h)
            else:
                downsample_stack(args.input_stack, dst, w, h)
            elapsed = time.perf_counter() - t0
            done = ri + 1
            eta = (elapsed / done) * (n_res - done) if done else 0
            print(f"  Done {label} in {elapsed/60:.1f} min; ETA ~{eta/60:.1f} min for remaining stacks", flush=True)
    else:
        for label, w, h in RESOLUTIONS:
            pth = os.path.join(stacks_dir, f"mars_global_input_stack_{label}.tif")
            if label == "32ppd" and os.path.isfile(args.input_stack):
                stack_paths[label] = args.input_stack
            elif os.path.isfile(pth):
                stack_paths[label] = pth

    # --- Batch scoring ---
    if not args.skip_batch:
        import shutil

        print("\n=== Batch ML suitability (per resolution) ===", flush=True)
        print("Note: 32 PPD can take 30–90+ min; 4/8 PPD are faster.", flush=True)
        n_batch = len(RESOLUTIONS)
        for bi, (label, _w, _h) in enumerate(RESOLUTIONS):
            stack_p = stack_paths.get(label)
            if not stack_p or not os.path.isfile(stack_p):
                print(f"error: missing stack for {label}: {stack_p}", file=sys.stderr)
                sys.exit(1)
            out_p = os.path.join(outputs_dir, f"mars_suitability_{label}.tif")
            if (
                label == "32ppd"
                and args.reuse_existing_32ppd
                and os.path.isfile(args.reuse_existing_32ppd)
            ):
                shutil.copy2(args.reuse_existing_32ppd, out_p)
                print(f"[batch {bi+1}/{n_batch}] {label} — copied existing {args.reuse_existing_32ppd}", flush=True)
                continue
            t0 = time.perf_counter()
            print(f"[batch {bi+1}/{n_batch}] {label} → {out_p}", flush=True)
            run_batch_on_stack(stack_p, out_p, block_size=args.block_size)
            elapsed = time.perf_counter() - t0
            eta = (elapsed / (bi + 1)) * (n_batch - bi - 1)
            print(f"  Finished {label} in {elapsed/60:.1f} min; ETA ~{eta/60:.1f} min", flush=True)

    # --- Comparison report ---
    if not args.skip_report:
        print("\n=== Comparison report ===", flush=True)
        out_paths = {
            label: os.path.join(outputs_dir, f"mars_suitability_{label}.tif")
            for label, _, _ in RESOLUTIONS
        }
        for label, path in out_paths.items():
            if not os.path.isfile(path):
                print(f"error: missing suitability output: {path}", file=sys.stderr)
                sys.exit(1)
        write_comparison_report(
            study_dir,
            out_paths,
            os.path.join(study_dir, "comparison_report.md"),
        )

    print(f"\nAll done in {(time.perf_counter()-t_total)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
