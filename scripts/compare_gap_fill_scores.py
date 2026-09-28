#!/usr/bin/env python3
"""
Compare landing scores across three arms:

  A · No corrections — data_gap_unfilled/
  B · Manual (IDW)   — data_gap_filled/
  C · ML fill        — data_gap_ml_filled/

Writes:
  scripts/gap_fill_score_compare.json
  frontend/3d_globe/public/data_gap_filled/score_compare.json
  frontend/3d_globe/public/data_gap_ml_filled/score_compare.json
  scripts/gap_fill_score_compare.md
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

REPO = Path(__file__).resolve().parents[1]
SRC = Path.home() / "Downloads" / "Mars Geoscience"
FILLED = REPO / "frontend" / "3d_globe" / "public" / "data_gap_filled"
UNFILLED = REPO / "frontend" / "3d_globe" / "public" / "data_gap_unfilled"
ML_FILLED = REPO / "frontend" / "3d_globe" / "public" / "data_gap_ml_filled"
OLD_PUBLIC = REPO / "frontend" / "3d_globe" / "public" / "data"
REPORT = REPO / "scripts" / "gap_fill_score_compare.json"
REPORT_MD = REPO / "scripts" / "gap_fill_score_compare.md"

sys.path.insert(0, str(REPO))

FILL_LAYERS = {
    "omega_ferric_nnphs.tif": "omega_ferric_nnphs.tif",
    "omega_albedo_r1080.tif": "omega_albedo_r1080.tif",
    "mola_roughness_0.6km_numeric.tif": "mola_roughness_0.6km_numeric.tif",
    "mars_yearly_temperature_range_v1.0.tif": "mars_yearly_temperature_range_v1.0.tif",
    "mola_hrsc_blend_topo_v2.tif": "MOLA_128ppd_topo.tif",
}
COPY_LAYERS = {
    "mola_hrsc_blend_slope_v2.tif": "mola_hrsc_blend_slope_v2.tif",
    "mars_yearly_avg_temperature_celsius.tif": "mars_yearly_avg_temperature_celsius.tif",
    "mars_crustal_thickness_gmm3_rm1.tif": "mars_crustal_thickness_gmm3_rm1.tif",
    "mars_odyssey_grs_mons_perc_wt.tif": "mars_odyssey_grs_mons_perc_wt.tif",
    "TES_Basalt_numeric.tif": "TES_Basalt_numeric.tif",
    "omega_pyroxene_bd2000.tif": "omega_pyroxene_bd2000.tif",
}
KEEP_FROM_PUBLIC = (
    "TES_Lambert_Albedo_numeric.tif",
    "tes_dayside_ti_putzig_2007.tif",
    "mars_landing_suitability_ml.tif",
)

LAYER_KEYS = (
    "elevation",
    "albedo",
    "roughness",
    "ferric",
    "tempRange",
)


def mars_transform(width: int, height: int):
    return from_bounds(-180, -90, 180, 90, width, height)


def write_geotiff(path: Path, data: np.ndarray) -> None:
    h, w = data.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": mars_transform(w, h),
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(data, dtype=np.float32), 1)


def build_unfilled_folder() -> None:
    if UNFILLED.exists():
        shutil.rmtree(UNFILLED)
    UNFILLED.mkdir(parents=True)
    for src_name, dst_name in {**FILL_LAYERS, **COPY_LAYERS}.items():
        with rasterio.open(SRC / src_name) as src:
            write_geotiff(UNFILLED / dst_name, src.read(1))
    for name in KEEP_FROM_PUBLIC:
        src = OLD_PUBLIC / name
        if src.is_file():
            shutil.copy2(src, UNFILLED / name)


def pixel_to_latlon(row: int, col: int, height: int, width: int) -> tuple[float, float]:
    lon = -180.0 + (col + 0.5) * 360.0 / width
    lat = 90.0 - (row + 0.5) * 180.0 / height
    return float(lat), float(lon)


def pick_gap_sites(n: int = 12) -> list[dict]:
    ferric_path = UNFILLED / "omega_ferric_nnphs.tif"
    with rasterio.open(ferric_path) as ds:
        ferric = ds.read(1).astype(np.float64)
        h, w = ferric.shape
    bad = (ferric == 0) | ~np.isfinite(ferric)
    for name in (
        "omega_albedo_r1080.tif",
        "mola_roughness_0.6km_numeric.tif",
        "mars_yearly_temperature_range_v1.0.tif",
    ):
        with rasterio.open(UNFILLED / name) as ds:
            a = ds.read(1).astype(np.float64)
        bad |= (a == 0) | ~np.isfinite(a)

    rows, cols = np.where(bad)
    if rows.size == 0:
        return []
    rng = np.random.default_rng(42)
    idx = rng.choice(rows.size, size=min(n, rows.size), replace=False)
    sites = []
    for i, j in zip(rows[idx], cols[idx]):
        lat, lon = pixel_to_latlon(int(i), int(j), h, w)
        sites.append(
            {
                "id": f"gap_{lat:.2f}_{lon:.2f}",
                "name": f"Former gap ({lat:.2f}°N, {lon:.2f}°E)",
                "lat": lat,
                "lon": lon,
                "kind": "gap",
            }
        )
    return sites


def load_models():
    os.chdir(REPO / "backend")
    warnings.filterwarnings("ignore")
    from backend.landing_predict import compute_landing_prediction
    from backend.mars_raster import sample_mars_data_at
    from backend.mars_sites import MARS_FAMOUS_SITES
    from backend.scoring import get_nn_models, load_scalers

    load_scalers()
    get_nn_models()

    regression_models: dict = {}
    try:
        import xgboost as xgb

        base = REPO / "backend" / "saved_models" / "regression_models"
        st_path = base / "surface_temp" / "xgb_model.json"
        ti_path = base / "thermal_inertia" / "xgb_model.json"
        if st_path.is_file() and ti_path.is_file():
            st = xgb.XGBRegressor()
            ti = xgb.XGBRegressor()
            st.load_model(str(st_path))
            ti.load_model(str(ti_path))
            regression_models = {
                "surface_temp": {"xgb": st},
                "thermal_inertia": {"xgb": ti},
            }
    except Exception as exc:  # noqa: BLE001
        print(f"XGB models skipped: {exc}", flush=True)

    return compute_landing_prediction, sample_mars_data_at, MARS_FAMOUS_SITES, regression_models


def score_at(predict_fn, sample_fn, lat: float, lon: float, data_dir: Path, regression_models: dict) -> dict:
    mars = sample_fn(lat, lon, data_dir=str(data_dir))
    result = predict_fn(
        mars,
        models_loaded=True,
        regression_models=regression_models,
    )
    return {
        "landing_score": result.get("landing_score"),
        "score_interpretation": result.get("score_interpretation"),
        "success": result.get("success"),
        "raw_layer_snapshot": {k: mars.get(k) for k in LAYER_KEYS if k in mars},
        "fused": (result.get("predictions") or {}).get("neural_networks"),
        "error": result.get("error"),
    }


def _delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(b) - float(a), 2)
    return None


def _mean(xs: list) -> float | None:
    return float(np.mean(xs)) if xs else None


def write_markdown(report: dict) -> None:
    s = report["summary"]
    lines = [
        "# Gap-fill score comparison (A / B / C)",
        "",
        "Three arms, same `/predict` pipeline:",
        "",
        "- **A · No corrections** — `data_gap_unfilled/`",
        "- **B · Manual (IDW)** — `data_gap_filled/`",
        "- **C · ML fill** — `data_gap_ml_filled/`",
        "",
        f"- Mean Δ(B−A): **{s.get('mean_delta_idw')}**",
        f"- Mean Δ(C−A): **{s.get('mean_delta_ml')}**",
        f"- Mean Δ(C−B): **{s.get('mean_delta_ml_minus_idw')}**",
        f"- Famous mean Δ(B−A) / Δ(C−A): **{s.get('mean_delta_idw_famous')}** / **{s.get('mean_delta_ml_famous')}**",
        f"- Gap-site mean Δ(B−A) / Δ(C−A): **{s.get('mean_delta_idw_gap')}** / **{s.get('mean_delta_ml_gap')}**",
        "",
        "## Largest |Δ(C−A)|",
        "",
        "| Site | A | B (IDW) | C (ML) | Δ IDW | Δ ML | Δ ML−IDW |",
        "|------|---|---------|--------|-------|-------|----------|",
    ]
    for j in s.get("largest_jumps_ml", [])[:10]:
        lines.append(
            f"| {j['name']} | {j['score_unfilled']} | {j['score_idw']} | {j['score_ml']} | "
            f"{j['delta_idw']:+.2f} | {j['delta_ml']:+.2f} | {j['delta_ml_minus_idw']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "Full JSON: `scripts/gap_fill_score_compare.json`.",
            "",
            "```bash",
            ".venv/bin/python scripts/compare_gap_fill_scores.py",
            "```",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines))


def main() -> int:
    if not FILLED.is_dir():
        print(f"ERROR: missing filled folder {FILLED}", file=sys.stderr)
        return 1
    if not ML_FILLED.is_dir():
        print(
            f"ERROR: missing ML filled folder {ML_FILLED} — run scripts/apply_ml_gap_fill.py",
            file=sys.stderr,
        )
        return 1

    if SRC.is_dir():
        print("Building unfilled baseline folder from Mars Geoscience…", flush=True)
        build_unfilled_folder()
    elif UNFILLED.is_dir():
        print(f"Using existing unfilled folder {UNFILLED}", flush=True)
    else:
        print(f"ERROR: missing {UNFILLED} and {SRC}", file=sys.stderr)
        return 1

    print("Loading models…", flush=True)
    predict_fn, sample_fn, famous, regression_models = load_models()

    sites = [
        {
            "id": s.id,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
            "kind": "famous",
        }
        for s in famous
    ]
    sites.extend(pick_gap_sites(16))

    rows = []
    for site in sites:
        print(f"  scoring {site['id']}…", flush=True)
        a = score_at(predict_fn, sample_fn, site["lat"], site["lon"], UNFILLED, regression_models)
        b = score_at(predict_fn, sample_fn, site["lat"], site["lon"], FILLED, regression_models)
        c = score_at(predict_fn, sample_fn, site["lat"], site["lon"], ML_FILLED, regression_models)
        sa, sb, sc = a.get("landing_score"), b.get("landing_score"), c.get("landing_score")
        rows.append(
            {
                **site,
                "score_unfilled": sa,
                "score_idw": sb,
                "score_ml": sc,
                # Back-compat aliases
                "score_filled": sb,
                "delta": _delta(sa, sb),
                "delta_idw": _delta(sa, sb),
                "delta_ml": _delta(sa, sc),
                "delta_ml_minus_idw": _delta(sb, sc),
                "arm_a": a,
                "arm_b": b,
                "arm_c": c,
            }
        )

    def collect(kind: str | None, key: str) -> list[float]:
        out = []
        for r in rows:
            if kind is not None and r["kind"] != kind:
                continue
            v = r.get(key)
            if v is not None:
                out.append(float(v))
        return out

    summary = {
        "n_sites": len(rows),
        "mean_delta_idw": _mean(collect(None, "delta_idw")),
        "mean_delta_ml": _mean(collect(None, "delta_ml")),
        "mean_delta_ml_minus_idw": _mean(collect(None, "delta_ml_minus_idw")),
        "mean_delta_idw_famous": _mean(collect("famous", "delta_idw")),
        "mean_delta_ml_famous": _mean(collect("famous", "delta_ml")),
        "mean_delta_idw_gap": _mean(collect("gap", "delta_idw")),
        "mean_delta_ml_gap": _mean(collect("gap", "delta_ml")),
        # legacy keys
        "mean_delta": _mean(collect(None, "delta_idw")),
        "mean_delta_famous": _mean(collect("famous", "delta_idw")),
        "mean_delta_gap_sites": _mean(collect("gap", "delta_idw")),
        "largest_jumps_ml": sorted(
            (
                {
                    "id": r["id"],
                    "name": r["name"],
                    "kind": r["kind"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "score_unfilled": r["score_unfilled"],
                    "score_idw": r["score_idw"],
                    "score_ml": r["score_ml"],
                    "delta_idw": r["delta_idw"],
                    "delta_ml": r["delta_ml"],
                    "delta_ml_minus_idw": r["delta_ml_minus_idw"],
                }
                for r in rows
                if r["delta_ml"] is not None
            ),
            key=lambda x: abs(x["delta_ml"]),
            reverse=True,
        )[:10],
        "largest_jumps": sorted(
            (
                {
                    "id": r["id"],
                    "name": r["name"],
                    "kind": r["kind"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "score_unfilled": r["score_unfilled"],
                    "score_filled": r["score_idw"],
                    "delta": r["delta_idw"],
                }
                for r in rows
                if r["delta_idw"] is not None
            ),
            key=lambda x: abs(x["delta"]),
            reverse=True,
        )[:10],
    }

    report = {
        "unfilled_dir": str(UNFILLED),
        "filled_dir": str(FILLED),
        "ml_filled_dir": str(ML_FILLED),
        "note": (
            "A = unfilled zeros; B = IDW+gaussian (data_gap_filled); "
            "C = HistGradientBoosting gap imputers (data_gap_ml_filled). "
            "Same /predict pipeline for all three."
        ),
        "summary": summary,
        "sites": [
            {
                "id": r["id"],
                "name": r["name"],
                "kind": r["kind"],
                "lat": r["lat"],
                "lon": r["lon"],
                "score_unfilled": r["score_unfilled"],
                "score_idw": r["score_idw"],
                "score_ml": r["score_ml"],
                "score_filled": r["score_idw"],
                "delta": r["delta_idw"],
                "delta_idw": r["delta_idw"],
                "delta_ml": r["delta_ml"],
                "delta_ml_minus_idw": r["delta_ml_minus_idw"],
                "layers_unfilled": r["arm_a"].get("raw_layer_snapshot"),
                "layers_idw": r["arm_b"].get("raw_layer_snapshot"),
                "layers_ml": r["arm_c"].get("raw_layer_snapshot"),
                "layers_filled": r["arm_b"].get("raw_layer_snapshot"),
            }
            for r in rows
        ],
    }

    REPORT.write_text(json.dumps(report, indent=2))
    shutil.copy2(REPORT, FILLED / "score_compare.json")
    shutil.copy2(REPORT, ML_FILLED / "score_compare.json")
    write_markdown(report)

    print("\n=== Score comparison A / B / C ===", flush=True)
    print(
        f"mean Δ(B−A)={summary['mean_delta_idw']}  "
        f"mean Δ(C−A)={summary['mean_delta_ml']}  "
        f"mean Δ(C−B)={summary['mean_delta_ml_minus_idw']}",
        flush=True,
    )
    print("\nLargest |Δ(C−A)|:", flush=True)
    for j in summary["largest_jumps_ml"][:8]:
        print(
            f"  {j['id']:28s}  A={j['score_unfilled']}  B={j['score_idw']}  C={j['score_ml']}  "
            f"(ΔML {j['delta_ml']:+.2f}, ΔML−IDW {j['delta_ml_minus_idw']:+.2f})",
            flush=True,
        )
    print(f"\nWrote {REPORT}", flush=True)
    print(f"Wrote {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
