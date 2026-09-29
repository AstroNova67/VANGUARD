"""Search global ML suitability raster and re-score candidates with custom weights."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

try:
    from backend.mars_raster import DATA_DIR, sample_mars_data_at
    from backend.mars_sites import MARS_FAMOUS_SITES
except ImportError:
    from mars_raster import DATA_DIR, sample_mars_data_at
    from mars_sites import MARS_FAMOUS_SITES

SUITABILITY_FILENAME = "mars_landing_suitability_ml.tif"


def _suitability_path(data_dir: str | None = None) -> str:
    return os.path.join(data_dir or DATA_DIR, SUITABILITY_FILENAME)


def _pixel_center_to_lat_lon(row: int, col: int, height: int, width: int) -> tuple[float, float]:
    lon = (col + 0.5) / width * 360.0 - 180.0
    lat = 90.0 - (row + 0.5) / height * 180.0
    return float(lat), float(lon)


def _top_map_candidates(
    *,
    path: str,
    count: int,
    stride: int,
    suppress_radius: int,
) -> list[tuple[float, float, float]]:
    """Return (lat, lon, ml_map_percent) peaks from a decimated suitability raster."""
    with rasterio.open(path) as src:
        width, height = src.width, src.height
        out_w = max(1, width // stride)
        out_h = max(1, height // stride)
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
        )
    grid = np.asarray(arr, dtype=np.float64)
    grid[~np.isfinite(grid)] = -1.0
    grid[(grid < 0) | (grid > 100)] = -1.0

    work = grid.copy()
    candidates: list[tuple[float, float, float]] = []
    want = max(1, count)
    for _ in range(want * 4):
        idx = int(np.argmax(work))
        peak = float(work.flat[idx])
        if peak < 0:
            break
        row_d, col_d = np.unravel_index(idx, work.shape)
        row = min(height - 1, int(row_d * stride + stride // 2))
        col = min(width - 1, int(col_d * stride + stride // 2))
        lat, lon = _pixel_center_to_lat_lon(row, col, height, width)
        candidates.append((lat, lon, peak))
        r0 = max(0, row_d - suppress_radius)
        r1 = min(work.shape[0], row_d + suppress_radius + 1)
        c0 = max(0, col_d - suppress_radius)
        c1 = min(work.shape[1], col_d + suppress_radius + 1)
        work[r0:r1, c0:c1] = -1.0
        if len(candidates) >= want:
            break
    return candidates


def _famous_site_seeds() -> list[tuple[float, float, float]]:
    """Always re-score curated sites — ML map peaks can disagree with live /predict."""
    return [(s.lat, s.lon, float("nan")) for s in MARS_FAMOUS_SITES]


def _contributions_from_breakdown(
    score_breakdown: dict[str, Any] | None,
    *,
    limit: int = 4,
) -> list[dict[str, float | str]]:
    if not score_breakdown:
        return []
    rows: list[tuple[str, float]] = []
    for key, entry in score_breakdown.items():
        if not isinstance(entry, dict):
            continue
        contrib = entry.get("contribution_percent")
        if contrib is None:
            contrib = (entry.get("contribution") or 0) * 100.0
        rows.append((str(key), float(contrib)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [
        {"property_name": key, "contribution_percent": round(val, 2)}
        for key, val in rows[:limit]
    ]


def find_best_landing_site(
    predict_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    data_dir: str | None = None,
    map_candidates: int = 24,
    rescore_limit: int = 40,
    stride: int = 12,
) -> dict[str, Any]:
    """
    Pick the best landing site by re-scoring ML-map peaks **and** famous sites
    with predict_fn (uses the caller's active scoring weights).

    Famous sites are always included because the archived suitability GeoTIFF can
    disagree with the live gap-filled /predict pipeline (map may show ~68% while
    live score is ~54%).
    """
    path = _suitability_path(data_dir)
    candidates: list[tuple[float, float, float]] = []
    if os.path.isfile(path):
        candidates.extend(
            _top_map_candidates(
                path=path,
                count=map_candidates,
                stride=stride,
                suppress_radius=3,
            )
        )
    candidates.extend(_famous_site_seeds())

    # De-dupe near-identical lat/lon (keep first = prefer map peak metadata when tied)
    seen: list[tuple[float, float]] = []
    unique: list[tuple[float, float, float]] = []
    for lat, lon, ml_score in candidates:
        if any(abs(lat - a) < 0.15 and abs(lon - b) < 0.15 for a, b in seen):
            continue
        seen.append((lat, lon))
        unique.append((lat, lon, ml_score))

    if not unique:
        return {
            "success": False,
            "error": (
                f"No candidates. Need {SUITABILITY_FILENAME} and/or famous sites. "
                "Run batch_global_landing_suitability.py if the map is missing."
            ),
        }

    scored: list[dict[str, Any]] = []
    for lat, lon, ml_score in unique[:rescore_limit]:
        mars_data = sample_mars_data_at(lat, lon, data_dir=data_dir)
        result = predict_fn(mars_data)
        if not result.get("success"):
            continue
        entry: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "landing_score_percent": result.get("landing_score"),
            "interpretation": result.get("score_interpretation"),
            "score_breakdown": result.get("score_breakdown"),
            "scoring_weights_percent": result.get("scoring_weights_percent"),
        }
        if ml_score == ml_score:  # not NaN
            entry["ml_map_score_percent"] = round(float(ml_score), 2)
        scored.append(entry)

    if not scored:
        return {"success": False, "error": "Could not score any candidate locations."}

    scored.sort(key=lambda r: float(r.get("landing_score_percent") or 0), reverse=True)
    best = scored[0]
    weights = best.get("scoring_weights_percent")
    return {
        "success": True,
        "best": {
            "latitude": best["latitude"],
            "longitude": best["longitude"],
            "landing_score_percent": best["landing_score_percent"],
            "interpretation": best["interpretation"],
            "region_description": (
                f"Mars {best['latitude']:.2f}°N, {best['longitude']:.2f}°E "
                f"(live re-score of {len(scored)} candidates: ML-map peaks + famous sites)"
            ),
            "top_contributions": _contributions_from_breakdown(best.get("score_breakdown")),
            "ml_map_score_percent": best.get("ml_map_score_percent"),
            "runner_up_scores": [
                {
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "landing_score_percent": r["landing_score_percent"],
                }
                for r in scored[1:4]
            ],
        },
        "scoring_weights_percent": weights,
        "candidates_evaluated": len(scored),
    }
