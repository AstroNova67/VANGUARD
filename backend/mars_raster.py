"""Sample bundled Mars GeoTIFFs at a lat/lon (same keys as the globe /predict payload)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import rasterio

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "3d_globe", "public", "data")

# marsDataKey → filename (matches frontend/3d_globe/index.js marsDatasets)
RASTER_LAYERS: tuple[tuple[str, str], ...] = (
    ("elevation", "MOLA_128ppd_topo.tif"),
    ("slope", "mola_hrsc_blend_slope_v2.tif"),
    ("roughness", "mola_roughness_0.6km_numeric.tif"),
    ("albedo", "omega_albedo_r1080.tif"),
    ("temperature", "mars_yearly_avg_temperature_celsius.tif"),
    ("tempRange", "mars_yearly_temperature_range_v1.0.tif"),
    ("crustalThickness", "mars_crustal_thickness_gmm3_rm1.tif"),
    ("ferric", "omega_ferric_nnphs.tif"),
    ("pyroxene", "omega_pyroxene_bd2000.tif"),
    ("basalt", "TES_Basalt_numeric.tif"),
    ("lambertAlbedo", "TES_Lambert_Albedo_numeric.tif"),
    ("thermalInertia", "tes_dayside_ti_putzig_2007.tif"),
    ("grsWaterWt", "mars_odyssey_grs_mons_perc_wt.tif"),
)


def lat_lon_to_pixel(lat: float, lon: float, width: int, height: int) -> tuple[int, int]:
    """Equirectangular pixel index (same convention as frontend latLonToPixel)."""
    x = int(((lon + 180.0) / 360.0) * width)
    y = int(((90.0 - lat) / 180.0) * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def _is_nodata(value: float, nodata: float | None) -> bool:
    if not np.isfinite(value):
        return True
    if nodata is None:
        return False
    try:
        nd = float(nodata)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(nd):
        return False
    tol = 1e-6 * max(1.0, abs(nd))
    return value == nd or abs(value - nd) <= tol


def _sample_one_tif(path: str, lat: float, lon: float) -> float | None:
    if not os.path.isfile(path):
        return None
    with rasterio.open(path) as src:
        x, y = lat_lon_to_pixel(lat, lon, src.width, src.height)
        arr = src.read(1, window=rasterio.windows.Window(x, y, 1, 1))
        raw = float(arr[0, 0])
        if _is_nodata(raw, src.nodata):
            return None
        return raw


def sample_mars_data_at(
    lat: float,
    lon: float,
    *,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """
    Build a /predict-compatible mars_data dict from GeoTIFF samples at (lat, lon).
    Longitude uses −180…180° (east positive), latitude °N.
    """
    base = data_dir or DATA_DIR
    out: dict[str, Any] = {"lat": float(lat), "lon": float(lon)}
    for key, filename in RASTER_LAYERS:
        path = os.path.join(base, filename)
        val = _sample_one_tif(path, lat, lon)
        if val is not None:
            out[key] = val
    ferric = out.get("ferric")
    if ferric is not None:
        out["dustObserved"] = ferric
    return out
