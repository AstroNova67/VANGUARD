"""Shared helpers for ML gap imputation (Arm C)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

REPO = Path(__file__).resolve().parents[1]
UNFILLED = REPO / "frontend" / "3d_globe" / "public" / "data_gap_unfilled"
IDW_FILLED = REPO / "frontend" / "3d_globe" / "public" / "data_gap_filled"
ML_FILLED = REPO / "frontend" / "3d_globe" / "public" / "data_gap_ml_filled"
MODEL_DIR = REPO / "backend" / "saved_models" / "gap_imputers"

# Target band key → GeoTIFF filename (same set as IDW FILL_LAYERS)
TARGET_BANDS: dict[str, str] = {
    "ferric": "omega_ferric_nnphs.tif",
    "albedo": "omega_albedo_r1080.tif",
    "roughness": "mola_roughness_0.6km_numeric.tif",
    "tempRange": "mars_yearly_temperature_range_v1.0.tif",
    "elevation": "MOLA_128ppd_topo.tif",
}

# Stable covariates (filenames under data_gap_unfilled)
COVARIATE_FILES: dict[str, str] = {
    "slope": "mola_hrsc_blend_slope_v2.tif",
    "temperature": "mars_yearly_avg_temperature_celsius.tif",
    "crustalThickness": "mars_crustal_thickness_gmm3_rm1.tif",
    "grsWaterWt": "mars_odyssey_grs_mons_perc_wt.tif",
    "thermalInertia": "tes_dayside_ti_putzig_2007.tif",
}

FEATURE_NAMES: tuple[str, ...] = (
    "lat",
    "sin_lon",
    "cos_lon",
    "slope",
    "temperature",
    "crustalThickness",
    "grsWaterWt",
    "thermalInertia",
)

GAUSS_SIGMA = 6.0
FEATHER_PX = 20.0
MAX_TRAIN_SAMPLES = 80_000
HOLDOUT_FRAC = 0.10
RANDOM_SEED = 42


def mars_transform(width: int, height: int):
    return from_bounds(-180, -90, 180, 90, width, height)


def null_mask(data: np.ndarray, nodata=None) -> np.ndarray:
    bad = (data == 0) | ~np.isfinite(data)
    if nodata is not None:
        try:
            nd = float(nodata)
            if np.isfinite(nd):
                bad |= data == nd
        except (TypeError, ValueError):
            pass
    return bad


def read_band(path: Path) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64), src.nodata


def resample_to_shape(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resample to target shape."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float64, copy=False)
    return zoom(arr, (out_h / h, out_w / w), order=0).astype(np.float64)


def lat_lon_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-center lat/lon grids (equirectangular, same as frontend)."""
    cols = (np.arange(width, dtype=np.float64) + 0.5) / width
    rows = (np.arange(height, dtype=np.float64) + 0.5) / height
    lon = -180.0 + cols * 360.0
    lat = 90.0 - rows * 180.0
    lon_g, lat_g = np.meshgrid(lon, lat)
    return lat_g, lon_g


def covariate_valid_mask(cov: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels where all non-geo covariates are usable (finite, non-zero for TI/slope/etc.)."""
    ok = np.ones(next(iter(cov.values())).shape, dtype=bool)
    for key in ("slope", "temperature", "crustalThickness", "grsWaterWt", "thermalInertia"):
        a = cov[key]
        ok &= np.isfinite(a)
        # temperature can be legitimately negative; only reject exact 0 / non-finite
        if key == "temperature":
            continue
        ok &= a != 0
    return ok


def load_covariate_stack(data_dir: Path, height: int, width: int) -> dict[str, np.ndarray]:
    lat_g, lon_g = lat_lon_grids(height, width)
    cov: dict[str, np.ndarray] = {
        "lat": lat_g,
        "sin_lon": np.sin(np.deg2rad(lon_g)),
        "cos_lon": np.cos(np.deg2rad(lon_g)),
    }
    for key, fname in COVARIATE_FILES.items():
        path = data_dir / fname
        if not path.is_file():
            raise FileNotFoundError(path)
        arr, _ = read_band(path)
        cov[key] = resample_to_shape(arr, height, width)
    return cov


def feature_matrix(cov: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    cols_list = [cov[name][rows, cols] for name in FEATURE_NAMES]
    return np.column_stack(cols_list).astype(np.float64)


def gaussian_blend(
    data: np.ndarray, bad: np.ndarray, sigma: float = GAUSS_SIGMA, feather_px: float = FEATHER_PX
) -> np.ndarray:
    smoothed = gaussian_filter(data.astype(np.float64), sigma=sigma, mode="nearest")
    dist_out = distance_transform_edt(~bad)
    alpha = np.zeros(data.shape, dtype=np.float64)
    alpha[bad] = 1.0
    edge = (~bad) & (dist_out <= feather_px)
    if feather_px > 0:
        alpha[edge] = 1.0 - dist_out[edge] / feather_px
    out = alpha * smoothed + (1.0 - alpha) * data.astype(np.float64)
    return out.astype(np.float32)


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
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(data, dtype=np.float32), 1)
