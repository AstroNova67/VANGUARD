#!/usr/bin/env bash
# Build a 13-band GeoTIFF in the order expected by backend/batch_global_landing_suitability.py
# (same order as frontend/3d_globe/index.js marsDatasets).
#
# Usage:
#   ./scripts/stack_mars_input_layers.sh
#   ./scripts/stack_mars_input_layers.sh /path/to/dir/with/tifs
#   ./scripts/stack_mars_input_layers.sh /path/to/dir/with/tifs /path/to/out.tif
#   ./scripts/stack_mars_input_layers.sh /path/to/dir /path/to/out.tif -- --pixel-stack-only
#
# With no arguments, reads layers from frontend/3d_globe/public/data/ and writes
# mars_global_input_stack_32ppd.tif there.
#
# Requires: Python with rasterio (uses repo .venv if present).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DATA_DIR="${REPO_ROOT}/frontend/3d_globe/public/data"

DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
if [ "$#" -ge 2 ]; then
  OUT="$2"
  shift 2 || true
else
  OUT="${DATA_DIR%/}/mars_global_input_stack_32ppd.tif"
  if [ "$#" -ge 1 ]; then
    shift 1 || true
  fi
fi

if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" "${SCRIPT_DIR}/stack_mars_layers.py" \
  --data-dir "$DATA_DIR" \
  --output "$OUT" \
  "$@"
