#!/bin/bash
# Start VANGUARD (Flask serves API + frontend).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Quieter TensorFlow C++ logs (app.py also sets this if unset).
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

cleanup() {
    echo ""
    echo "Stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

echo ""
echo "  VANGUARD"
echo "  --------"
echo "  Starting server (first load can take a minute)…"
echo "  Verbose logs: export VANGUARD_VERBOSE=1"
echo ""

uv run python backend/app.py

