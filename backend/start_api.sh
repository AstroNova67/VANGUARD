#!/bin/bash

# Mars Landing Suitability API Startup Script

echo "🚀 Starting Mars Landing Suitability API Server..."

# Navigate to project root directory
cd "$(dirname "$0")/.."

# Preferred: use uv if available (matches README).
if command -v uv &> /dev/null; then
    echo "🌐 Starting Flask server with uv..."
    exec uv run python backend/app.py
fi

# Fallback: use an existing venv if present.
VENV_DIR=""
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
fi

if [ -z "$VENV_DIR" ]; then
    echo "❌ No virtual environment found."
    echo ""
    echo "Create one and install deps:"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "Or install uv and run:"
    echo "   ./start_api.sh"
    exit 1
fi

echo "🔧 Activating virtual environment ($VENV_DIR)..."
source "$VENV_DIR/bin/activate"

echo "🌍 Starting Flask API server on http://localhost:5002"
echo "Press Ctrl+C to stop the server"
echo ""

exec python backend/app.py
