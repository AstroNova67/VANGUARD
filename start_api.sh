#!/bin/bash
# Start the VANGUARD Mars Landing System API
echo "🚀 Starting VANGUARD Mars Landing System API..."
cd "$(dirname "$0")"
uv run python backend/app.py
