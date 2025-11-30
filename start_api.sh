#!/bin/bash
# Start the VANGUARD Mars Landing System Website
# Flask now serves both the API and frontend together

echo "🚀 Starting VANGUARD Mars Landing System Website..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down website..."
    exit 0
}

# Set up signal handlers for cleanup
trap cleanup SIGINT SIGTERM

# Start Flask server (serves both API and frontend)
echo "🌐 Starting Flask server on port 5002..."
echo ""
echo "🎉 VANGUARD Mars Landing System is running!"
echo "🌐 Website: http://localhost:5002"
echo "📡 API: http://localhost:5002/predict"
echo ""
echo "Press Ctrl+C to stop"

# Run Flask (this will block until Ctrl+C)
uv run python backend/app.py

