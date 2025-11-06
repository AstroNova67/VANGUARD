#!/bin/bash
# Start the VANGUARD Mars Landing System API
echo "🚀 Starting VANGUARD Mars Landing System API..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to cleanup background processes on exit
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend API stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend server stopped"
    fi
    exit 0
}

# Set up signal handlers for cleanup
trap cleanup SIGINT SIGTERM

# Check if live-server is installed globally
if ! command -v live-server &> /dev/null; then
    echo "📦 Installing live-server globally..."
    npm install -g live-server
fi

# Start backend API in background
echo "🔧 Starting backend API on port 5002..."
uv run python backend/app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend server in background
echo "🌐 Starting frontend server on port 8080..."
cd frontend/3d_globe
live-server --port=8080 --host=0.0.0.0 &
FRONTEND_PID=$!

# Go back to project root
cd "$SCRIPT_DIR"

echo ""
echo "🎉 VANGUARD Mars Landing System is running!"
echo "📡 Backend API: http://localhost:5002"
echo "🌐 Frontend: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for background processes
wait $BACKEND_PID $FRONTEND_PID

