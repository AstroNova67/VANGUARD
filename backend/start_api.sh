#!/bin/bash

# Mars Landing Suitability API Startup Script

echo "🚀 Starting Mars Landing Suitability API Server..."

# Navigate to project root directory
cd "$(dirname "$0")/.."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r backend/api_requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Start the Flask server
echo "🌍 Starting Flask API server on http://localhost:5002"
echo "📊 API endpoints:"
echo "   - POST /predict - Get landing suitability prediction"
echo "   - GET /health - Health check"
echo "   - GET /models - Model information"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python app.py
