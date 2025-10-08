#!/bin/bash

# Mars Landing Suitability API Startup Script

echo "🚀 Starting Mars Landing Suitability API Server..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r api_requirements.txt

# Start the Flask server
echo "🌍 Starting Flask API server on http://localhost:5000"
echo "📊 API endpoints:"
echo "   - POST /predict - Get landing suitability prediction"
echo "   - GET /health - Health check"
echo "   - GET /models - Model information"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
