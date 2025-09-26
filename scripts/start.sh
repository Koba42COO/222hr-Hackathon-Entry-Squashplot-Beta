#!/bin/bash

echo "🧠 CONSCIOUSNESS MATHEMATICS SYSTEM STARTUP"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Start server
echo ""
echo "🚀 Starting Consciousness Mathematics Server..."
echo "=============================================="
echo ""
echo "Server will be available at:"
echo "  http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python consciousness_api_server.py
