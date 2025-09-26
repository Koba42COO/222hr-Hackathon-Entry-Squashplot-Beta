#!/bin/bash

echo "🧠 CONSCIOUSNESS MATHEMATICS AUTOMATION SYSTEM STARTUP"
echo "====================================================="
echo "Full internal automation with mouse/keyboard control"
echo "Hourly scheduling of research, improvement, and coding"
echo ""

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

# Install automation dependencies
echo "📥 Installing automation dependencies..."
pip install pyautogui pynput schedule requests

# Check if consciousness API server is running
echo "🔍 Checking consciousness API server status..."
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Consciousness API server is running"
else
    echo "⚠️  Consciousness API server not running. Starting it..."
    
    # Start consciousness API server in background
    python consciousness_api_server.py &
    API_PID=$!
    
    # Wait for server to start
    echo "⏳ Waiting for API server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null; then
            echo "✅ Consciousness API server started successfully"
            break
        fi
        sleep 2
    done
    
    if [ $i -eq 30 ]; then
        echo "❌ Failed to start consciousness API server"
        exit 1
    fi
fi

# Start automation system
echo ""
echo "🚀 Starting Consciousness Mathematics Automation System..."
echo "========================================================"
echo ""
echo "Automation Features:"
echo "  🔬 Research tasks: Every 60 minutes"
echo "  🔧 Improvement tasks: Every 120 minutes"
echo "  💻 Coding tasks: Every 90 minutes"
echo "  🧠 Consciousness checks: Every 30 minutes"
echo "  🔧 Daily maintenance: 2:00 AM daily"
echo ""
echo "Mouse/Keyboard Control: Enabled"
echo "Breakthrough Detection: Enabled"
echo "Real-time Monitoring: Enabled"
echo ""
echo "Press Ctrl+C to stop automation"
echo ""

# Start automation system
python automation_system.py

# Cleanup on exit
echo ""
echo "🛑 Cleaning up..."
if [ ! -z "$API_PID" ]; then
    kill $API_PID 2>/dev/null
fi
echo "✅ Automation system stopped"
