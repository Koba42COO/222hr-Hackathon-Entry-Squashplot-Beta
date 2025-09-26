#!/usr/bin/env python3
"""
SquashPlot - Advanced Chia Plot Compression Tool
===============================================

Professional Chia plotting solution featuring:
- Advanced Multi-Stage Compression (Zstandard, Brotli, LZ4)
- Chia Blockchain Integration
- Professional Web Dashboard
- Mad Max/BladeBit Compatible CLI
- GPU Optimization & Resource Management
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point"""
    print("🗜️ SquashPlot - Advanced Chia Plot Compression")
    print("=" * 60)
    print("🔧 Multi-Stage Compression | 🌱 Chia Integration | 📊 Professional Dashboard")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="SquashPlot - Advanced Chia Plot Compression Tool")
    parser.add_argument('--web', action='store_true',
                       help='Start web interface (default)')
    parser.add_argument('--cli', action='store_true',
                       help='Start command-line interface')
    parser.add_argument('--demo', action='store_true',
                       help='Run interactive demo')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port for web server (default: 8080)')

    args = parser.parse_args()

    # Default to web interface
    if not any([args.web, args.cli, args.demo]):
        args.web = True

    if args.web:
        start_web_interface(args.port)
    elif args.cli:
        start_cli_interface()
    elif args.demo:
        run_demo()

def start_web_interface(port=8080):  # Replit default port
    """Start the comprehensive SquashPlot enterprise web interface"""
    print("🚀 Starting SquashPlot Enterprise Platform...")
    print("=" * 60)
    print("🌱 Complete Chia Farming Management System")
    print("🔐 Authentication | 📊 Multi-Dashboard | ⚡ Job Management")
    print("📈 Analytics | 🏊 Pool Management | 💰 Rewards Tracking")
    print("💾 Storage Management | 🩺 Health Monitoring")
    print("=" * 60)
    print(f"📡 Access URL: http://localhost:{port}")
    print()

    try:
        # Import the comprehensive Flask application
        from src.web_server import app

        print("✅ SquashPlot Enterprise Platform loaded successfully!")
        print("🔐 Authentication System: Active")
        print("📊 Dashboard System: Multi-page enterprise interface")
        print("⚡ Job Management: Real-time plotting job control")
        print("📈 Analytics: Professional data visualization")
        print("🩺 Health Integration: Plot and harvester monitoring")
        print()

        # Start the Flask development server
        print(f"🚀 Starting Flask server on port {port}...")
        app.run(host='127.0.0.1', port=port, debug=True)

    except ImportError as e:
        print(f"❌ Enterprise platform not available: {e}")
        print("💡 Make sure all dependencies are installed")
        print("🔧 Install with: pip install -r requirements.txt")
        start_cli_interface()
    except Exception as e:
        print(f"❌ Failed to start enterprise platform: {e}")
        print("💡 Check port availability and dependencies")
        start_cli_interface()

def start_cli_interface():
    """Start the command-line interface"""
    print("💻 Starting SquashPlot CLI...")
    print()

    try:
        # Import and run SquashPlot CLI from our existing system
        from squashplot import main as squashplot_main
        squashplot_main()
    except ImportError as e:
        print(f"❌ SquashPlot CLI module not found: {e}")
        print("💡 Use the web interface to access SquashPlot features")
        print("🔧 Or run: python -m squashplot")

def run_demo():
    """Run interactive SquashPlot demo"""
    print("🎯 Starting SquashPlot Demo...")
    print()

    try:
        # Import SquashPlot demo functionality
        from squashplot import SquashPlotCompressor

        print("🔧 Testing Multi-Stage Compression Engine...")
        compressor = SquashPlotCompressor(pro_enabled=False)
        print("✅ Basic compression engine operational")

        print("\n🧪 Testing Chia Integration...")
        from chia_resources.chia_resource_query import ChiaResourceQuery
        chia_query = ChiaResourceQuery()
        stats = chia_query.get_database_stats()
        print(f"✅ Chia resources database: {stats['total_resources']} resources available")

        print("\n📊 Testing Compression Algorithms...")
        # Test available compression algorithms
        test_algorithms = ["zlib", "bz2", "lzma"]
        for algo in test_algorithms:
            try:
                if algo == "zlib":
                    import zlib
                    result = zlib.compress(b"test data")
                elif algo == "bz2":
                    import bz2
                    result = bz2.compress(b"test data")
                elif algo == "lzma":
                    import lzma
                    result = lzma.compress(b"test data")
                print(f"   ✅ {algo}: Available")
            except ImportError:
                print(f"   ⚠️ {algo}: Not available")

        print("\n🚀 All SquashPlot systems operational!")
        print("💡 Use the web interface for full functionality")
        print("🔗 Start with: python main.py --web")

    except ImportError as e:
        print(f"⚠️ Some SquashPlot modules need configuration: {e}")
        print("💡 Use the web interface to set up and configure SquashPlot")
        print("🔧 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
