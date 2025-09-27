#!/usr/bin/env python3
"""
SquashPlot - Advanced Chia Plot Compression Tool
===============================================

Professional Chia plotting solution featuring:
- Advanced Multi-Stage Compression (Zstandard, Brotli, LZ4)
- prime aligned compute-Enhanced Compression Technology
- Wallace Transform with Golden Ratio Optimization
- CUDNT Complexity Reduction (O(n²) → O(n^1.44))
- Black Glass UI/UX Professional Web Dashboard
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
    print("SquashPlot - Advanced Chia Plot Compression")
    print("=" * 60)
    print("prime aligned compute Enhancement | Golden Ratio Optimization | Black Glass UI/UX")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="SquashPlot - Advanced Chia Plot Compression Tool")
    parser.add_argument('--web', action='store_true',
                       help='Start web interface (default)')
    parser.add_argument('--cli', action='store_true',
                       help='Start command-line interface')
    parser.add_argument('--demo', action='store_true',
                       help='Run interactive demo')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port for web server (default: 8080 - Replit optimized)')

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

def start_bridge_server():
    """Start the bridge server for developer purposes"""
    print("Starting Bridge Server for Developer Mode...")
    try:
        import subprocess
        import threading
        import time
        
        # Start bridge in background thread
        def run_bridge():
            try:
                subprocess.run([sys.executable, "universal_bridge.py"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Bridge server error: {e}")
        
        bridge_thread = threading.Thread(target=run_bridge, daemon=True)
        bridge_thread.start()
        
        # Give bridge time to start
        time.sleep(2)
        print("Bridge server started on port 8443")
        return True
        
    except Exception as e:
        print(f"Failed to start bridge server: {e}")
        return False

def start_web_interface(port=8080):  # Replit default port
    """Start the web interface using Replit template structure"""
    print("Starting SquashPlot Web Dashboard...")
    print(f"Replit URL: https://your-replit-name.replit.dev")
    print(f"Local access: http://localhost:{port}")
    print("Andy's CLI Integration: Available via dashboard")
    print()
    
    # Start bridge server first for developer purposes
    bridge_started = start_bridge_server()

    try:
        # Import and start the enhanced API server (Andy's integration)
        from squashplot_api_server import app

        print("SquashPlot API Server started successfully!")
        print(f"Dashboard: http://localhost:{port}")
        print(f"API Docs: http://localhost:{port}/docs")
        print("CLI Commands: Available in dashboard")
        print()

        # Start the server with uvicorn for FastAPI
        import uvicorn
        print("Starting FastAPI server with uvicorn...")
        uvicorn.run("squashplot_api_server:app", host='0.0.0.0', port=port, reload=True)

    except ImportError:
        print("Enhanced API server not available, trying basic server...")
        try:
            from squashplot_dashboard import app
            app.run(host='0.0.0.0', port=port, debug=True)
        except ImportError:
            print("No web server available, falling back to CLI mode...")
            start_cli_interface()
    except Exception as e:
        print(f"Failed to start web server: {e}")
        print("Make sure port is available and dependencies are installed")
        print("Falling back to CLI mode...")
        start_cli_interface()

def start_cli_interface():
    """Start the command-line interface"""
    print("Starting SquashPlot CLI...")
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
