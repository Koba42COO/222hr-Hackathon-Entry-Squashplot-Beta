#!/usr/bin/env python3
"""
SquashPlot - Main Entry Point for Replit
========================================

This is the main entry point for running SquashPlot on Replit.
It provides a simple interface to access all SquashPlot features.
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point"""
    print("🌟 SquashPlot - Chia Plot Compression Tool")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="SquashPlot - Advanced Chia Plot Compression")
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

def start_web_interface(port=8080):
    """Start the web interface"""
    print("🚀 Starting SquashPlot Web Interface...")
    print(f"📡 Server will be available at: https://your-replit-url.replit.dev")
    print(f"🔗 Or locally at: http://localhost:{port}")
    print()

    try:
        # Import and start web server
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import webbrowser

        # Change to the directory containing the web interface
        web_dir = Path(__file__).parent
        os.chdir(web_dir)

        # Start server
        server_address = ('', port)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

        print("✅ Web interface started successfully!")
        print("🌐 Open your browser to access SquashPlot")
        print("💡 Click 'Run' in Replit to start the server")
        print()

        # Try to open web interface automatically
        try:
            webbrowser.open(f"http://localhost:{port}/squashplot_web_interface.html")
        except:
            pass

        httpd.serve_forever()

    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        print("💡 Make sure the port is available and try again")

def start_cli_interface():
    """Start the command-line interface"""
    print("💻 Starting SquashPlot CLI...")
    print()

    try:
        # Import and run CLI
        from squashplot import main as cli_main
        cli_main()
    except ImportError:
        print("❌ CLI module not found")
        print("💡 Make sure all SquashPlot files are in the project")

def run_demo():
    """Run interactive demo"""
    print("🎯 Starting SquashPlot Interactive Demo...")
    print()

    try:
        # Import demo functionality
        from squashplot import SquashPlotCompressor

        print("🗜️ Testing Basic Compression...")
        compressor = SquashPlotCompressor(pro_enabled=False)
        print("✅ Basic compressor initialized")

        print("\n🧠 Testing Pro Compression...")
        compressor_pro = SquashPlotCompressor(pro_enabled=True)
        print("✅ Pro compressor initialized")

        print("\n📊 Running Benchmark...")
        from squashplot import main as cli_main
        # Run benchmark with --benchmark flag
        sys.argv = ['squashplot.py', '--benchmark']
        cli_main()

    except ImportError as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure all SquashPlot files are installed")

if __name__ == "__main__":
    main()
