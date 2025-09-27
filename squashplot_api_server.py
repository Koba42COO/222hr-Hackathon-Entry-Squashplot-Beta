#!/usr/bin/env python3
"""
SquashPlot API Server - Production-Ready Backend
===============================================

FastAPI-based server providing REST API endpoints for SquashPlot operations.
Built following Replit template architecture with Andy's CLI improvements integrated.

Features:
- Real-time server monitoring (Andy's check_server.py logic)
- CLI command execution and templates
- Compression operations and validation
- WebSocket support for live updates
- Professional error handling and logging
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse

# SquashPlot Core Imports - Andy's Graceful Fallback Approach
try:
    from squashplot import SquashPlotCompressor
    SQUASHPLOT_AVAILABLE = True
    print("SquashPlot core compression engine loaded")
except (ImportError, NameError) as e:
    SQUASHPLOT_AVAILABLE = False
    print(f"SquashPlot compression engine not available: {e}")
    print("Running in demo mode with CLI integration")

# Andy's check_server utility
try:
    from check_server import check_server
    CHECK_SERVER_AVAILABLE = True
    print("Andy's check_server utility loaded")
except ImportError:
    CHECK_SERVER_AVAILABLE = False
    print("check_server utility not available - using fallback")

# Configuration
class Config:
    TITLE = "SquashPlot API Server"
    VERSION = "2.0.0"
    DESCRIPTION = "Professional Chia Plot Compression API with Andy's CLI Integration"

    # Server settings
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", "8080"))

    # Replit-specific optimizations
    REPLIT_MODE = os.getenv("REPLIT", False)

    # Andy's CLI command templates
    CLI_COMMANDS = {
        "web": "python main.py --web",
        "cli": "python main.py --cli",
        "demo": "python main.py --demo",
        "check_server": "python check_server.py",
        "basic_plotting": "python squashplot.py -t /tmp/plot1 -d /plots -f YOUR_FARMER_KEY -p YOUR_POOL_KEY",
        "dual_temp": "python squashplot.py -t /tmp/plot1 -2 /tmp/plot2 -d /plots -f YOUR_FARMER_KEY -p YOUR_POOL_KEY -n 2",
        "with_compression": "python squashplot.py --compress 3 -t /tmp/plot1 -d /plots -f YOUR_FARMER_KEY -p YOUR_POOL_KEY"
    }

# Initialize FastAPI app
app = FastAPI(
    title=Config.TITLE,
    version=Config.VERSION,
    description=Config.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Replit deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replit handles CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Root route - serve the enhanced dashboard
@app.get("/")
async def root():
    """Serve the enhanced SquashPlot dashboard"""
    try:
        dashboard_path = Path("squashplot_dashboard_enhanced.html")
        if dashboard_path.exists():
            return FileResponse(dashboard_path)
        else:
            # Fallback to basic dashboard
            basic_dashboard = Path("squashplot_dashboard.html")
            if basic_dashboard.exists():
                return FileResponse(basic_dashboard)
            else:
                return HTMLResponse("""
                <html>
                    <head><title>SquashPlot Dashboard</title></head>
                    <body>
                        <h1>SquashPlot Dashboard</h1>
                        <p>Dashboard files not found. Please ensure the HTML files are in the correct location.</p>
                        <p><a href="/docs">API Documentation</a></p>
                    </body>
                </html>
                """)
    except Exception as e:
        return HTMLResponse(f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>")

# Bridge endpoint for CLI command execution
@app.post("/api/bridge")
async def bridge_endpoint(request: dict):
    """Bridge endpoint for CLI command execution - works across all platforms"""
    try:
        command = request.get("command", "").strip()
        
        if command == "PING":
            return {"status": "success", "response": "PONG"}
        elif command == "STATUS":
            return {
                "status": "success", 
                "response": {
                    "server": "online",
                    "platform": "api_server",
                    "timestamp": datetime.now().isoformat()
                }
            }
        elif command == "STOP":
            # This would stop the bridge, but we'll just return success
            return {"status": "success", "response": "STOPPED"}
        else:
            return {"status": "error", "response": "Unknown command"}
            
    except Exception as e:
        return {"status": "error", "response": str(e)}

# Multi-OS Installer Download endpoints
@app.get("/download/installers")
async def get_installer_info():
    """Get information about available installers for all platforms"""
    try:
        import json
        checksums_path = Path("installers/checksums/checksums.json")
        
        if checksums_path.exists():
            with open(checksums_path, 'r') as f:
                installer_info = json.load(f)
            return installer_info
        else:
            # Fallback if checksums file doesn't exist
            return {
                "error": "Installer information not available",
                "message": "Please contact support for installer downloads"
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get installer info: {str(e)}"}
        )

@app.get("/secure-download/{platform}/{filename}")
@app.head("/secure-download/{platform}/{filename}")
async def secure_download(platform: str, filename: str):
    """Secure download endpoint with integrity verification"""
    try:
        # Security checks
        if platform not in ["windows", "macos", "linux"]:
            raise HTTPException(status_code=400, detail="Invalid platform")
        
        # Load checksums
        import json
        import hashlib
        checksums_path = Path("installers/checksums/checksums.json")
        
        if not checksums_path.exists():
            raise HTTPException(status_code=404, detail="Installer information not found")
        
        with open(checksums_path, 'r') as f:
            installer_info = json.load(f)
        
        # Verify file exists and matches expected checksum
        installer_path = Path(f"installers/{platform}/{filename}")
        if not installer_path.exists():
            raise HTTPException(status_code=404, detail="Installer file not found")
        
        # Verify file integrity
        expected_checksum = installer_info["installers"][platform]["sha256"]
        actual_checksum = calculate_file_checksum(installer_path)
        
        if actual_checksum != expected_checksum:
            raise HTTPException(status_code=500, detail="File integrity check failed")
        
        # Return file with security headers
        response = FileResponse(
            path=installer_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        response.headers["X-Download-Signature"] = expected_checksum
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

# Legacy endpoint for backward compatibility
@app.get("/download/exe-installer")
async def download_exe_installer():
    """Download the SquashPlot EXE installer"""
    try:
        # Look for EXE installer files
        exe_files = [
            "squashplot_bridge_professional/SquashPlotBridgeEXEInstaller.py",
            "squashplot_bridge_professional/SquashPlotBridgeInstaller.py",
            "bridge_installer.py"
        ]
        
        installer_found = None
        for exe_file in exe_files:
            if os.path.exists(exe_file):
                installer_found = exe_file
                break
        
        if installer_found:
            # Return the installer file
            return FileResponse(
                path=installer_found,
                filename=f"SquashPlotInstaller_{datetime.now().strftime('%Y%m%d')}.py",
                media_type="application/octet-stream"
            )
        else:
            # Create a simple installer script if none found
            installer_content = '''#!/usr/bin/env python3
"""
SquashPlot Bridge Installer
===========================
Simple installer script for SquashPlot Bridge
"""

import os
import sys
import subprocess
import platform

def install_bridge():
    """Install SquashPlot Bridge"""
    print("SquashPlot Bridge Installer")
    print("=" * 40)
    
    system = platform.system().lower()
    print(f"Detected system: {system}")
    
    if system == "windows":
        print("Installing for Windows...")
        # Windows-specific installation
        install_windows()
    elif system == "linux":
        print("Installing for Linux...")
        # Linux-specific installation
        install_linux()
    elif system == "darwin":
        print("Installing for macOS...")
        # macOS-specific installation
        install_macos()
    else:
        print(f"Unsupported system: {system}")
        return False
    
    print("Installation completed!")
    return True

def install_windows():
    """Install for Windows"""
    try:
        # Install Python dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully")
        
        # Create desktop shortcut (if possible)
        create_desktop_shortcut()
        
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        return False

def install_linux():
    """Install for Linux"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        return False

def install_macos():
    """Install for macOS"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "SquashPlot Bridge.lnk")
        target = os.path.join(os.getcwd(), "universal_bridge.py")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{target}"'
        shortcut.WorkingDirectory = os.getcwd()
        shortcut.save()
        
        print("Desktop shortcut created")
    except ImportError:
        print("Shortcut creation skipped (required packages not available)")
    except Exception as e:
        print(f"Shortcut creation failed: {e}")

if __name__ == "__main__":
    success = install_bridge()
    if success:
        print("\\nBridge installed successfully!")
        print("You can now run: python universal_bridge.py")
    else:
        print("\\nInstallation failed. Please check the error messages above.")
    
    input("Press Enter to exit...")
'''
            
            # Return the generated installer
            return HTMLResponse(
                content=installer_content,
                headers={
                    "Content-Disposition": f"attachment; filename=SquashPlotInstaller_{datetime.now().strftime('%Y%m%d')}.py"
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to prepare installer: {str(e)}"}
        )

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models
class CompressionRequest(BaseModel):
    input_file: str
    output_file: str
    level: int = 3
    algorithm: str = "auto"

class CLICommandRequest(BaseModel):
    command: str
    timeout: int = 30

class ServerStatus(BaseModel):
    status: str
    uptime: float
    timestamp: str
    version: str = Config.VERSION

# Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interface selection landing page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SquashPlot - Choose Your Interface</title>
        <style>
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1a4d3a 0%, #2e7d3e 50%, #f0ad4e 100%);
                min-height: 100vh;
                margin: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
            }
            .container {
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                max-width: 600px;
                width: 90%;
            }
            h1 {
                font-size: 3rem;
                margin-bottom: 10px;
                background: linear-gradient(135deg, white, #f0ad4e);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                margin-bottom: 30px;
            }
            .interface-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .interface-card {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 25px;
                transition: all 0.3s ease;
                cursor: pointer;
                text-decoration: none;
                color: white;
                display: block;
            }
            .interface-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                background: rgba(255, 255, 255, 0.15);
            }
            .interface-title {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .interface-desc {
                font-size: 0.9rem;
                opacity: 0.8;
                line-height: 1.4;
            }
            .status {
                margin-top: 20px;
                padding: 10px;
                background: rgba(0, 255, 0, 0.1);
                border-radius: 10px;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SquashPlot</h1>
            <div class="subtitle">Advanced Chia Plot Compression with Andy's Enhancements</div>

            <div class="interface-grid">
                <a href="/dashboard" class="interface-card">
                    <div class="interface-title">Enhanced Dashboard</div>
                    <div class="interface-desc">
                        Andy's professional UI with real-time monitoring,
                        CLI integration, and modern design system
                    </div>
                </a>

                <a href="/original" class="interface-card">
                    <div class="interface-title">📊 Original Interface</div>
                    <div class="interface-desc">
                        Classic SquashPlot interface with compression tools
                        and farming calculators
                    </div>
                </a>

                <a href="/docs" class="interface-card">
                    <div class="interface-title">API Documentation</div>
                    <div class="interface-desc">
                        Interactive API docs for developers and integrations
                        with FastAPI/Swagger UI
                    </div>
                </a>

                <a href="/health" class="interface-card">
                    <div class="interface-title">🔍 System Status</div>
                    <div class="interface-desc">
                        Real-time system health, API status, and monitoring
                        information
                    </div>
                </a>

                        <a href="/bridge-download" class="interface-card">
                            <div class="interface-title">🔒 Secure Bridge App</div>
                            <div class="interface-desc">
                                Download the advanced CLI automation bridge for
                                professional users with top-level security
                            </div>
                        </a>

                        <a href="/command-control" class="interface-card">
                            <div class="interface-title">Command Control</div>
                            <div class="interface-desc">
                                Manage which commands are allowed on your system
                                with user-friendly checkboxes and security controls
                            </div>
                        </a>
            </div>

            <div class="status">
                Server Online | Authentication Ready | CLI Integration Active
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": Config.VERSION,
        "squashplot_available": SQUASHPLOT_AVAILABLE
    }

@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get comprehensive server status (Andy's check_server.py logic)"""
    start_time = time.time() - psutil.boot_time()

    # System information
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Check SquashPlot availability
    status = "online" if SQUASHPLOT_AVAILABLE else "limited"

    return ServerStatus(
        status=status,
        uptime=start_time,
        timestamp=datetime.now().isoformat(),
        version=Config.VERSION
    )

@app.get("/system-info")
async def system_info():
    """Detailed system information"""
    return {
        "cpu": {
            "cores": psutil.cpu_count(),
            "usage_percent": psutil.cpu_percent(interval=0.1)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
            "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        },
        "replit_mode": Config.REPLIT_MODE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cli-commands")
async def get_cli_commands():
    """Get available CLI commands (Andy's templates)"""
    return {
        "commands": Config.CLI_COMMANDS,
        "note": "These are command templates. Execute them in your local terminal for full functionality.",
        "web_execution": "Available through web interface for command preview and templates"
    }

@app.post("/cli/execute")
async def execute_cli_command(request: CLICommandRequest):
    """Execute CLI command (simulation - actual execution requires terminal)"""
    command = request.command.strip()

    # Security check - only allow safe SquashPlot commands
    allowed_commands = [
        "python main.py",
        "python check_server.py",
        "python squashplot.py"
    ]

    if not any(cmd in command for cmd in allowed_commands):
        raise HTTPException(status_code=400, detail="Command not allowed for web execution")

    # Simulate command execution (in real deployment, this would execute safely)
    return {
        "command": command,
        "status": "simulated",
        "output": f"Command would execute: {command}\n\nNote: For full functionality, run this command in your local terminal.",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/compress")
async def compress_file(request: CompressionRequest):
    """Compress a file using SquashPlot"""
    if not SQUASHPLOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="SquashPlot compression engine not available")

    try:
        # Initialize compressor
        compressor = SquashPlotCompressor()

        # Perform compression
        result = compressor.compress_plot(
            request.input_file,
            request.output_file,
            compression_level=request.level
        )

        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle WebSocket messages
                response = {"type": "echo", "data": message, "timestamp": datetime.now().isoformat()}
                await manager.send_personal_message(json.dumps(response), websocket)
            except json.JSONDecodeError:
                await manager.send_personal_message(json.dumps({"error": "Invalid JSON"}), websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(json.dumps({
            "type": "disconnect",
            "message": "Client disconnected",
            "timestamp": datetime.now().isoformat()
        }))

@app.get("/api/status")
async def api_status():
    """Simple API status for frontend checks"""
    return {"status": "online", "version": Config.VERSION, "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def status():
    """Status endpoint for developer bridge compatibility"""
    return {
        "status": "online", 
        "version": Config.VERSION, 
        "timestamp": datetime.now().isoformat(),
        "developer_mode": True,
        "allowed_commands": ["hello-world"]
    }

@app.post("/execute")
async def execute_command(request: dict):
    """Execute command endpoint for developer bridge"""
    try:
        import subprocess
        import os
        
        command = request.get("command", "").strip()
        local_execution = request.get("local_execution", False)
        
        # Developer bridge lockdown - only allow hello-world and notepad when connected
        if command == "hello-world":
            return {
                "success": True,
                "output": "Hello World! SquashPlot Developer Bridge is working correctly.",
                "command": command,
                "timestamp": datetime.now().isoformat(),
                "developer_mode": True
            }
        elif command == "notepad" and local_execution:
            # Actually execute notepad
            try:
                subprocess.Popen(["notepad.exe"], shell=True)
                return {
                    "success": True,
                    "output": "Notepad opened successfully!",
                    "command": command,
                    "timestamp": datetime.now().isoformat(),
                    "developer_mode": True,
                    "local_execution": True
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to open notepad: {str(e)}",
                    "command": command,
                    "timestamp": datetime.now().isoformat(),
                    "developer_mode": True
                }
        else:
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "error": f"Command '{command}' not allowed in developer mode. Only 'hello-world' and 'notepad' (when connected) are permitted.",
                    "allowed_commands": ["hello-world", "notepad"],
                    "timestamp": datetime.now().isoformat(),
                    "developer_mode": True
                }
            )
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "developer_mode": True
        }, 500

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the enhanced SquashPlot dashboard"""
    try:
        with open("squashplot_dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <h1>Dashboard Not Found</h1>
        <p>The dashboard file is not available. Please check if squashplot_dashboard.html exists.</p>
        <a href="/">Back to main interface</a>
        """)

@app.get("/original", response_class=HTMLResponse)
async def original_interface():
    """Serve the original SquashPlot interface"""
    try:
        with open("squashplot_web_interface.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <h1>Original Interface Not Found</h1>
        <p>The original interface file is not available.</p>
        <a href="/">Back to main interface</a>
        """)

@app.get("/bridge-download", response_class=HTMLResponse)
async def bridge_download():
    """Serve the Secure Bridge App download page"""
    try:
        with open("bridge_download.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <h1>Bridge Download Not Available</h1>
        <p>The Secure Bridge App download page is not available.</p>
        <a href="/">Back to main interface</a>
        """)

@app.get("/eula", response_class=HTMLResponse)
async def eula():
    """Serve the End User License Agreement"""
    try:
        with open("EULA.md", "r", encoding="utf-8") as f:
            content = f.read()
            # Convert markdown to HTML (simplified)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SquashPlot Bridge App - EULA</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
                    h1 {{ color: #333; border-bottom: 2px solid #00d4ff; }}
                    h2 {{ color: #555; margin-top: 30px; }}
                    .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="warning">
                    <strong>IMPORTANT:</strong> This is a legal document. Please read carefully before using the software.
                </div>
                <pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">{content}</pre>
            </body>
            </html>
            """
            return HTMLResponse(html_content)
    except FileNotFoundError:
        return HTMLResponse("<h1>EULA Not Available</h1><p>The End User License Agreement is not available.</p>")

@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    """Serve the Privacy Policy"""
    try:
        with open("PRIVACY_POLICY.md", "r", encoding="utf-8") as f:
            content = f.read()
            # Convert markdown to HTML (simplified)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SquashPlot Bridge App - Privacy Policy</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
                    h1 {{ color: #333; border-bottom: 2px solid #00d4ff; }}
                    h2 {{ color: #555; margin-top: 30px; }}
                    .notice {{ background: #e3f2fd; border: 1px solid #2196f3; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="notice">
                    <strong>📋 Privacy Notice:</strong> This policy explains how we collect, use, and protect your information.
                </div>
                <pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">{content}</pre>
            </body>
            </html>
            """
            return HTMLResponse(html_content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Privacy Policy Not Available</h1><p>The Privacy Policy is not available.</p>")

@app.get("/security-warning", response_class=HTMLResponse)
async def security_warning():
    """Serve the security warning modal"""
    try:
        with open("security_warning_modal.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Security Warning Not Available</h1><p>The security warning page is not available.</p>")

@app.get("/download/bridge_installer.py")
async def download_bridge_installer():
    """Download the bridge installer script"""
    try:
        if os.path.exists("bridge_installer.py"):
            return FileResponse(
                "bridge_installer.py",
                media_type="application/octet-stream",
                filename="bridge_installer.py"
            )
        else:
            raise HTTPException(status_code=404, detail="Bridge installer not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving installer: {str(e)}")

@app.get("/download/uninstall_bridge.py")
async def download_uninstaller():
    """Download the bridge uninstaller script"""
    try:
        if os.path.exists("uninstall_bridge.py"):
            return FileResponse(
                "uninstall_bridge.py",
                media_type="application/octet-stream",
                filename="uninstall_bridge.py"
            )
        else:
            raise HTTPException(status_code=404, detail="Bridge uninstaller not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving uninstaller: {str(e)}")

        @app.get("/download/system_scanner.py")
        async def download_system_scanner():
            """Download the system scanner script"""
            try:
                if os.path.exists("system_scanner.py"):
                    return FileResponse(
                        "system_scanner.py",
                        media_type="application/octet-stream",
                        filename="system_scanner.py"
                    )
                else:
                    raise HTTPException(status_code=404, detail="System scanner not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error serving system scanner: {str(e)}")

@app.get("/command-control", response_class=HTMLResponse)
async def command_control():
    """Serve the user command control interface"""
    try:
        with open("user_command_control.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <h1>Command Control Not Available</h1>
        <p>The command control interface is not available.</p>
        <a href="/">Back to main interface</a>
        """)

@app.get("/squashplot_bridge_professional/SquashPlotDeveloperBridge.html", response_class=HTMLResponse)
async def developer_bridge():
    """Serve the SquashPlot Developer Bridge interface"""
    try:
        with open("squashplot_bridge_professional/SquashPlotDeveloperBridge.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <h1>Developer Bridge Not Available</h1>
        <p>The SquashPlot Developer Bridge interface is not available.</p>
        <a href="/">Back to main interface</a>
        """)

        @app.get("/api/command-preferences")
        async def get_command_preferences():
            """Get current command preferences"""
            try:
                from user_command_controller import UserCommandController
                controller = UserCommandController()
                return controller.get_web_interface_data()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting preferences: {str(e)}")

        @app.post("/api/command-preferences")
        async def update_command_preferences(request: dict):
            """Update command preferences"""
            try:
                from user_command_controller import UserCommandController
                controller = UserCommandController()
                
                if "user_preferences" in request:
                    controller.import_configuration(request)
                    controller.update_bridge_config()
                    return {"status": "success", "message": "Preferences updated"}
                else:
                    raise HTTPException(status_code=400, detail="Invalid request format")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error updating preferences: {str(e)}")

        @app.post("/api/command-preferences/reset")
        async def reset_command_preferences():
            """Reset command preferences to defaults"""
            try:
                from user_command_controller import UserCommandController
                controller = UserCommandController()
                controller.reset_to_defaults()
                controller.update_bridge_config()
                return {"status": "success", "message": "Preferences reset to defaults"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error resetting preferences: {str(e)}")

# Replit-specific optimizations
if Config.REPLIT_MODE:
    print("Running in Replit mode - optimizing for Replit environment")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    print("SquashPlot API Server starting up...")
    print(f"Server will be available at: https://your-replit-url.replit.dev")
    print(f"Local access at: http://localhost:{Config.PORT}")
    print("API documentation at: /docs")
    # Broadcast startup message
    await manager.broadcast(json.dumps({
        "type": "startup",
        "message": "SquashPlot API Server started",
        "version": Config.VERSION,
        "timestamp": datetime.now().isoformat()
    }))

if __name__ == "__main__":
    print("Starting SquashPlot API Server...")
    print(f"Port: {Config.PORT}")
    print(f"Replit Mode: {Config.REPLIT_MODE}")

    uvicorn.run(
        "squashplot_api_server:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
        log_level="info"
    )
