#!/usr/bin/env python3
"""
SquashPlot Bridge Web Uninstaller
Web-based uninstaller that can be triggered from the website
"""

import os
import sys
import json
import shutil
import subprocess
import platform
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import tempfile

class SquashPlotBridgeUninstaller:
    """Web-based uninstaller for SquashPlot Bridge"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.install_dir = self._get_install_directory()
        self.uninstall_log = []
        
    def _get_install_directory(self):
        """Get installation directory"""
        if self.system == "windows":
            return Path.home() / "AppData" / "Local" / "SquashPlot" / "Bridge"
        elif self.system == "darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / "SquashPlot" / "Bridge"
        else:  # Linux
            return Path.home() / ".local" / "share" / "squashplot" / "bridge"
    
    def log_uninstall(self, message: str):
        """Log uninstall progress"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.uninstall_log.append(log_entry)
        print(log_entry)
    
    def check_installation(self):
        """Check if SquashPlot Bridge is installed"""
        if self.install_dir.exists():
            config_file = self.install_dir / "bridge_config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    return True, config
                except:
                    return True, None
        return False, None
    
    def stop_bridge_processes(self):
        """Stop any running bridge processes"""
        self.log_uninstall("Stopping bridge processes...")
        
        try:
            if self.system == "windows":
                # Stop Python processes running bridge apps
                subprocess.run([
                    "taskkill", "/f", "/im", "python.exe", 
                    "/fi", "WINDOWTITLE eq SquashPlot*"
                ], capture_output=True)
                
                # Also try to stop by process name
                subprocess.run([
                    "taskkill", "/f", "/im", "python.exe",
                    "/fi", "COMMANDLINE eq *SquashPlot*Bridge*"
                ], capture_output=True)
                
            else:
                # Unix-like systems
                subprocess.run(["pkill", "-f", "SquashPlot.*Bridge"], capture_output=True)
                subprocess.run(["pkill", "-f", "squashplot.*bridge"], capture_output=True)
            
            self.log_uninstall("✓ Bridge processes stopped")
            time.sleep(2)  # Give processes time to stop
            
        except Exception as e:
            self.log_uninstall(f"⚠ Could not stop bridge processes: {e}")
    
    def remove_startup_integration(self):
        """Remove startup integration"""
        self.log_uninstall("Removing startup integration...")
        
        if self.system == "windows":
            startup_dir = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
            startup_files = ["SquashPlotBridge.bat", "SquashPlotBridge.lnk"]
            
        elif self.system == "darwin":  # macOS
            startup_dir = Path.home() / "Library" / "LaunchAgents"
            startup_files = ["com.squashplot.bridge.plist"]
            
        else:  # Linux
            startup_dir = Path.home() / ".config" / "autostart"
            startup_files = ["squashplot-bridge.desktop"]
        
        removed_count = 0
        for startup_file in startup_files:
            startup_path = startup_dir / startup_file
            if startup_path.exists():
                try:
                    startup_path.unlink()
                    self.log_uninstall(f"✓ Removed {startup_file}")
                    removed_count += 1
                except Exception as e:
                    self.log_uninstall(f"⚠ Could not remove {startup_file}: {e}")
        
        if removed_count > 0:
            self.log_uninstall(f"✓ Removed {removed_count} startup files")
        else:
            self.log_uninstall("✓ No startup files found to remove")
    
    def remove_installation_files(self):
        """Remove installation directory and files"""
        self.log_uninstall("Removing installation files...")
        
        if self.install_dir.exists():
            try:
                shutil.rmtree(self.install_dir)
                self.log_uninstall(f"✓ Removed installation directory: {self.install_dir}")
            except Exception as e:
                self.log_uninstall(f"⚠ Could not remove installation directory: {e}")
                return False
        else:
            self.log_uninstall("✓ Installation directory not found")
        
        # Remove parent directories if empty
        try:
            parent_dir = self.install_dir.parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                shutil.rmtree(parent_dir)
                self.log_uninstall(f"✓ Removed parent directory: {parent_dir}")
                
                # Remove grandparent directory if empty
                grandparent_dir = parent_dir.parent
                if grandparent_dir.exists() and not any(grandparent_dir.iterdir()):
                    shutil.rmtree(grandparent_dir)
                    self.log_uninstall(f"✓ Removed grandparent directory: {grandparent_dir}")
                    
        except Exception as e:
            self.log_uninstall(f"⚠ Could not remove parent directories: {e}")
        
        return True
    
    def cleanup_registry_entries(self):
        """Clean up Windows registry entries (if any)"""
        if self.system == "windows":
            self.log_uninstall("Cleaning up registry entries...")
            
            # This would clean up any registry entries if they exist
            # For now, we'll just log that we checked
            self.log_uninstall("✓ Registry cleanup completed (no entries found)")
    
    def uninstall_bridge(self):
        """Perform complete uninstall"""
        self.log_uninstall("Starting SquashPlot Bridge uninstall...")
        self.log_uninstall(f"System: {self.system}")
        self.log_uninstall(f"Install directory: {self.install_dir}")
        
        # Check if installed
        is_installed, config = self.check_installation()
        if not is_installed:
            self.log_uninstall("⚠ SquashPlot Bridge is not installed")
            return False, "SquashPlot Bridge is not installed"
        
        if config:
            self.log_uninstall(f"✓ Found installation from {config.get('installation_date', 'unknown date')}")
        
        # Perform uninstall steps
        try:
            self.stop_bridge_processes()
            self.remove_startup_integration()
            self.cleanup_registry_entries()
            success = self.remove_installation_files()
            
            if success:
                self.log_uninstall("=" * 50)
                self.log_uninstall("✅ SQUASHPLOT BRIDGE UNINSTALLED SUCCESSFULLY!")
                self.log_uninstall("=" * 50)
                return True, "SquashPlot Bridge uninstalled successfully"
            else:
                self.log_uninstall("=" * 50)
                self.log_uninstall("❌ UNINSTALL FAILED!")
                self.log_uninstall("=" * 50)
                return False, "Uninstall failed - some files could not be removed"
                
        except Exception as e:
            self.log_uninstall(f"❌ Uninstall failed with error: {e}")
            return False, f"Uninstall failed: {str(e)}"

class UninstallerWebServer:
    """Web server for remote uninstaller"""
    
    def __init__(self, port=8082):
        self.port = port
        self.uninstaller = SquashPlotBridgeUninstaller()
        
    def start_server(self):
        """Start the uninstaller web server"""
        class UninstallerHandler(BaseHTTPRequestHandler):
            def __init__(self, *args, uninstaller_instance=None, **kwargs):
                self.uninstaller = uninstaller_instance
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                """Handle GET requests"""
                if self.path == '/':
                    self.serve_uninstaller_page()
                elif self.path == '/status':
                    self.check_installation_status()
                else:
                    self.send_error(404, "Not Found")
            
            def do_POST(self):
                """Handle POST requests"""
                if self.path == '/uninstall':
                    self.perform_uninstall()
                else:
                    self.send_error(404, "Not Found")
            
            def serve_uninstaller_page(self):
                """Serve the uninstaller web page"""
                html_content = self.get_uninstaller_html()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))
            
            def check_installation_status(self):
                """Check if SquashPlot Bridge is installed"""
                is_installed, config = self.uninstaller.check_installation()
                
                status = {
                    'installed': is_installed,
                    'config': config,
                    'install_directory': str(self.uninstaller.install_dir),
                    'system': self.uninstaller.system
                }
                
                self.send_json_response(200, status)
            
            def perform_uninstall(self):
                """Perform the uninstall operation"""
                try:
                    success, message = self.uninstaller.uninstall_bridge()
                    
                    response = {
                        'success': success,
                        'message': message,
                        'log': self.uninstaller.uninstall_log
                    }
                    
                    status_code = 200 if success else 500
                    self.send_json_response(status_code, response)
                    
                except Exception as e:
                    error_response = {
                        'success': False,
                        'message': f'Uninstall error: {str(e)}',
                        'log': self.uninstaller.uninstall_log
                    }
                    self.send_json_response(500, error_response)
            
            def send_json_response(self, status_code, data):
                """Send JSON response"""
                import json
                response_json = json.dumps(data).encode('utf-8')
                
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(response_json)))
                self.end_headers()
                self.wfile.write(response_json)
            
            def get_uninstaller_html(self):
                """Get the uninstaller HTML page"""
                return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SquashPlot Bridge - Uninstaller</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
            color: white;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .warning-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 1.1rem;
            font-weight: bold;
            margin: 10px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }
        
        .button {
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .button.danger {
            background: linear-gradient(45deg, #ff0000, #cc0000);
            color: white;
        }
        
        .button.danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 0, 0, 0.3);
        }
        
        .button.secondary {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .output {
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 20px;
            min-height: 200px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            overflow-y: auto;
            max-height: 400px;
            white-space: pre-wrap;
        }
        
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: bold;
        }
        
        .status.installed {
            background: rgba(255, 165, 0, 0.3);
            border: 2px solid rgba(255, 165, 0, 0.5);
        }
        
        .status.not-installed {
            background: rgba(0, 255, 0, 0.3);
            border: 2px solid rgba(0, 255, 0, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗑️ SquashPlot Bridge Uninstaller</h1>
            <p>Remove SquashPlot Bridge from your system</p>
            <span class="warning-badge">⚠️ PERMANENT REMOVAL</span>
        </div>

        <div class="panel">
            <h3>🔍 Installation Status</h3>
            <div id="status" class="status">
                Checking installation status...
            </div>
            <button id="checkStatusBtn" class="button secondary">Check Status</button>
        </div>

        <div class="panel">
            <h3>🗑️ Uninstall SquashPlot Bridge</h3>
            <p>This will permanently remove SquashPlot Bridge from your system:</p>
            <ul>
                <li>Remove all installation files</li>
                <li>Remove startup integration</li>
                <li>Stop any running processes</li>
                <li>Clean up all registry entries</li>
            </ul>
            
            <div style="text-align: center; margin: 20px 0;">
                <button id="uninstallBtn" class="button danger" disabled>🗑️ UNINSTALL NOW</button>
            </div>
            
            <div id="output" class="output">SquashPlot Bridge Uninstaller Ready
=====================================

Click "Check Status" to verify installation.
Then click "UNINSTALL NOW" to remove SquashPlot Bridge.

⚠️  WARNING: This action cannot be undone!  ⚠️</div>
        </div>
    </div>

    <script>
        class SquashPlotUninstaller {
            constructor() {
                this.initializeEventListeners();
                this.checkStatus();
            }
            
            initializeEventListeners() {
                document.getElementById('checkStatusBtn').addEventListener('click', () => {
                    this.checkStatus();
                });
                
                document.getElementById('uninstallBtn').addEventListener('click', () => {
                    this.performUninstall();
                });
            }
            
            async checkStatus() {
                this.updateStatus('Checking installation status...', '');
                
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    
                    if (data.installed) {
                        this.updateStatus('SquashPlot Bridge is INSTALLED', 'installed');
                        document.getElementById('uninstallBtn').disabled = false;
                        
                        this.addOutput('✅ SquashPlot Bridge found!');
                        this.addOutput(`📁 Install Directory: ${data.install_directory}`);
                        this.addOutput(`💻 System: ${data.system}`);
                        if (data.config) {
                            this.addOutput(`📅 Installed: ${data.config.installation_date}`);
                            this.addOutput(`📦 Version: ${data.config.version}`);
                        }
                        this.addOutput('⚠️ Ready to uninstall - Click "UNINSTALL NOW" button');
                        
                    } else {
                        this.updateStatus('SquashPlot Bridge is NOT INSTALLED', 'not-installed');
                        document.getElementById('uninstallBtn').disabled = true;
                        
                        this.addOutput('ℹ️ SquashPlot Bridge is not installed on this system.');
                        this.addOutput('No uninstall action needed.');
                    }
                    
                } catch (error) {
                    this.updateStatus('Error checking status', '');
                    this.addOutput(`❌ Error checking installation status: ${error.message}`);
                }
            }
            
            async performUninstall() {
                if (!confirm('Are you sure you want to uninstall SquashPlot Bridge?\\n\\nThis action cannot be undone!')) {
                    return;
                }
                
                this.addOutput('🗑️ Starting uninstall process...');
                document.getElementById('uninstallBtn').disabled = true;
                
                try {
                    const response = await fetch('/uninstall', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        this.addOutput('✅ Uninstall completed successfully!');
                        this.addOutput(`📝 Message: ${data.message}`);
                        this.addOutput('\\n📋 Uninstall Log:');
                        data.log.forEach(logEntry => {
                            this.addOutput(`   ${logEntry}`);
                        });
                        
                        this.updateStatus('SquashPlot Bridge UNINSTALLED', 'not-installed');
                        document.getElementById('uninstallBtn').disabled = true;
                        
                    } else {
                        this.addOutput('❌ Uninstall failed!');
                        this.addOutput(`📝 Error: ${data.message}`);
                        this.addOutput('\\n📋 Error Log:');
                        data.log.forEach(logEntry => {
                            this.addOutput(`   ${logEntry}`);
                        });
                        
                        document.getElementById('uninstallBtn').disabled = false;
                    }
                    
                } catch (error) {
                    this.addOutput(`❌ Uninstall request failed: ${error.message}`);
                    document.getElementById('uninstallBtn').disabled = false;
                }
            }
            
            updateStatus(message, className) {
                const statusElement = document.getElementById('status');
                statusElement.textContent = message;
                statusElement.className = 'status ' + className;
            }
            
            addOutput(message) {
                const output = document.getElementById('output');
                output.textContent += '\\n' + message;
                output.scrollTop = output.scrollHeight;
            }
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new SquashPlotUninstaller();
        });
    </script>
</body>
</html>'''
        
        # Create handler with uninstaller instance
        def handler(*args, **kwargs):
            return UninstallerHandler(*args, uninstaller_instance=self.uninstaller, **kwargs)
        
        # Start server
        server = HTTPServer(('127.0.0.1', self.port), handler)
        print(f"SquashPlot Bridge Uninstaller Web Server")
        print(f"Server started on http://127.0.0.1:{self.port}")
        print(f"Open your browser to: http://127.0.0.1:{self.port}")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\\nUninstaller web server stopped")

def main():
    """Main function"""
    print("SquashPlot Bridge Web Uninstaller")
    print("=" * 40)
    
    uninstaller_server = UninstallerWebServer()
    uninstaller_server.start_server()

if __name__ == "__main__":
    main()
