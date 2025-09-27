#!/usr/bin/env python3
"""
SquashPlot Bridge - Professional Version
Cross-platform secure bridge with web interface and local execution
"""

import os
import sys
import json
import time
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser

class SquashPlotBridgeHandler(BaseHTTPRequestHandler):
    """HTTP handler for SquashPlot Bridge web interface"""
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_interface()
        elif self.path == '/status':
            self.get_status()
        elif self.path.startswith('/execute'):
            self.execute_command()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/execute':
            self.execute_command()
        else:
            self.send_error(404, "Not Found")
    
    def serve_interface(self):
        """Serve the web interface"""
        try:
            with open('SquashPlotBridge.html', 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except:
            self.send_error(404, "Interface not found")
    
    def get_status(self):
        """Get bridge status"""
        status = {
            'status': 'active',
            'platform': self.detect_platform(),
            'bridge_port': 8443,
            'web_port': 8080,
            'timestamp': time.time(),
            'version': '1.0.0'
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode('utf-8'))
    
    def execute_command(self):
        """Execute command with whitelist security"""
        try:
            # Get command from request
            if self.command == 'GET':
                parsed_path = urlparse(self.path)
                params = parse_qs(parsed_path.query)
                command = params.get('command', [''])[0]
            else:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    command = data.get('command', '')
                else:
                    command = ''
            
            # Security: Only allow whitelisted commands
            if command != 'hello-world':
                response = {
                    'success': False,
                    'error': 'Command not in whitelist. Only "hello-world" allowed.',
                    'received_command': command,
                    'whitelist': ['hello-world']
                }
            else:
                # Execute whitelisted command
                result = self.execute_hello_world_demo()
                response = {
                    'success': True,
                    'result': result,
                    'command': command,
                    'platform': self.detect_platform()
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e),
                'type': 'execution_error'
            }
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def execute_hello_world_demo(self):
        """Execute the hello-world demonstration script"""
        try:
            platform = self.detect_platform()
            
            if platform == 'Windows':
                # Windows: Create and execute batch file
                script_content = '''@echo off
echo Hello World! SquashPlot Bridge is working! > squashplot_demo.txt
start notepad squashplot_demo.txt
echo SquashPlot Bridge demo executed successfully!'''
                
                # Write to temporary file
                with open('squashplot_demo_temp.bat', 'w') as f:
                    f.write(script_content)
                
                # Execute non-blocking
                subprocess.Popen(['squashplot_demo_temp.bat'], shell=True)
                
                # Clean up after delay
                def cleanup():
                    time.sleep(3)
                    try:
                        os.remove('squashplot_demo_temp.bat')
                    except:
                        pass
                
                threading.Thread(target=cleanup, daemon=True).start()
                
                return {
                    'message': 'SquashPlot Bridge demo executed successfully!',
                    'details': 'Notepad opened with SquashPlot Bridge demonstration',
                    'platform': 'Windows',
                    'security': 'whitelist_protected'
                }
                
            elif platform == 'macOS':
                # macOS: Execute via bash
                subprocess.Popen([
                    'bash', '-c', 
                    'echo "Hello World! SquashPlot Bridge is working!" > squashplot_demo.txt && open -e squashplot_demo.txt'
                ])
                
                return {
                    'message': 'SquashPlot Bridge demo executed successfully!',
                    'details': 'TextEdit opened with SquashPlot Bridge demonstration',
                    'platform': 'macOS',
                    'security': 'whitelist_protected'
                }
                
            elif platform == 'Linux':
                # Linux: Execute via bash
                subprocess.Popen([
                    'bash', '-c',
                    'echo "Hello World! SquashPlot Bridge is working!" > squashplot_demo.txt && gedit squashplot_demo.txt &'
                ])
                
                return {
                    'message': 'SquashPlot Bridge demo executed successfully!',
                    'details': 'Text editor opened with SquashPlot Bridge demonstration',
                    'platform': 'Linux',
                    'security': 'whitelist_protected'
                }
                
            else:
                return {
                    'message': 'SquashPlot Bridge demo ready for execution',
                    'details': f'Platform-specific execution for {platform}',
                    'platform': platform,
                    'security': 'whitelist_protected'
                }
                
        except Exception as e:
            return {
                'message': 'Demo execution failed',
                'error': str(e),
                'platform': self.detect_platform()
            }
    
    def detect_platform(self):
        """Detect the current platform"""
        if sys.platform.startswith('win'):
            return 'Windows'
        elif sys.platform.startswith('darwin'):
            return 'macOS'
        elif sys.platform.startswith('linux'):
            return 'Linux'
        else:
            return 'Unknown'

class SquashPlotBridge:
    """Main SquashPlot Bridge class"""
    
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start(self):
        """Start the SquashPlot Bridge server"""
        try:
            self.server = HTTPServer((self.host, self.port), SquashPlotBridgeHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print("=" * 60)
            print("SquashPlot Bridge - Professional Version")
            print("=" * 60)
            print(f"Server started on http://{self.host}:{self.port}")
            print(f"Web Interface: http://{self.host}:{self.port}")
            print(f"API Endpoint: http://{self.host}:{self.port}/execute?command=hello-world")
            print(f"Status Endpoint: http://{self.host}:{self.port}/status")
            print("Security: Whitelist Protected")
            print("=" * 60)
            
            # Auto-open browser
            time.sleep(2)
            webbrowser.open(f'http://{self.host}:{self.port}')
            
            return True
            
        except Exception as e:
            print(f"Failed to start SquashPlot Bridge: {e}")
            return False
    
    def stop(self):
        """Stop the SquashPlot Bridge server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("SquashPlot Bridge stopped")
    
    def is_running(self):
        """Check if server is running"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except:
            return False

def main():
    """Main function"""
    bridge = SquashPlotBridge()
    
    if bridge.start():
        try:
            print("SquashPlot Bridge is running... Press Ctrl+C to stop")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down SquashPlot Bridge...")
            bridge.stop()
    else:
        print("Failed to start SquashPlot Bridge")

if __name__ == "__main__":
    main()
