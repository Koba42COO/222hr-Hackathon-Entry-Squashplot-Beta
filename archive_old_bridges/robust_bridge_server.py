#!/usr/bin/env python3
"""
Robust Bridge Server - Better Error Handling and Logging
Handles HTTP requests from web interface and executes commands via bridge
"""

import os
import sys
import json
import time
import socket
import threading
import subprocess
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser

class RobustBridgeHandler(BaseHTTPRequestHandler):
    """Robust HTTP handler for bridge web interface"""
    
    def log_message(self, format, *args):
        """Override to add timestamp and better logging"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/':
                self.serve_interface()
            elif parsed_path.path == '/status':
                self.get_status()
            elif parsed_path.path == '/execute':
                self.execute_command()
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            print(f"GET Error: {e}")
            self.send_error(500, f"Internal Server Error: {e}")
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/execute':
                self.execute_command()
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            print(f"POST Error: {e}")
            self.send_error(500, f"Internal Server Error: {e}")
    
    def serve_interface(self):
        """Serve the web interface"""
        try:
            # Serve the bridge connected interface
            with open('bridge_connected_interface.html', 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
            print("Served web interface successfully")
            
        except FileNotFoundError:
            print("Interface file not found")
            self.send_error(404, "Interface file not found")
        except Exception as e:
            print(f"Error serving interface: {e}")
            self.send_error(500, f"Error serving interface: {e}")
    
    def get_status(self):
        """Get bridge status"""
        try:
            status = {
                'status': 'active',
                'platform': self.detect_platform(),
                'bridge_port': 8443,
                'web_port': 8080,
                'timestamp': time.time(),
                'uptime': time.time() - getattr(self.server, 'start_time', time.time())
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode('utf-8'))
            print("Status request handled successfully")
            
        except Exception as e:
            print(f"Status error: {e}")
            self.send_error(500, f"Status error: {e}")
    
    def execute_command(self):
        """Execute command via bridge"""
        try:
            # Get command from URL parameters or POST data
            command = None
            
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
            
            print(f"Received command: {command}")
            
            # Validate command (only allow hello-world script)
            if command != 'hello-world':
                response = {
                    'success': False,
                    'error': 'Command not allowed. Only "hello-world" is permitted.',
                    'allowed_commands': ['hello-world'],
                    'received_command': command
                }
            else:
                # Execute the hello-world script
                result = self.execute_hello_world()
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
            print(f"Command execution response sent: {response['success']}")
            
        except Exception as e:
            print(f"Execute command error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            error_response = {
                'success': False,
                'error': f'Execution failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def execute_hello_world(self):
        """Execute the hello-world script with better error handling"""
        platform = self.detect_platform()
        print(f"Executing hello-world script on {platform}")
        
        try:
            if platform == 'Windows':
                # Create and execute Windows batch script
                script_content = '''@echo off
echo Hello World! This really works! > hello_world.txt
notepad hello_world.txt
echo Script executed successfully - Notepad opened with Hello World message!
pause'''
                
                # Write script to temporary file
                script_path = 'temp_hello_world.bat'
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                print(f"Created batch script: {script_path}")
                
                # Execute the script
                result = subprocess.run([script_path], capture_output=True, text=True, shell=True, timeout=30)
                
                print(f"Script execution result: {result.returncode}")
                print(f"Script stdout: {result.stdout}")
                print(f"Script stderr: {result.stderr}")
                
                # Clean up
                try:
                    os.remove(script_path)
                    print("Cleaned up temporary script file")
                except:
                    print("Warning: Could not clean up temporary script file")
                
                return {
                    'message': 'Hello World script executed successfully!',
                    'details': 'Notepad should have opened with Hello World message',
                    'output': result.stdout,
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'platform': 'Windows'
                }
                
            elif platform == 'macOS':
                # Execute macOS script
                script_content = '''#!/bin/bash
echo "Hello World! This really works!" > hello_world.txt
open -e hello_world.txt
echo "Script executed successfully - TextEdit should have opened!"'''
                
                result = subprocess.run(['bash', '-c', script_content], capture_output=True, text=True, timeout=30)
                
                return {
                    'message': 'Hello World script executed successfully!',
                    'details': 'TextEdit should have opened with Hello World message',
                    'output': result.stdout,
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'platform': 'macOS'
                }
                
            elif platform == 'Linux':
                # Execute Linux script
                script_content = '''#!/bin/bash
echo "Hello World! This really works!" > hello_world.txt
gedit hello_world.txt &
echo "Script executed successfully - Text editor should have opened!"'''
                
                result = subprocess.run(['bash', '-c', script_content], capture_output=True, text=True, timeout=30)
                
                return {
                    'message': 'Hello World script executed successfully!',
                    'details': 'Text editor should have opened with Hello World message',
                    'output': result.stdout,
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'platform': 'Linux'
                }
                
            else:
                return {
                    'message': 'Hello World script would execute on this platform',
                    'details': f'Platform-specific execution for {platform}',
                    'platform': platform
                }
                
        except subprocess.TimeoutExpired:
            print("Script execution timed out")
            return {
                'message': 'Script execution timed out',
                'error': 'Script took too long to execute',
                'platform': platform
            }
        except Exception as e:
            print(f"Script execution error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'message': 'Script execution failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'platform': platform
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

class RobustBridgeServer:
    """Robust Bridge Web Server with better error handling"""
    
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.start_time = time.time()
        
    def start(self):
        """Start the web server with better error handling"""
        try:
            self.server = HTTPServer((self.host, self.port), RobustBridgeHandler)
            self.server.start_time = self.start_time
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print(f"Robust Bridge Server started on http://{self.host}:{self.port}")
            print(f"Web Interface: http://{self.host}:{self.port}")
            print(f"API Endpoint: http://{self.host}:{self.port}/execute?command=hello-world")
            print(f"Status Endpoint: http://{self.host}:{self.port}/status")
            print("Server is running with improved error handling...")
            
            # Auto-open browser
            time.sleep(2)
            try:
                webbrowser.open(f'http://{self.host}:{self.port}')
                print("Browser opened automatically")
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
            
            return True
            
        except Exception as e:
            print(f"Failed to start web server: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def stop(self):
        """Stop the web server"""
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
                print("Robust Bridge Server stopped")
            except Exception as e:
                print(f"Error stopping server: {e}")
    
    def is_running(self):
        """Check if server is running"""
        try:
            # Try to connect to the server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except:
            return False

def main():
    """Main function with better error handling"""
    print("Robust Bridge Server - Better Error Handling")
    print("=" * 60)
    
    # Start web server
    web_server = RobustBridgeServer()
    
    if web_server.start():
        try:
            print("\nServer is running... Press Ctrl+C to stop")
            print("Monitoring server health...")
            
            while True:
                time.sleep(5)
                
                # Check if server is still running
                if not web_server.is_running():
                    print("Server appears to have stopped unexpectedly!")
                    break
                    
                print(f"Server health check: OK (uptime: {time.time() - web_server.start_time:.1f}s)")
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            web_server.stop()
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            web_server.stop()
    else:
        print("Failed to start web server")

if __name__ == "__main__":
    main()
