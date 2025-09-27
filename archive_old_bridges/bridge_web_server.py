#!/usr/bin/env python3
"""
Bridge Web Server - Connects Web Interface to Bridge
Handles HTTP requests from web interface and executes commands via bridge
"""

import os
import sys
import json
import time
import socket
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser

class BridgeWebHandler(BaseHTTPRequestHandler):
    """HTTP handler for bridge web interface"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_interface()
        elif parsed_path.path == '/status':
            self.get_status()
        elif parsed_path.path == '/execute':
            self.execute_command()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/execute':
            self.execute_command()
        else:
            self.send_error(404, "Not Found")
    
    def serve_interface(self):
        """Serve the web interface"""
        try:
            # Serve the bridge connected interface
            with open('bridge_connected_interface.html', 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
            
        except FileNotFoundError:
            self.send_error(404, "Interface file not found")
    
    def get_status(self):
        """Get bridge status"""
        status = {
            'status': 'active',
            'platform': self.detect_platform(),
            'bridge_port': 8443,
            'web_port': 8080,
            'timestamp': time.time()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode('utf-8'))
    
    def execute_command(self):
        """Execute command via bridge"""
        # Get command from URL parameters or POST data
        if self.command == 'GET':
            parsed_path = urlparse(self.path)
            params = parse_qs(parsed_path.query)
            command = params.get('command', [''])[0]
        else:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            command = data.get('command', '')
        
        # Validate command (only allow hello-world script)
        if command != 'hello-world':
            response = {
                'success': False,
                'error': 'Command not allowed. Only "hello-world" is permitted.',
                'allowed_commands': ['hello-world']
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
    
    def execute_hello_world(self):
        """Execute the hello-world script"""
        platform = self.detect_platform()
        
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
                
                # Execute the script
                result = subprocess.run([script_path], capture_output=True, text=True, shell=True)
                
                # Clean up
                try:
                    os.remove(script_path)
                except:
                    pass
                
                return {
                    'message': 'Hello World script executed successfully!',
                    'details': 'Notepad should have opened with Hello World message',
                    'output': result.stdout,
                    'platform': 'Windows'
                }
                
            elif platform == 'macOS':
                # Execute macOS script
                script_content = '''#!/bin/bash
echo "Hello World! This really works!" > hello_world.txt
open -e hello_world.txt
echo "Script executed successfully - TextEdit should have opened!"'''
                
                result = subprocess.run(['bash', '-c', script_content], capture_output=True, text=True)
                
                return {
                    'message': 'Hello World script executed successfully!',
                    'details': 'TextEdit should have opened with Hello World message',
                    'output': result.stdout,
                    'platform': 'macOS'
                }
                
            elif platform == 'Linux':
                # Execute Linux script
                script_content = '''#!/bin/bash
echo "Hello World! This really works!" > hello_world.txt
gedit hello_world.txt &
echo "Script executed successfully - Text editor should have opened!"'''
                
                result = subprocess.run(['bash', '-c', script_content], capture_output=True, text=True)
                
                return {
                    'message': 'Hello World script executed successfully!',
                    'details': 'Text editor should have opened with Hello World message',
                    'output': result.stdout,
                    'platform': 'Linux'
                }
                
            else:
                return {
                    'message': 'Hello World script would execute on this platform',
                    'details': f'Platform-specific execution for {platform}',
                    'platform': platform
                }
                
        except Exception as e:
            return {
                'message': 'Script execution failed',
                'error': str(e),
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
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass

class BridgeWebServer:
    """Bridge Web Server"""
    
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start(self):
        """Start the web server"""
        try:
            self.server = HTTPServer((self.host, self.port), BridgeWebHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print(f"Bridge Web Server started on http://{self.host}:{self.port}")
            print(f"Web Interface: http://{self.host}:{self.port}")
            print(f"API Endpoint: http://{self.host}:{self.port}/execute?command=hello-world")
            
            # Auto-open browser
            time.sleep(1)
            webbrowser.open(f'http://{self.host}:{self.port}')
            
            return True
            
        except Exception as e:
            print(f"Failed to start web server: {e}")
            return False
    
    def stop(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("Bridge Web Server stopped")

def main():
    """Main function"""
    print("Bridge Web Server - Connects Web Interface to Bridge")
    print("=" * 60)
    
    # Start web server
    web_server = BridgeWebServer()
    
    if web_server.start():
        try:
            print("\nWeb server running... Press Ctrl+C to stop")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            web_server.stop()
    else:
        print("Failed to start web server")

if __name__ == "__main__":
    main()
