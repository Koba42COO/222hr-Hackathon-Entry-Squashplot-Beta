#!/usr/bin/env python3
"""
Simple Bridge Server - Minimal and Reliable
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

class SimpleBridgeHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler"""
    
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
            with open('bridge_connected_interface.html', 'r', encoding='utf-8') as f:
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
            'platform': 'Windows' if sys.platform.startswith('win') else 'Unix',
            'bridge_port': 8443,
            'web_port': 8080,
            'timestamp': time.time()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode('utf-8'))
    
    def execute_command(self):
        """Execute command"""
        try:
            # Get command
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
            
            print(f"Executing command: {command}")
            
            # Only allow hello-world
            if command != 'hello-world':
                response = {
                    'success': False,
                    'error': 'Only hello-world command allowed',
                    'received': command
                }
            else:
                # Execute hello world script
                result = self.run_hello_world()
                response = {
                    'success': True,
                    'result': result
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"Error: {e}")
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def run_hello_world(self):
        """Run hello world script"""
        try:
            if sys.platform.startswith('win'):
                # Windows: Create and run batch file
                script_content = '''@echo off
echo Hello World! This really works! > hello_world.txt
start notepad hello_world.txt
echo Script executed!'''
                
                # Write to temp file
                with open('hello_temp.bat', 'w') as f:
                    f.write(script_content)
                
                # Run it
                subprocess.Popen(['hello_temp.bat'], shell=True)
                
                # Clean up after a delay
                def cleanup():
                    time.sleep(2)
                    try:
                        os.remove('hello_temp.bat')
                    except:
                        pass
                
                threading.Thread(target=cleanup, daemon=True).start()
                
                return {
                    'message': 'Hello World script executed!',
                    'details': 'Notepad should have opened with Hello World message',
                    'platform': 'Windows'
                }
            else:
                # Unix-like systems
                subprocess.Popen(['echo', 'Hello World! This really works!'], shell=True)
                return {
                    'message': 'Hello World script executed!',
                    'platform': 'Unix'
                }
                
        except Exception as e:
            return {
                'message': 'Script execution failed',
                'error': str(e)
            }

def main():
    """Main function"""
    print("Simple Bridge Server")
    print("=" * 40)
    
    try:
        server = HTTPServer(('127.0.0.1', 8080), SimpleBridgeHandler)
        print("Server started on http://127.0.0.1:8080")
        
        # Open browser
        time.sleep(1)
        webbrowser.open('http://127.0.0.1:8080')
        
        print("Server running... Press Ctrl+C to stop")
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
