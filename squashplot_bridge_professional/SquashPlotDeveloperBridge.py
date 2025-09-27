#!/usr/bin/env python3
"""
SquashPlot Developer Bridge - Locked Down Single Command Version
ONLY FOR DEVELOPMENT AND TESTING - NOT FOR PRODUCTION USE
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
import logging

class DeveloperBridgeConfig:
    """Developer bridge configuration - Locked down"""
    
    # Single allowed command
    ALLOWED_COMMAND = 'hello-world'
    
    # Security warnings
    SECURITY_WARNING = """
    ⚠️  WARNING: DEVELOPER BRIDGE - NOT FOR PRODUCTION ⚠️
    
    This is a LOCKED-DOWN developer version with:
    - ONLY ONE ALLOWED COMMAND: hello-world
    - NO AUTHENTICATION (for development ease)
    - NO ENCRYPTION (for debugging)
    - LIMITED FUNCTIONALITY (for testing only)
    
    DO NOT USE IN PRODUCTION OR DISTRIBUTE TO END USERS!
    
    This version is intended ONLY for:
    - Development testing
    - Proof of concept demonstration
    - Educational purposes
    - Internal evaluation
    
    For production use, use SquashPlotSecureBridge.py instead.
    """
    
    # Rate limiting (very permissive for development)
    MAX_REQUESTS_PER_MINUTE = 100
    MAX_FAILED_ATTEMPTS = 10

class DeveloperCommandValidator:
    """Locked down command validator - Only allows hello-world"""
    
    def __init__(self):
        self.allowed_commands = {
            'hello-world': {
                'description': 'Hello World demonstration script (ONLY ALLOWED COMMAND)',
                'execution_timeout': 30,
                'max_attempts': 10
            }
        }
        
        # Block ALL other commands
        self.blocked_patterns = [
            r'.*',  # Block everything by default
        ]
    
    def validate_command(self, command: str) -> tuple[bool, str, str]:
        """Validate command - ONLY allows hello-world"""
        
        # Log all command attempts for security monitoring
        logging.info(f"Command validation attempt: '{command}' from IP")
        
        # ONLY allow hello-world command
        if command == self.ALLOWED_COMMAND:
            return True, "Command approved", "Hello world command is the only allowed command"
        
        # Block everything else
        return False, "Command blocked", f"Only '{self.ALLOWED_COMMAND}' command is allowed in developer mode"

class DeveloperRateLimiter:
    """Simple rate limiter for developer version"""
    
    def __init__(self):
        self.request_counts = {}
        self.max_requests = DeveloperBridgeConfig.MAX_REQUESTS_PER_MINUTE
        self.window_size = 60  # 1 minute
    
    def check_rate_limit(self, client_id: str) -> tuple[bool, str]:
        """Check rate limit - Very permissive for development"""
        current_time = time.time()
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        requests = self.request_counts[client_id]
        
        # Remove old requests
        requests[:] = [req_time for req_time in requests if current_time - req_time < self.window_size]
        
        # Check limit
        if len(requests) >= self.max_requests:
            return False, f"Rate limit exceeded (max {self.max_requests} requests per minute)"
        
        # Add current request
        requests.append(current_time)
        return True, "Rate limit OK"

class SquashPlotDeveloperBridge:
    """Locked Down Developer Bridge - Single Command Only"""
    
    def __init__(self, host='127.0.0.1', port=8081):  # Different port from secure version
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        
        # Components
        self.command_validator = DeveloperCommandValidator()
        self.rate_limiter = DeveloperRateLimiter()
        
        # Setup logging
        self.setup_logging()
        
        # Display security warning
        self.display_security_warning()
    
    def setup_logging(self):
        """Setup logging for security monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - DEVELOPER-BRIDGE - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('squashplot_developer_bridge.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SquashPlotDeveloperBridge')
    
    def display_security_warning(self):
        """Display security warning"""
        print("\n" + "="*80)
        print(DeveloperBridgeConfig.SECURITY_WARNING)
        print("="*80)
        print()
        
        # Log the warning
        self.logger.warning("DEVELOPER BRIDGE STARTED - NOT FOR PRODUCTION USE")
    
    def start(self):
        """Start the locked down developer bridge"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            
            class DeveloperBridgeHandler(BaseHTTPRequestHandler):
                def __init__(self, *args, bridge_instance=None, **kwargs):
                    self.bridge = bridge_instance
                    super().__init__(*args, **kwargs)
                
                def log_message(self, format, *args):
                    """Override to reduce log noise"""
                    pass
                
                def do_GET(self):
                    self.handle_request()
                
                def do_POST(self):
                    self.handle_request()
                
                def handle_request(self):
                    """Handle request with locked down security"""
                    try:
                        # Extract client info
                        client_info = {
                            'ip': self.client_address[0],
                            'port': self.client_address[1],
                            'user_agent': self.headers.get('User-Agent', ''),
                            'timestamp': time.time()
                        }
                        
                        # Rate limiting
                        client_id = f"{client_info['ip']}:{client_info['port']}"
                        rate_ok, rate_msg = self.bridge.rate_limiter.check_rate_limit(client_id)
                        if not rate_ok:
                            self.send_error_response(429, rate_msg)
                            return
                        
                        # Log all requests for security monitoring
                        self.bridge.logger.info(f"Request from {client_info['ip']}: {self.path}")
                        
                        # Parse request
                        if self.path == '/':
                            self.serve_interface()
                        elif self.path == '/status':
                            self.get_status()
                        elif self.path.startswith('/execute'):
                            self.execute_command()
                        elif self.path == '/security-warning':
                            self.show_security_warning()
                        else:
                            self.send_error_response(404, "Endpoint not found")
                            
                    except Exception as e:
                        self.bridge.logger.error(f"Request handling error: {e}")
                        self.send_error_response(500, "Internal server error")
                
                def serve_interface(self):
                    """Serve the locked down developer interface"""
                    try:
                        with open('SquashPlotDeveloperBridge.html', 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(content.encode('utf-8'))
                        
                        self.bridge.logger.info("Developer interface served")
                        
                    except Exception as e:
                        self.send_error_response(404, f"Interface error: {e}")
                
                def get_status(self):
                    """Get developer bridge status"""
                    status = {
                        'status': 'developer_active',
                        'security_level': 'locked_down',
                        'platform': self.bridge.detect_platform(),
                        'bridge_port': 8443,
                        'web_port': self.port,
                        'timestamp': time.time(),
                        'version': '1.0.0-developer',
                        'warning': 'DEVELOPER VERSION - NOT FOR PRODUCTION',
                        'allowed_commands': ['hello-world'],
                        'features': [
                            'Single command only (hello-world)',
                            'No authentication (development ease)',
                            'No encryption (debugging)',
                            'Rate limiting',
                            'Security logging'
                        ]
                    }
                    
                    self.send_json_response(200, status)
                
                def show_security_warning(self):
                    """Show security warning"""
                    warning = {
                        'warning': DeveloperBridgeConfig.SECURITY_WARNING,
                        'allowed_commands': ['hello-world'],
                        'restrictions': [
                            'Only hello-world command allowed',
                            'No authentication required',
                            'No encryption',
                            'Development use only'
                        ]
                    }
                    
                    self.send_json_response(200, warning)
                
                def execute_command(self):
                    """Execute command with locked down validation"""
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
                        
                        # Log command attempt
                        self.bridge.logger.info(f"Command execution attempt: '{command}' from {self.client_address[0]}")
                        
                        # Validate command (ONLY allows hello-world)
                        valid, valid_msg, valid_reason = self.bridge.command_validator.validate_command(command)
                        if not valid:
                            self.send_error_response(403, f"Command blocked: {valid_reason}")
                            self.bridge.logger.warning(f"Command blocked: '{command}' from {self.client_address[0]}")
                            return
                        
                        # Execute ONLY the hello-world command
                        if command == 'hello-world':
                            result = self.execute_hello_world_demo()
                        else:
                            # This should never happen due to validation, but just in case
                            self.send_error_response(403, "Only hello-world command allowed")
                            return
                        
                        response = {
                            'success': True,
                            'result': result,
                            'command': command,
                            'platform': self.bridge.detect_platform(),
                            'security_level': 'locked_down_developer',
                            'warning': 'This is a developer version - not for production use'
                        }
                        
                        self.send_json_response(200, response)
                        self.bridge.logger.info(f"Command '{command}' executed successfully for {self.client_address[0]}")
                        
                    except Exception as e:
                        self.send_error_response(500, f"Execution error: {e}")
                        self.bridge.logger.error(f"Command execution error: {e}")
                
                def execute_hello_world_demo(self):
                    """Execute hello world demo - ONLY allowed command"""
                    try:
                        platform = self.bridge.detect_platform()
                        
                        if platform == 'Windows':
                            script_content = '''@echo off
echo SquashPlot Developer Bridge - Hello World Demo! > developer_demo.txt
start notepad developer_demo.txt
echo Developer bridge demo executed successfully!'''
                            
                            with open('developer_demo_temp.bat', 'w') as f:
                                f.write(script_content)
                            
                            subprocess.Popen(['developer_demo_temp.bat'], shell=True)
                            
                            def cleanup():
                                time.sleep(3)
                                try:
                                    os.remove('developer_demo_temp.bat')
                                except:
                                    pass
                            
                            threading.Thread(target=cleanup, daemon=True).start()
                            
                            return {
                                'message': 'SquashPlot Developer Bridge demo executed successfully!',
                                'details': 'Notepad opened with developer bridge demonstration',
                                'platform': 'Windows',
                                'security': 'locked_down_developer',
                                'warning': 'This is a developer version - not for production use'
                            }
                        elif platform == 'macOS':
                            subprocess.Popen([
                                'bash', '-c', 
                                'echo "SquashPlot Developer Bridge - Hello World Demo!" > developer_demo.txt && open -e developer_demo.txt'
                            ])
                            
                            return {
                                'message': 'SquashPlot Developer Bridge demo executed successfully!',
                                'details': 'TextEdit opened with developer bridge demonstration',
                                'platform': 'macOS',
                                'security': 'locked_down_developer',
                                'warning': 'This is a developer version - not for production use'
                            }
                        elif platform == 'Linux':
                            subprocess.Popen([
                                'bash', '-c',
                                'echo "SquashPlot Developer Bridge - Hello World Demo!" > developer_demo.txt && gedit developer_demo.txt &'
                            ])
                            
                            return {
                                'message': 'SquashPlot Developer Bridge demo executed successfully!',
                                'details': 'Text editor opened with developer bridge demonstration',
                                'platform': 'Linux',
                                'security': 'locked_down_developer',
                                'warning': 'This is a developer version - not for production use'
                            }
                        else:
                            return {
                                'message': 'SquashPlot Developer Bridge demo ready for execution',
                                'details': f'Platform-specific execution for {platform}',
                                'platform': platform,
                                'security': 'locked_down_developer',
                                'warning': 'This is a developer version - not for production use'
                            }
                            
                    except Exception as e:
                        return {
                            'message': 'Demo execution failed',
                            'error': str(e),
                            'platform': self.bridge.detect_platform(),
                            'warning': 'This is a developer version - not for production use'
                        }
                
                def send_json_response(self, status_code: int, data: dict):
                    """Send JSON response"""
                    response_json = json.dumps(data).encode('utf-8')
                    
                    self.send_response(status_code)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Length', str(len(response_json)))
                    self.end_headers()
                    self.wfile.write(response_json)
                
                def send_error_response(self, status_code: int, message: str):
                    """Send error response"""
                    error_data = {
                        'success': False,
                        'error': message,
                        'status_code': status_code,
                        'timestamp': time.time(),
                        'warning': 'This is a developer version - not for production use'
                    }
                    self.send_json_response(status_code, error_data)
            
            # Create server with custom handler
            def handler(*args, **kwargs):
                return DeveloperBridgeHandler(*args, bridge_instance=self, **kwargs)
            
            self.server = HTTPServer((self.host, self.port), handler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print("SquashPlot Developer Bridge - Locked Down Version")
            print("=" * 60)
            print(f"Server started on http://{self.host}:{self.port}")
            print(f"Web Interface: http://{self.host}:{self.port}")
            print(f"Security Level: LOCKED DOWN (Developer Only)")
            print(f"Allowed Commands: ONLY hello-world")
            print(f"Warning: NOT FOR PRODUCTION USE")
            print("=" * 60)
            
            # Auto-open browser
            time.sleep(2)
            webbrowser.open(f'http://{self.host}:{self.port}')
            
            return True
            
        except Exception as e:
            print(f"Failed to start SquashPlot Developer Bridge: {e}")
            return False
    
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
    
    def stop(self):
        """Stop the developer bridge server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("SquashPlot Developer Bridge stopped")

def main():
    """Main function"""
    bridge = SquashPlotDeveloperBridge()
    
    if bridge.start():
        try:
            print("SquashPlot Developer Bridge is running... Press Ctrl+C to stop")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down SquashPlot Developer Bridge...")
            bridge.stop()
    else:
        print("Failed to start SquashPlot Developer Bridge")

if __name__ == "__main__":
    main()
