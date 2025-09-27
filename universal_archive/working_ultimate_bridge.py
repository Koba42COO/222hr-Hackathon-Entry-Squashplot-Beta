#!/usr/bin/env python3
"""
Working Ultimate Secure Bridge
Simplified version that works reliably on Windows
"""

import json
import logging
import os
import secrets
import socket
import ssl
import subprocess
import threading
import time
import hashlib
import hmac
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

class WorkingSecurityConfig:
    """Working security configuration"""
    
    # Simplified but still secure settings
    MAX_COMMAND_LENGTH = 500
    MAX_EXECUTION_TIME = 30
    MAX_REQUESTS_PER_MINUTE = 10
    SESSION_TIMEOUT = 900  # 15 minutes
    MAX_AUTH_ATTEMPTS = 3
    LOCKOUT_DURATION = 300  # 5 minutes
    
    # Allowed commands
    ALLOWED_COMMANDS = [
        'squashplot',
        'python',
        'pip',
        'ls',
        'pwd',
        'cat',
        'grep',
        'find',
        'du',
        'df',
        'ps',
        'top',
        'htop',
        'free',
        'uptime',
        'whoami',
        'date',
        'echo'
    ]
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'[;&|`$]',  # Command injection
        r'\.\./',    # Path traversal
        r'~',        # Home directory access
        r'rm\s+-rf', # Dangerous deletion
        r'sudo',     # Privilege escalation
        r'su\s+',   # User switching
        r'chmod\s+777',  # Dangerous permissions
        r'mkdir\s+-p\s+/',  # System directory creation
    ]

class WorkingCrypto:
    """Working cryptographic operations"""
    
    def __init__(self):
        self.master_key = self._generate_key()
        
    def _generate_key(self) -> bytes:
        """Generate a secure key"""
        return hashlib.sha256(secrets.token_bytes(32)).digest()
    
    def encrypt(self, data: str) -> str:
        """Encrypt data using AES-like encryption"""
        try:
            import cryptography
            from cryptography.fernet import Fernet
            
            # Create Fernet key from our master key
            key = base64.urlsafe_b64encode(self.master_key)
            fernet = Fernet(key)
            
            # Encrypt the data
            encrypted_data = fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
            
        except ImportError:
            # Fallback to simple XOR encryption if cryptography not available
            return self._simple_encrypt(data)
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            import cryptography
            from cryptography.fernet import Fernet
            
            # Create Fernet key from our master key
            key = base64.urlsafe_b64encode(self.master_key)
            fernet = Fernet(key)
            
            # Decrypt the data
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
            
        except ImportError:
            # Fallback to simple XOR decryption
            return self._simple_decrypt(encrypted_data)
    
    def _simple_encrypt(self, data: str) -> str:
        """Simple XOR encryption fallback"""
        data_bytes = data.encode()
        encrypted = bytearray()
        
        for i, byte in enumerate(data_bytes):
            key_byte = self.master_key[i % len(self.master_key)]
            encrypted.append(byte ^ key_byte)
        
        return base64.b64encode(encrypted).decode()
    
    def _simple_decrypt(self, encrypted_data: str) -> str:
        """Simple XOR decryption fallback"""
        encrypted = base64.b64decode(encrypted_data.encode())
        decrypted = bytearray()
        
        for i, byte in enumerate(encrypted):
            key_byte = self.master_key[i % len(self.master_key)]
            decrypted.append(byte ^ key_byte)
        
        return decrypted.decode()

class WorkingSessionManager:
    """Working session management"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = WorkingSecurityConfig.SESSION_TIMEOUT
        
    def create_session(self, client_ip: str) -> Dict:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        
        self.sessions[session_id] = {
            'created': time.time(),
            'last_activity': time.time(),
            'client_ip': client_ip,
            'command_count': 0,
            'is_active': True
        }
        
        return {
            'session_id': session_id,
            'created': self.sessions[session_id]['created'],
            'timeout': self.session_timeout
        }
    
    def validate_session(self, session_id: str, client_ip: str) -> Tuple[bool, str]:
        """Validate a session"""
        if not session_id or session_id not in self.sessions:
            return False, "Invalid session"
        
        session = self.sessions[session_id]
        
        # Check if session is active
        if not session['is_active']:
            return False, "Session inactive"
        
        # Check IP address
        if session['client_ip'] != client_ip:
            return False, "IP address mismatch"
        
        # Check session timeout
        current_time = time.time()
        if current_time - session['last_activity'] > self.session_timeout:
            self.sessions[session_id]['is_active'] = False
            return False, "Session expired"
        
        # Update last activity
        session['last_activity'] = current_time
        
        return True, "valid"
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get session information"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        current_time = time.time()
        time_remaining = self.session_timeout - (current_time - session['last_activity'])
        
        return {
            'session_id': session_id,
            'created': session['created'],
            'last_activity': session['last_activity'],
            'time_remaining': max(0, time_remaining),
            'command_count': session['command_count'],
            'is_active': session['is_active']
        }

class WorkingCommandSanitizer:
    """Working command sanitizer"""
    
    def sanitize_command(self, command: str) -> Tuple[bool, str, str]:
        """Sanitize and validate command"""
        import re
        
        # Check command length
        if len(command) > WorkingSecurityConfig.MAX_COMMAND_LENGTH:
            return False, "Command too long", ""
        
        # Check for dangerous patterns
        for pattern in WorkingSecurityConfig.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Dangerous pattern detected: {pattern}", ""
        
        # Check if command starts with allowed command
        command_parts = command.strip().split()
        if not command_parts:
            return False, "Empty command", ""
        
        base_command = command_parts[0].lower()
        if base_command not in WorkingSecurityConfig.ALLOWED_COMMANDS:
            return False, f"Command not allowed: {base_command}", ""
        
        return True, "Command is safe", command.strip()

class WorkingCommandExecutor:
    """Working command executor"""
    
    def execute_command(self, command: str) -> Dict:
        """Execute command safely"""
        try:
            # Execute command with timeout
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=WorkingSecurityConfig.MAX_EXECUTION_TIME
            )
            
            return {
                'success': True,
                'output': result.stdout,
                'error': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Command timed out after {WorkingSecurityConfig.MAX_EXECUTION_TIME} seconds",
                'output': '',
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Command execution failed: {str(e)}",
                'output': '',
                'return_code': -1
            }

class WorkingRateLimiter:
    """Working rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.max_requests = WorkingSecurityConfig.MAX_REQUESTS_PER_MINUTE
        
    def is_allowed(self, client_ip: str) -> Tuple[bool, str]:
        """Check if client is within rate limits"""
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60  # Keep only last minute
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False, "Rate limit exceeded"
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        return True, "Rate limit OK"

class WorkingSecureBridge:
    """Working secure bridge"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8443):
        self.host = host
        self.port = port
        self.running = False
        
        # Security components
        self.crypto = WorkingCrypto()
        self.session_manager = WorkingSessionManager()
        self.sanitizer = WorkingCommandSanitizer()
        self.executor = WorkingCommandExecutor()
        self.rate_limiter = WorkingRateLimiter()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_commands': 0,
            'blocked_commands': 0,
            'auth_failures': 0,
            'rate_limited': 0,
            'active_sessions': 0
        }
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("Working Secure Bridge initialized")
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('working_bridge_security.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('WorkingSecureBridge')
    
    def start_server(self):
        """Start the working secure bridge server"""
        try:
            self.running = True
            
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(5)
            
            self.logger.info(f"Working Secure Bridge started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = sock.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Error accepting connection: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    def _handle_client(self, client_socket, address):
        """Handle client connection"""
        client_ip = address[0]
        
        try:
            self.logger.info(f"Client connected from {address}")
            
            # Set timeout
            client_socket.settimeout(30)
            
            # Receive data
            data = client_socket.recv(4096).decode()
            if not data:
                return
            
            # Handle special commands
            if data.strip() == "PING":
                client_socket.send("PONG".encode())
                return
            
            if data.strip() == "STOP":
                client_socket.send("STOPPED".encode())
                self.logger.info("Stop request received")
                threading.Timer(1.0, self.stop_server).start()
                return
            
            if data.strip() == "CREATE_SESSION":
                session_data = self.session_manager.create_session(client_ip)
                client_socket.send(json.dumps({
                    'success': True,
                    'session': session_data
                }).encode())
                return
            
            # Parse JSON request
            try:
                request_data = json.loads(data)
            except json.JSONDecodeError:
                client_socket.send(json.dumps({
                    'error': 'Invalid request format'
                }).encode())
                return
            
            # Process request
            response = self._process_request(client_ip, request_data)
            client_socket.send(json.dumps(response).encode())
            
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def _process_request(self, client_ip: str, request_data: Dict) -> Dict:
        """Process client request"""
        try:
            request_type = request_data.get('type', 'command')
            
            if request_type == 'session_auth':
                session_id = request_data.get('session_id', '')
                command = request_data.get('command', '')
                
                # Validate session
                session_valid, session_message = self.session_manager.validate_session(session_id, client_ip)
                if not session_valid:
                    self.stats['auth_failures'] += 1
                    return {
                        'error': 'Session validation failed',
                        'message': session_message,
                        'requires_login': True
                    }
                
                # Execute command
                return self._execute_command(client_ip, session_id, command)
            
            else:
                return {
                    'error': 'Invalid request type',
                    'message': 'Request type must be session_auth'
                }
                
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return {
                'error': 'Request processing failed',
                'message': str(e)
            }
    
    def _execute_command(self, client_ip: str, session_id: str, command: str) -> Dict:
        """Execute command with security checks"""
        try:
            # Check rate limiting
            rate_ok, rate_message = self.rate_limiter.is_allowed(client_ip)
            if not rate_ok:
                self.stats['rate_limited'] += 1
                return {
                    'error': 'Rate limit exceeded',
                    'message': rate_message
                }
            
            # Sanitize command
            is_valid, validation_message, clean_command = self.sanitizer.sanitize_command(command)
            if not is_valid:
                self.stats['blocked_commands'] += 1
                return {
                    'error': 'Command blocked',
                    'message': validation_message
                }
            
            # Execute command
            result = self.executor.execute_command(clean_command)
            self.stats['total_requests'] += 1
            
            if result['success']:
                self.stats['successful_commands'] += 1
                # Update session
                if session_id in self.session_manager.sessions:
                    self.session_manager.sessions[session_id]['command_count'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                'error': 'Command execution failed',
                'message': str(e)
            }
    
    def stop_server(self):
        """Stop the bridge server"""
        self.running = False
        self.logger.info("Working Secure Bridge stopped")

def main():
    """Main entry point"""
    print("Working Secure SquashPlot Bridge")
    print("=" * 40)
    
    # Start the bridge
    try:
        bridge = WorkingSecureBridge()
        bridge.start_server()
    except KeyboardInterrupt:
        print("\nShutting down bridge...")
    except Exception as e:
        print(f"Error starting bridge: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
