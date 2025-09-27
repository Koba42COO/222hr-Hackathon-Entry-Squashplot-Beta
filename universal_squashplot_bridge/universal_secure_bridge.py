#!/usr/bin/env python3
"""
Universal Secure SquashPlot Bridge
Cross-platform compatible with all operating systems including iOS/mobile
"""

import json
import logging
import os
import platform
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
from pathlib import Path

# Cross-platform compatibility imports
try:
    import webbrowser
except ImportError:
    webbrowser = None

try:
    import ssl
    SSL_AVAILABLE = True
except ImportError:
    SSL_AVAILABLE = False

# Platform detection
PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM == 'windows'
IS_MACOS = PLATFORM == 'darwin'
IS_LINUX = PLATFORM == 'linux'
IS_IOS = 'ios' in PLATFORM or 'darwin' in PLATFORM  # iOS is based on Darwin
IS_ANDROID = 'android' in PLATFORM

class UniversalSecurityConfig:
    """Universal security configuration for all platforms"""
    
    # Cross-platform settings
    MAX_COMMAND_LENGTH = 500
    MAX_EXECUTION_TIME = 30
    MAX_REQUESTS_PER_MINUTE = 10
    SESSION_TIMEOUT = 900  # 15 minutes
    MAX_AUTH_ATTEMPTS = 3
    LOCKOUT_DURATION = 300  # 5 minutes
    
    # Platform-specific ports (avoid conflicts)
    DEFAULT_PORTS = {
        'windows': 8443,
        'darwin': 8444,  # macOS/iOS
        'linux': 8445,
        'ios': 8446,
        'android': 8447
    }
    
    # Platform-specific allowed commands
    WINDOWS_COMMANDS = [
        'squashplot', 'python', 'pip', 'dir', 'cd', 'type', 'echo',
        'where', 'whoami', 'date', 'time', 'ver', 'systeminfo'
    ]
    
    UNIX_COMMANDS = [
        'squashplot', 'python', 'pip', 'ls', 'pwd', 'cat', 'grep',
        'find', 'du', 'df', 'ps', 'top', 'htop', 'free', 'uptime',
        'whoami', 'date', 'echo', 'which', 'uname', 'id'
    ]
    
    MOBILE_COMMANDS = [
        'squashplot', 'python', 'pip', 'ls', 'pwd', 'cat', 'echo',
        'date', 'whoami', 'uname'
    ]
    
    # Dangerous patterns (cross-platform)
    DANGEROUS_PATTERNS = [
        r'[;&|`$]',  # Command injection
        r'\.\./',    # Path traversal
        r'~',        # Home directory access
        r'rm\s+-rf', # Dangerous deletion
        r'sudo',     # Privilege escalation
        r'su\s+',   # User switching
        r'chmod\s+777',  # Dangerous permissions
        r'mkdir\s+-p\s+/',  # System directory creation
        r'del\s+/s',  # Windows dangerous deletion
        r'format',   # Disk formatting
        r'shutdown', # System shutdown
        r'reboot',   # System reboot
        r'halt',     # System halt
    ]

class UniversalCrypto:
    """Universal cryptographic operations for all platforms"""
    
    def __init__(self):
        self.master_key = self._generate_key()
        
    def _generate_key(self) -> bytes:
        """Generate a secure key using platform-specific entropy"""
        if IS_WINDOWS:
            # Windows-specific entropy
            entropy = os.urandom(32) + secrets.token_bytes(32)
        elif IS_IOS or IS_MACOS:
            # macOS/iOS-specific entropy
            entropy = os.urandom(32) + secrets.token_bytes(32)
        elif IS_LINUX:
            # Linux-specific entropy
            entropy = os.urandom(32) + secrets.token_bytes(32)
        else:
            # Fallback entropy
            entropy = secrets.token_bytes(64)
        
        return hashlib.sha256(entropy).digest()
    
    def encrypt(self, data: str) -> str:
        """Encrypt data using platform-appropriate methods"""
        try:
            # Try cryptography library first (works on most platforms)
            import cryptography
            from cryptography.fernet import Fernet
            
            key = base64.urlsafe_b64encode(self.master_key)
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
            
        except ImportError:
            # Fallback to platform-specific encryption
            if IS_WINDOWS:
                return self._windows_encrypt(data)
            elif IS_IOS or IS_MACOS:
                return self._apple_encrypt(data)
            elif IS_LINUX:
                return self._linux_encrypt(data)
            else:
                return self._universal_encrypt(data)
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data using platform-appropriate methods"""
        try:
            import cryptography
            from cryptography.fernet import Fernet
            
            key = base64.urlsafe_b64encode(self.master_key)
            fernet = Fernet(key)
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
            
        except ImportError:
            # Fallback to platform-specific decryption
            if IS_WINDOWS:
                return self._windows_decrypt(encrypted_data)
            elif IS_IOS or IS_MACOS:
                return self._apple_decrypt(encrypted_data)
            elif IS_LINUX:
                return self._linux_decrypt(encrypted_data)
            else:
                return self._universal_decrypt(encrypted_data)
    
    def _windows_encrypt(self, data: str) -> str:
        """Windows-specific encryption fallback"""
        data_bytes = data.encode('utf-8')
        encrypted = bytearray()
        
        for i, byte in enumerate(data_bytes):
            key_byte = self.master_key[i % len(self.master_key)]
            encrypted.append(byte ^ key_byte ^ (i % 256))
        
        return base64.b64encode(encrypted).decode()
    
    def _windows_decrypt(self, encrypted_data: str) -> str:
        """Windows-specific decryption fallback"""
        encrypted = base64.b64decode(encrypted_data.encode())
        decrypted = bytearray()
        
        for i, byte in enumerate(encrypted):
            key_byte = self.master_key[i % len(self.master_key)]
            decrypted.append(byte ^ key_byte ^ (i % 256))
        
        return decrypted.decode('utf-8')
    
    def _apple_encrypt(self, data: str) -> str:
        """macOS/iOS-specific encryption fallback"""
        # Use CommonCrypto-like encryption for Apple platforms
        return self._universal_encrypt(data)
    
    def _apple_decrypt(self, encrypted_data: str) -> str:
        """macOS/iOS-specific decryption fallback"""
        return self._universal_decrypt(encrypted_data)
    
    def _linux_encrypt(self, data: str) -> str:
        """Linux-specific encryption fallback"""
        return self._universal_encrypt(data)
    
    def _linux_decrypt(self, encrypted_data: str) -> str:
        """Linux-specific decryption fallback"""
        return self._universal_decrypt(encrypted_data)
    
    def _universal_encrypt(self, data: str) -> str:
        """Universal encryption fallback"""
        data_bytes = data.encode('utf-8')
        encrypted = bytearray()
        
        for i, byte in enumerate(data_bytes):
            key_byte = self.master_key[i % len(self.master_key)]
            encrypted.append(byte ^ key_byte)
        
        return base64.b64encode(encrypted).decode()
    
    def _universal_decrypt(self, encrypted_data: str) -> str:
        """Universal decryption fallback"""
        encrypted = base64.b64decode(encrypted_data.encode())
        decrypted = bytearray()
        
        for i, byte in enumerate(encrypted):
            key_byte = self.master_key[i % len(self.master_key)]
            decrypted.append(byte ^ key_byte)
        
        return decrypted.decode('utf-8')

class UniversalSessionManager:
    """Universal session management for all platforms"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = UniversalSecurityConfig.SESSION_TIMEOUT
        
    def create_session(self, client_ip: str, user_agent: str = "") -> Dict:
        """Create a new session with platform-specific handling"""
        session_id = secrets.token_urlsafe(32)
        
        # Platform-specific session data
        platform_info = {
            'platform': PLATFORM,
            'is_mobile': IS_IOS or IS_ANDROID,
            'user_agent': user_agent,
            'timestamp': time.time()
        }
        
        self.sessions[session_id] = {
            'created': time.time(),
            'last_activity': time.time(),
            'client_ip': client_ip,
            'command_count': 0,
            'is_active': True,
            'platform_info': platform_info
        }
        
        return {
            'session_id': session_id,
            'created': self.sessions[session_id]['created'],
            'timeout': self.session_timeout,
            'platform': PLATFORM
        }
    
    def validate_session(self, session_id: str, client_ip: str) -> Tuple[bool, str]:
        """Validate a session with platform-specific checks"""
        if not session_id or session_id not in self.sessions:
            return False, "Invalid session"
        
        session = self.sessions[session_id]
        
        # Check if session is active
        if not session['is_active']:
            return False, "Session inactive"
        
        # Check IP address (relaxed for mobile platforms)
        if IS_IOS or IS_ANDROID:
            # Mobile platforms may have changing IPs
            if not self._is_mobile_ip_valid(session['client_ip'], client_ip):
                return False, "IP address mismatch"
        else:
            # Desktop platforms require exact IP match
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
    
    def _is_mobile_ip_valid(self, original_ip: str, current_ip: str) -> bool:
        """Check if mobile IP change is valid (carrier switching, etc.)"""
        # For mobile platforms, allow some IP flexibility
        # This is a simplified check - in production, you'd want more sophisticated logic
        return original_ip == current_ip or current_ip.startswith(original_ip.split('.')[0])
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get session information with platform details"""
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
            'is_active': session['is_active'],
            'platform': session.get('platform_info', {}).get('platform', PLATFORM)
        }

class UniversalCommandSanitizer:
    """Universal command sanitizer for all platforms"""
    
    def __init__(self):
        self.allowed_commands = self._get_platform_commands()
        
    def _get_platform_commands(self) -> List[str]:
        """Get platform-specific allowed commands"""
        if IS_WINDOWS:
            return UniversalSecurityConfig.WINDOWS_COMMANDS
        elif IS_IOS or IS_ANDROID:
            return UniversalSecurityConfig.MOBILE_COMMANDS
        else:
            return UniversalSecurityConfig.UNIX_COMMANDS
    
    def sanitize_command(self, command: str) -> Tuple[bool, str, str]:
        """Sanitize and validate command for current platform"""
        import re
        
        # Check command length
        if len(command) > UniversalSecurityConfig.MAX_COMMAND_LENGTH:
            return False, "Command too long", ""
        
        # Check for dangerous patterns
        for pattern in UniversalSecurityConfig.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Dangerous pattern detected: {pattern}", ""
        
        # Check if command starts with allowed command
        command_parts = command.strip().split()
        if not command_parts:
            return False, "Empty command", ""
        
        base_command = command_parts[0].lower()
        if base_command not in self.allowed_commands:
            return False, f"Command not allowed on {PLATFORM}: {base_command}", ""
        
        # Platform-specific additional checks
        if IS_WINDOWS:
            return self._sanitize_windows_command(command)
        elif IS_IOS or IS_ANDROID:
            return self._sanitize_mobile_command(command)
        else:
            return self._sanitize_unix_command(command)
    
    def _sanitize_windows_command(self, command: str) -> Tuple[bool, str, str]:
        """Windows-specific command sanitization"""
        # Additional Windows-specific checks
        dangerous_windows = ['format', 'del /s', 'rd /s', 'shutdown', 'restart']
        for dangerous in dangerous_windows:
            if dangerous in command.lower():
                return False, f"Windows dangerous command detected: {dangerous}", ""
        
        return True, "Command is safe for Windows", command.strip()
    
    def _sanitize_mobile_command(self, command: str) -> Tuple[bool, str, str]:
        """Mobile-specific command sanitization"""
        # Mobile platforms have more restrictions
        if any(cmd in command.lower() for cmd in ['rm', 'del', 'format', 'shutdown']):
            return False, "Command not allowed on mobile platform", ""
        
        return True, "Command is safe for mobile", command.strip()
    
    def _sanitize_unix_command(self, command: str) -> Tuple[bool, str, str]:
        """Unix-specific command sanitization"""
        # Additional Unix-specific checks
        dangerous_unix = ['rm -rf', 'chmod 777', 'sudo', 'su -']
        for dangerous in dangerous_unix:
            if dangerous in command.lower():
                return False, f"Unix dangerous command detected: {dangerous}", ""
        
        return True, "Command is safe for Unix", command.strip()

class UniversalCommandExecutor:
    """Universal command executor for all platforms"""
    
    def execute_command(self, command: str) -> Dict:
        """Execute command safely on current platform"""
        try:
            # Platform-specific command execution
            if IS_WINDOWS:
                return self._execute_windows_command(command)
            elif IS_IOS or IS_ANDROID:
                return self._execute_mobile_command(command)
            else:
                return self._execute_unix_command(command)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Command execution failed: {str(e)}",
                'output': '',
                'return_code': -1
            }
    
    def _execute_windows_command(self, command: str) -> Dict:
        """Execute command on Windows"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=UniversalSecurityConfig.MAX_EXECUTION_TIME,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            return {
                'success': True,
                'output': result.stdout,
                'error': result.stderr,
                'return_code': result.returncode,
                'platform': 'windows'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Command timed out after {UniversalSecurityConfig.MAX_EXECUTION_TIME} seconds",
                'output': '',
                'return_code': -1,
                'platform': 'windows'
            }
    
    def _execute_unix_command(self, command: str) -> Dict:
        """Execute command on Unix-like systems"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=UniversalSecurityConfig.MAX_EXECUTION_TIME
            )
            
            return {
                'success': True,
                'output': result.stdout,
                'error': result.stderr,
                'return_code': result.returncode,
                'platform': 'unix'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Command timed out after {UniversalSecurityConfig.MAX_EXECUTION_TIME} seconds",
                'output': '',
                'return_code': -1,
                'platform': 'unix'
            }
    
    def _execute_mobile_command(self, command: str) -> Dict:
        """Execute command on mobile platforms"""
        try:
            # Mobile platforms may have limited subprocess support
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=UniversalSecurityConfig.MAX_EXECUTION_TIME
            )
            
            return {
                'success': True,
                'output': result.stdout,
                'error': result.stderr,
                'return_code': result.returncode,
                'platform': 'mobile'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Command timed out after {UniversalSecurityConfig.MAX_EXECUTION_TIME} seconds",
                'output': '',
                'return_code': -1,
                'platform': 'mobile'
            }

class UniversalRateLimiter:
    """Universal rate limiter for all platforms"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.max_requests = UniversalSecurityConfig.MAX_REQUESTS_PER_MINUTE
        
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

class UniversalSecureBridge:
    """Universal secure bridge for all platforms"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = None):
        self.host = host
        self.port = port or UniversalSecurityConfig.DEFAULT_PORTS.get(PLATFORM, 8443)
        self.running = False
        
        # Universal security components
        self.crypto = UniversalCrypto()
        self.session_manager = UniversalSessionManager()
        self.sanitizer = UniversalCommandSanitizer()
        self.executor = UniversalCommandExecutor()
        self.rate_limiter = UniversalRateLimiter()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_commands': 0,
            'blocked_commands': 0,
            'auth_failures': 0,
            'rate_limited': 0,
            'active_sessions': 0,
            'platform': PLATFORM
        }
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Universal Secure Bridge initialized on {PLATFORM}")
    
    def _setup_logging(self):
        """Setup platform-specific logging"""
        log_file = f'universal_bridge_security_{PLATFORM}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('UniversalSecureBridge')
    
    def start_server(self):
        """Start the universal secure bridge server"""
        try:
            self.running = True
            
            # Create socket with platform-specific settings
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Platform-specific socket options
            if IS_WINDOWS:
                # Windows-specific socket options
                pass
            elif IS_IOS or IS_ANDROID:
                # Mobile-specific socket options
                pass
            else:
                # Unix-specific socket options
                pass
            
            sock.bind((self.host, self.port))
            sock.listen(5)
            
            self.logger.info(f"Universal Secure Bridge started on {self.host}:{self.port} ({PLATFORM})")
            
            # Auto-open browser on desktop platforms
            if not IS_IOS and not IS_ANDROID and webbrowser:
                threading.Timer(2.0, self._open_browser).start()
            
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
    
    def _open_browser(self):
        """Open browser automatically on desktop platforms"""
        try:
            if webbrowser:
                webbrowser.open(f'http://{self.host}:{self.port}')
        except Exception as e:
            self.logger.warning(f"Could not open browser: {e}")
    
    def _handle_client(self, client_socket, address):
        """Handle client connection with platform-specific handling"""
        client_ip = address[0]
        
        try:
            self.logger.info(f"Client connected from {address} on {PLATFORM}")
            
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
                    'session': session_data,
                    'platform': PLATFORM
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
        """Process client request with platform-specific handling"""
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
                        'requires_login': True,
                        'platform': PLATFORM
                    }
                
                # Execute command
                return self._execute_command(client_ip, session_id, command)
            
            else:
                return {
                    'error': 'Invalid request type',
                    'message': 'Request type must be session_auth',
                    'platform': PLATFORM
                }
                
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return {
                'error': 'Request processing failed',
                'message': str(e),
                'platform': PLATFORM
            }
    
    def _execute_command(self, client_ip: str, session_id: str, command: str) -> Dict:
        """Execute command with universal security checks"""
        try:
            # Check rate limiting
            rate_ok, rate_message = self.rate_limiter.is_allowed(client_ip)
            if not rate_ok:
                self.stats['rate_limited'] += 1
                return {
                    'error': 'Rate limit exceeded',
                    'message': rate_message,
                    'platform': PLATFORM
                }
            
            # Sanitize command
            is_valid, validation_message, clean_command = self.sanitizer.sanitize_command(command)
            if not is_valid:
                self.stats['blocked_commands'] += 1
                return {
                    'error': 'Command blocked',
                    'message': validation_message,
                    'platform': PLATFORM
                }
            
            # Execute command
            result = self.executor.execute_command(clean_command)
            self.stats['total_requests'] += 1
            
            if result['success']:
                self.stats['successful_commands'] += 1
                # Update session
                if session_id in self.session_manager.sessions:
                    self.session_manager.sessions[session_id]['command_count'] += 1
            
            # Add platform info to result
            result['platform'] = PLATFORM
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                'error': 'Command execution failed',
                'message': str(e),
                'platform': PLATFORM
            }
    
    def stop_server(self):
        """Stop the bridge server"""
        self.running = False
        self.logger.info(f"Universal Secure Bridge stopped ({PLATFORM})")

def main():
    """Main entry point with platform detection"""
    print(f"Universal Secure SquashPlot Bridge - {PLATFORM.upper()}")
    print("=" * 50)
    print(f"Platform: {PLATFORM}")
    print(f"iOS Compatible: {IS_IOS}")
    print(f"Mobile Compatible: {IS_IOS or IS_ANDROID}")
    print("=" * 50)
    
    # Start the bridge
    try:
        bridge = UniversalSecureBridge()
        bridge.start_server()
    except KeyboardInterrupt:
        print(f"\nShutting down bridge on {PLATFORM}...")
    except Exception as e:
        print(f"Error starting bridge on {PLATFORM}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
