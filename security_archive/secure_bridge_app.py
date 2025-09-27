#!/usr/bin/env python3
"""
SquashPlot Secure Bridge App - Security Hardened Version
Implements comprehensive security measures for safe command execution
"""

import json
import logging
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import resource
import signal
import sys
from pathlib import Path

class SecurityConfig:
    """Security configuration and constants"""
    MAX_COMMAND_LENGTH = 500
    MAX_EXECUTION_TIME = 30  # seconds
    MAX_MEMORY_MB = 100
    MAX_REQUESTS_PER_MINUTE = 10
    SESSION_TIMEOUT = 900  # 15 minutes default (configurable)
    MAX_AUTH_ATTEMPTS = 3
    LOCKOUT_DURATION = 300  # 5 minutes
    SESSION_WARNING_TIME = 300  # 5 minutes before timeout
    CSRF_TOKEN_LENGTH = 32
    MAX_SESSIONS_PER_IP = 3
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
    ALLOWED_COMMANDS = [
        'squashplot',
        'chia',
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
        'nvidia-smi',
        'free',
        'uptime',
        'whoami',
        'date',
        'echo'
    ]

class SessionManager:
    """Advanced session management with timeout and security features"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = SecurityConfig.SESSION_TIMEOUT
        self.warning_time = SecurityConfig.SESSION_WARNING_TIME
        self.max_sessions_per_ip = SecurityConfig.MAX_SESSIONS_PER_IP
        self.ip_sessions: Dict[str, List[str]] = {}
        
    def create_session(self, client_ip: str, user_agent: str = "") -> str:
        """Create a new secure session"""
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(SecurityConfig.CSRF_TOKEN_LENGTH)
        
        # Check session limits per IP
        if client_ip not in self.ip_sessions:
            self.ip_sessions[client_ip] = []
        
        # Remove old sessions for this IP if at limit
        if len(self.ip_sessions[client_ip]) >= self.max_sessions_per_ip:
            oldest_session = self.ip_sessions[client_ip][0]
            if oldest_session in self.sessions:
                del self.sessions[oldest_session]
            self.ip_sessions[client_ip].remove(oldest_session)
        
        # Create new session
        self.sessions[session_id] = {
            'created': time.time(),
            'last_activity': time.time(),
            'client_ip': client_ip,
            'user_agent': user_agent,
            'csrf_token': csrf_token,
            'command_count': 0,
            'is_active': True,
            'warning_sent': False
        }
        
        # Track session for this IP
        self.ip_sessions[client_ip].append(session_id)
        
        return session_id
    
    def validate_session(self, session_id: str, client_ip: str, csrf_token: str = None) -> Tuple[bool, str]:
        """Validate session with comprehensive security checks"""
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
            self._expire_session(session_id)
            return False, "Session expired"
        
        # Check CSRF token if provided
        if csrf_token and session['csrf_token'] != csrf_token:
            return False, "Invalid CSRF token"
        
        # Update last activity
        session['last_activity'] = current_time
        
        # Check if warning should be sent
        time_remaining = self.session_timeout - (current_time - session['last_activity'])
        if time_remaining <= self.warning_time and not session['warning_sent']:
            session['warning_sent'] = True
            return True, "warning_soon"
        
        return True, "valid"
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        if session_id in self.sessions:
            session = self.sessions[session_id].copy()
            # Calculate time remaining
            current_time = time.time()
            time_remaining = self.session_timeout - (current_time - session['last_activity'])
            session['time_remaining'] = max(0, time_remaining)
            return session
        return None
    
    def extend_session(self, session_id: str) -> bool:
        """Extend session timeout"""
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = time.time()
            self.sessions[session_id]['warning_sent'] = False
            return True
        return False
    
    def _expire_session(self, session_id: str):
        """Expire a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            client_ip = session['client_ip']
            
            # Remove from IP tracking
            if client_ip in self.ip_sessions and session_id in self.ip_sessions[client_ip]:
                self.ip_sessions[client_ip].remove(session_id)
            
            # Mark as inactive
            self.sessions[session_id]['is_active'] = False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._expire_session(session_id)
    
    def get_active_sessions_count(self, client_ip: str = None) -> int:
        """Get count of active sessions"""
        if client_ip:
            return len([s for s in self.ip_sessions.get(client_ip, []) 
                       if s in self.sessions and self.sessions[s]['is_active']])
        return len([s for s in self.sessions.values() if s['is_active']])

class SecureAuthentication:
    """Secure authentication system with token validation"""
    
    def __init__(self):
        self.auth_tokens: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.master_key = self._generate_master_key()
        self.session_manager = SessionManager()
        
    def _generate_master_key(self) -> str:
        """Generate a secure master key for token validation"""
        return secrets.token_urlsafe(32)
    
    def generate_auth_token(self, client_ip: str) -> str:
        """Generate a secure authentication token (legacy method)"""
        token = secrets.token_urlsafe(32)
        self.auth_tokens[token] = {
            'created': time.time(),
            'client_ip': client_ip,
            'attempts': 0,
            'locked': False,
            'last_activity': time.time()
        }
        return token
    
    def create_session(self, client_ip: str, user_agent: str = "") -> Dict:
        """Create a new secure session"""
        session_id = self.session_manager.create_session(client_ip, user_agent)
        session_info = self.session_manager.get_session_info(session_id)
        return {
            'session_id': session_id,
            'csrf_token': session_info['csrf_token'],
            'timeout': SecurityConfig.SESSION_TIMEOUT,
            'warning_time': SecurityConfig.SESSION_WARNING_TIME
        }
    
    def validate_auth(self, token: str, client_ip: str) -> Tuple[bool, str]:
        """Validate authentication token with security checks (legacy method)"""
        if not token or token not in self.auth_tokens:
            return False, "Invalid token"
        
        auth_data = self.auth_tokens[token]
        
        # Check IP address
        if auth_data['client_ip'] != client_ip:
            return False, "IP address mismatch"
        
        # Check if locked
        if auth_data['locked']:
            if time.time() - auth_data.get('lockout_time', 0) > SecurityConfig.LOCKOUT_DURATION:
                auth_data['locked'] = False
                auth_data['attempts'] = 0
            else:
                return False, "Account locked due to failed attempts"
        
        # Check session timeout
        if time.time() - auth_data['created'] > SecurityConfig.SESSION_TIMEOUT:
            del self.auth_tokens[token]
            return False, "Session expired"
        
        # Update last activity
        auth_data['last_activity'] = time.time()
        return True, "Valid token"
    
    def validate_session(self, session_id: str, client_ip: str, csrf_token: str = None) -> Tuple[bool, str]:
        """Validate session with comprehensive security checks"""
        return self.session_manager.validate_session(session_id, client_ip, csrf_token)
    
    def extend_session(self, session_id: str) -> bool:
        """Extend session timeout"""
        return self.session_manager.extend_session(session_id)
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.session_manager.get_session_info(session_id)
    
    def record_failed_attempt(self, client_ip: str):
        """Record failed authentication attempt"""
        now = time.time()
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        
        self.failed_attempts[client_ip].append(now)
        
        # Clean old attempts
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip]
            if now - attempt < SecurityConfig.LOCKOUT_DURATION
        ]
        
        # Check if should be locked
        if len(self.failed_attempts[client_ip]) >= SecurityConfig.MAX_AUTH_ATTEMPTS:
            # Lock all tokens for this IP
            for token, data in self.auth_tokens.items():
                if data['client_ip'] == client_ip:
                    data['locked'] = True
                    data['lockout_time'] = now

class CommandSanitizer:
    """Advanced command sanitization and validation"""
    
    def __init__(self):
        self.dangerous_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in SecurityConfig.DANGEROUS_PATTERNS]
        self.allowed_commands = set(SecurityConfig.ALLOWED_COMMANDS)
    
    def sanitize_command(self, command: str) -> Tuple[bool, str, str]:
        """Sanitize and validate command with comprehensive checks"""
        if not command or not isinstance(command, str):
            return False, "Invalid command", ""
        
        # Check length
        if len(command) > SecurityConfig.MAX_COMMAND_LENGTH:
            return False, "Command too long", ""
        
        # Remove leading/trailing whitespace
        command = command.strip()
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(command):
                return False, f"Dangerous pattern detected: {pattern.pattern}", ""
        
        # Extract base command
        base_command = command.split()[0] if command.split() else ""
        
        # Check if command is in whitelist
        if base_command not in self.allowed_commands:
            return False, f"Command not allowed: {base_command}", ""
        
        # Additional security checks
        if not self._is_safe_command(command):
            return False, "Command failed safety checks", ""
        
        return True, "Command validated", command
    
    def _is_safe_command(self, command: str) -> bool:
        """Additional safety checks for commands"""
        # Check for path traversal attempts
        if '..' in command or '~' in command:
            return False
        
        # Check for system directory access
        dangerous_paths = ['/etc/', '/sys/', '/proc/', '/dev/', '/root/']
        for path in dangerous_paths:
            if path in command:
                return False
        
        # Check for dangerous file operations
        dangerous_ops = ['rm -rf', 'chmod 777', 'chown', 'dd if=', 'mkfs']
        for op in dangerous_ops:
            if op in command.lower():
                return False
        
        return True

class RateLimiter:
    """Rate limiting to prevent DoS attacks"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.max_requests = SecurityConfig.MAX_REQUESTS_PER_MINUTE
        self.time_window = 60  # 1 minute
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, str]:
        """Check if client is within rate limits"""
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.time_window
        ]
        
        if len(self.requests[client_ip]) >= self.max_requests:
            return False, "Rate limit exceeded"
        
        self.requests[client_ip].append(now)
        return True, "Rate limit OK"

class SecureCommandExecutor:
    """Secure command execution with process isolation"""
    
    def __init__(self, working_dir: str = "/tmp/squashplot_bridge"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        self.max_execution_time = SecurityConfig.MAX_EXECUTION_TIME
        self.max_memory = SecurityConfig.MAX_MEMORY_MB * 1024 * 1024
    
    def execute_command(self, command: str) -> Dict:
        """Execute command with security restrictions"""
        try:
            # Set up isolated environment
            env = os.environ.copy()
            env['PATH'] = '/usr/bin:/bin'  # Restricted PATH
            env['HOME'] = str(self.working_dir)
            env['USER'] = os.getenv('USER', 'squashplot')
            
            # Set resource limits
            def set_limits():
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, 
                                     (self.max_execution_time, self.max_execution_time))
                    resource.setrlimit(resource.RLIMIT_AS, 
                                     (self.max_memory, self.max_memory))
                    resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))  # 50MB file size limit
                except Exception as e:
                    logging.warning(f"Could not set resource limits: {e}")
            
            # Execute command with restrictions
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=str(self.working_dir),
                env=env,
                preexec_fn=set_limits,
                user=os.getuid(),
                group=os.getgid()
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'return_code': result.returncode,
                'execution_time': time.time()
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Command execution timeout',
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Execution error: {str(e)}',
                'return_code': -1
            }

class SecurityAuditLogger:
    """Comprehensive security audit logging"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.max_log_size = 10 * 1024 * 1024  # 10MB
        self.log_rotation = 5
    
    def log_security_event(self, event_type: str, details: Dict):
        """Log security events with comprehensive details"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'client_ip': details.get('client_ip', 'unknown'),
            'user_agent': details.get('user_agent', 'unknown'),
            'command': details.get('command', ''),
            'success': details.get('success', False),
            'error': details.get('error', ''),
            'risk_level': self._assess_risk(event_type, details),
            'session_id': details.get('session_id', ''),
            'auth_token': details.get('auth_token', '')[:8] + '...' if details.get('auth_token') else None
        }
        
        # Write to secure log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")
        
        # Rotate logs if needed
        self._rotate_logs()
    
    def _assess_risk(self, event_type: str, details: Dict) -> str:
        """Assess risk level of security event"""
        if event_type == 'authentication_failure':
            return 'HIGH'
        elif event_type == 'command_execution' and not details.get('success', False):
            return 'MEDIUM'
        elif event_type == 'rate_limit_exceeded':
            return 'MEDIUM'
        elif event_type == 'dangerous_command_blocked':
            return 'HIGH'
        else:
            return 'LOW'
    
    def _rotate_logs(self):
        """Rotate log files when they get too large"""
        try:
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > self.max_log_size:
                # Rotate existing logs
                for i in range(self.log_rotation - 1, 0, -1):
                    old_file = f"{self.log_file}.{i}"
                    new_file = f"{self.log_file}.{i + 1}"
                    if os.path.exists(old_file):
                        os.rename(old_file, new_file)
                
                # Move current log
                os.rename(self.log_file, f"{self.log_file}.1")
        except Exception as e:
            logging.error(f"Log rotation failed: {e}")

class SecureBridgeApp:
    """Security-hardened SquashPlot Bridge App"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8443):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
        # Security components
        self.auth = SecureAuthentication()
        self.sanitizer = CommandSanitizer()
        self.rate_limiter = RateLimiter()
        self.executor = SecureCommandExecutor()
        self.audit_logger = SecurityAuditLogger()
        
        # Setup logging
        self._setup_logging()
        
        # Security statistics
        self.stats = {
            'total_requests': 0,
            'successful_commands': 0,
            'blocked_commands': 0,
            'auth_failures': 0,
            'rate_limited': 0,
            'active_sessions': 0
        }
        
        # Start session cleanup thread
        self._start_session_cleanup()
    
    def _setup_logging(self):
        """Setup secure logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bridge_security.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SecureBridge')
    
    def _start_session_cleanup(self):
        """Start background thread for session cleanup"""
        def cleanup_worker():
            while self.running:
                try:
                    self.auth.session_manager.cleanup_expired_sessions()
                    self.stats['active_sessions'] = self.auth.session_manager.get_active_sessions_count()
                    time.sleep(60)  # Cleanup every minute
                except Exception as e:
                    self.logger.error(f"Session cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def start_server(self):
        """Start the secure bridge server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            
            self.logger.info(f"Secure Bridge App started on {self.host}:{self.port}")
            self.audit_logger.log_security_event('server_start', {
                'host': self.host,
                'port': self.port
            })
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
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
        """Handle client connection with security measures"""
        client_ip = address[0]
        session_id = secrets.token_urlsafe(16)
        
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
                self.audit_logger.log_security_event('ping_request', {
                    'client_ip': client_ip,
                    'session_id': session_id
                })
                return
            
            if data.strip() == "STOP":
                client_socket.send("STOPPED".encode())
                self.logger.info("Stop request received")
                self.audit_logger.log_security_event('stop_request', {
                    'client_ip': client_ip,
                    'session_id': session_id
                })
                threading.Timer(1.0, self.stop_server).start()
                return
            
            # Handle session creation request
            if data.strip() == "CREATE_SESSION":
                user_agent = "Bridge Client"  # Could be passed in request
                session_data = self.auth.create_session(client_ip, user_agent)
                client_socket.send(json.dumps({
                    'success': True,
                    'session': session_data
                }).encode())
                self.audit_logger.log_security_event('session_created', {
                    'client_ip': client_ip,
                    'session_id': session_data['session_id']
                })
                return
            
            # Parse JSON request
            try:
                request_data = json.loads(data)
                request_type = request_data.get('type', 'command')
                
                # Handle session-based authentication
                if request_type == 'session_auth':
                    session_id = request_data.get('session_id', '')
                    csrf_token = request_data.get('csrf_token', '')
                    
                    # Validate session
                    session_valid, session_message = self.auth.validate_session(session_id, client_ip, csrf_token)
                    if not session_valid:
                        self.auth.record_failed_attempt(client_ip)
                        self.stats['auth_failures'] += 1
                        self.audit_logger.log_security_event('session_validation_failure', {
                            'client_ip': client_ip,
                            'session_id': session_id,
                            'error': session_message
                        })
                        client_socket.send(json.dumps({
                            'error': 'Session validation failed',
                            'message': session_message,
                            'requires_login': True
                        }).encode())
                        return
                    
                    # Check for session warning
                    if session_message == "warning_soon":
                        client_socket.send(json.dumps({
                            'success': True,
                            'warning': 'Session will expire soon',
                            'time_remaining': self.auth.get_session_info(session_id)['time_remaining']
                        }).encode())
                        return
                    
                    # Session is valid, proceed with command execution
                    command = request_data.get('command', '')
                    self._execute_secure_command(client_socket, client_ip, session_id, command)
                    return
                
                # Legacy token-based authentication
                elif request_type == 'token_auth':
                    auth_token = request_data.get('auth_token', '')
                    command = request_data.get('command', '')
                    
                    # Validate authentication
                    auth_valid, auth_message = self.auth.validate_auth(auth_token, client_ip)
                    if not auth_valid:
                        self.auth.record_failed_attempt(client_ip)
                        self.stats['auth_failures'] += 1
                        self.audit_logger.log_security_event('authentication_failure', {
                            'client_ip': client_ip,
                            'session_id': session_id,
                            'error': auth_message
                        })
                        client_socket.send(json.dumps({
                            'error': 'Authentication failed',
                            'message': auth_message
                        }).encode())
                        return
                    
                    # Execute command with legacy auth
                    self._execute_secure_command(client_socket, client_ip, session_id, command)
                    return
                
                else:
                    client_socket.send(json.dumps({
                        'error': 'Invalid request type',
                        'message': 'Request type must be session_auth or token_auth'
                    }).encode())
                    return
                
            except json.JSONDecodeError:
                self.audit_logger.log_security_event('invalid_request', {
                    'client_ip': client_ip,
                    'session_id': session_id,
                    'data': data[:100]  # First 100 chars only
                })
                client_socket.send(json.dumps({
                    'error': 'Invalid request format'
                }).encode())
                
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
            self.audit_logger.log_security_event('client_error', {
                'client_ip': client_ip,
                'session_id': session_id,
                'error': str(e)
            })
        finally:
            client_socket.close()
    
    def _execute_secure_command(self, client_socket, client_ip: str, session_id: str, command: str):
        """Execute command with comprehensive security checks"""
        try:
            # Check rate limiting
            rate_ok, rate_message = self.rate_limiter.is_allowed(client_ip)
            if not rate_ok:
                self.stats['rate_limited'] += 1
                self.audit_logger.log_security_event('rate_limit_exceeded', {
                    'client_ip': client_ip,
                    'session_id': session_id
                })
                client_socket.send(json.dumps({
                    'error': 'Rate limit exceeded',
                    'message': rate_message
                }).encode())
                return
            
            # Sanitize command
            is_valid, validation_message, clean_command = self.sanitizer.sanitize_command(command)
            if not is_valid:
                self.stats['blocked_commands'] += 1
                self.audit_logger.log_security_event('dangerous_command_blocked', {
                    'client_ip': client_ip,
                    'session_id': session_id,
                    'command': command,
                    'reason': validation_message
                })
                client_socket.send(json.dumps({
                    'error': 'Command blocked',
                    'message': validation_message
                }).encode())
                return
            
            # Execute command
            result = self.executor.execute_command(clean_command)
            self.stats['total_requests'] += 1
            if result['success']:
                self.stats['successful_commands'] += 1
            
            # Log execution
            self.audit_logger.log_security_event('command_execution', {
                'client_ip': client_ip,
                'session_id': session_id,
                'command': clean_command,
                'success': result['success'],
                'error': result.get('error', '')
            })
            
            # Send response
            client_socket.send(json.dumps(result).encode())
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            client_socket.send(json.dumps({
                'error': 'Command execution failed',
                'message': str(e)
            }).encode())
    
    def stop_server(self):
        """Stop the bridge server"""
        self.running = False
        if self.socket:
            self.socket.close()
        self.logger.info("Bridge server stopped")
        self.audit_logger.log_security_event('server_stop', {
            'stats': self.stats
        })

def main():
    """Main entry point for the secure bridge app"""
    print("🛡️ SquashPlot Secure Bridge App - Security Hardened")
    print("=" * 60)
    
    # Security warning
    print("⚠️  SECURITY WARNING:")
    print("This software executes commands on your local machine.")
    print("Ensure you understand the security implications before proceeding.")
    print()
    
    # Get user confirmation
    response = input("Do you understand the security risks and want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Exiting for security reasons.")
        sys.exit(1)
    
    # Start the secure bridge
    try:
        bridge = SecureBridgeApp()
        bridge.start_server()
    except KeyboardInterrupt:
        print("\nShutting down secure bridge...")
    except Exception as e:
        print(f"Error starting bridge: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
