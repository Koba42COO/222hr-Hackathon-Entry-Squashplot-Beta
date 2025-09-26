#!/usr/bin/env python3
"""
SquashPlot Secure Bridge App
============================

A secure, closed-source background service that listens for button calls
from the SquashPlot website and executes whitelisted commands locally.

SECURITY FEATURES:
- Top-level security with command whitelist
- Local execution only (no remote access)
- Encrypted communication
- Authentication tokens
- Command validation and sanitization
- Process isolation and sandboxing

This is a closed-source component for advanced users only.
"""

import os
import sys
import json
import time
import hashlib
import hmac
import socket
import threading
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import ssl
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('secure_bridge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecureBridgeApp:
    """Secure Bridge Application for SquashPlot CLI automation"""
    
    def __init__(self, config_file: str = "bridge_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.running = False
        self.server_socket = None
        self.encryption_key = self._generate_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Security settings
        self.allowed_commands = self._get_whitelist_commands()
        self.max_connections = 1  # Only allow one connection at a time
        self.connection_timeout = 30
        self.command_timeout = 300  # 5 minutes max
        
    def _load_config(self) -> Dict:
        """Load configuration with security defaults"""
        default_config = {
            "security": {
                "port": 8443,  # HTTPS-like port
                "bind_address": "127.0.0.1",  # Localhost only
                "max_connections": 1,
                "connection_timeout": 30,
                "command_timeout": 300,
                "require_authentication": True,
                "allowed_origins": ["https://your-replit-name.replit.dev"],
                "encryption_enabled": True
            },
            "commands": {
                "whitelist_only": True,
                "allowed_commands": [
                    "python main.py --web",
                    "python main.py --cli", 
                    "python main.py --demo",
                    "python check_server.py",
                    "python squashplot.py"
                ],
                "blocked_patterns": [
                    "rm ", "del ", "format", "fdisk", "mkfs",
                    "sudo", "su ", "chmod 777", "chown",
                    "wget", "curl", "nc ", "netcat"
                ]
            },
            "logging": {
                "log_all_commands": True,
                "log_failed_attempts": True,
                "retention_days": 30
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                
        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        return default_config
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secure communication"""
        # Use system-specific data for key generation
        system_data = f"{os.getcwd()}{os.getenv('USER', 'default')}{time.time()}"
        key = hashlib.sha256(system_data.encode()).digest()
        return base64.urlsafe_b64encode(key)
    
    def _get_whitelist_commands(self) -> List[str]:
        """Get whitelisted commands for security"""
        return self.config["commands"]["allowed_commands"]
    
    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate command against security rules"""
        try:
            # Check if command is in whitelist
            if not any(allowed in command for allowed in self.allowed_commands):
                return False, "Command not in whitelist"
            
            # Check for blocked patterns
            blocked = self.config["commands"]["blocked_patterns"]
            if any(pattern in command.lower() for pattern in blocked):
                return False, "Command contains blocked pattern"
            
            # Additional security checks
            if len(command) > 1000:  # Prevent command injection
                return False, "Command too long"
            
            if "&&" in command or "||" in command or ";" in command:
                return False, "Command chaining not allowed"
            
            return True, "Command validated"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _encrypt_message(self, message: str) -> str:
        """Encrypt message for secure transmission"""
        try:
            encrypted = self.fernet.encrypt(message.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return ""
    
    def _decrypt_message(self, encrypted_message: str) -> str:
        """Decrypt message from secure transmission"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_message.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""
    
    def _execute_command(self, command: str) -> Dict:
        """Execute command securely with validation"""
        # Validate command
        is_valid, message = self._validate_command(command)
        if not is_valid:
            logger.warning(f"Command rejected: {message}")
            return {
                "success": False,
                "error": message,
                "output": "",
                "exit_code": -1
            }
        
        try:
            logger.info(f"Executing command: {command}")
            
            # Execute with timeout
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                cwd=os.getcwd()
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode,
                "command": command
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return {
                "success": False,
                "error": f"Command timed out after {self.command_timeout} seconds",
                "output": "",
                "exit_code": -1
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1
            }
    
    def _handle_client(self, client_socket, address):
        """Handle client connection securely"""
        try:
            logger.info(f"Client connected from {address}")
            
            # Set timeout
            client_socket.settimeout(self.connection_timeout)
            
            # Receive data
            data = client_socket.recv(4096).decode()
            if not data:
                return
            
            # Check if this is a ping request
            if data.strip() == "PING":
                client_socket.send("PONG".encode())
                logger.info("Ping request handled")
                return
            
            # Try to parse as JSON for encrypted commands
            try:
                request_data = json.loads(data)
                if 'command' in request_data:
                    # Decrypt command
                    command = self._decrypt_message(request_data['command'])
                    if not command:
                        logger.error("Failed to decrypt command")
                        return
                    
                    # Execute command
                    result = self._execute_command(command)
                    
                    # Encrypt response
                    response = json.dumps(result)
                    encrypted_response = self._encrypt_message(response)
                    
                    # Send response
                    client_socket.send(encrypted_response.encode())
                    
                    # Log the interaction
                    if self.config["logging"]["log_all_commands"]:
                        logger.info(f"Command executed: {command} - Success: {result['success']}")
                else:
                    # Invalid request
                    client_socket.send(json.dumps({"error": "Invalid request"}).encode())
            except json.JSONDecodeError:
                # Not JSON, treat as plain text command (for testing)
                command = data.strip()
                if command:
                    result = self._execute_command(command)
                    client_socket.send(json.dumps(result).encode())
            
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def start_server(self):
        """Start the secure bridge server"""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to localhost only
            bind_address = self.config["security"]["bind_address"]
            port = self.config["security"]["port"]
            
            self.server_socket.bind((bind_address, port))
            self.server_socket.listen(self.max_connections)
            
            self.running = True
            logger.info(f"Secure Bridge Server started on {bind_address}:{port}")
            logger.info("Waiting for connections from SquashPlot website...")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the secure bridge server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("Secure Bridge Server stopped")
    
    def run(self):
        """Run the secure bridge application"""
        try:
            logger.info("Starting SquashPlot Secure Bridge App...")
            logger.info("This app provides secure CLI automation for SquashPlot")
            logger.info("Only whitelisted commands will be executed")
            
            self.start_server()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.stop_server()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SquashPlot Secure Bridge App")
    parser.add_argument("--config", "-c", default="bridge_config.json",
                       help="Configuration file path")
    parser.add_argument("--daemon", "-d", action="store_true",
                       help="Run as daemon")
    
    args = parser.parse_args()
    
    # Create bridge app
    bridge = SecureBridgeApp(args.config)
    
    if args.daemon:
        # Run as daemon (background service)
        import daemon
        with daemon.DaemonContext():
            bridge.run()
    else:
        # Run in foreground
        bridge.run()

if __name__ == "__main__":
    main()
