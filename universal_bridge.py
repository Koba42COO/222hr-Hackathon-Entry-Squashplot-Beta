#!/usr/bin/env python3
"""
Universal SquashPlot Bridge
============================
Platform-agnostic bridge that works on:
- Local development (127.0.0.1)
- Replit (replit.dev domain)
- Cloud hosting (various domains)
- Docker containers
- Any other platform
"""

import os
import sys
import json
import socket
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path

class PlatformDetector:
    """Detects the current platform and provides appropriate configuration"""
    
    @staticmethod
    def detect_platform() -> Dict[str, str]:
        """Detect current platform and return configuration"""
        
        # Check for Replit
        if os.environ.get('REPL_ID') or os.environ.get('REPL_SLUG'):
            return {
                'platform': 'replit',
                'host': '0.0.0.0',  # Bind to all interfaces
                'port': '8443',
                'base_url': f"https://{os.environ.get('REPL_SLUG', 'your-repl')}.replit.dev",
                'bridge_url': f"https://{os.environ.get('REPL_SLUG', 'your-repl')}.replit.dev:8443"
            }
        
        # Check for Docker
        elif os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
            return {
                'platform': 'docker',
                'host': '0.0.0.0',
                'port': '8443',
                'base_url': 'http://localhost',
                'bridge_url': 'http://localhost:8443'
            }
        
        # Check for Heroku
        elif os.environ.get('DYNO'):
            return {
                'platform': 'heroku',
                'host': '0.0.0.0',
                'port': os.environ.get('PORT', '8443'),
                'base_url': f"https://{os.environ.get('HEROKU_APP_NAME', 'your-app')}.herokuapp.com",
                'bridge_url': f"https://{os.environ.get('HEROKU_APP_NAME', 'your-app')}.herokuapp.com"
            }
        
        # Check for Railway
        elif os.environ.get('RAILWAY_ENVIRONMENT'):
            return {
                'platform': 'railway',
                'host': '0.0.0.0',
                'port': os.environ.get('PORT', '8443'),
                'base_url': f"https://{os.environ.get('RAILWAY_STATIC_URL', 'your-app')}.up.railway.app",
                'bridge_url': f"https://{os.environ.get('RAILWAY_STATIC_URL', 'your-app')}.up.railway.app"
            }
        
        # Check for Vercel
        elif os.environ.get('VERCEL'):
            return {
                'platform': 'vercel',
                'host': '0.0.0.0',
                'port': os.environ.get('PORT', '8443'),
                'base_url': f"https://{os.environ.get('VERCEL_URL', 'your-app')}.vercel.app",
                'bridge_url': f"https://{os.environ.get('VERCEL_URL', 'your-app')}.vercel.app"
            }
        
        # Check for local development
        else:
            return {
                'platform': 'local',
                'host': '127.0.0.1',
                'port': '8443',
                'base_url': 'http://localhost',
                'bridge_url': 'http://127.0.0.1:8443'
            }

class UniversalBridge:
    """Universal bridge that adapts to different platforms"""
    
    def __init__(self):
        self.config = PlatformDetector.detect_platform()
        self.host = self.config['host']
        self.port = int(self.config['port'])
        self.running = False
        self.server_socket = None
        self.start_time = datetime.now()
        self.logger = self._setup_logging()
        
        # Platform-specific settings
        self._apply_platform_config()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the bridge"""
        logger = logging.getLogger('UniversalBridge')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _apply_platform_config(self):
        """Apply platform-specific configuration"""
        platform = self.config['platform']
        
        if platform == 'replit':
            # Replit-specific settings
            self.logger.info("Running on Replit platform")
            
        elif platform == 'docker':
            # Docker-specific settings
            self.logger.info("Running in Docker container")
            
        elif platform in ['heroku', 'railway', 'vercel']:
            # Cloud platform settings
            self.logger.info(f"Running on {platform} platform")
            
        else:
            # Local development
            self.logger.info("Running in local development mode")
    
    def handle_client(self, client_socket: socket.socket, client_address: Tuple[str, int]):
        """Handle individual client connections"""
        try:
            # Receive data from client
            data = client_socket.recv(1024).decode('utf-8').strip()
            self.logger.info(f"Received from {client_address}: {data}")
            
            # Handle PING requests
            if data == 'PING':
                response = 'PONG'
                client_socket.send(response.encode('utf-8'))
                self.logger.info(f"Sent PONG to {client_address}")
            
            # Handle STOP requests
            elif data == 'STOP':
                response = 'STOPPED'
                client_socket.send(response.encode('utf-8'))
                self.logger.info("Received STOP command")
                self.running = False
            
            # Handle STATUS requests
            elif data == 'STATUS':
                uptime = datetime.now() - self.start_time
                status = {
                    'status': 'online',
                    'platform': self.config['platform'],
                    'uptime': str(uptime),
                    'port': self.port,
                    'host': self.host,
                    'base_url': self.config['base_url'],
                    'bridge_url': self.config['bridge_url'],
                    'timestamp': datetime.now().isoformat()
                }
                response = json.dumps(status, indent=2)
                client_socket.send(response.encode('utf-8'))
                self.logger.info(f"Sent status to {client_address}")
            
            # Handle PLATFORM requests
            elif data == 'PLATFORM':
                response = json.dumps(self.config, indent=2)
                client_socket.send(response.encode('utf-8'))
                self.logger.info(f"Sent platform info to {client_address}")
            
            # Default response
            else:
                response = 'OK'
                client_socket.send(response.encode('utf-8'))
                self.logger.info(f"Sent OK to {client_address}")
                
        except Exception as e:
            self.logger.error(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
    
    def start(self):
        """Start the universal bridge server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            self.logger.info(f"Universal Bridge started on {self.host}:{self.port}")
            self.logger.info(f"Platform: {self.config['platform']}")
            self.logger.info(f"Base URL: {self.config['base_url']}")
            self.logger.info(f"Bridge URL: {self.config['bridge_url']}")
            self.logger.info(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("Ready to accept connections...")
            self.logger.info("Commands: PING, STOP, STATUS, PLATFORM")
            self.logger.info("-" * 60)
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    self.logger.info(f"New connection from {client_address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        self.logger.error(f"Socket error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Failed to start bridge: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bridge server"""
        self.logger.info("Stopping Universal Bridge...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        self.logger.info("Universal Bridge stopped")

def main():
    """Main function"""
    print("Universal SquashPlot Bridge")
    print("=" * 60)
    
    # Show platform detection
    config = PlatformDetector.detect_platform()
    print(f"Platform: {config['platform']}")
    print(f"Host: {config['host']}")
    print(f"Port: {config['port']}")
    print(f"Base URL: {config['base_url']}")
    print(f"Bridge URL: {config['bridge_url']}")
    print("-" * 60)
    
    bridge = UniversalBridge()
    
    try:
        bridge.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        bridge.stop()
    except Exception as e:
        print(f"Error: {e}")
        bridge.stop()

if __name__ == "__main__":
    main()



