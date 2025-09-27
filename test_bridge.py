#!/usr/bin/env python3
"""
Simple Test Bridge for Dashboard Status Bar Testing
Responds to PING requests on port 8443
"""

import socket
import threading
import time
import json
from datetime import datetime

class TestBridge:
    def __init__(self, host='127.0.0.1', port=8443):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.start_time = datetime.now()
        
    def handle_client(self, client_socket, client_address):
        """Handle individual client connections"""
        try:
            # Receive data from client
            data = client_socket.recv(1024).decode('utf-8').strip()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Received from {client_address}: {data}")
            
            # Handle PING requests
            if data == 'PING':
                response = 'PONG'
                client_socket.send(response.encode('utf-8'))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent PONG to {client_address}")
            
            # Handle STOP requests
            elif data == 'STOP':
                response = 'STOPPED'
                client_socket.send(response.encode('utf-8'))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Received STOP command")
                self.running = False
            
            # Handle status requests
            elif data == 'STATUS':
                uptime = datetime.now() - self.start_time
                status = {
                    'status': 'online',
                    'uptime': str(uptime),
                    'port': self.port,
                    'timestamp': datetime.now().isoformat()
                }
                response = json.dumps(status)
                client_socket.send(response.encode('utf-8'))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent status to {client_address}")
            
            # Default response
            else:
                response = 'OK'
                client_socket.send(response.encode('utf-8'))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent OK to {client_address}")
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
    
    def start(self):
        """Start the bridge server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"Test Bridge started on {self.host}:{self.port}")
            print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("Ready to accept connections...")
            print("Send 'PING' to test connection")
            print("Send 'STOP' to shutdown")
            print("Send 'STATUS' for bridge info")
            print("-" * 50)
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] New connection from {client_address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        print(f"Socket error: {e}")
                    break
                    
        except Exception as e:
            print(f"Failed to start bridge: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bridge server"""
        print(f"\nStopping Test Bridge...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Test Bridge stopped")

def main():
    """Main function"""
    print("SquashPlot Test Bridge")
    print("=" * 50)
    
    bridge = TestBridge()
    
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
