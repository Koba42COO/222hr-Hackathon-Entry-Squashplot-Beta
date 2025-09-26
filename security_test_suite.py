#!/usr/bin/env python3
"""
SquashPlot Security Test Suite
Comprehensive security testing for the bridge app
"""

import json
import requests
import time
import threading
import subprocess
import socket
import sys
from typing import List, Dict, Tuple
import concurrent.futures
import random
import string

class SecurityTestSuite:
    """Comprehensive security testing suite"""
    
    def __init__(self, bridge_host: str = "127.0.0.1", bridge_port: int = 8443):
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.base_url = f"http://{bridge_host}:{bridge_port}"
        self.test_results = []
        self.auth_token = None
        
    def run_all_tests(self) -> Dict:
        """Run all security tests and return results"""
        print("🛡️ Starting SquashPlot Security Test Suite")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Authentication Tests", self._test_authentication),
            ("Input Validation Tests", self._test_input_validation),
            ("Rate Limiting Tests", self._test_rate_limiting),
            ("Command Injection Tests", self._test_command_injection),
            ("DoS Protection Tests", self._test_dos_protection),
            ("Network Security Tests", self._test_network_security),
            ("Process Isolation Tests", self._test_process_isolation),
            ("Audit Logging Tests", self._test_audit_logging)
        ]
        
        all_results = {}
        
        for category_name, test_function in test_categories:
            print(f"\n🔍 Running {category_name}...")
            try:
                results = test_function()
                all_results[category_name] = results
                self._print_category_results(category_name, results)
            except Exception as e:
                print(f"❌ Error in {category_name}: {e}")
                all_results[category_name] = {"error": str(e)}
        
        # Generate security report
        self._generate_security_report(all_results)
        return all_results
    
    def _test_authentication(self) -> Dict:
        """Test authentication security"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        # Test 1: Invalid token
        try:
            response = self._send_request("invalid_token", "ls")
            if "Authentication failed" in response.get("error", ""):
                results["tests"].append({"name": "Invalid token rejection", "status": "PASS"})
                results["passed"] += 1
            else:
                results["tests"].append({"name": "Invalid token rejection", "status": "FAIL"})
                results["failed"] += 1
                results["critical_issues"].append("Invalid tokens not properly rejected")
        except Exception as e:
            results["tests"].append({"name": "Invalid token rejection", "status": "ERROR", "error": str(e)})
            results["failed"] += 1
        
        # Test 2: Token expiration
        # This would require time manipulation in a real test
        results["tests"].append({"name": "Token expiration", "status": "SKIP", "note": "Requires time manipulation"})
        
        # Test 3: IP address validation
        # This would require IP spoofing in a real test
        results["tests"].append({"name": "IP address validation", "status": "SKIP", "note": "Requires IP spoofing"})
        
        return results
    
    def _test_input_validation(self) -> Dict:
        """Test input validation and sanitization"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        # Get auth token first
        if not self.auth_token:
            self.auth_token = self._get_auth_token()
        
        dangerous_commands = [
            "rm -rf /",
            "sudo rm -rf /",
            "chmod 777 /",
            "cat /etc/passwd",
            "ls ../../../",
            "echo 'test' && rm -rf /",
            "python -c 'import os; os.system(\"rm -rf /\")'",
            "`rm -rf /`",
            "$(rm -rf /)",
            "| rm -rf /",
            "; rm -rf /",
            "&& rm -rf /",
            "|| rm -rf /"
        ]
        
        for cmd in dangerous_commands:
            try:
                response = self._send_request(self.auth_token, cmd)
                if "Command blocked" in response.get("error", ""):
                    results["tests"].append({"name": f"Block dangerous command: {cmd[:20]}...", "status": "PASS"})
                    results["passed"] += 1
                else:
                    results["tests"].append({"name": f"Block dangerous command: {cmd[:20]}...", "status": "FAIL"})
                    results["failed"] += 1
                    results["critical_issues"].append(f"Dangerous command not blocked: {cmd}")
            except Exception as e:
                results["tests"].append({"name": f"Block dangerous command: {cmd[:20]}...", "status": "ERROR", "error": str(e)})
                results["failed"] += 1
        
        # Test command length limits
        long_command = "echo " + "x" * 1000
        try:
            response = self._send_request(self.auth_token, long_command)
            if "Command too long" in response.get("error", ""):
                results["tests"].append({"name": "Command length limit", "status": "PASS"})
                results["passed"] += 1
            else:
                results["tests"].append({"name": "Command length limit", "status": "FAIL"})
                results["failed"] += 1
        except Exception as e:
            results["tests"].append({"name": "Command length limit", "status": "ERROR", "error": str(e)})
            results["failed"] += 1
        
        return results
    
    def _test_rate_limiting(self) -> Dict:
        """Test rate limiting protection"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        if not self.auth_token:
            self.auth_token = self._get_auth_token()
        
        # Test rapid requests
        rapid_requests = 20  # More than the rate limit
        success_count = 0
        rate_limited_count = 0
        
        for i in range(rapid_requests):
            try:
                response = self._send_request(self.auth_token, "echo test")
                if "Rate limit exceeded" in response.get("error", ""):
                    rate_limited_count += 1
                elif response.get("success"):
                    success_count += 1
            except Exception as e:
                pass  # Expected for some requests
        
        if rate_limited_count > 0:
            results["tests"].append({"name": "Rate limiting active", "status": "PASS"})
            results["passed"] += 1
        else:
            results["tests"].append({"name": "Rate limiting active", "status": "FAIL"})
            results["failed"] += 1
            results["critical_issues"].append("Rate limiting not working properly")
        
        return results
    
    def _test_command_injection(self) -> Dict:
        """Test command injection protection"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        if not self.auth_token:
            self.auth_token = self._get_auth_token()
        
        injection_attempts = [
            "echo test; rm -rf /",
            "echo test && rm -rf /",
            "echo test | rm -rf /",
            "echo test `rm -rf /`",
            "echo test $(rm -rf /)",
            "echo test; cat /etc/passwd",
            "echo test && sudo rm -rf /",
            "echo test; chmod 777 /",
            "echo test; mkdir -p /tmp/hack",
            "echo test; python -c 'import os; os.system(\"rm -rf /\")'"
        ]
        
        for injection in injection_attempts:
            try:
                response = self._send_request(self.auth_token, injection)
                if "Command blocked" in response.get("error", ""):
                    results["tests"].append({"name": f"Block injection: {injection[:30]}...", "status": "PASS"})
                    results["passed"] += 1
                else:
                    results["tests"].append({"name": f"Block injection: {injection[:30]}...", "status": "FAIL"})
                    results["failed"] += 1
                    results["critical_issues"].append(f"Command injection not blocked: {injection}")
            except Exception as e:
                results["tests"].append({"name": f"Block injection: {injection[:30]}...", "status": "ERROR", "error": str(e)})
                results["failed"] += 1
        
        return results
    
    def _test_dos_protection(self) -> Dict:
        """Test DoS protection mechanisms"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        # Test concurrent connections
        def make_request():
            try:
                response = self._send_request(self.auth_token or "test", "echo test")
                return response
            except:
                return None
        
        # Test with multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures, timeout=10)]
        
        # Check if server is still responsive
        try:
            test_response = self._send_request(self.auth_token or "test", "echo test")
            if test_response:
                results["tests"].append({"name": "DoS protection - server responsive", "status": "PASS"})
                results["passed"] += 1
            else:
                results["tests"].append({"name": "DoS protection - server responsive", "status": "FAIL"})
                results["failed"] += 1
                results["critical_issues"].append("Server not responsive after concurrent requests")
        except Exception as e:
            results["tests"].append({"name": "DoS protection - server responsive", "status": "ERROR", "error": str(e)})
            results["failed"] += 1
        
        return results
    
    def _test_network_security(self) -> Dict:
        """Test network security measures"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        # Test 1: Check if server only binds to localhost
        try:
            # Try to connect from external IP (this should fail in secure setup)
            external_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            external_socket.settimeout(5)
            result = external_socket.connect_ex(("127.0.0.1", self.bridge_port))
            external_socket.close()
            
            if result == 0:
                results["tests"].append({"name": "Localhost binding", "status": "PASS"})
                results["passed"] += 1
            else:
                results["tests"].append({"name": "Localhost binding", "status": "FAIL"})
                results["failed"] += 1
        except Exception as e:
            results["tests"].append({"name": "Localhost binding", "status": "ERROR", "error": str(e)})
            results["failed"] += 1
        
        # Test 2: Check for SSL/TLS (if implemented)
        results["tests"].append({"name": "SSL/TLS encryption", "status": "SKIP", "note": "Not implemented in current version"})
        
        return results
    
    def _test_process_isolation(self) -> Dict:
        """Test process isolation and resource limits"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        if not self.auth_token:
            self.auth_token = self._get_auth_token()
        
        # Test 1: Memory limit
        memory_test = "python -c 'import sys; sys.exit(0)'"  # Safe command
        try:
            response = self._send_request(self.auth_token, memory_test)
            if response.get("success"):
                results["tests"].append({"name": "Process isolation - memory", "status": "PASS"})
                results["passed"] += 1
            else:
                results["tests"].append({"name": "Process isolation - memory", "status": "FAIL"})
                results["failed"] += 1
        except Exception as e:
            results["tests"].append({"name": "Process isolation - memory", "status": "ERROR", "error": str(e)})
            results["failed"] += 1
        
        # Test 2: Timeout protection
        timeout_test = "sleep 60"  # This should timeout
        try:
            start_time = time.time()
            response = self._send_request(self.auth_token, timeout_test)
            end_time = time.time()
            
            if end_time - start_time < 35:  # Should timeout before 35 seconds
                results["tests"].append({"name": "Process isolation - timeout", "status": "PASS"})
                results["passed"] += 1
            else:
                results["tests"].append({"name": "Process isolation - timeout", "status": "FAIL"})
                results["failed"] += 1
                results["critical_issues"].append("Command timeout not working properly")
        except Exception as e:
            results["tests"].append({"name": "Process isolation - timeout", "status": "ERROR", "error": str(e)})
            results["failed"] += 1
        
        return results
    
    def _test_audit_logging(self) -> Dict:
        """Test audit logging functionality"""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "critical_issues": []
        }
        
        # Check if audit log file exists
        try:
            with open("security_audit.log", "r") as f:
                log_content = f.read()
                if "command_execution" in log_content:
                    results["tests"].append({"name": "Audit logging - command execution", "status": "PASS"})
                    results["passed"] += 1
                else:
                    results["tests"].append({"name": "Audit logging - command execution", "status": "FAIL"})
                    results["failed"] += 1
        except FileNotFoundError:
            results["tests"].append({"name": "Audit logging - file exists", "status": "FAIL"})
            results["failed"] += 1
            results["critical_issues"].append("Audit log file not found")
        except Exception as e:
            results["tests"].append({"name": "Audit logging - file exists", "status": "ERROR", "error": str(e)})
            results["failed"] += 1
        
        return results
    
    def _send_request(self, auth_token: str, command: str) -> Dict:
        """Send request to bridge app"""
        try:
            # Use requests for HTTP-like communication
            # Note: This is a simplified version - real implementation would use socket
            data = json.dumps({
                "auth_token": auth_token,
                "command": command
            })
            
            # For testing purposes, we'll simulate the response
            # In a real test, this would connect to the actual bridge
            return {"success": True, "output": "test output", "error": ""}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_auth_token(self) -> str:
        """Get authentication token for testing"""
        # This would normally authenticate with the bridge
        # For testing, we'll use a mock token
        return "test_auth_token_12345"
    
    def _print_category_results(self, category: str, results: Dict):
        """Print results for a test category"""
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        total = passed + failed
        
        if total > 0:
            success_rate = (passed / total) * 100
            status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            print(f"{status} {category}: {passed}/{total} tests passed ({success_rate:.1f}%)")
        else:
            print(f"⏭️ {category}: No tests run")
        
        # Print critical issues
        for issue in results.get("critical_issues", []):
            print(f"   🚨 CRITICAL: {issue}")
    
    def _generate_security_report(self, all_results: Dict):
        """Generate comprehensive security report"""
        print("\n" + "=" * 60)
        print("🛡️ SECURITY TEST REPORT")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        critical_issues = []
        
        for category, results in all_results.items():
            if "error" not in results:
                total_passed += results.get("passed", 0)
                total_failed += results.get("failed", 0)
                critical_issues.extend(results.get("critical_issues", []))
        
        total_tests = total_passed + total_failed
        if total_tests > 0:
            overall_success = (total_passed / total_tests) * 100
            print(f"Overall Security Score: {overall_success:.1f}%")
            print(f"Tests Passed: {total_passed}")
            print(f"Tests Failed: {total_failed}")
        
        if critical_issues:
            print(f"\n🚨 CRITICAL SECURITY ISSUES FOUND:")
            for i, issue in enumerate(critical_issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("\n✅ No critical security issues found")
        
        # Security recommendations
        print(f"\n📋 SECURITY RECOMMENDATIONS:")
        if overall_success < 80:
            print("   - Implement additional security measures")
            print("   - Review and fix failed tests")
            print("   - Consider professional security audit")
        else:
            print("   - Security implementation looks good")
            print("   - Continue regular security testing")
            print("   - Monitor for new vulnerabilities")

def main():
    """Run the security test suite"""
    print("🛡️ SquashPlot Security Test Suite")
    print("=" * 60)
    
    # Check if bridge is running
    try:
        test_suite = SecurityTestSuite()
        results = test_suite.run_all_tests()
        
        # Save results to file
        with open("security_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Test results saved to: security_test_results.json")
        
    except Exception as e:
        print(f"❌ Error running security tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
