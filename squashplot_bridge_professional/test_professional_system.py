#!/usr/bin/env python3
"""
Professional System Test - Comprehensive Testing
Tests the complete SquashPlot Bridge Professional system
"""

import os
import sys
import time
import json
import subprocess
import threading
import requests
from pathlib import Path

class ProfessionalSystemTest:
    """Comprehensive test suite for SquashPlot Bridge Professional"""
    
    def __init__(self):
        self.bridge_host = '127.0.0.1'
        self.bridge_port = 8080
        self.bridge_url = f'http://{self.bridge_host}:{self.bridge_port}'
        self.test_results = []
        
    def log_test(self, test_name, success, message=""):
        """Log test results"""
        status = "PASS" if success else "FAIL"
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {message}")
        
    def test_file_structure(self):
        """Test that all required files exist"""
        required_files = [
            'SquashPlotBridge.py',
            'SquashPlotBridge.html',
            'PROFESSIONAL_SUMMARY.md',
            'SIMPLE_SUMMARY.md'
        ]
        
        for file in required_files:
            if os.path.exists(file):
                self.log_test(f"File exists: {file}", True, "File found")
            else:
                self.log_test(f"File exists: {file}", False, "File missing")
                
    def test_imports(self):
        """Test that all required imports work"""
        try:
            # Test standard library imports
            import http.server
            import threading
            import subprocess
            import json
            import time
            import os
            import sys
            import webbrowser
            from urllib.parse import urlparse, parse_qs
            
            self.log_test("Standard library imports", True, "All imports successful")
            
        except ImportError as e:
            self.log_test("Standard library imports", False, f"Import error: {e}")
            
    def test_bridge_initialization(self):
        """Test bridge initialization"""
        try:
            # Import the bridge class
            sys.path.append('.')
            from SquashPlotBridge import SquashPlotBridge
            
            # Test initialization
            bridge = SquashPlotBridge()
            self.log_test("Bridge initialization", True, "Bridge class created successfully")
            
            # Test methods exist
            methods = ['start', 'stop', 'is_running']
            for method in methods:
                if hasattr(bridge, method):
                    self.log_test(f"Method exists: {method}", True, "Method found")
                else:
                    self.log_test(f"Method exists: {method}", False, "Method missing")
                    
        except Exception as e:
            self.log_test("Bridge initialization", False, f"Error: {e}")
            
    def test_html_interface(self):
        """Test HTML interface file"""
        try:
            with open('SquashPlotBridge.html', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for essential elements
            required_elements = [
                '<title>SquashPlot Bridge - Professional Interface</title>',
                'class SquashPlotBridgeInterface',
                'executeDemo',
                'checkBridgeStatus',
                'testConnection'
            ]
            
            for element in required_elements:
                if element in content:
                    self.log_test(f"HTML element: {element[:30]}...", True, "Element found")
                else:
                    self.log_test(f"HTML element: {element[:30]}...", False, "Element missing")
                    
        except Exception as e:
            self.log_test("HTML interface", False, f"Error: {e}")
            
    def test_bridge_server_startup(self):
        """Test bridge server startup"""
        try:
            # Start bridge server in background
            process = subprocess.Popen([
                sys.executable, 'SquashPlotBridge.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for startup
            time.sleep(3)
            
            # Check if process is running
            if process.poll() is None:
                self.log_test("Bridge server startup", True, "Server process running")
                
                # Test HTTP connectivity
                try:
                    response = requests.get(f'{self.bridge_url}/status', timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        self.log_test("HTTP status endpoint", True, f"Status: {data.get('status')}")
                    else:
                        self.log_test("HTTP status endpoint", False, f"Status code: {response.status_code}")
                except requests.RequestException as e:
                    self.log_test("HTTP status endpoint", False, f"Request error: {e}")
                
                # Test web interface
                try:
                    response = requests.get(f'{self.bridge_url}/', timeout=5)
                    if response.status_code == 200:
                        self.log_test("Web interface serving", True, "HTML interface accessible")
                    else:
                        self.log_test("Web interface serving", False, f"Status code: {response.status_code}")
                except requests.RequestException as e:
                    self.log_test("Web interface serving", False, f"Request error: {e}")
                
                # Test command execution
                try:
                    response = requests.post(
                        f'{self.bridge_url}/execute',
                        json={'command': 'hello-world'},
                        timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success'):
                            self.log_test("Command execution", True, "Hello world command executed")
                        else:
                            self.log_test("Command execution", False, f"Execution failed: {data.get('error')}")
                    else:
                        self.log_test("Command execution", False, f"Status code: {response.status_code}")
                except requests.RequestException as e:
                    self.log_test("Command execution", False, f"Request error: {e}")
                
                # Clean up - stop the process
                process.terminate()
                process.wait(timeout=5)
                self.log_test("Bridge server shutdown", True, "Server stopped cleanly")
                
            else:
                self.log_test("Bridge server startup", False, "Server process failed to start")
                
        except Exception as e:
            self.log_test("Bridge server startup", False, f"Error: {e}")
            
    def test_security_features(self):
        """Test security features"""
        try:
            # Test unauthorized command
            response = requests.post(
                f'{self.bridge_url}/execute',
                json={'command': 'malicious-command'},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data.get('success') and 'not in whitelist' in data.get('error', ''):
                    self.log_test("Security: Command whitelist", True, "Unauthorized command blocked")
                else:
                    self.log_test("Security: Command whitelist", False, "Unauthorized command allowed")
            else:
                self.log_test("Security: Command whitelist", True, f"Unauthorized command rejected (status: {response.status_code})")
                
        except Exception as e:
            self.log_test("Security: Command whitelist", False, f"Error: {e}")
            
    def test_documentation(self):
        """Test documentation files"""
        docs = [
            'PROFESSIONAL_SUMMARY.md',
            'SIMPLE_SUMMARY.md'
        ]
        
        for doc in docs:
            try:
                with open(doc, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if len(content) > 100:  # Basic content check
                    self.log_test(f"Documentation: {doc}", True, "Document has content")
                else:
                    self.log_test(f"Documentation: {doc}", False, "Document appears empty")
                    
            except Exception as e:
                self.log_test(f"Documentation: {doc}", False, f"Error: {e}")
                
    def run_all_tests(self):
        """Run all tests"""
        print("SquashPlot Bridge Professional - Comprehensive Test Suite")
        print("=" * 70)
        
        # Run tests (already in the correct directory)
        self.test_file_structure()
        self.test_imports()
        self.test_bridge_initialization()
        self.test_html_interface()
        self.test_bridge_server_startup()
        self.test_security_features()
        self.test_documentation()
        
        # Generate report
        self.generate_report()
            
    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 70)
        print("TEST REPORT SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['test']}: {result['message']}")
        
        # Save report
        report_data = {
            'timestamp': time.time(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'results': self.test_results
        }
        
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\nDetailed report saved to: test_report.json")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! SquashPlot Bridge Professional is ready!")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix issues.")

def main():
    """Main test function"""
    try:
        tester = ProfessionalSystemTest()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test suite error: {e}")

if __name__ == "__main__":
    main()
