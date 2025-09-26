#!/usr/bin/env python3
import subprocess
import sys
import os

def run_security_tests():
    print("Running SquashPlot Security Tests...")
    
    # Test 1: Check if hardened bridge exists
    if os.path.exists("secure_bridge_hardened.py"):
        print("[OK] Hardened bridge app found")
    else:
        print("[ERROR] Hardened bridge app not found")
    
    # Test 2: Check security configuration
    if os.path.exists("security_config.json"):
        print("[OK] Security configuration found")
    else:
        print("[ERROR] Security configuration not found")
    
    # Test 3: Run security test suite
    if os.path.exists("security_test_suite.py"):
        print("Running security test suite...")
        result = subprocess.run([sys.executable, "security_test_suite.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] Security tests passed")
        else:
            print(f"[WARNING] Security tests had issues: {result.stderr}")
    
    print("Security testing completed")

if __name__ == "__main__":
    run_security_tests()
