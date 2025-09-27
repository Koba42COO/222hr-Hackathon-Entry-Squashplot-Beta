#!/usr/bin/env python3
"""
Test Working Bridge
Simple test for the working secure bridge
"""

import json
import time
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if we can import the working bridge modules"""
    print("Testing imports...")
    
    try:
        from working_ultimate_bridge import WorkingSecureBridge, WorkingCrypto, WorkingSessionManager
        print("OK - Working bridge modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"FAIL - Import failed: {e}")
        return False
    except Exception as e:
        print(f"FAIL - Unexpected error: {e}")
        return False

def test_crypto():
    """Test working crypto"""
    print("\nTesting Working Crypto...")
    
    try:
        from working_ultimate_bridge import WorkingCrypto
        
        crypto = WorkingCrypto()
        print("OK - WorkingCrypto initialized")
        
        # Test encryption/decryption
        test_data = "This is a test message"
        encrypted = crypto.encrypt(test_data)
        print("OK - Encryption successful")
        
        decrypted = crypto.decrypt(encrypted)
        print("OK - Decryption successful")
        
        if decrypted == test_data:
            print("PASS - Crypto test PASSED")
            return True
        else:
            print("FAIL - Crypto test FAILED")
            return False
            
    except Exception as e:
        print(f"FAIL - Crypto test failed: {e}")
        return False

def test_session_manager():
    """Test session manager"""
    print("\nTesting Session Manager...")
    
    try:
        from working_ultimate_bridge import WorkingSessionManager
        
        session_manager = WorkingSessionManager()
        print("OK - SessionManager initialized")
        
        # Test session creation
        session_data = session_manager.create_session("127.0.0.1")
        print("OK - Session created")
        
        session_id = session_data['session_id']
        
        # Test session validation
        is_valid, message = session_manager.validate_session(session_id, "127.0.0.1")
        if is_valid:
            print("OK - Session validation successful")
        else:
            print(f"FAIL - Session validation failed: {message}")
            return False
        
        # Test session info
        info = session_manager.get_session_info(session_id)
        if info:
            print("OK - Session info retrieved")
        else:
            print("FAIL - Session info not retrieved")
            return False
        
        print("PASS - Session manager test PASSED")
        return True
        
    except Exception as e:
        print(f"FAIL - Session manager test failed: {e}")
        return False

def test_command_sanitizer():
    """Test command sanitizer"""
    print("\nTesting Command Sanitizer...")
    
    try:
        from working_ultimate_bridge import WorkingCommandSanitizer
        
        sanitizer = WorkingCommandSanitizer()
        print("OK - CommandSanitizer initialized")
        
        # Test safe command
        is_valid, message, clean_command = sanitizer.sanitize_command("squashplot --help")
        if is_valid:
            print("OK - Safe command accepted")
        else:
            print(f"FAIL - Safe command rejected: {message}")
            return False
        
        # Test dangerous command
        is_valid, message, clean_command = sanitizer.sanitize_command("rm -rf /")
        if not is_valid:
            print("OK - Dangerous command blocked")
        else:
            print("FAIL - Dangerous command allowed")
            return False
        
        print("PASS - Command sanitizer test PASSED")
        return True
        
    except Exception as e:
        print(f"FAIL - Command sanitizer test failed: {e}")
        return False

def test_bridge_initialization():
    """Test bridge initialization"""
    print("\nTesting Bridge Initialization...")
    
    try:
        from working_ultimate_bridge import WorkingSecureBridge
        
        bridge = WorkingSecureBridge()
        print("OK - WorkingSecureBridge initialized")
        
        # Check components
        if hasattr(bridge, 'crypto') and bridge.crypto is not None:
            print("OK - Crypto component initialized")
        else:
            print("FAIL - Crypto component not initialized")
            return False
            
        if hasattr(bridge, 'session_manager') and bridge.session_manager is not None:
            print("OK - Session manager component initialized")
        else:
            print("FAIL - Session manager component not initialized")
            return False
            
        if hasattr(bridge, 'sanitizer') and bridge.sanitizer is not None:
            print("OK - Sanitizer component initialized")
        else:
            print("FAIL - Sanitizer component not initialized")
            return False
            
        print("PASS - Bridge initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"FAIL - Bridge initialization test failed: {e}")
        return False

def test_html_interface():
    """Test HTML interface file"""
    print("\nTesting HTML Interface...")
    
    try:
        interface_file = "ultimate_security_interface.html"
        
        if os.path.exists(interface_file):
            print("OK - HTML interface file exists")
            
            with open(interface_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for key elements
            required_elements = [
                'Ultimate Impenetrable SquashPlot Bridge',
                'Create Impossible Security Session',
                'Execute Commands with Ultimate Protection'
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if not missing_elements:
                print("OK - All required HTML elements present")
                print("PASS - HTML interface test PASSED")
                return True
            else:
                print(f"FAIL - Missing HTML elements: {missing_elements}")
                return False
        else:
            print("FAIL - HTML interface file not found")
            return False
            
    except Exception as e:
        print(f"FAIL - HTML interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Working Bridge Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Crypto Test", test_crypto),
        ("Session Manager Test", test_session_manager),
        ("Command Sanitizer Test", test_command_sanitizer),
        ("Bridge Initialization Test", test_bridge_initialization),
        ("HTML Interface Test", test_html_interface)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"FAIL - {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*40)
    print("TEST RESULTS SUMMARY")
    print("="*40)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"PASS - {test_name}")
            passed += 1
        else:
            print(f"FAIL - {test_name}")
            failed += 1
    
    print(f"\nTotal: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("\nSUCCESS! ALL TESTS PASSED!")
        print("Your working secure bridge is ready to use!")
        print("The bridge has solid security features!")
    else:
        print(f"\nWARNING: {failed} test(s) failed. Please check the issues above.")
    
    print("="*40)

if __name__ == "__main__":
    main()
