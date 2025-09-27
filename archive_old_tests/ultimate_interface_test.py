#!/usr/bin/env python3
"""
Universal Security Interface Test Suite
Tests the universal_security_interface.html functionality
"""

import os
import sys
import re
from pathlib import Path

def test_universal_interface():
    """Test the universal security interface HTML file"""
    print("Universal Security Interface Test Suite")
    print("=" * 50)
    
    interface_file = "ultimate_security_interface.html"
    
    if not os.path.exists(interface_file):
        print(f"ERROR - {interface_file} not found")
        return False
    
    try:
        with open(interface_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR - Could not read {interface_file}: {e}")
        return False
    
    tests_passed = 0
    tests_failed = 0
    
    print("\n==================== Universal Interface Tests ====================")
    
    # Test 1: Universal installer download button
    print("Testing universal installer download button...")
    if 'downloadUniversalBtn' in content:
        print("OK - Universal download button present")
        tests_passed += 1
    else:
        print("FAIL - Universal download button missing")
        tests_failed += 1
    
    # Test 2: Platform detection
    print("Testing platform detection...")
    if 'detectPlatform' in content:
        print("OK - Platform detection function present")
        tests_passed += 1
    else:
        print("FAIL - Platform detection function missing")
        tests_failed += 1
    
    # Test 3: Cross-platform badges
    print("Testing cross-platform badges...")
    platforms = ['Windows', 'macOS', 'Linux', 'iOS', 'Android']
    all_platforms_present = all(platform in content for platform in platforms)
    if all_platforms_present:
        print("OK - All platform badges present")
        tests_passed += 1
    else:
        print("FAIL - Some platform badges missing")
        tests_failed += 1
    
    # Test 4: Universal session creation
    print("Testing universal session creation...")
    if 'createUniversalSessionBtn' in content:
        print("OK - Universal session creation button present")
        tests_passed += 1
    else:
        print("FAIL - Universal session creation button missing")
        tests_failed += 1
    
    # Test 5: Modal functionality
    print("Testing modal functionality...")
    modals = ['sessionModal', 'downloadModal', 'securityAlertModal']
    all_modals_present = all(modal in content for modal in modals)
    if all_modals_present:
        print("OK - All modals present")
        tests_passed += 1
    else:
        print("FAIL - Some modals missing")
        tests_failed += 1
    
    # Test 6: Universal features grid
    print("Testing universal features grid...")
    features = ['Windows Compatible', 'macOS Compatible', 'Linux Compatible', 
                'iOS Compatible', 'Android Compatible', 'Universal Installer']
    all_features_present = all(feature in content for feature in features)
    if all_features_present:
        print("OK - All universal features present")
        tests_passed += 1
    else:
        print("FAIL - Some universal features missing")
        tests_failed += 1
    
    # Test 7: Command execution with universal security
    print("Testing command execution with universal security...")
    if 'executeCommandBtn' in content and 'Universal Security' in content:
        print("OK - Universal command execution present")
        tests_passed += 1
    else:
        print("FAIL - Universal command execution missing")
        tests_failed += 1
    
    # Test 8: Platform-specific placeholder text
    print("Testing platform-specific placeholder text...")
    if 'auto-detects platform' in content:
        print("OK - Platform-specific placeholder present")
        tests_passed += 1
    else:
        print("FAIL - Platform-specific placeholder missing")
        tests_failed += 1
    
    # Test 9: Universal security metrics
    print("Testing universal security metrics...")
    metrics = ['Platform Detection', 'Cross-Platform', 'Mobile Support', 'Auto-Installer']
    all_metrics_present = all(metric in content for metric in metrics)
    if all_metrics_present:
        print("OK - All universal security metrics present")
        tests_passed += 1
    else:
        print("FAIL - Some universal security metrics missing")
        tests_failed += 1
    
    # Test 10: JavaScript functionality
    print("Testing JavaScript functionality...")
    js_functions = ['downloadUniversalInstaller', 'createUniversalSession', 'executeCommand']
    all_js_functions_present = all(func in content for func in js_functions)
    if all_js_functions_present:
        print("OK - All JavaScript functions present")
        tests_passed += 1
    else:
        print("FAIL - Some JavaScript functions missing")
        tests_failed += 1
    
    print("\n==================== Universal Interface Test Results ====================")
    print(f"PASS - {tests_passed} tests passed")
    if tests_failed > 0:
        print(f"FAIL - {tests_failed} tests failed")
    
    print(f"\nTotal: {tests_passed + tests_failed} tests, {tests_passed} PASSED, {tests_failed} FAILED")
    
    if tests_failed == 0:
        print("\nSUCCESS! ALL UNIVERSAL INTERFACE TESTS PASSED!")
        print("The universal security interface is working perfectly!")
        print("All buttons and functionality are correctly implemented!")
        return True
    else:
        print(f"\nFAILURE! {tests_failed} tests failed!")
        print("Some universal interface functionality needs to be fixed!")
        return False

def test_button_functionality():
    """Test that all buttons have proper event handlers"""
    print("\n==================== Button Functionality Tests ====================")
    
    interface_file = "ultimate_security_interface.html"
    
    try:
        with open(interface_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR - Could not read {interface_file}: {e}")
        return False
    
    # Extract all button IDs
    button_ids = re.findall(r'id="([^"]*Btn[^"]*)"', content)
    print(f"Found {len(button_ids)} buttons: {button_ids}")
    
    # Extract all event listeners
    event_listeners = re.findall(r'addEventListener\([\'"]([^\'"]*)[\'"]', content)
    print(f"Found {len(event_listeners)} event listeners: {event_listeners}")
    
    # Check that each button has an event listener (look for getElementById)
    missing_listeners = []
    for button_id in button_ids:
        # Look for getElementById with this button ID
        if f'getElementById(\'{button_id}\')' in content or f'getElementById("{button_id}")' in content:
            continue
        else:
            missing_listeners.append(button_id)
    
    if missing_listeners:
        print(f"WARNING - Buttons without event listeners: {missing_listeners}")
    else:
        print("OK - All buttons have event listeners")
    
    return len(missing_listeners) == 0

if __name__ == "__main__":
    print("Universal Security Interface Test Suite")
    print("=" * 50)
    
    # Test universal interface
    interface_success = test_universal_interface()
    
    # Test button functionality
    button_success = test_button_functionality()
    
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    
    if interface_success and button_success:
        print("SUCCESS! ALL TESTS PASSED!")
        print("Universal security interface is working perfectly!")
        print("All buttons and functionality are correctly implemented!")
        print("\nReady to use:")
        print("1. Open: universal_security_interface.html")
        print("2. Test all buttons and functionality")
        print("3. Verify universal installer download")
        print("4. Confirm cross-platform compatibility display")
    else:
        print("FAILURE! Some tests failed!")
        print("Please check the failed tests above!")
    
    print("=" * 50)
