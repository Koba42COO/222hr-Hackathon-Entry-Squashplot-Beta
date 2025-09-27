#!/usr/bin/env python3
"""
Simple HTML Test - Windows Compatible
Checks the HTML file for basic issues
"""

import re
import os

def test_html_simple():
    """Test HTML file for basic issues"""
    html_file = "ultimate_security_interface.html"
    
    if not os.path.exists(html_file):
        print(f"ERROR: {html_file} not found")
        return False
    
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Could not read {html_file}: {e}")
        return False
    
    print("HTML Simple Test")
    print("=" * 40)
    
    issues = []
    
    # Check 1: Basic HTML structure
    if '<!DOCTYPE html>' not in content:
        issues.append("Missing DOCTYPE declaration")
    else:
        print("OK - DOCTYPE found")
    
    if '<html' not in content:
        issues.append("Missing <html> tag")
    else:
        print("OK - <html> tag found")
    
    if '</html>' not in content:
        issues.append("Missing </html> closing tag")
    else:
        print("OK - </html> closing tag found")
    
    # Check 2: Required elements
    required_elements = [
        'downloadUniversalBtn',
        'createUniversalSessionBtn', 
        'executeCommandBtn',
        'commandInput',
        'commandOutput'
    ]
    
    for element in required_elements:
        if f'id="{element}"' not in content:
            issues.append(f"Missing required element: {element}")
        else:
            print(f"OK - {element} found")
    
    # Check 3: JavaScript class
    if 'UniversalSecurityInterface' not in content:
        issues.append("Missing UniversalSecurityInterface class")
    else:
        print("OK - UniversalSecurityInterface class found")
    
    # Check 4: Event listeners
    if 'addEventListener' not in content:
        issues.append("No event listeners found")
    else:
        print("OK - Event listeners found")
    
    # Check 5: Bridge connectivity
    if 'connectToBridge' not in content:
        issues.append("Missing connectToBridge function")
    else:
        print("OK - connectToBridge function found")
    
    if 'sendCommandToBridge' not in content:
        issues.append("Missing sendCommandToBridge function")
    else:
        print("OK - sendCommandToBridge function found")
    
    # Print results
    print("\n" + "=" * 40)
    if not issues:
        print("SUCCESS - HTML file is valid!")
        print("No syntax errors found")
        print("All required elements present")
        return True
    else:
        print("FAILURE - HTML file has issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False

if __name__ == "__main__":
    success = test_html_simple()
    
    if success:
        print("\nHTML file is ready to use!")
        print("Try opening it in your browser:")
        print("   - Double-click ultimate_security_interface.html")
    else:
        print("\nHTML file needs to be fixed!")
        print("Please check the issues listed above.")
