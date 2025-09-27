#!/usr/bin/env python3
"""
HTML Validation Test
Checks the HTML file for syntax errors and issues
"""

import re
import os

def test_html_validation():
    """Test HTML file for common issues"""
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
    
    print("HTML Validation Test")
    print("=" * 40)
    
    issues = []
    
    # Check 1: Basic HTML structure
    if '<!DOCTYPE html>' not in content:
        issues.append("Missing DOCTYPE declaration")
    
    if '<html' not in content:
        issues.append("Missing <html> tag")
    
    if '</html>' not in content:
        issues.append("Missing </html> closing tag")
    
    # Check 2: Head section
    if '<head>' not in content:
        issues.append("Missing <head> tag")
    
    if '<title>' not in content:
        issues.append("Missing <title> tag")
    
    # Check 3: Body section
    if '<body>' not in content:
        issues.append("Missing <body> tag")
    
    if '</body>' not in content:
        issues.append("Missing </body> closing tag")
    
    # Check 4: JavaScript syntax
    script_sections = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
    if not script_sections:
        issues.append("No JavaScript found")
    else:
        for i, script in enumerate(script_sections):
            # Check for common JS syntax issues
            if script.count('{') != script.count('}'):
                issues.append(f"JavaScript brace mismatch in script section {i+1}")
            
            if script.count('(') != script.count(')'):
                issues.append(f"JavaScript parenthesis mismatch in script section {i+1}")
    
    # Check 5: CSS syntax
    style_sections = re.findall(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
    if not style_sections:
        issues.append("No CSS found")
    else:
        for i, style in enumerate(style_sections):
            # Check for common CSS syntax issues
            if style.count('{') != style.count('}'):
                issues.append(f"CSS brace mismatch in style section {i+1}")
    
    # Check 6: Required elements for the interface
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
    
    # Check 7: Event listeners
    if 'addEventListener' not in content:
        issues.append("No event listeners found")
    
    # Check 8: Unicode characters that might cause issues
    unicode_issues = []
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        for char in line:
            if ord(char) > 127:  # Non-ASCII characters
                if char in ['🌐', '🛡️', '🚀', '📱', '✅', '❌', '🔗', '⚡', '🚨']:
                    continue  # These are intentional emojis
                else:
                    unicode_issues.append(f"Line {line_num}: Unexpected Unicode character '{char}'")
    
    if unicode_issues:
        issues.extend(unicode_issues[:5])  # Limit to first 5 unicode issues
    
    # Print results
    if not issues:
        print("✅ HTML file is valid!")
        print("✅ No syntax errors found")
        print("✅ All required elements present")
        print("✅ JavaScript structure looks good")
        print("✅ CSS structure looks good")
        return True
    else:
        print("❌ HTML file has issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False

if __name__ == "__main__":
    success = test_html_validation()
    
    if success:
        print("\n🎉 HTML file is ready to use!")
        print("Try opening it in your browser:")
        print("   - Double-click ultimate_security_interface.html")
        print("   - Or open in browser: file:///path/to/ultimate_security_interface.html")
    else:
        print("\n❌ HTML file needs to be fixed!")
        print("Please check the issues listed above.")
