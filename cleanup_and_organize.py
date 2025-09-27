#!/usr/bin/env python3
"""
Cleanup and Organize Script
Moves unnecessary files to archive and organizes the professional structure
"""

import os
import shutil
from pathlib import Path

def cleanup_and_organize():
    """Clean up and organize all files"""
    print("SquashPlot Bridge - Cleanup and Organization")
    print("=" * 60)
    
    # Create archive directories
    archives = [
        'archive_old_interfaces',
        'archive_old_bridges', 
        'archive_old_tests',
        'archive_old_docs'
    ]
    
    for archive in archives:
        os.makedirs(archive, exist_ok=True)
        print(f"Created archive directory: {archive}")
    
    # Files to archive - Old interfaces
    old_interfaces = [
        'ultimate_security_interface.html',
        'developer_security_interface.html',
        'secure_developer_interface.html',
        'working_developer_interface.html',
        'clean_developer_interface.html',
        'simple_security_interface.html',
        'bridge_connected_interface.html'
    ]
    
    # Files to archive - Old bridges
    old_bridges = [
        'bridge_web_server.py',
        'robust_bridge_server.py',
        'simple_bridge_server.py',
        'ultimate_impenetrable_bridge.py',
        'ultimate_secure_bridge.py',
        'working_ultimate_bridge.py',
        'secure_bridge_app.py',
        'enterprise_secure_bridge.py',
        'military_grade_secure_bridge.py'
    ]
    
    # Files to archive - Old tests
    old_tests = [
        'test_ultimate_security.py',
        'simple_security_test.py',
        'windows_security_test.py',
        'test_working_bridge.py',
        'test_bridge_connectivity.py',
        'test_universal_compatibility.py',
        'test_universal_interface.py',
        'test_cross_platform_compatibility.py',
        'ultimate_compatibility_test.py',
        'ultimate_interface_test.py',
        'test_html_validation.py',
        'test_html_simple.py',
        'test_complete_system.py'
    ]
    
    # Files to archive - Old docs
    old_docs = [
        'IMPOSSIBLE_TO_BREAK_SECURITY.md',
        'ULTIMATE_SECURITY_SUMMARY.md',
        'ULTIMATE_SECURITY_ACHIEVEMENT.md',
        'WORLD_CLASS_SECURITY_AUDIT.md',
        'WORLD_CLASS_SECURITY_IMPLEMENTATION.md',
        'SECURITY_SESSION_IMPLEMENTATION.md',
        'ULTIMATE_CROSS_PLATFORM_SUMMARY.md',
        'ULTIMATE_DOWNLOAD_INSTRUCTIONS.md',
        'ULTIMATE_CLEAN_STRUCTURE_SUMMARY.md',
        'FINAL_CLEAN_STRUCTURE_COMPLETE.md',
        'START_COMPLETE_SYSTEM.md',
        'FINAL_SECURITY_TEST_RESULTS.md'
    ]
    
    # Archive files
    def archive_files(file_list, archive_dir):
        for file in file_list:
            if os.path.exists(file):
                try:
                    shutil.move(file, archive_dir)
                    print(f"Moved {file} to {archive_dir}")
                except Exception as e:
                    print(f"Failed to move {file}: {e}")
    
    print("\nArchiving old interfaces...")
    archive_files(old_interfaces, 'archive_old_interfaces')
    
    print("\nArchiving old bridges...")
    archive_files(old_bridges, 'archive_old_bridges')
    
    print("\nArchiving old tests...")
    archive_files(old_tests, 'archive_old_tests')
    
    print("\nArchiving old docs...")
    archive_files(old_docs, 'archive_old_docs')
    
    # Create final structure
    print("\nCreating final structure...")
    
    # Keep essential files in root
    essential_files = [
        'squashplot_bridge_professional',
        'bridge_installer.py',
        'README.md'
    ]
    
    # Create README for the main directory
    readme_content = """# SquashPlot Bridge - Professional System

## Quick Start

### Professional Version (Recommended)
```bash
cd squashplot_bridge_professional
python SquashPlotBridge.py
```

Then open your browser to: `http://127.0.0.1:8080`

## What You Get

- **Professional Web Interface**: Clean, modern interface
- **Secure Command Execution**: Whitelist-protected commands
- **Cross-Platform Support**: Works on Windows, Mac, Linux
- **Hello World Demo**: Proves the system works by opening Notepad/TextEdit

## Documentation

- **Professional Summary**: `squashplot_bridge_professional/PROFESSIONAL_SUMMARY.md`
- **Simple Summary**: `squashplot_bridge_professional/SIMPLE_SUMMARY.md`

## Architecture

- **SquashPlotBridge.py**: Main server application
- **SquashPlotBridge.html**: Professional web interface
- **Security**: Whitelist-based command validation

## Security Features

- Only approved commands can execute
- Server-side validation
- Cross-platform security controls
- Safe demonstration script

## Perfect For

- Demonstrations
- Development
- Education
- Prototyping secure bridges

---

**Status**: Production Ready  
**Version**: 1.0.0
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("Created main README.md")
    
    # Create archive README
    archive_readme = """# Archive Directory

This directory contains archived versions of SquashPlot Bridge development files.

## Contents

- **archive_old_interfaces/**: Previous web interface versions
- **archive_old_bridges/**: Previous bridge server versions  
- **archive_old_tests/**: Previous test files
- **archive_old_docs/**: Previous documentation

## Current Active Version

The current active version is in: `squashplot_bridge_professional/`

All archived files are kept for reference but are not part of the current system.
"""
    
    with open('ARCHIVE_README.md', 'w') as f:
        f.write(archive_readme)
    
    print("Created ARCHIVE_README.md")
    
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE!")
    print("=" * 60)
    print("✅ Professional structure created")
    print("✅ Old files archived")
    print("✅ Documentation organized")
    print("✅ README files created")
    print("\nCurrent active system: squashplot_bridge_professional/")
    print("Start with: cd squashplot_bridge_professional && python SquashPlotBridge.py")

if __name__ == "__main__":
    cleanup_and_organize()
