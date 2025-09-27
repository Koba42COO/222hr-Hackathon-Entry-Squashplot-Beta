#!/usr/bin/env python3
"""
SquashPlot Bridge EXE Installer Generator
Creates a standalone executable installer for end users
"""

import os
import sys
import shutil
import subprocess
import platform
import tempfile
from pathlib import Path

class EXEInstallerGenerator:
    """Generates standalone executable installer using PyInstaller"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.output_dir = Path("executable_installer")
        self.installer_script = "SquashPlotBridgeInstaller.py"
        
    def check_pyinstaller(self):
        """Check if PyInstaller is installed"""
        try:
            import PyInstaller
            print(f"✓ PyInstaller {PyInstaller.__version__} found")
            return True
        except ImportError:
            print("⚠ PyInstaller not found. Installing...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
                print("✓ PyInstaller installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install PyInstaller: {e}")
                return False
    
    def create_installer_spec(self):
        """Create PyInstaller spec file for the installer"""
        spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['SquashPlotBridgeInstaller.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('SquashPlotSecureBridge.py', '.'),
        ('SquashPlotDeveloperBridge.py', '.'),
        ('SquashPlotSecureBridge.html', '.'),
        ('SquashPlotDeveloperBridge.html', '.'),
        ('PROFESSIONAL_SUMMARY.md', '.'),
        ('SIMPLE_SUMMARY.md', '.'),
        ('SECURITY_WARNING_README.md', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'cryptography',
        'numpy',
        'threading',
        'webbrowser',
        'http.server',
        'socketserver',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SquashPlotBridgeInstaller',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='installer_icon.ico' if os.path.exists('installer_icon.ico') else None,
)
'''
        
        spec_path = self.output_dir / "installer.spec"
        with open(spec_path, 'w') as f:
            f.write(spec_content)
        
        print(f"✓ Created PyInstaller spec file: {spec_path}")
        return spec_path
    
    def create_installer_icon(self):
        """Create a simple installer icon"""
        # This would normally be a proper .ico file
        # For now, we'll skip the icon
        print("⚠ No installer icon provided (optional)")
        return None
    
    def build_executable(self):
        """Build the executable installer"""
        print("🔨 Building executable installer...")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Check PyInstaller
        if not self.check_pyinstaller():
            return False
        
        # Create spec file
        spec_path = self.create_installer_spec()
        
        # Build executable
        try:
            cmd = [
                sys.executable, "-m", "PyInstaller",
                "--clean",
                "--noconfirm",
                "--distpath", str(self.output_dir / "dist"),
                "--workpath", str(self.output_dir / "build"),
                str(spec_path)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=Path("."), capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Executable installer built successfully!")
                
                # Find the created executable
                dist_dir = self.output_dir / "dist"
                if dist_dir.exists():
                    exe_files = list(dist_dir.glob("SquashPlotBridgeInstaller*"))
                    if exe_files:
                        exe_path = exe_files[0]
                        print(f"✓ Executable created: {exe_path}")
                        return exe_path
                
                print("⚠ Executable built but location unclear")
                return True
            else:
                print(f"❌ Build failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False
    
    def create_installer_package(self):
        """Create complete installer package"""
        print("📦 Creating installer package...")
        
        # Build executable
        exe_path = self.build_executable()
        if not exe_path:
            return False
        
        # Create package directory
        package_dir = self.output_dir / "package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy executable
        if isinstance(exe_path, Path):
            package_exe = package_dir / exe_path.name
            shutil.copy2(exe_path, package_exe)
            print(f"✓ Copied executable to package: {package_exe}")
        
        # Create user documentation
        self.create_user_documentation(package_dir)
        
        # Create installer manifest
        self.create_installer_manifest(package_dir)
        
        print(f"✓ Installer package created in: {package_dir}")
        return package_dir
    
    def create_user_documentation(self, package_dir):
        """Create user-friendly documentation"""
        
        # User README
        user_readme = '''# SquashPlot Bridge - Easy Installation

## 🚀 Quick Start

### For End Users (No Technical Knowledge Required):

1. **Download** the installer: `SquashPlotBridgeInstaller.exe`
2. **Double-click** the installer file
3. **Follow** the installation wizard
4. **Click** "Start SquashPlot Bridge" when installation completes
5. **Done!** SquashPlot Bridge will start automatically

### What You Get:
- ✅ **Secure Bridge Application** - Professional-grade security
- ✅ **Easy Web Interface** - Simple point-and-click interface  
- ✅ **Auto-Startup** - Starts automatically when you turn on your computer
- ✅ **Cross-Platform** - Works on Windows, Mac, and Linux
- ✅ **Hello World Demo** - Proves the system works safely

### System Requirements:
- **Windows**: Windows 10 or later
- **Mac**: macOS 10.14 or later  
- **Linux**: Ubuntu 18.04 or similar
- **Memory**: 100 MB available RAM
- **Disk Space**: 50 MB available space

## 🛡️ Security Features

Your SquashPlot Bridge includes:
- **Quantum-Safe Encryption** - Military-grade security
- **AI Threat Detection** - Intelligent threat monitoring
- **Behavioral Biometrics** - Advanced user authentication
- **Safe Command Execution** - Only approved commands allowed

## 🗑️ Uninstalling

To remove SquashPlot Bridge:
1. **Go to** the SquashPlot website
2. **Click** the uninstall button
3. **Confirm** the uninstall action
4. **Restart** your computer

The uninstaller will remove all files and settings automatically.

## 📞 Support

- **Website**: [SquashPlot Website]
- **Documentation**: See files in installation folder
- **Security Info**: Read SECURITY_WARNING_README.md

## ⚠️ Important Notes

- This is a **secure application** for legitimate use only
- **Do not download** from unofficial sources
- **Always verify** you're using the official installer
- **Contact support** if you have any questions

---

**SquashPlot Bridge** - Professional Secure Bridge Application  
**Version**: 2.0.0-Secure  
**Status**: Production Ready
'''
        
        readme_path = package_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(user_readme)
        
        print(f"✓ Created user documentation: {readme_path}")
        
        # User Quick Start Guide
        quick_start = '''SquashPlot Bridge - Quick Start Guide
=====================================

STEP 1: INSTALLATION
-------------------
1. Double-click "SquashPlotBridgeInstaller.exe"
2. Wait for installation to complete
3. Click "Start SquashPlot Bridge" button
4. Your browser will open automatically

STEP 2: USING THE BRIDGE
-----------------------
1. The bridge opens at: http://127.0.0.1:8080
2. Click "Execute Demo Script" to test
3. Notepad will open with "Hello World!" message
4. This proves the bridge is working correctly

STEP 3: AUTOMATIC STARTUP
-------------------------
- The bridge will start automatically when you turn on your computer
- No need to run the installer again
- The bridge runs in the background

STEP 4: UNINSTALLING (If Needed)
-------------------------------
1. Go to the SquashPlot website
2. Click the uninstall button
3. Follow the uninstall instructions
4. Restart your computer

TROUBLESHOOTING
---------------
- If the bridge doesn't start: Restart your computer
- If the browser doesn't open: Go to http://127.0.0.1:8080 manually
- If you get errors: Check the installation log in the completion window

SUPPORT
-------
- Read README.txt for detailed information
- Check the SquashPlot website for updates
- Contact support if you need help

That's it! SquashPlot Bridge is now installed and ready to use.
'''
        
        quick_start_path = package_dir / "QUICK_START.txt"
        with open(quick_start_path, 'w', encoding='utf-8') as f:
            f.write(quick_start)
        
        print(f"✓ Created quick start guide: {quick_start_path}")
    
    def create_installer_manifest(self, package_dir):
        """Create installer manifest with file information"""
        manifest = {
            "installer": {
                "name": "SquashPlot Bridge Installer",
                "version": "2.0.0",
                "type": "executable",
                "platform": self.system,
                "created": "2025-09-26",
                "description": "Professional SquashPlot Bridge installation package"
            },
            "files": [
                {
                    "name": "SquashPlotBridgeInstaller.exe",
                    "type": "executable",
                    "description": "Main installer executable"
                },
                {
                    "name": "README.txt",
                    "type": "documentation", 
                    "description": "User documentation"
                },
                {
                    "name": "QUICK_START.txt",
                    "type": "guide",
                    "description": "Quick start guide"
                }
            ],
            "requirements": {
                "os": ["Windows 10+", "macOS 10.14+", "Ubuntu 18.04+"],
                "memory": "100 MB RAM",
                "disk": "50 MB space",
                "network": "Local network access"
            }
        }
        
        manifest_path = package_dir / "installer_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Created installer manifest: {manifest_path}")

def main():
    """Main function to create executable installer"""
    print("SquashPlot Bridge EXE Installer Generator")
    print("=" * 50)
    
    generator = EXEInstallerGenerator()
    
    try:
        package_dir = generator.create_installer_package()
        if package_dir:
            print("\n" + "=" * 50)
            print("🎉 EXECUTABLE INSTALLER CREATED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📁 Package location: {package_dir}")
            print("📦 Files included:")
            for file in package_dir.iterdir():
                print(f"   - {file.name}")
            print("\n✅ Ready for distribution to end users!")
        else:
            print("\n❌ Failed to create executable installer")
            
    except Exception as e:
        print(f"\n❌ Error creating installer: {e}")

if __name__ == "__main__":
    main()



