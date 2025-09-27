#!/usr/bin/env python3
"""
Universal Secure Installer
Cross-platform installer with ultimate security features
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import time
import hashlib
import secrets
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Platform detection
PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM == 'windows'
IS_MACOS = PLATFORM == 'darwin'
IS_LINUX = PLATFORM == 'linux'
IS_IOS = 'ios' in PLATFORM or 'darwin' in PLATFORM

class UniversalSecurityInstaller:
    """Universal secure installer for all platforms"""
    
    def __init__(self):
        self.platform = PLATFORM
        self.install_dir = self._get_install_directory()
        self.config = self._load_installer_config()
        
    def _get_install_directory(self) -> Path:
        """Get platform-specific install directory"""
        if IS_WINDOWS:
            # Windows: Program Files or user directory
            try:
                return Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')) / 'SquashPlotBridge'
            except:
                return Path.home() / 'SquashPlotBridge'
        elif IS_MACOS:
            # macOS: Applications or user directory
            return Path.home() / 'Applications' / 'SquashPlotBridge'
        elif IS_LINUX:
            # Linux: /opt or user directory
            return Path('/opt/SquashPlotBridge') if os.access('/', os.W_OK) else Path.home() / 'SquashPlotBridge'
        else:
            # Fallback
            return Path.home() / 'SquashPlotBridge'
    
    def _load_installer_config(self) -> Dict:
        """Load installer configuration"""
        return {
            'version': '1.0.0',
            'platform': self.platform,
            'security_level': 'universal',
            'features': [
                'cross_platform_compatibility',
                'mobile_support',
                'session_security',
                'command_sanitization',
                'rate_limiting',
                'encryption',
                'auto_startup',
                'web_interface'
            ],
            'required_packages': self._get_required_packages(),
            'optional_packages': self._get_optional_packages()
        }
    
    def _get_required_packages(self) -> List[str]:
        """Get platform-specific required packages"""
        base_packages = ['cryptography']
        
        if IS_WINDOWS:
            return base_packages + ['pywin32']
        elif IS_MACOS or IS_IOS:
            return base_packages + ['pyobjc-core']
        elif IS_LINUX:
            return base_packages + ['python3-dev']
        else:
            return base_packages
    
    def _get_optional_packages(self) -> List[str]:
        """Get platform-specific optional packages"""
        return ['numpy', 'requests', 'psutil']
    
    def install(self) -> bool:
        """Perform universal installation"""
        try:
            print(f"Universal Secure Installer - {self.platform.upper()}")
            print("=" * 50)
            
            # Check system requirements
            if not self._check_system_requirements():
                return False
            
            # Create install directory
            if not self._create_install_directory():
                return False
            
            # Install dependencies
            if not self._install_dependencies():
                return False
            
            # Copy files
            if not self._copy_files():
                return False
            
            # Setup security
            if not self._setup_security():
                return False
            
            # Configure startup
            if not self._configure_startup():
                return False
            
            # Create shortcuts
            if not self._create_shortcuts():
                return False
            
            # Finalize installation
            if not self._finalize_installation():
                return False
            
            print("\n" + "=" * 50)
            print("✅ INSTALLATION COMPLETE!")
            print("=" * 50)
            print(f"Platform: {self.platform.upper()}")
            print(f"Install Directory: {self.install_dir}")
            print("Security Level: UNIVERSAL")
            print("Features: Cross-platform, Mobile Support, Session Security")
            print("\n🚀 Ready to use:")
            print(f"1. Run: {self.install_dir / 'universal_secure_bridge.py'}")
            print(f"2. Open: {self.install_dir / 'universal_security_interface.html'}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def _check_system_requirements(self) -> bool:
        """Check system requirements"""
        print("\n🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check platform
        print(f"✅ Platform: {self.platform.upper()}")
        
        # Check write permissions
        if not os.access(self.install_dir.parent, os.W_OK):
            print(f"❌ No write permission to {self.install_dir.parent}")
            return False
        print("✅ Write permissions OK")
        
        # Check network
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("✅ Network connectivity OK")
        except:
            print("⚠️ No network connectivity (some features may be limited)")
        
        return True
    
    def _create_install_directory(self) -> bool:
        """Create installation directory"""
        print(f"\n📁 Creating install directory: {self.install_dir}")
        
        try:
            self.install_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.install_dir / 'logs').mkdir(exist_ok=True)
            (self.install_dir / 'config').mkdir(exist_ok=True)
            (self.install_dir / 'temp').mkdir(exist_ok=True)
            
            print("✅ Install directory created")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create install directory: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Install platform-specific dependencies"""
        print("\n📦 Installing dependencies...")
        
        try:
            # Install required packages
            for package in self.config['required_packages']:
                print(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {package} installed")
                else:
                    print(f"⚠️ {package} installation warning: {result.stderr}")
            
            # Install optional packages
            for package in self.config['optional_packages']:
                print(f"Installing optional {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {package} installed")
                else:
                    print(f"⚠️ {package} skipped (optional)")
            
            print("✅ Dependencies installed")
            return True
            
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def _copy_files(self) -> bool:
        """Copy bridge files to install directory"""
        print("\n📋 Copying files...")
        
        try:
            # Files to copy
            files_to_copy = [
                'universal_secure_bridge.py',
                'ultimate_security_interface.html',
                'working_ultimate_bridge.py',
                'test_working_bridge.py'
            ]
            
            current_dir = Path.cwd()
            
            for file_name in files_to_copy:
                source_file = current_dir / file_name
                if source_file.exists():
                    dest_file = self.install_dir / file_name
                    shutil.copy2(source_file, dest_file)
                    print(f"✅ Copied {file_name}")
                else:
                    print(f"⚠️ {file_name} not found (skipping)")
            
            # Copy documentation
            doc_files = [
                'FINAL_SECURITY_TEST_RESULTS.md',
                'IMPOSSIBLE_TO_BREAK_SECURITY.md',
                'ULTIMATE_SECURITY_SUMMARY.md'
            ]
            
            for doc_file in doc_files:
                source_file = current_dir / doc_file
                if source_file.exists():
                    dest_file = self.install_dir / doc_file
                    shutil.copy2(source_file, dest_file)
                    print(f"✅ Copied {doc_file}")
            
            print("✅ Files copied")
            return True
            
        except Exception as e:
            print(f"❌ Failed to copy files: {e}")
            return False
    
    def _setup_security(self) -> bool:
        """Setup security configuration"""
        print("\n🔒 Setting up security...")
        
        try:
            # Create security config
            security_config = {
                'platform': self.platform,
                'install_time': time.time(),
                'security_level': 'universal',
                'features': {
                    'session_security': True,
                    'command_sanitization': True,
                    'rate_limiting': True,
                    'encryption': True,
                    'mobile_support': True,
                    'cross_platform': True
                },
                'ports': {
                    'windows': 8443,
                    'darwin': 8444,
                    'linux': 8445,
                    'ios': 8446,
                    'android': 8447
                }
            }
            
            config_file = self.install_dir / 'config' / 'security_config.json'
            with open(config_file, 'w') as f:
                json.dump(security_config, f, indent=2)
            
            print("✅ Security configuration created")
            
            # Create startup script
            if not self._create_startup_script():
                return False
            
            print("✅ Security setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup security: {e}")
            return False
    
    def _create_startup_script(self) -> bool:
        """Create platform-specific startup script"""
        try:
            if IS_WINDOWS:
                return self._create_windows_startup()
            elif IS_MACOS:
                return self._create_macos_startup()
            elif IS_LINUX:
                return self._create_linux_startup()
            else:
                return self._create_generic_startup()
        except Exception as e:
            print(f"❌ Failed to create startup script: {e}")
            return False
    
    def _create_windows_startup(self) -> bool:
        """Create Windows startup script"""
        startup_script = f'''@echo off
cd /d "{self.install_dir}"
python universal_secure_bridge.py
pause
'''
        
        script_file = self.install_dir / 'start_bridge.bat'
        with open(script_file, 'w') as f:
            f.write(startup_script)
        
        # Create PowerShell script for advanced users
        ps_script = f'''# SquashPlot Universal Secure Bridge Startup Script
Set-Location "{self.install_dir}"
python universal_secure_bridge.py
'''
        
        ps_file = self.install_dir / 'start_bridge.ps1'
        with open(ps_file, 'w') as f:
            f.write(ps_script)
        
        print("✅ Windows startup scripts created")
        return True
    
    def _create_macos_startup(self) -> bool:
        """Create macOS startup script"""
        startup_script = f'''#!/bin/bash
cd "{self.install_dir}"
python3 universal_secure_bridge.py
'''
        
        script_file = self.install_dir / 'start_bridge.sh'
        with open(script_file, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        print("✅ macOS startup script created")
        return True
    
    def _create_linux_startup(self) -> bool:
        """Create Linux startup script"""
        startup_script = f'''#!/bin/bash
cd "{self.install_dir}"
python3 universal_secure_bridge.py
'''
        
        script_file = self.install_dir / 'start_bridge.sh'
        with open(script_file, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        # Create systemd service
        service_content = f'''[Unit]
Description=SquashPlot Universal Secure Bridge
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'root')}
WorkingDirectory={self.install_dir}
ExecStart={sys.executable} universal_secure_bridge.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
'''
        
        service_file = self.install_dir / 'squashplot-bridge.service'
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print("✅ Linux startup script and systemd service created")
        return True
    
    def _create_generic_startup(self) -> bool:
        """Create generic startup script"""
        startup_script = f'''#!/bin/bash
cd "{self.install_dir}"
python universal_secure_bridge.py
'''
        
        script_file = self.install_dir / 'start_bridge.sh'
        with open(script_file, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        try:
            os.chmod(script_file, 0o755)
        except:
            pass
        
        print("✅ Generic startup script created")
        return True
    
    def _configure_startup(self) -> bool:
        """Configure automatic startup"""
        print("\n🚀 Configuring startup...")
        
        try:
            if IS_WINDOWS:
                return self._configure_windows_startup()
            elif IS_MACOS:
                return self._configure_macos_startup()
            elif IS_LINUX:
                return self._configure_linux_startup()
            else:
                print("⚠️ Auto-startup not configured for this platform")
                return True
        except Exception as e:
            print(f"⚠️ Startup configuration warning: {e}")
            return True  # Non-critical
    
    def _configure_windows_startup(self) -> bool:
        """Configure Windows startup"""
        try:
            # Create registry entry for startup
            import winreg
            
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE
            )
            
            winreg.SetValueEx(
                key,
                "SquashPlotBridge",
                0,
                winreg.REG_SZ,
                str(self.install_dir / 'start_bridge.bat')
            )
            
            winreg.CloseKey(key)
            print("✅ Windows startup configured")
            return True
            
        except Exception as e:
            print(f"⚠️ Windows startup configuration failed: {e}")
            return True  # Non-critical
    
    def _configure_macos_startup(self) -> bool:
        """Configure macOS startup"""
        try:
            # Create LaunchAgent
            launch_agent_dir = Path.home() / 'Library' / 'LaunchAgents'
            launch_agent_dir.mkdir(exist_ok=True)
            
            plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.squashplot.bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{self.install_dir / 'universal_secure_bridge.py'}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>{self.install_dir}</string>
</dict>
</plist>
'''
            
            plist_file = launch_agent_dir / 'com.squashplot.bridge.plist'
            with open(plist_file, 'w') as f:
                f.write(plist_content)
            
            print("✅ macOS startup configured")
            return True
            
        except Exception as e:
            print(f"⚠️ macOS startup configuration failed: {e}")
            return True  # Non-critical
    
    def _configure_linux_startup(self) -> bool:
        """Configure Linux startup"""
        try:
            # Copy systemd service to system directory
            system_service_dir = Path('/etc/systemd/system')
            if system_service_dir.exists() and os.access(system_service_dir, os.W_OK):
                service_source = self.install_dir / 'squashplot-bridge.service'
                service_dest = system_service_dir / 'squashplot-bridge.service'
                
                shutil.copy2(service_source, service_dest)
                
                # Enable service
                subprocess.run(['systemctl', 'enable', 'squashplot-bridge'], check=True)
                print("✅ Linux startup configured (systemd)")
            else:
                print("⚠️ Linux startup configuration skipped (no permissions)")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Linux startup configuration failed: {e}")
            return True  # Non-critical
    
    def _create_shortcuts(self) -> bool:
        """Create platform-specific shortcuts"""
        print("\n🔗 Creating shortcuts...")
        
        try:
            if IS_WINDOWS:
                return self._create_windows_shortcuts()
            elif IS_MACOS:
                return self._create_macos_shortcuts()
            elif IS_LINUX:
                return self._create_linux_shortcuts()
            else:
                print("⚠️ Shortcuts not created for this platform")
                return True
        except Exception as e:
            print(f"⚠️ Shortcut creation warning: {e}")
            return True  # Non-critical
    
    def _create_windows_shortcuts(self) -> bool:
        """Create Windows shortcuts"""
        try:
            # Create desktop shortcut
            desktop = Path.home() / 'Desktop'
            if desktop.exists():
                shortcut_content = f'''[InternetShortcut]
URL=file:///{self.install_dir / 'ultimate_security_interface.html'}
IDList=
IconFile={self.install_dir / 'start_bridge.bat'}
IconIndex=0
'''
                shortcut_file = desktop / 'SquashPlot Bridge.url'
                with open(shortcut_file, 'w') as f:
                    f.write(shortcut_content)
                
                print("✅ Windows shortcuts created")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Windows shortcut creation failed: {e}")
            return True
    
    def _create_macos_shortcuts(self) -> bool:
        """Create macOS shortcuts"""
        try:
            # Create Applications folder entry
            applications = Path.home() / 'Applications'
            if applications.exists():
                app_bundle = applications / 'SquashPlot Bridge.app'
                app_bundle.mkdir(exist_ok=True)
                
                # Create Info.plist
                info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>start_bridge.sh</string>
    <key>CFBundleIdentifier</key>
    <string>com.squashplot.bridge</string>
    <key>CFBundleName</key>
    <string>SquashPlot Bridge</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
</dict>
</plist>
'''
                
                with open(app_bundle / 'Info.plist', 'w') as f:
                    f.write(info_plist)
                
                # Copy startup script
                shutil.copy2(self.install_dir / 'start_bridge.sh', app_bundle / 'start_bridge.sh')
                os.chmod(app_bundle / 'start_bridge.sh', 0o755)
                
                print("✅ macOS shortcuts created")
            
            return True
            
        except Exception as e:
            print(f"⚠️ macOS shortcut creation failed: {e}")
            return True
    
    def _create_linux_shortcuts(self) -> bool:
        """Create Linux shortcuts"""
        try:
            # Create desktop entry
            desktop = Path.home() / 'Desktop'
            if desktop.exists():
                desktop_entry = f'''[Desktop Entry]
Version=1.0
Type=Application
Name=SquashPlot Bridge
Comment=Universal Secure SquashPlot Bridge
Exec={self.install_dir / 'start_bridge.sh'}
Icon=applications-internet
Terminal=true
Categories=Network;Security;
'''
                
                desktop_file = desktop / 'SquashPlot Bridge.desktop'
                with open(desktop_file, 'w') as f:
                    f.write(desktop_entry)
                
                os.chmod(desktop_file, 0o755)
                print("✅ Linux shortcuts created")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Linux shortcut creation failed: {e}")
            return True
    
    def _finalize_installation(self) -> bool:
        """Finalize installation"""
        print("\n🎯 Finalizing installation...")
        
        try:
            # Create installation record
            install_record = {
                'install_time': time.time(),
                'platform': self.platform,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'install_directory': str(self.install_dir),
                'version': self.config['version'],
                'features': self.config['features']
            }
            
            record_file = self.install_dir / 'install_record.json'
            with open(record_file, 'w') as f:
                json.dump(install_record, f, indent=2)
            
            # Set permissions
            if not IS_WINDOWS:
                # Make scripts executable
                for script_file in self.install_dir.glob('*.sh'):
                    os.chmod(script_file, 0o755)
            
            print("✅ Installation finalized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to finalize installation: {e}")
            return False

def main():
    """Main installer entry point"""
    print("Universal Secure SquashPlot Bridge Installer")
    print("=" * 50)
    
    installer = UniversalSecurityInstaller()
    
    # Confirm installation
    print(f"Platform: {installer.platform.upper()}")
    print(f"Install Directory: {installer.install_dir}")
    print("\nThis will install:")
    print("- Universal Secure Bridge (cross-platform)")
    print("- Mobile/iOS compatibility")
    print("- Session-based security")
    print("- Command sanitization")
    print("- Rate limiting and encryption")
    print("- Auto-startup configuration")
    print("- Web interface")
    
    response = input("\nProceed with installation? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    # Perform installation
    if installer.install():
        print("\n🎉 Installation completed successfully!")
        print("Your SquashPlot Bridge is ready to use with universal security!")
    else:
        print("\n❌ Installation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
