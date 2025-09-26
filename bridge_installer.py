#!/usr/bin/env python3
"""
SquashPlot Bridge App Installer
Automated installation with startup integration for all operating systems
"""

import os
import sys
import json
import shutil
import subprocess
import platform
from pathlib import Path
from datetime import datetime

class BridgeInstaller:
    """Cross-platform bridge app installer with startup integration"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.install_dir = self._get_install_directory()
        self.bridge_app = "secure_bridge_app.py"
        self.config_file = "bridge_config.json"
        self.startup_script = None
        
    def _get_install_directory(self):
        """Get appropriate installation directory for the OS"""
        if self.system == "windows":
            return Path.home() / "AppData" / "Local" / "SquashPlot" / "Bridge"
        elif self.system == "darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / "SquashPlot" / "Bridge"
        else:  # Linux
            return Path.home() / ".local" / "share" / "squashplot" / "bridge"
    
    def safe_print(self, message: str):
        """Print message with safe encoding for all OS"""
        try:
            print(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    def check_dependencies(self):
        """Check for required dependencies"""
        self.safe_print("Checking system dependencies...")
        
        # Check Python installation
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.safe_print(f"[ERROR] Python 3.8+ required. Found: {python_version.major}.{python_version.minor}")
            self.safe_print("Please install Python 3.8 or later from https://python.org")
            return False
        
        self.safe_print(f"[OK] Python {python_version.major}.{python_version.minor} detected")
        
        # Check required Python packages
        required_packages = ["cryptography", "requests"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.safe_print(f"[OK] {package} package available")
            except ImportError:
                missing_packages.append(package)
                self.safe_print(f"[MISSING] {package} package not found")
        
        if missing_packages:
            self.safe_print(f"[INFO] Installing missing packages: {', '.join(missing_packages)}")
            try:
                import subprocess
                for package in missing_packages:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True)
                    self.safe_print(f"[OK] Installed {package}")
            except subprocess.CalledProcessError as e:
                self.safe_print(f"[ERROR] Failed to install packages: {e}")
                self.safe_print("Please install manually: pip install " + " ".join(missing_packages))
                return False
        
        return True
    
    def run_system_scan(self):
        """Run comprehensive system scan before installation"""
        self.safe_print("Running system scan...")
        
        try:
            # Import and run system scanner
            from system_scanner import SystemScanner
            scanner = SystemScanner()
            scan_results = scanner.run_comprehensive_scan()
            
            # Check if system is ready
            critical_issues = len([r for r in scan_results.get("recommendations", []) if r.get("type") == "critical"])
            
            if critical_issues > 0:
                self.safe_print(f"[WARNING] {critical_issues} critical issues found")
                response = input("Continue with installation despite issues? (yes/no): ")
                if response.lower() != 'yes':
                    self.safe_print("Installation cancelled due to system issues.")
                    return False
            
            return True
            
        except ImportError:
            self.safe_print("[WARNING] System scanner not available - skipping scan")
            return True
        except Exception as e:
            self.safe_print(f"[WARNING] System scan failed: {str(e)}")
            return True
    
    def install_bridge_app(self):
        """Install bridge app with startup integration"""
        self.safe_print("SquashPlot Bridge App Installer")
        self.safe_print("=" * 50)
        
        try:
            # Run system scan first
            if not self.run_system_scan():
                return False
            
            # Check dependencies
            if not self.check_dependencies():
                self.safe_print("[ERROR] Dependency check failed. Please install required dependencies.")
                return False
            
            # Create installation directory
            self.install_dir.mkdir(parents=True, exist_ok=True)
            self.safe_print(f"[OK] Created installation directory: {self.install_dir}")
            
            # Copy bridge app
            if os.path.exists(self.bridge_app):
                shutil.copy2(self.bridge_app, self.install_dir / self.bridge_app)
                self.safe_print(f"[OK] Installed bridge app")
            else:
                self.safe_print(f"[ERROR] Bridge app not found: {self.bridge_app}")
                return False
            
            # Create configuration
            self._create_bridge_config()
            self.safe_print(f"[OK] Created bridge configuration")
            
            # Create startup script
            self._create_startup_script()
            self.safe_print(f"[OK] Created startup script")
            
            # Install startup integration
            self._install_startup_integration()
            self.safe_print(f"[OK] Added to system startup")
            
            # Create uninstaller
            self._create_uninstaller()
            self.safe_print(f"[OK] Created uninstaller")
            
            # Test installation
            if self._test_installation():
                self.safe_print(f"[OK] Installation test passed")
            else:
                self.safe_print(f"[WARNING] Installation test failed")
            
            self.safe_print(f"\n[SUCCESS] Bridge app installed successfully!")
            self.safe_print(f"Installation directory: {self.install_dir}")
            self.safe_print(f"Bridge will start automatically on system boot")
            self.safe_print(f"To uninstall, run: python uninstall_bridge.py")
            
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Installation failed: {str(e)}")
            return False
    
    def _create_bridge_config(self):
        """Create bridge configuration file"""
        config = {
            "bridge_settings": {
                "host": "127.0.0.1",
                "port": 8443,
                "quiet_mode": True,
                "auto_start": True,
                "log_level": "INFO",
                "max_connections": 5,
                "timeout": 30
            },
            "security_settings": {
                "authentication_required": True,
                "whitelist_commands": [
                    "squashplot --help",
                    "squashplot --version", 
                    "squashplot --status",
                    "squashplot --compress",
                    "squashplot --decompress"
                ],
                "dangerous_patterns": [
                    "rm -rf",
                    "sudo",
                    "chmod 777",
                    "format",
                    "del /f"
                ]
            },
            "startup_settings": {
                "start_on_boot": True,
                "minimize_to_tray": True,
                "show_notifications": True,
                "auto_connect": True
            }
        }
        
        config_path = self.install_dir / self.config_file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def _create_startup_script(self):
        """Create OS-specific startup script"""
        if self.system == "windows":
            self._create_windows_startup()
        elif self.system == "darwin":  # macOS
            self._create_macos_startup()
        else:  # Linux
            self._create_linux_startup()
    
    def _create_windows_startup(self):
        """Create Windows startup script and registry entry"""
        # Create batch file for startup
        startup_script = f'''@echo off
cd /d "{self.install_dir}"
python {self.bridge_app} --quiet --config {self.config_file}
'''
        
        startup_file = self.install_dir / "start_bridge.bat"
        with open(startup_file, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # Create VBS script to run silently
        vbs_script = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "{startup_file}", 0, False
'''
        
        vbs_file = self.install_dir / "start_bridge_silent.vbs"
        with open(vbs_file, 'w', encoding='utf-8') as f:
            f.write(vbs_script)
        
        self.startup_script = vbs_file
        
        # Add to Windows startup registry
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                               r"Software\Microsoft\Windows\CurrentVersion\Run", 
                               0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, "SquashPlotBridge", 0, winreg.REG_SZ, str(vbs_file))
            winreg.CloseKey(key)
            self.safe_print(f"[OK] Added to Windows startup registry")
        except Exception as e:
            self.safe_print(f"[WARNING] Could not add to Windows startup: {str(e)}")
    
    def _create_macos_startup(self):
        """Create macOS startup script and LaunchAgent"""
        # Create startup script
        startup_script = f'''#!/bin/bash
cd "{self.install_dir}"
python3 {self.bridge_app} --quiet --config {self.config_file} &
'''
        
        startup_file = self.install_dir / "start_bridge.sh"
        with open(startup_file, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(startup_file, 0o755)
        
        # Create LaunchAgent plist
        plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.squashplot.bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>{startup_file}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{self.install_dir}/bridge.log</string>
    <key>StandardErrorPath</key>
    <string>{self.install_dir}/bridge_error.log</string>
</dict>
</plist>'''
        
        plist_file = Path.home() / "Library" / "LaunchAgents" / "com.squashplot.bridge.plist"
        plist_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(plist_file, 'w', encoding='utf-8') as f:
            f.write(plist_content)
        
        self.startup_script = startup_file
        
        # Load LaunchAgent
        try:
            subprocess.run(["launchctl", "load", str(plist_file)], check=True)
            self.safe_print(f"[OK] Added to macOS LaunchAgents")
        except Exception as e:
            self.safe_print(f"[WARNING] Could not load LaunchAgent: {str(e)}")
    
    def _create_linux_startup(self):
        """Create Linux startup script and systemd service"""
        # Create startup script
        startup_script = f'''#!/bin/bash
cd "{self.install_dir}"
python3 {self.bridge_app} --quiet --config {self.config_file} &
'''
        
        startup_file = self.install_dir / "start_bridge.sh"
        with open(startup_file, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(startup_file, 0o755)
        
        # Create systemd service
        service_content = f'''[Unit]
Description=SquashPlot Bridge Service
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'squashplot')}
WorkingDirectory={self.install_dir}
ExecStart={startup_file}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
        
        service_file = Path.home() / ".config" / "systemd" / "user" / "squashplot-bridge.service"
        service_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(service_file, 'w', encoding='utf-8') as f:
            f.write(service_content)
        
        self.startup_script = startup_file
        
        # Enable and start service
        try:
            subprocess.run(["systemctl", "--user", "enable", "squashplot-bridge.service"], check=True)
            subprocess.run(["systemctl", "--user", "start", "squashplot-bridge.service"], check=True)
            self.safe_print(f"[OK] Added to Linux systemd services")
        except Exception as e:
            self.safe_print(f"[WARNING] Could not enable systemd service: {str(e)}")
    
    def _install_startup_integration(self):
        """Install startup integration for the current OS"""
        if self.system == "windows":
            self._install_windows_startup()
        elif self.system == "darwin":
            self._install_macos_startup()
        else:
            self._install_linux_startup()
    
    def _install_windows_startup(self):
        """Install Windows startup integration"""
        # Already handled in _create_windows_startup
        pass
    
    def _install_macos_startup(self):
        """Install macOS startup integration"""
        # Already handled in _create_macos_startup
        pass
    
    def _install_linux_startup(self):
        """Install Linux startup integration"""
        # Already handled in _create_linux_startup
        pass
    
    def _create_uninstaller(self):
        """Create uninstaller script"""
        uninstaller_content = f'''#!/usr/bin/env python3
"""
SquashPlot Bridge App Uninstaller
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def uninstall_bridge():
    print("SquashPlot Bridge App Uninstaller")
    print("=" * 40)
    
    system = platform.system().lower()
    install_dir = Path.home() / "AppData" / "Local" / "SquashPlot" / "Bridge" if system == "windows" else Path.home() / "Library" / "Application Support" / "SquashPlot" / "Bridge" if system == "darwin" else Path.home() / ".local" / "share" / "squashplot" / "bridge"
    
    try:
        # Stop service
        if system == "windows":
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
        elif system == "darwin":
            subprocess.run(["launchctl", "unload", str(Path.home() / "Library" / "LaunchAgents" / "com.squashplot.bridge.plist")], capture_output=True)
        else:
            subprocess.run(["systemctl", "--user", "stop", "squashplot-bridge.service"], capture_output=True)
        
        print("[OK] Stopped bridge service")
        
        # Remove startup entries
        if system == "windows":
            import winreg
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\\Microsoft\\Windows\\CurrentVersion\\Run", 0, winreg.KEY_SET_VALUE)
                winreg.DeleteValue(key, "SquashPlotBridge")
                winreg.CloseKey(key)
                print("[OK] Removed from Windows startup")
            except:
                pass
        elif system == "darwin":
            plist_file = Path.home() / "Library" / "LaunchAgents" / "com.squashplot.bridge.plist"
            if plist_file.exists():
                plist_file.unlink()
                print("[OK] Removed from macOS LaunchAgents")
        else:
            service_file = Path.home() / ".config" / "systemd" / "user" / "squashplot-bridge.service"
            if service_file.exists():
                service_file.unlink()
                subprocess.run(["systemctl", "--user", "disable", "squashplot-bridge.service"], capture_output=True)
                print("[OK] Removed from Linux systemd services")
        
        # Remove installation directory
        if install_dir.exists():
            import shutil
            shutil.rmtree(install_dir)
            print(f"[OK] Removed installation directory: {{install_dir}}")
        
        print("\\n[SUCCESS] Bridge app uninstalled successfully!")
        
    except Exception as e:
        print(f"[ERROR] Uninstall failed: {{str(e)}}")

if __name__ == "__main__":
    uninstall_bridge()
'''
        
        uninstaller_file = self.install_dir / "uninstall_bridge.py"
        with open(uninstaller_file, 'w', encoding='utf-8') as f:
            f.write(uninstaller_content)
        
        # Copy uninstaller to a more accessible location
        accessible_uninstaller = Path.home() / "Desktop" / "Uninstall_SquashPlot_Bridge.py"
        shutil.copy2(uninstaller_file, accessible_uninstaller)
    
    def _test_installation(self):
        """Test the installation"""
        try:
            # Check if files exist
            bridge_file = self.install_dir / self.bridge_app
            config_file = self.install_dir / self.config_file
            
            if not bridge_file.exists() or not config_file.exists():
                return False
            
            # Test configuration
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if "bridge_settings" not in config:
                return False
            
            return True
            
        except Exception:
            return False
    
    def start_bridge_service(self):
        """Start the bridge service immediately"""
        try:
            if self.system == "windows":
                subprocess.Popen([sys.executable, str(self.install_dir / self.bridge_app), "--quiet", "--config", str(self.install_dir / self.config_file)], 
                               cwd=str(self.install_dir), creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen([sys.executable, str(self.install_dir / self.bridge_app), "--quiet", "--config", str(self.install_dir / self.config_file)], 
                               cwd=str(self.install_dir))
            
            self.safe_print(f"[OK] Bridge service started")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to start bridge service: {str(e)}")
            return False

def main():
    """Main installation entry point"""
    print("SquashPlot Bridge App Installer")
    print("=" * 50)
    
    installer = BridgeInstaller()
    
    # Check if already installed
    if installer.install_dir.exists() and (installer.install_dir / "secure_bridge_app.py").exists():
        response = input("Bridge app already installed. Reinstall? (yes/no): ")
        if response.lower() != 'yes':
            print("Installation cancelled.")
            return
    
    # Run installation
    if installer.install_bridge_app():
        print("\n" + "=" * 50)
        print("INSTALLATION COMPLETE")
        print("=" * 50)
        print("The SquashPlot Bridge App has been installed and configured.")
        print("It will start automatically when you restart your computer.")
        print("")
        print("To start the bridge service now, run:")
        print(f"  cd {installer.install_dir}")
        print(f"  python {installer.bridge_app}")
        print("")
        print("To uninstall later, run:")
        print("  python uninstall_bridge.py")
        print("")
        print("The bridge will listen quietly on port 8443 for dashboard commands.")
    else:
        print("\n[ERROR] Installation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
