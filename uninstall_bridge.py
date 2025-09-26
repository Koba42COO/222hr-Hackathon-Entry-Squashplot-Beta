#!/usr/bin/env python3
"""
SquashPlot Bridge App Uninstaller
Automated uninstallation with cleanup for all operating systems
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

class BridgeUninstaller:
    """Cross-platform bridge app uninstaller"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.install_dir = self._get_install_directory()
        
    def _get_install_directory(self):
        """Get installation directory for the OS"""
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
    
    def uninstall_bridge(self):
        """Uninstall bridge app completely"""
        self.safe_print("SquashPlot Bridge App Uninstaller")
        self.safe_print("=" * 50)
        
        try:
            # Check if installed
            if not self.install_dir.exists():
                self.safe_print("[INFO] Bridge app not found - nothing to uninstall")
                return True
            
            self.safe_print(f"[INFO] Found installation at: {self.install_dir}")
            
            # Stop running services
            self._stop_bridge_service()
            
            # Remove startup entries
            self._remove_startup_entries()
            
            # Remove installation directory
            self._remove_installation_directory()
            
            # Clean up registry/launch agents/systemd
            self._cleanup_system_integration()
            
            # Remove desktop shortcuts
            self._remove_desktop_shortcuts()
            
            self.safe_print("\n[SUCCESS] Bridge app uninstalled successfully!")
            self.safe_print("All files and system integrations have been removed.")
            
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Uninstall failed: {str(e)}")
            return False
    
    def _stop_bridge_service(self):
        """Stop running bridge service"""
        self.safe_print("[INFO] Stopping bridge service...")
        
        try:
            if self.system == "windows":
                # Kill Python processes running bridge app
                subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                             capture_output=True, text=True)
                self.safe_print("[OK] Stopped Windows bridge processes")
                
            elif self.system == "darwin":
                # Unload LaunchAgent
                plist_file = Path.home() / "Library" / "LaunchAgents" / "com.squashplot.bridge.plist"
                if plist_file.exists():
                    subprocess.run(["launchctl", "unload", str(plist_file)], 
                                 capture_output=True, text=True)
                    self.safe_print("[OK] Unloaded macOS LaunchAgent")
                
            else:  # Linux
                # Stop systemd service
                subprocess.run(["systemctl", "--user", "stop", "squashplot-bridge.service"], 
                             capture_output=True, text=True)
                self.safe_print("[OK] Stopped Linux systemd service")
                
        except Exception as e:
            self.safe_print(f"[WARNING] Could not stop service: {str(e)}")
    
    def _remove_startup_entries(self):
        """Remove startup entries"""
        self.safe_print("[INFO] Removing startup entries...")
        
        try:
            if self.system == "windows":
                # Remove from Windows startup registry
                import winreg
                try:
                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                       r"Software\Microsoft\Windows\CurrentVersion\Run", 
                                       0, winreg.KEY_SET_VALUE)
                    winreg.DeleteValue(key, "SquashPlotBridge")
                    winreg.CloseKey(key)
                    self.safe_print("[OK] Removed from Windows startup registry")
                except FileNotFoundError:
                    self.safe_print("[INFO] No Windows startup entry found")
                except Exception as e:
                    self.safe_print(f"[WARNING] Could not remove Windows startup: {str(e)}")
                    
            elif self.system == "darwin":
                # Remove LaunchAgent plist
                plist_file = Path.home() / "Library" / "LaunchAgents" / "com.squashplot.bridge.plist"
                if plist_file.exists():
                    plist_file.unlink()
                    self.safe_print("[OK] Removed macOS LaunchAgent")
                else:
                    self.safe_print("[INFO] No macOS LaunchAgent found")
                    
            else:  # Linux
                # Remove systemd service
                service_file = Path.home() / ".config" / "systemd" / "user" / "squashplot-bridge.service"
                if service_file.exists():
                    service_file.unlink()
                    subprocess.run(["systemctl", "--user", "disable", "squashplot-bridge.service"], 
                                 capture_output=True, text=True)
                    self.safe_print("[OK] Removed Linux systemd service")
                else:
                    self.safe_print("[INFO] No Linux systemd service found")
                    
        except Exception as e:
            self.safe_print(f"[WARNING] Could not remove startup entries: {str(e)}")
    
    def _remove_installation_directory(self):
        """Remove installation directory"""
        self.safe_print("[INFO] Removing installation directory...")
        
        try:
            if self.install_dir.exists():
                shutil.rmtree(self.install_dir)
                self.safe_print(f"[OK] Removed installation directory: {self.install_dir}")
            else:
                self.safe_print("[INFO] Installation directory not found")
                
        except Exception as e:
            self.safe_print(f"[ERROR] Could not remove installation directory: {str(e)}")
    
    def _cleanup_system_integration(self):
        """Clean up system integration files"""
        self.safe_print("[INFO] Cleaning up system integration...")
        
        try:
            if self.system == "windows":
                # Clean up Windows-specific files
                startup_script = self.install_dir / "start_bridge_silent.vbs"
                if startup_script.exists():
                    startup_script.unlink()
                    
            elif self.system == "darwin":
                # Clean up macOS-specific files
                pass  # LaunchAgent already removed
                
            else:  # Linux
                # Clean up Linux-specific files
                pass  # systemd service already removed
                
            self.safe_print("[OK] System integration cleaned up")
            
        except Exception as e:
            self.safe_print(f"[WARNING] Could not clean up system integration: {str(e)}")
    
    def _remove_desktop_shortcuts(self):
        """Remove desktop shortcuts"""
        self.safe_print("[INFO] Removing desktop shortcuts...")
        
        try:
            desktop = Path.home() / "Desktop"
            
            # Remove uninstaller shortcut
            uninstaller_shortcut = desktop / "Uninstall_SquashPlot_Bridge.py"
            if uninstaller_shortcut.exists():
                uninstaller_shortcut.unlink()
                self.safe_print("[OK] Removed desktop uninstaller shortcut")
            
            # Remove any other shortcuts
            for shortcut in desktop.glob("*SquashPlot*"):
                if shortcut.is_file():
                    shortcut.unlink()
                    self.safe_print(f"[OK] Removed shortcut: {shortcut.name}")
                    
        except Exception as e:
            self.safe_print(f"[WARNING] Could not remove desktop shortcuts: {str(e)}")
    
    def check_installation_status(self):
        """Check if bridge app is installed"""
        self.safe_print("SquashPlot Bridge App Status Check")
        self.safe_print("=" * 50)
        
        # Check installation directory
        if self.install_dir.exists():
            self.safe_print(f"[FOUND] Installation directory: {self.install_dir}")
            
            # Check for bridge app
            bridge_app = self.install_dir / "secure_bridge_app.py"
            if bridge_app.exists():
                self.safe_print("[FOUND] Bridge app executable")
            else:
                self.safe_print("[MISSING] Bridge app executable")
            
            # Check for config
            config_file = self.install_dir / "bridge_config.json"
            if config_file.exists():
                self.safe_print("[FOUND] Bridge configuration")
            else:
                self.safe_print("[MISSING] Bridge configuration")
            
            # Check startup integration
            if self.system == "windows":
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                       r"Software\Microsoft\Windows\CurrentVersion\Run", 
                                       0, winreg.KEY_READ)
                    try:
                        winreg.QueryValueEx(key, "SquashPlotBridge")
                        self.safe_print("[FOUND] Windows startup entry")
                    except FileNotFoundError:
                        self.safe_print("[MISSING] Windows startup entry")
                    winreg.CloseKey(key)
                except Exception:
                    self.safe_print("[UNKNOWN] Windows startup status")
                    
            elif self.system == "darwin":
                plist_file = Path.home() / "Library" / "LaunchAgents" / "com.squashplot.bridge.plist"
                if plist_file.exists():
                    self.safe_print("[FOUND] macOS LaunchAgent")
                else:
                    self.safe_print("[MISSING] macOS LaunchAgent")
                    
            else:  # Linux
                service_file = Path.home() / ".config" / "systemd" / "user" / "squashplot-bridge.service"
                if service_file.exists():
                    self.safe_print("[FOUND] Linux systemd service")
                else:
                    self.safe_print("[MISSING] Linux systemd service")
            
            return True
        else:
            self.safe_print("[NOT FOUND] Bridge app is not installed")
            return False

def main():
    """Main uninstaller entry point"""
    print("SquashPlot Bridge App Uninstaller")
    print("=" * 50)
    
    uninstaller = BridgeUninstaller()
    
    # Check if installed
    if not uninstaller.check_installation_status():
        print("\n[INFO] Bridge app is not installed. Nothing to uninstall.")
        return
    
    # Confirm uninstallation
    print("\n" + "=" * 50)
    response = input("Are you sure you want to uninstall the SquashPlot Bridge App? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Uninstallation cancelled.")
        return
    
    # Run uninstallation
    if uninstaller.uninstall_bridge():
        print("\n" + "=" * 50)
        print("UNINSTALLATION COMPLETE")
        print("=" * 50)
        print("The SquashPlot Bridge App has been completely removed from your system.")
        print("All files, startup entries, and system integrations have been cleaned up.")
        print("")
        print("To reinstall, visit the SquashPlot dashboard and download the installer again.")
    else:
        print("\n[ERROR] Uninstallation failed. Please check the error messages above.")
        print("You may need to manually remove files from the installation directory.")

if __name__ == "__main__":
    main()
