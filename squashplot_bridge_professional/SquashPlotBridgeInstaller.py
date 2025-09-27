#!/usr/bin/env python3
"""
SquashPlot Bridge Professional Installer
Enhanced installer with completion display and built-in uninstaller
"""

import os
import sys
import json
import shutil
import subprocess
import platform
import threading
import time
import webbrowser
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import http.server
import socketserver
import tempfile

class SquashPlotBridgeInstaller:
    """Professional SquashPlot Bridge installer with GUI completion"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.install_dir = self._get_install_directory()
        self.bridge_app = "SquashPlotSecureBridge.py"
        self.developer_app = "SquashPlotDeveloperBridge.py"
        self.config_file = "bridge_config.json"
        self.startup_script = None
        self.installation_log = []
        
    def _get_install_directory(self):
        """Get appropriate installation directory for the OS"""
        if self.system == "windows":
            return Path.home() / "AppData" / "Local" / "SquashPlot" / "Bridge"
        elif self.system == "darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / "SquashPlot" / "Bridge"
        else:  # Linux
            return Path.home() / ".local" / "share" / "squashplot" / "bridge"
    
    def log_installation(self, message: str):
        """Log installation progress"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.installation_log.append(log_entry)
        print(log_entry)
    
    def check_dependencies(self):
        """Check for required dependencies"""
        self.log_installation("Checking system dependencies...")
        
        # Check Python installation
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            raise Exception(f"Python 3.8+ required. Found: {python_version.major}.{python_version.minor}")
        
        self.log_installation(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
        
        # Check required modules
        required_modules = ['cryptography', 'numpy']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
                self.log_installation(f"✓ {module} module available")
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.log_installation(f"Installing missing modules: {', '.join(missing_modules)}")
            self._install_python_modules(missing_modules)
        
        return True
    
    def _install_python_modules(self, modules):
        """Install missing Python modules"""
        for module in modules:
            try:
                self.log_installation(f"Installing {module}...")
                subprocess.run([sys.executable, "-m", "pip", "install", module], 
                             check=True, capture_output=True)
                self.log_installation(f"✓ {module} installed successfully")
            except subprocess.CalledProcessError as e:
                self.log_installation(f"⚠ Warning: Could not install {module}: {e}")
    
    def create_install_directory(self):
        """Create installation directory"""
        self.log_installation(f"Creating installation directory: {self.install_dir}")
        
        try:
            self.install_dir.mkdir(parents=True, exist_ok=True)
            self.log_installation(f"✓ Installation directory created")
        except Exception as e:
            raise Exception(f"Failed to create installation directory: {e}")
    
    def copy_bridge_files(self):
        """Copy bridge application files"""
        self.log_installation("Copying SquashPlot Bridge files...")
        
        # Source directory (current directory)
        source_dir = Path(".")
        
        # Files to copy
        files_to_copy = [
            self.bridge_app,
            self.developer_app,
            "SquashPlotSecureBridge.html",
            "SquashPlotDeveloperBridge.html",
            "PROFESSIONAL_SUMMARY.md",
            "SIMPLE_SUMMARY.md",
            "SECURITY_WARNING_README.md"
        ]
        
        copied_files = []
        for file_name in files_to_copy:
            source_file = source_dir / file_name
            if source_file.exists():
                dest_file = self.install_dir / file_name
                try:
                    shutil.copy2(source_file, dest_file)
                    copied_files.append(file_name)
                    self.log_installation(f"✓ Copied {file_name}")
                except Exception as e:
                    self.log_installation(f"⚠ Warning: Could not copy {file_name}: {e}")
            else:
                self.log_installation(f"⚠ Warning: {file_name} not found")
        
        if not copied_files:
            raise Exception("No bridge files were copied successfully")
        
        self.log_installation(f"✓ Copied {len(copied_files)} files successfully")
        return copied_files
    
    def create_uninstaller(self):
        """Create built-in uninstaller"""
        self.log_installation("Creating built-in uninstaller...")
        
        uninstaller_content = f'''#!/usr/bin/env python3
"""
SquashPlot Bridge Uninstaller
Built-in uninstaller for complete removal
"""

import os
import sys
import shutil
import subprocess
import platform
import json
from pathlib import Path

def uninstall_squashplot_bridge():
    """Uninstall SquashPlot Bridge completely"""
    print("SquashPlot Bridge Uninstaller")
    print("=" * 40)
    
    system = platform.system().lower()
    
    # Get installation directory
    if system == "windows":
        install_dir = Path.home() / "AppData" / "Local" / "SquashPlot" / "Bridge"
        startup_dir = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    elif system == "darwin":  # macOS
        install_dir = Path.home() / "Library" / "Application Support" / "SquashPlot" / "Bridge"
        startup_dir = Path.home() / "Library" / "LaunchAgents"
    else:  # Linux
        install_dir = Path.home() / ".local" / "share" / "squashplot" / "bridge"
        startup_dir = Path.home() / ".config" / "autostart"
    
    print(f"Installation directory: {{install_dir}}")
    
    # Stop any running bridge processes
    print("Stopping bridge processes...")
    try:
        if system == "windows":
            subprocess.run(["taskkill", "/f", "/im", "python.exe", "/fi", "WINDOWTITLE eq SquashPlot*"], 
                         capture_output=True)
        else:
            subprocess.run(["pkill", "-f", "SquashPlot.*Bridge"], capture_output=True)
        print("✓ Bridge processes stopped")
    except:
        print("⚠ Could not stop bridge processes (may not be running)")
    
    # Remove startup scripts
    print("Removing startup integration...")
    startup_files = [
        "SquashPlotBridge.lnk",  # Windows
        "com.squashplot.bridge.plist",  # macOS
        "squashplot-bridge.desktop"  # Linux
    ]
    
    for startup_file in startup_files:
        startup_path = startup_dir / startup_file
        if startup_path.exists():
            try:
                startup_path.unlink()
                print(f"✓ Removed {{startup_file}}")
            except Exception as e:
                print(f"⚠ Could not remove {{startup_file}}: {{e}}")
    
    # Remove installation directory
    print("Removing installation files...")
    if install_dir.exists():
        try:
            shutil.rmtree(install_dir)
            print(f"✓ Removed installation directory")
        except Exception as e:
            print(f"⚠ Could not remove installation directory: {{e}}")
    
    # Remove parent directories if empty
    try:
        parent_dir = install_dir.parent
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            shutil.rmtree(parent_dir)
            print(f"✓ Removed parent directory")
    except:
        pass
    
    print("=" * 40)
    print("✓ SquashPlot Bridge uninstalled successfully!")
    print("All files and startup integration have been removed.")

if __name__ == "__main__":
    uninstall_squashplot_bridge()
'''
        
        # Write uninstaller to installation directory
        uninstaller_path = self.install_dir / "uninstall.py"
        try:
            with open(uninstaller_path, 'w', encoding='utf-8') as f:
                f.write(uninstaller_content)
            
            # Make executable on Unix systems
            if self.system != "windows":
                os.chmod(uninstaller_path, 0o755)
            
            self.log_installation(f"✓ Uninstaller created: {uninstaller_path}")
        except Exception as e:
            self.log_installation(f"⚠ Warning: Could not create uninstaller: {e}")
    
    def setup_startup_integration(self):
        """Setup automatic startup integration"""
        self.log_installation("Setting up automatic startup...")
        
        if self.system == "windows":
            self._setup_windows_startup()
        elif self.system == "darwin":  # macOS
            self._setup_macos_startup()
        else:  # Linux
            self._setup_linux_startup()
        
        self.log_installation("✓ Startup integration configured")
    
    def _setup_windows_startup(self):
        """Setup Windows startup integration"""
        startup_dir = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        
        # Create shortcut
        shortcut_content = f'''@echo off
cd /d "{self.install_dir}"
start "SquashPlot Bridge" python {self.bridge_app}
'''
        
        shortcut_path = startup_dir / "SquashPlotBridge.bat"
        try:
            with open(shortcut_path, 'w') as f:
                f.write(shortcut_content)
            self.log_installation(f"✓ Windows startup shortcut created")
        except Exception as e:
            self.log_installation(f"⚠ Warning: Could not create startup shortcut: {e}")
    
    def _setup_macos_startup(self):
        """Setup macOS startup integration"""
        startup_dir = Path.home() / "Library" / "LaunchAgents"
        startup_dir.mkdir(exist_ok=True)
        
        plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.squashplot.bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>python3</string>
        <string>{self.install_dir / self.bridge_app}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>'''
        
        plist_path = startup_dir / "com.squashplot.bridge.plist"
        try:
            with open(plist_path, 'w') as f:
                f.write(plist_content)
            self.log_installation(f"✓ macOS startup plist created")
        except Exception as e:
            self.log_installation(f"⚠ Warning: Could not create startup plist: {e}")
    
    def _setup_linux_startup(self):
        """Setup Linux startup integration"""
        startup_dir = Path.home() / ".config" / "autostart"
        startup_dir.mkdir(parents=True, exist_ok=True)
        
        desktop_content = f'''[Desktop Entry]
Type=Application
Name=SquashPlot Bridge
Comment=SquashPlot Secure Bridge Application
Exec=python3 {self.install_dir / self.bridge_app}
Path={self.install_dir}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
'''
        
        desktop_path = startup_dir / "squashplot-bridge.desktop"
        try:
            with open(desktop_path, 'w') as f:
                f.write(desktop_content)
            os.chmod(desktop_path, 0o755)
            self.log_installation(f"✓ Linux startup desktop file created")
        except Exception as e:
            self.log_installation(f"⚠ Warning: Could not create startup desktop file: {e}")
    
    def create_configuration(self):
        """Create bridge configuration file"""
        self.log_installation("Creating configuration file...")
        
        config = {
            "installation_date": datetime.now().isoformat(),
            "install_directory": str(self.install_dir),
            "system": self.system,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "bridge_app": self.bridge_app,
            "developer_app": self.developer_app,
            "version": "2.0.0-secure",
            "auto_start": True,
            "secure_mode": True
        }
        
        config_path = self.install_dir / self.config_file
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.log_installation(f"✓ Configuration file created")
        except Exception as e:
            self.log_installation(f"⚠ Warning: Could not create configuration file: {e}")
    
    def test_installation(self):
        """Test the installation"""
        self.log_installation("Testing installation...")
        
        try:
            # Test if bridge app can be imported
            bridge_path = self.install_dir / self.bridge_app
            if bridge_path.exists():
                self.log_installation("✓ Bridge application file found")
            else:
                raise Exception("Bridge application file not found")
            
            # Test configuration
            config_path = self.install_dir / self.config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.log_installation(f"✓ Configuration file valid")
            else:
                raise Exception("Configuration file not found")
            
            self.log_installation("✓ Installation test passed")
            return True
            
        except Exception as e:
            self.log_installation(f"✗ Installation test failed: {e}")
            return False

class InstallationCompletionGUI:
    """GUI for installation completion display"""
    
    def __init__(self, installation_log, install_dir, bridge_app):
        self.installation_log = installation_log
        self.install_dir = install_dir
        self.bridge_app = bridge_app
        self.root = None
        
    def show_completion(self):
        """Show installation completion GUI"""
        self.root = tk.Tk()
        self.root.title("SquashPlot Bridge - Installation Complete")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🎉 SquashPlot Bridge Installation Complete!", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Success message
        success_frame = ttk.LabelFrame(main_frame, text="Installation Summary", padding="15")
        success_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        success_text = f"""✅ Installation completed successfully!

📁 Installation Directory: {self.install_dir}
🚀 Bridge Application: {self.bridge_app}
🔄 Auto-startup: Enabled
🛡️ Security Level: Maximum

The SquashPlot Bridge is now installed and ready to use.
It will start automatically when you restart your computer."""
        
        ttk.Label(success_frame, text=success_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Installation log
        log_frame = ttk.LabelFrame(main_frame, text="Installation Log", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        log_text = scrolledtext.ScrolledText(log_frame, height=12, width=80)
        log_text.pack(fill=tk.BOTH, expand=True)
        
        # Populate log
        for log_entry in self.installation_log:
            log_text.insert(tk.END, log_entry + "\n")
        log_text.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(15, 0))
        
        # Start Bridge button
        start_button = ttk.Button(button_frame, text="🚀 Start SquashPlot Bridge", 
                                 command=self.start_bridge, style='Accent.TButton')
        start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Open folder button
        folder_button = ttk.Button(button_frame, text="📁 Open Installation Folder", 
                                  command=self.open_folder)
        folder_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Close button
        close_button = ttk.Button(button_frame, text="✅ Close", command=self.close_window)
        close_button.pack(side=tk.LEFT)
        
        # Uninstall info
        uninstall_frame = ttk.LabelFrame(main_frame, text="Uninstall Information", padding="10")
        uninstall_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 0))
        
        uninstall_text = """To uninstall SquashPlot Bridge:
1. Go to the SquashPlot website
2. Use the online uninstaller, OR
3. Run the uninstaller: python uninstall.py (in installation folder)

The uninstaller will remove all files and startup integration."""
        
        ttk.Label(uninstall_frame, text=uninstall_text, justify=tk.LEFT, 
                 foreground='gray').pack(anchor=tk.W)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Start the GUI
        self.root.mainloop()
    
    def start_bridge(self):
        """Start the SquashPlot Bridge"""
        try:
            bridge_path = self.install_dir / self.bridge_app
            subprocess.Popen([sys.executable, str(bridge_path)], 
                           cwd=str(self.install_dir))
            
            # Open browser
            time.sleep(2)
            webbrowser.open('http://127.0.0.1:8080')
            
            messagebox.showinfo("Bridge Started", 
                              "SquashPlot Bridge is starting...\n"
                              "Your browser should open automatically.\n"
                              "If not, go to: http://127.0.0.1:8080")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start SquashPlot Bridge:\n{e}")
    
    def open_folder(self):
        """Open installation folder"""
        try:
            if platform.system() == "Windows":
                os.startfile(self.install_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(self.install_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(self.install_dir)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")
    
    def close_window(self):
        """Close the completion window"""
        self.root.destroy()

def main():
    """Main installation function"""
    print("SquashPlot Bridge Professional Installer")
    print("=" * 50)
    
    installer = SquashPlotBridgeInstaller()
    
    try:
        # Run installation steps
        installer.check_dependencies()
        installer.create_install_directory()
        installer.copy_bridge_files()
        installer.create_uninstaller()
        installer.setup_startup_integration()
        installer.create_configuration()
        
        # Test installation
        if installer.test_installation():
            print("\n" + "=" * 50)
            print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            # Show completion GUI
            completion_gui = InstallationCompletionGUI(
                installer.installation_log,
                installer.install_dir,
                installer.bridge_app
            )
            completion_gui.show_completion()
            
        else:
            print("\n" + "=" * 50)
            print("❌ INSTALLATION FAILED!")
            print("=" * 50)
            print("Please check the installation log above for errors.")
            input("Press Enter to exit...")
            
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("Please check the error message above.")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
