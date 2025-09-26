#!/usr/bin/env python3
"""
SquashPlot Dependency Installer
Handles installation of Python and required packages for all operating systems
"""

import os
import sys
import platform
import subprocess
import urllib.request
import shutil
from pathlib import Path

class DependencyInstaller:
    """Cross-platform dependency installer"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.python_required = (3, 8)
        self.required_packages = [
            "cryptography",
            "requests", 
            "psutil",
            "fastapi",
            "uvicorn"
        ]
    
    def safe_print(self, message: str):
        """Print message with safe encoding for all OS"""
        try:
            print(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    def check_python_installation(self):
        """Check if Python is installed and meets requirements"""
        self.safe_print("Checking Python installation...")
        
        try:
            version = sys.version_info
            if version.major >= self.python_required[0] and version.minor >= self.python_required[1]:
                self.safe_print(f"[OK] Python {version.major}.{version.minor} detected")
                return True
            else:
                self.safe_print(f"[ERROR] Python {self.python_required[0]}.{self.python_required[1]}+ required")
                self.safe_print(f"Found: Python {version.major}.{version.minor}")
                return False
        except Exception as e:
            self.safe_print(f"[ERROR] Python check failed: {str(e)}")
            return False
    
    def install_python_windows(self):
        """Install Python on Windows"""
        self.safe_print("Installing Python on Windows...")
        
        try:
            # Check if Python is already installed
            if shutil.which("python") or shutil.which("python3"):
                self.safe_print("[INFO] Python already installed")
                return True
            
            # Download Python installer
            python_url = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
            installer_path = Path.home() / "Downloads" / "python-installer.exe"
            
            self.safe_print(f"Downloading Python installer from {python_url}")
            urllib.request.urlretrieve(python_url, installer_path)
            
            # Run installer with silent installation
            self.safe_print("Running Python installer...")
            subprocess.run([
                str(installer_path),
                "/quiet",
                "InstallAllUsers=1",
                "PrependPath=1",
                "Include_test=0"
            ], check=True)
            
            # Clean up installer
            installer_path.unlink()
            
            self.safe_print("[OK] Python installed successfully")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Python installation failed: {str(e)}")
            self.safe_print("Please install Python manually from https://python.org")
            return False
    
    def install_python_macos(self):
        """Install Python on macOS"""
        self.safe_print("Installing Python on macOS...")
        
        try:
            # Check if Homebrew is installed
            if not shutil.which("brew"):
                self.safe_print("Installing Homebrew...")
                subprocess.run([
                    "/bin/bash", "-c", 
                    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                ], check=True)
            
            # Install Python via Homebrew
            self.safe_print("Installing Python via Homebrew...")
            subprocess.run(["brew", "install", "python@3.11"], check=True)
            
            # Add to PATH
            subprocess.run(["brew", "link", "python@3.11"], check=True)
            
            self.safe_print("[OK] Python installed successfully")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Python installation failed: {str(e)}")
            self.safe_print("Please install Python manually:")
            self.safe_print("1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            self.safe_print("2. Install Python: brew install python@3.11")
            return False
    
    def install_python_linux(self):
        """Install Python on Linux"""
        self.safe_print("Installing Python on Linux...")
        
        try:
            # Detect package manager
            if shutil.which("apt"):
                self.safe_print("Using apt package manager...")
                subprocess.run(["sudo", "apt", "update"], check=True)
                subprocess.run(["sudo", "apt", "install", "-y", "python3.11", "python3.11-pip"], check=True)
                
            elif shutil.which("yum"):
                self.safe_print("Using yum package manager...")
                subprocess.run(["sudo", "yum", "install", "-y", "python3.11", "python3.11-pip"], check=True)
                
            elif shutil.which("dnf"):
                self.safe_print("Using dnf package manager...")
                subprocess.run(["sudo", "dnf", "install", "-y", "python3.11", "python3.11-pip"], check=True)
                
            elif shutil.which("pacman"):
                self.safe_print("Using pacman package manager...")
                subprocess.run(["sudo", "pacman", "-S", "--noconfirm", "python", "python-pip"], check=True)
                
            else:
                self.safe_print("[ERROR] Unsupported package manager")
                self.safe_print("Please install Python manually:")
                self.safe_print("Ubuntu/Debian: sudo apt install python3.11 python3.11-pip")
                self.safe_print("CentOS/RHEL: sudo yum install python3.11 python3.11-pip")
                self.safe_print("Fedora: sudo dnf install python3.11 python3.11-pip")
                self.safe_print("Arch: sudo pacman -S python python-pip")
                return False
            
            self.safe_print("[OK] Python installed successfully")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Python installation failed: {str(e)}")
            self.safe_print("Please install Python manually using your package manager")
            return False
    
    def install_python(self):
        """Install Python based on operating system"""
        if self.system == "windows":
            return self.install_python_windows()
        elif self.system == "darwin":
            return self.install_python_macos()
        else:  # Linux
            return self.install_python_linux()
    
    def install_python_packages(self):
        """Install required Python packages"""
        self.safe_print("Installing Python packages...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install required packages
            for package in self.required_packages:
                self.safe_print(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                self.safe_print(f"[OK] Installed {package}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.safe_print(f"[ERROR] Package installation failed: {e}")
            self.safe_print("Please install packages manually:")
            self.safe_print(f"pip install {' '.join(self.required_packages)}")
            return False
    
    def install_all_dependencies(self):
        """Install all required dependencies"""
        self.safe_print("SquashPlot Dependency Installer")
        self.safe_print("=" * 50)
        
        # Check if Python is already installed
        if self.check_python_installation():
            self.safe_print("[OK] Python installation is sufficient")
        else:
            self.safe_print("[INFO] Python installation required")
            if not self.install_python():
                return False
        
        # Install Python packages
        if not self.install_python_packages():
            return False
        
        self.safe_print("\n[SUCCESS] All dependencies installed successfully!")
        self.safe_print("You can now run the SquashPlot Bridge App installer.")
        
        return True
    
    def check_system_requirements(self):
        """Check system requirements and provide recommendations"""
        self.safe_print("System Requirements Check")
        self.safe_print("=" * 50)
        
        # Check operating system
        self.safe_print(f"Operating System: {platform.system()} {platform.release()}")
        
        # Check architecture
        self.safe_print(f"Architecture: {platform.machine()}")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            self.safe_print(f"Available Memory: {memory_gb:.1f} GB")
            
            if memory_gb < 2:
                self.safe_print("[WARNING] Less than 2GB RAM available. Performance may be affected.")
        except ImportError:
            self.safe_print("[INFO] psutil not available for memory check")
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(Path.home())
            free_gb = disk_usage.free / (1024**3)
            self.safe_print(f"Available Disk Space: {free_gb:.1f} GB")
            
            if free_gb < 1:
                self.safe_print("[WARNING] Less than 1GB free disk space. Installation may fail.")
        except Exception:
            self.safe_print("[INFO] Could not check disk space")
        
        # System requirements
        self.safe_print("\nSystem Requirements:")
        self.safe_print("- Python 3.8 or later")
        self.safe_print("- 2GB RAM minimum (4GB recommended)")
        self.safe_print("- 1GB free disk space")
        self.safe_print("- Internet connection for package installation")
        
        return True

def main():
    """Main dependency installer entry point"""
    print("SquashPlot Dependency Installer")
    print("=" * 50)
    
    installer = DependencyInstaller()
    
    # Check system requirements
    installer.check_system_requirements()
    
    # Ask user if they want to proceed
    response = input("\nProceed with dependency installation? (yes/no): ")
    if response.lower() != 'yes':
        print("Installation cancelled.")
        return
    
    # Install all dependencies
    if installer.install_all_dependencies():
        print("\n" + "=" * 50)
        print("DEPENDENCY INSTALLATION COMPLETE")
        print("=" * 50)
        print("All required dependencies have been installed.")
        print("You can now run the SquashPlot Bridge App installer.")
        print("")
        print("Next steps:")
        print("1. Download bridge_installer.py")
        print("2. Run: python bridge_installer.py")
    else:
        print("\n[ERROR] Dependency installation failed.")
        print("Please install dependencies manually and try again.")

if __name__ == "__main__":
    main()
