#!/usr/bin/env python3
"""
SquashPlot System Scanner
Comprehensive system analysis for installation requirements
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

class SystemScanner:
    """Comprehensive system scanner for installation requirements"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.scan_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "python_info": {},
            "packages": {},
            "network": {},
            "security": {},
            "recommendations": []
        }
    
    def safe_print(self, message: str):
        """Print message with safe encoding for all OS"""
        try:
            print(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    def scan_system_info(self):
        """Scan basic system information"""
        self.safe_print("Scanning system information...")
        
        try:
            import psutil
            
            self.scan_results["system_info"] = {
                "os": platform.system(),
                "os_version": platform.release(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "disk_free_gb": round(shutil.disk_usage(Path.home()).free / (1024**3), 2),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1)
            }
            
            self.safe_print(f"[OK] System: {self.scan_results['system_info']['os']} {self.scan_results['system_info']['os_version']}")
            self.safe_print(f"[OK] Architecture: {self.scan_results['system_info']['architecture']}")
            self.safe_print(f"[OK] Memory: {self.scan_results['system_info']['memory_total_gb']} GB")
            self.safe_print(f"[OK] Disk Space: {self.scan_results['system_info']['disk_free_gb']} GB free")
            
        except ImportError:
            self.scan_results["system_info"] = {
                "os": platform.system(),
                "os_version": platform.release(),
                "architecture": platform.machine(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "psutil_available": False
            }
            self.safe_print("[WARNING] psutil not available - limited system info")
    
    def scan_python_installation(self):
        """Scan Python installation and requirements"""
        self.safe_print("Scanning Python installation...")
        
        python_info = {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "executable": sys.executable,
            "path": sys.path,
            "pip_available": False,
            "pip_version": None,
            "requirements_met": False
        }
        
        # Check if Python version meets requirements
        if sys.version_info >= (3, 8):
            python_info["requirements_met"] = True
            self.safe_print(f"[OK] Python {python_info['version']} meets requirements")
        else:
            self.safe_print(f"[ERROR] Python {python_info['version']} does not meet requirements (3.8+ needed)")
            self.scan_results["recommendations"].append({
                "type": "critical",
                "issue": "Python version too old",
                "solution": "Install Python 3.8 or later from https://python.org"
            })
        
        # Check pip availability
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                python_info["pip_available"] = True
                python_info["pip_version"] = result.stdout.strip()
                self.safe_print(f"[OK] pip available: {python_info['pip_version']}")
            else:
                self.safe_print("[ERROR] pip not available")
                self.scan_results["recommendations"].append({
                    "type": "critical",
                    "issue": "pip not available",
                    "solution": "Install pip: python -m ensurepip --upgrade"
                })
        except Exception as e:
            self.safe_print(f"[ERROR] pip check failed: {str(e)}")
            self.scan_results["recommendations"].append({
                "type": "critical",
                "issue": "pip check failed",
                "solution": "Reinstall Python with pip included"
            })
        
        self.scan_results["python_info"] = python_info
    
    def scan_required_packages(self):
        """Scan for required Python packages"""
        self.safe_print("Scanning required packages...")
        
        required_packages = {
            "cryptography": {"required": True, "version": None, "installed": False},
            "requests": {"required": True, "version": None, "installed": False},
            "psutil": {"required": True, "version": None, "installed": False},
            "fastapi": {"required": False, "version": None, "installed": False},
            "uvicorn": {"required": False, "version": None, "installed": False}
        }
        
        for package, info in required_packages.items():
            try:
                module = __import__(package)
                info["installed"] = True
                
                # Try to get version
                if hasattr(module, "__version__"):
                    info["version"] = module.__version__
                    self.safe_print(f"[OK] {package} {info['version']} installed")
                else:
                    self.safe_print(f"[OK] {package} installed (version unknown)")
                    
            except ImportError:
                if info["required"]:
                    self.safe_print(f"[MISSING] {package} (required)")
                    self.scan_results["recommendations"].append({
                        "type": "critical" if info["required"] else "warning",
                        "issue": f"Missing package: {package}",
                        "solution": f"Install with: pip install {package}"
                    })
                else:
                    self.safe_print(f"[MISSING] {package} (optional)")
        
        self.scan_results["packages"] = required_packages
    
    def scan_network_connectivity(self):
        """Scan network connectivity and requirements"""
        self.safe_print("Scanning network connectivity...")
        
        network_info = {
            "internet_available": False,
            "pip_connectivity": False,
            "github_connectivity": False,
            "python_org_connectivity": False
        }
        
        # Test internet connectivity
        try:
            import urllib.request
            urllib.request.urlopen("https://www.google.com", timeout=5)
            network_info["internet_available"] = True
            self.safe_print("[OK] Internet connectivity available")
        except Exception:
            self.safe_print("[ERROR] No internet connectivity")
            self.scan_results["recommendations"].append({
                "type": "critical",
                "issue": "No internet connectivity",
                "solution": "Check internet connection and firewall settings"
            })
            return
        
        # Test pip connectivity
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "requests"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                network_info["pip_connectivity"] = True
                self.safe_print("[OK] pip connectivity available")
            else:
                self.safe_print("[WARNING] pip connectivity issues")
        except Exception:
            self.safe_print("[WARNING] pip connectivity test failed")
        
        # Test GitHub connectivity (for potential downloads)
        try:
            urllib.request.urlopen("https://github.com", timeout=5)
            network_info["github_connectivity"] = True
            self.safe_print("[OK] GitHub connectivity available")
        except Exception:
            self.safe_print("[WARNING] GitHub connectivity issues")
        
        # Test Python.org connectivity
        try:
            urllib.request.urlopen("https://python.org", timeout=5)
            network_info["python_org_connectivity"] = True
            self.safe_print("[OK] Python.org connectivity available")
        except Exception:
            self.safe_print("[WARNING] Python.org connectivity issues")
        
        self.scan_results["network"] = network_info
    
    def scan_security_settings(self):
        """Scan security-related settings"""
        self.safe_print("Scanning security settings...")
        
        security_info = {
            "firewall_detected": False,
            "antivirus_detected": False,
            "admin_privileges": False,
            "install_path_writable": False
        }
        
        # Check admin privileges
        try:
            if self.system == "windows":
                import ctypes
                security_info["admin_privileges"] = ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                security_info["admin_privileges"] = os.geteuid() == 0
                
            if security_info["admin_privileges"]:
                self.safe_print("[OK] Administrator privileges available")
            else:
                self.safe_print("[INFO] Running as regular user")
        except Exception:
            self.safe_print("[WARNING] Could not determine privilege level")
        
        # Check if install path is writable
        try:
            test_path = Path.home() / "AppData" / "Local" / "SquashPlot" if self.system == "windows" else Path.home() / ".local" / "share" / "squashplot"
            test_path.mkdir(parents=True, exist_ok=True)
            test_file = test_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            security_info["install_path_writable"] = True
            self.safe_print("[OK] Install path is writable")
        except Exception as e:
            self.safe_print(f"[ERROR] Install path not writable: {str(e)}")
            self.scan_results["recommendations"].append({
                "type": "critical",
                "issue": "Install path not writable",
                "solution": "Run installer as administrator or choose different install location"
            })
        
        # Check for common antivirus processes (Windows)
        if self.system == "windows":
            try:
                result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq *antivirus*"], 
                                      capture_output=True, text=True)
                if "antivirus" in result.stdout.lower():
                    security_info["antivirus_detected"] = True
                    self.safe_print("[INFO] Antivirus software detected")
            except Exception:
                pass
        
        self.scan_results["security"] = security_info
    
    def generate_installation_plan(self):
        """Generate installation plan based on scan results"""
        self.safe_print("Generating installation plan...")
        
        plan = {
            "python_installation_needed": not self.scan_results["python_info"].get("requirements_met", False),
            "pip_installation_needed": not self.scan_results["python_info"].get("pip_available", False),
            "packages_to_install": [],
            "estimated_time": "5-15 minutes",
            "complexity": "low"
        }
        
        # Check which packages need installation
        for package, info in self.scan_results["packages"].items():
            if info["required"] and not info["installed"]:
                plan["packages_to_install"].append(package)
        
        # Determine complexity
        if plan["python_installation_needed"]:
            plan["complexity"] = "high"
            plan["estimated_time"] = "15-30 minutes"
        elif plan["packages_to_install"]:
            plan["complexity"] = "medium"
            plan["estimated_time"] = "5-10 minutes"
        
        self.scan_results["installation_plan"] = plan
        
        # Generate recommendations
        if plan["python_installation_needed"]:
            self.scan_results["recommendations"].append({
                "type": "critical",
                "issue": "Python installation required",
                "solution": "Download and install Python 3.8+ from https://python.org"
            })
        
        if plan["pip_installation_needed"]:
            self.scan_results["recommendations"].append({
                "type": "critical", 
                "issue": "pip installation required",
                "solution": "Reinstall Python with pip included or run: python -m ensurepip --upgrade"
            })
        
        if plan["packages_to_install"]:
            self.scan_results["recommendations"].append({
                "type": "warning",
                "issue": f"Missing packages: {', '.join(plan['packages_to_install'])}",
                "solution": f"Install with: pip install {' '.join(plan['packages_to_install'])}"
            })
    
    def run_comprehensive_scan(self):
        """Run comprehensive system scan"""
        self.safe_print("SquashPlot System Scanner")
        self.safe_print("=" * 50)
        self.safe_print("Analyzing system for SquashPlot Bridge App installation...")
        self.safe_print("")
        
        # Run all scans
        self.scan_system_info()
        self.scan_python_installation()
        self.scan_required_packages()
        self.scan_network_connectivity()
        self.scan_security_settings()
        self.generate_installation_plan()
        
        # Generate report
        self._generate_scan_report()
        
        return self.scan_results
    
    def _generate_scan_report(self):
        """Generate comprehensive scan report"""
        self.safe_print("\n" + "=" * 50)
        self.safe_print("SYSTEM SCAN REPORT")
        self.safe_print("=" * 50)
        
        # System readiness
        critical_issues = len([r for r in self.scan_results["recommendations"] if r["type"] == "critical"])
        warning_issues = len([r for r in self.scan_results["recommendations"] if r["type"] == "warning"])
        
        if critical_issues == 0:
            readiness = "READY"
            status_icon = "[OK]"
        elif critical_issues <= 2:
            readiness = "MOSTLY READY"
            status_icon = "[WARNING]"
        else:
            readiness = "NOT READY"
            status_icon = "[ERROR]"
        
        self.safe_print(f"{status_icon} System Readiness: {readiness}")
        self.safe_print(f"Critical Issues: {critical_issues}")
        self.safe_print(f"Warnings: {warning_issues}")
        
        # Installation plan
        plan = self.scan_results.get("installation_plan", {})
        if plan:
            self.safe_print(f"\nInstallation Plan:")
            self.safe_print(f"Complexity: {plan.get('complexity', 'unknown').upper()}")
            self.safe_print(f"Estimated Time: {plan.get('estimated_time', 'unknown')}")
            
            if plan.get("python_installation_needed"):
                self.safe_print("• Python installation required")
            if plan.get("pip_installation_needed"):
                self.safe_print("• pip installation required")
            if plan.get("packages_to_install"):
                self.safe_print(f"• Packages to install: {', '.join(plan['packages_to_install'])}")
        
        # Recommendations
        if self.scan_results["recommendations"]:
            self.safe_print(f"\nRecommendations:")
            for i, rec in enumerate(self.scan_results["recommendations"], 1):
                severity = "[CRITICAL]" if rec["type"] == "critical" else "[WARNING]"
                self.safe_print(f"  {i}. {severity} {rec['issue']}")
                self.safe_print(f"     Solution: {rec['solution']}")
        
        # Save detailed report
        report_file = f"system_scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.scan_results, f, indent=2)
            self.safe_print(f"\nDetailed report saved to: {report_file}")
        except Exception as e:
            self.safe_print(f"\nError saving report: {str(e)}")

def main():
    """Main system scanner entry point"""
    print("SquashPlot System Scanner")
    print("=" * 50)
    
    scanner = SystemScanner()
    results = scanner.run_comprehensive_scan()
    
    # Ask if user wants to proceed with installation
    if results.get("installation_plan", {}).get("complexity") != "low":
        response = input("\nProceed with installation? (yes/no): ")
        if response.lower() == 'yes':
            print("\nYou can now run the bridge installer:")
            print("python bridge_installer.py")
        else:
            print("Scan completed. Address any issues and run the scanner again.")

if __name__ == "__main__":
    main()
