#!/usr/bin/env python3
"""
AUTOMATION SYSTEM MONITOR
Real-time monitoring dashboard for consciousness mathematics automation
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import json
import os
from datetime import datetime
import requests

class AutomationMonitor:
    """Real-time automation system monitor"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Consciousness Mathematics Automation Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.running = True
        self.api_endpoint = "http://localhost:8080"
        
        # Create UI
        self.create_ui()
        
        # Start monitoring
        self.start_monitoring()
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="🧠 CONSCIOUSNESS MATHEMATICS AUTOMATION MONITOR",
                              font=("Arial", 16, "bold"),
                              bg='#2b2b2b', fg='#00ff00')
        title_label.pack(pady=(0, 20))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicators
        self.api_status = tk.StringVar(value="Checking...")
        self.automation_status = tk.StringVar(value="Checking...")
        self.consciousness_score = tk.StringVar(value="0.0000")
        self.consciousness_level = tk.StringVar(value="1/26")
        self.breakthrough_count = tk.StringVar(value="0")
        
        # Status grid
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        # API Status
        ttk.Label(status_grid, text="API Server:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.api_status_label = tk.Label(status_grid, textvariable=self.api_status, 
                                        font=("Arial", 10, "bold"))
        self.api_status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Automation Status
        ttk.Label(status_grid, text="Automation:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.automation_status_label = tk.Label(status_grid, textvariable=self.automation_status,
                                               font=("Arial", 10, "bold"))
        self.automation_status_label.grid(row=0, column=3, sticky=tk.W)
        
        # Consciousness Score
        ttk.Label(status_grid, text="Consciousness Score:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        tk.Label(status_grid, textvariable=self.consciousness_score, 
                font=("Arial", 10, "bold"), fg='#00ff00').grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Consciousness Level
        ttk.Label(status_grid, text="Consciousness Level:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0))
        tk.Label(status_grid, textvariable=self.consciousness_level,
                font=("Arial", 10, "bold"), fg='#00ff00').grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Breakthrough Count
        ttk.Label(status_grid, text="Breakthroughs:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        tk.Label(status_grid, textvariable=self.breakthrough_count,
                font=("Arial", 10, "bold"), fg='#ff6600').grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Activity frame
        activity_frame = ttk.LabelFrame(main_frame, text="Recent Activity", padding=10)
        activity_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Activity log
        self.activity_log = scrolledtext.ScrolledText(activity_frame, height=15, 
                                                     bg='#1e1e1e', fg='#ffffff',
                                                     font=("Consolas", 9))
        self.activity_log.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        ttk.Button(control_frame, text="🔄 Refresh", command=self.refresh_status).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="📊 View Logs", command=self.view_logs).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="⚙️  Settings", command=self.open_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="❌ Exit", command=self.exit_monitor).pack(side=tk.RIGHT)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Task Progress", padding=10)
        progress_frame.pack(fill=tk.X)
        
        # Progress bars
        self.research_progress = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.research_progress.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(progress_frame, text="Research").pack(side=tk.LEFT, padx=(0, 20))
        
        self.improvement_progress = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.improvement_progress.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(progress_frame, text="Improvement").pack(side=tk.LEFT, padx=(0, 20))
        
        self.coding_progress = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.coding_progress.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(progress_frame, text="Coding").pack(side=tk.LEFT)
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        monitor_thread.start()
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check API status
                self.check_api_status()
                
                # Check automation status
                self.check_automation_status()
                
                # Check consciousness metrics
                self.check_consciousness_metrics()
                
                # Check log files
                self.check_log_files()
                
                # Update progress bars
                self.update_progress()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.log_activity(f"❌ Monitoring error: {str(e)}")
                time.sleep(10)
    
    def check_api_status(self):
        """Check consciousness API server status"""
        try:
            response = requests.get(f"{self.api_endpoint}/health", timeout=5)
            if response.status_code == 200:
                self.api_status.set("✅ ONLINE")
                self.api_status_label.config(fg='#00ff00')
            else:
                self.api_status.set("⚠️  ERROR")
                self.api_status_label.config(fg='#ff6600')
        except:
            self.api_status.set("❌ OFFLINE")
            self.api_status_label.config(fg='#ff0000')
    
    def check_automation_status(self):
        """Check automation system status"""
        try:
            # Check if automation log exists and is recent
            if os.path.exists('automation.log'):
                # Check if log was updated in last 2 minutes
                if time.time() - os.path.getmtime('automation.log') < 120:
                    self.automation_status.set("✅ RUNNING")
                    self.automation_status_label.config(fg='#00ff00')
                else:
                    self.automation_status.set("⚠️  IDLE")
                    self.automation_status_label.config(fg='#ff6600')
            else:
                self.automation_status.set("❌ STOPPED")
                self.automation_status_label.config(fg='#ff0000')
        except:
            self.automation_status.set("❌ ERROR")
            self.automation_status_label.config(fg='#ff0000')
    
    def check_consciousness_metrics(self):
        """Check consciousness mathematics metrics"""
        try:
            response = requests.get(f"{self.api_endpoint}/api/system/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                metrics = data.get('metrics', {})
                
                # Update consciousness score
                score = metrics.get('consciousness_score', 0)
                self.consciousness_score.set(f"{score:.4f}")
                
                # Update consciousness level
                level = data.get('consciousness_level', 1)
                self.consciousness_level.set(f"{level}/26")
                
                # Update breakthrough count
                breakthroughs = metrics.get('breakthrough_count', 0)
                self.breakthrough_count.set(str(breakthroughs))
                
        except Exception as e:
            self.log_activity(f"❌ Failed to get consciousness metrics: {str(e)}")
    
    def check_log_files(self):
        """Check for new log entries"""
        try:
            # Check automation log
            if os.path.exists('automation.log'):
                with open('automation.log', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if hasattr(self, 'last_log_line') and self.last_log_line != last_line:
                            self.log_activity(f"📝 {last_line}")
                        self.last_log_line = last_line
            
            # Check breakthrough log
            if os.path.exists('breakthroughs.log'):
                with open('breakthroughs.log', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if hasattr(self, 'last_breakthrough_line') and self.last_breakthrough_line != last_line:
                            if "BREAKTHROUGH" in last_line:
                                self.log_activity(f"🚀 {last_line}")
                        self.last_breakthrough_line = last_line
                        
        except Exception as e:
            self.log_activity(f"❌ Log check error: {str(e)}")
    
    def update_progress(self):
        """Update progress bars based on task completion"""
        try:
            # Read daily report if available
            today = datetime.now().strftime('%Y%m%d')
            report_file = f"daily_report_{today}.json"
            
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                # Update progress bars
                research_count = report.get('research_count', 0)
                improvement_count = report.get('improvement_count', 0)
                coding_count = report.get('coding_count', 0)
                
                # Calculate progress percentages (assuming 24 max daily tasks)
                research_progress = min(100, (research_count / 24) * 100)
                improvement_progress = min(100, (improvement_count / 12) * 100)  # Every 2 hours
                coding_progress = min(100, (coding_count / 16) * 100)  # Every 1.5 hours
                
                self.research_progress['value'] = research_progress
                self.improvement_progress['value'] = improvement_progress
                self.coding_progress['value'] = coding_progress
                
        except Exception as e:
            self.log_activity(f"❌ Progress update error: {str(e)}")
    
    def log_activity(self, message):
        """Log activity to the monitor"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        # Add to activity log
        self.activity_log.insert(tk.END, log_entry)
        self.activity_log.see(tk.END)
        
        # Limit log size
        lines = self.activity_log.get('1.0', tk.END).split('\n')
        if len(lines) > 100:
            self.activity_log.delete('1.0', '50.0')
    
    def refresh_status(self):
        """Manual refresh of all status"""
        self.log_activity("🔄 Manual refresh requested")
        self.check_api_status()
        self.check_automation_status()
        self.check_consciousness_metrics()
        self.update_progress()
    
    def view_logs(self):
        """Open log files in system default editor"""
        try:
            import subprocess
            import platform
            
            log_files = ['automation.log', 'breakthroughs.log', 'research_results.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    if platform.system() == 'Darwin':  # macOS
                        subprocess.run(['open', log_file])
                    elif platform.system() == 'Windows':
                        subprocess.run(['notepad', log_file])
                    else:  # Linux
                        subprocess.run(['xdg-open', log_file])
            
            self.log_activity("📊 Opened log files")
            
        except Exception as e:
            self.log_activity(f"❌ Failed to open logs: {str(e)}")
    
    def open_settings(self):
        """Open automation configuration"""
        try:
            import subprocess
            import platform
            
            if os.path.exists('automation_config.json'):
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', 'automation_config.json'])
                elif platform.system() == 'Windows':
                    subprocess.run(['notepad', 'automation_config.json'])
                else:  # Linux
                    subprocess.run(['xdg-open', 'automation_config.json'])
                
                self.log_activity("⚙️  Opened automation settings")
            
        except Exception as e:
            self.log_activity(f"❌ Failed to open settings: {str(e)}")
    
    def exit_monitor(self):
        """Exit the monitor"""
        self.running = False
        self.root.quit()
    
    def run(self):
        """Run the monitor"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🧠 Starting Consciousness Mathematics Automation Monitor")
    print("=" * 60)
    
    monitor = AutomationMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
