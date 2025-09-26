#!/usr/bin/env python3
"""
SquashPlot Security Monitor
Real-time security monitoring and alerting system
"""

import json
import time
import threading
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class SecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self, config_file: str = "security_config.json"):
        self.config = self._load_config(config_file)
        self.running = False
        self.alerts = []
        self.metrics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'auth_failures': 0,
            'suspicious_activity': 0,
            'critical_events': 0
        }
        self.threat_level = "LOW"
        
    def _load_config(self, config_file: str) -> Dict:
        """Load security configuration"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "log_file": "security_audit.log",
                "alert_threshold": 10,
                "critical_threshold": 5
            },
            "alerts": {
                "email_enabled": False,
                "email_smtp": "smtp.gmail.com",
                "email_port": 587,
                "email_user": "",
                "email_password": "",
                "alert_recipients": []
            },
            "threat_detection": {
                "suspicious_patterns": [
                    r"rm\s+-rf",
                    r"sudo\s+",
                    r"chmod\s+777",
                    r"cat\s+/etc/passwd",
                    r"wget\s+",
                    r"curl\s+",
                    r"nc\s+",
                    r"bash\s+-i"
                ],
                "rate_limit_threshold": 20,
                "consecutive_failures": 5
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        return default_config
    
    def start_monitoring(self):
        """Start the security monitoring system"""
        self.running = True
        print("🛡️ Starting SquashPlot Security Monitor")
        print("=" * 50)
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._monitor_logs, daemon=True),
            threading.Thread(target=self._monitor_metrics, daemon=True),
            threading.Thread(target=self._monitor_threats, daemon=True),
            threading.Thread(target=self._generate_reports, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print("✅ Security monitoring active")
        print("📊 Monitoring logs, metrics, and threats")
        print("🚨 Alerts will be sent for critical events")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping security monitor...")
            self.running = False
    
    def _monitor_logs(self):
        """Monitor security audit logs"""
        log_file = self.config["monitoring"]["log_file"]
        last_position = 0
        
        while self.running:
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                        for line in new_lines:
                            self._process_log_entry(line.strip())
                            
            except Exception as e:
                print(f"Error monitoring logs: {e}")
            
            time.sleep(1)
    
    def _process_log_entry(self, log_entry: str):
        """Process a single log entry for security analysis"""
        try:
            entry = json.loads(log_entry)
            event_type = entry.get('event_type', '')
            risk_level = entry.get('risk_level', 'LOW')
            
            # Update metrics
            self.metrics['total_requests'] += 1
            
            if event_type == 'authentication_failure':
                self.metrics['auth_failures'] += 1
                self._check_auth_brute_force(entry)
            
            elif event_type == 'dangerous_command_blocked':
                self.metrics['blocked_requests'] += 1
                self._check_suspicious_activity(entry)
            
            elif risk_level == 'HIGH':
                self.metrics['critical_events'] += 1
                self._handle_critical_event(entry)
            
            # Check threat level
            self._update_threat_level()
            
        except json.JSONDecodeError:
            pass  # Skip invalid JSON
        except Exception as e:
            print(f"Error processing log entry: {e}")
    
    def _check_auth_brute_force(self, entry: Dict):
        """Check for brute force authentication attempts"""
        client_ip = entry.get('client_ip', '')
        timestamp = entry.get('timestamp', '')
        
        # Count recent failures from same IP
        recent_failures = self._count_recent_failures(client_ip, 300)  # 5 minutes
        
        if recent_failures >= self.config["threat_detection"]["consecutive_failures"]:
            self._create_alert({
                'type': 'BRUTE_FORCE',
                'severity': 'HIGH',
                'message': f'Brute force attack detected from {client_ip}',
                'client_ip': client_ip,
                'timestamp': timestamp,
                'details': f'{recent_failures} failed attempts in 5 minutes'
            })
    
    def _check_suspicious_activity(self, entry: Dict):
        """Check for suspicious activity patterns"""
        command = entry.get('command', '')
        client_ip = entry.get('client_ip', '')
        
        # Check for suspicious patterns
        suspicious_patterns = self.config["threat_detection"]["suspicious_patterns"]
        for pattern in suspicious_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                self.metrics['suspicious_activity'] += 1
                self._create_alert({
                    'type': 'SUSPICIOUS_COMMAND',
                    'severity': 'MEDIUM',
                    'message': f'Suspicious command detected: {command[:50]}...',
                    'client_ip': client_ip,
                    'command': command,
                    'pattern': pattern
                })
                break
    
    def _handle_critical_event(self, entry: Dict):
        """Handle critical security events"""
        self._create_alert({
            'type': 'CRITICAL_EVENT',
            'severity': 'CRITICAL',
            'message': f'Critical security event: {entry.get("event_type", "unknown")}',
            'details': entry
        })
    
    def _count_recent_failures(self, client_ip: str, time_window: int) -> int:
        """Count recent authentication failures from an IP"""
        # This would normally query the log file
        # For demo purposes, we'll simulate
        return 0
    
    def _create_alert(self, alert_data: Dict):
        """Create a security alert"""
        alert = {
            'id': f"alert_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'threat_level': self.threat_level,
            **alert_data
        }
        
        self.alerts.append(alert)
        
        # Send alert if configured
        if self.config["alerts"]["email_enabled"]:
            self._send_email_alert(alert)
        
        # Print alert
        severity_icon = "🔴" if alert['severity'] == 'CRITICAL' else "🟠" if alert['severity'] == 'HIGH' else "🟡"
        print(f"{severity_icon} ALERT: {alert['message']}")
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["alerts"]["email_user"]
            msg['To'] = ", ".join(self.config["alerts"]["alert_recipients"])
            msg['Subject'] = f"SquashPlot Security Alert: {alert['type']}"
            
            body = f"""
Security Alert Details:
- Type: {alert['type']}
- Severity: {alert['severity']}
- Message: {alert['message']}
- Timestamp: {alert['timestamp']}
- Threat Level: {self.threat_level}

Details: {json.dumps(alert.get('details', {}), indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config["alerts"]["email_smtp"], self.config["alerts"]["email_port"])
            server.starttls()
            server.login(self.config["alerts"]["email_user"], self.config["alerts"]["email_password"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def _monitor_metrics(self):
        """Monitor system metrics and performance"""
        while self.running:
            try:
                # Check system resources
                self._check_system_resources()
                
                # Check for anomalies
                self._check_anomalies()
                
            except Exception as e:
                print(f"Error monitoring metrics: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # Check disk space
            disk_usage = os.statvfs('/')
            free_space = disk_usage.f_frsize * disk_usage.f_bavail
            total_space = disk_usage.f_frsize * disk_usage.f_blocks
            free_percent = (free_space / total_space) * 100
            
            if free_percent < 10:  # Less than 10% free
                self._create_alert({
                    'type': 'LOW_DISK_SPACE',
                    'severity': 'MEDIUM',
                    'message': f'Low disk space: {free_percent:.1f}% free',
                    'free_percent': free_percent
                })
        except Exception as e:
            pass  # Skip if can't check disk space
    
    def _check_anomalies(self):
        """Check for anomalous activity"""
        # Check for unusual request patterns
        if self.metrics['total_requests'] > 0:
            blocked_rate = (self.metrics['blocked_requests'] / self.metrics['total_requests']) * 100
            
            if blocked_rate > 50:  # More than 50% blocked
                self._create_alert({
                    'type': 'HIGH_BLOCK_RATE',
                    'severity': 'MEDIUM',
                    'message': f'High block rate: {blocked_rate:.1f}% of requests blocked',
                    'blocked_rate': blocked_rate
                })
    
    def _monitor_threats(self):
        """Monitor for emerging threats"""
        while self.running:
            try:
                # Check for new threat patterns
                self._check_threat_intelligence()
                
                # Update threat level
                self._update_threat_level()
                
            except Exception as e:
                print(f"Error monitoring threats: {e}")
            
            time.sleep(60)  # Check every minute
    
    def _check_threat_intelligence(self):
        """Check for new threat intelligence"""
        # This would normally connect to threat intelligence feeds
        # For demo purposes, we'll simulate
        pass
    
    def _update_threat_level(self):
        """Update the current threat level"""
        critical_events = self.metrics['critical_events']
        suspicious_activity = self.metrics['suspicious_activity']
        
        if critical_events > 10 or suspicious_activity > 50:
            self.threat_level = "CRITICAL"
        elif critical_events > 5 or suspicious_activity > 20:
            self.threat_level = "HIGH"
        elif critical_events > 2 or suspicious_activity > 10:
            self.threat_level = "MEDIUM"
        else:
            self.threat_level = "LOW"
    
    def _generate_reports(self):
        """Generate periodic security reports"""
        while self.running:
            try:
                # Generate hourly reports
                self._generate_hourly_report()
                
            except Exception as e:
                print(f"Error generating reports: {e}")
            
            time.sleep(3600)  # Every hour
    
    def _generate_hourly_report(self):
        """Generate hourly security report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'threat_level': self.threat_level,
            'metrics': self.metrics.copy(),
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'summary': self._generate_summary()
        }
        
        # Save report
        report_file = f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Security report generated: {report_file}")
    
    def _generate_summary(self) -> str:
        """Generate security summary"""
        total_requests = self.metrics['total_requests']
        blocked_requests = self.metrics['blocked_requests']
        auth_failures = self.metrics['auth_failures']
        critical_events = self.metrics['critical_events']
        
        summary = f"""
Security Summary:
- Total Requests: {total_requests}
- Blocked Requests: {blocked_requests}
- Auth Failures: {auth_failures}
- Critical Events: {critical_events}
- Threat Level: {self.threat_level}
        """
        
        return summary.strip()
    
    def get_security_status(self) -> Dict:
        """Get current security status"""
        return {
            'threat_level': self.threat_level,
            'metrics': self.metrics,
            'recent_alerts': self.alerts[-5:],
            'monitoring_active': self.running
        }

def main():
    """Main entry point for security monitor"""
    print("🛡️ SquashPlot Security Monitor")
    print("=" * 50)
    
    try:
        monitor = SecurityMonitor()
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Security monitor stopped")
    except Exception as e:
        print(f"❌ Error starting security monitor: {e}")

if __name__ == "__main__":
    main()
