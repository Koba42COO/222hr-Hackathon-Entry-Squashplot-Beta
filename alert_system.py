#!/usr/bin/env python3
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class AlertSystem:
    def __init__(self, config_file="monitoring_config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
    
    def send_alert(self, alert_type, message, severity="MEDIUM"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_message = f"[{timestamp}] {severity}: {message}"
        
        print(f"ALERT: {alert_message}")
        
        # Log alert
        with open("alerts.log", "a") as f:
            f.write(f"{alert_message}\n")
        
        # Send email if configured
        if self.config["alerting"]["email_enabled"]:
            self._send_email_alert(alert_type, message, severity)
    
    def _send_email_alert(self, alert_type, message, severity):
        # Email sending logic here
        pass

if __name__ == "__main__":
    alert_system = AlertSystem()
    alert_system.send_alert("test", "This is a test alert", "LOW")
