#!/usr/bin/env python3
"""
SquashPlot Compliance Monitor
Automated compliance checking and reporting system
"""

import json
import os
import re
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class ComplianceMonitor:
    """Automated compliance monitoring system"""
    
    def __init__(self, config_file: str = "compliance_config.json"):
        self.config = self._load_config(config_file)
        self.compliance_status = {}
        self.violations = []
        self.reports = []
        self.running = False
        
    def _load_config(self, config_file: str) -> Dict:
        """Load compliance configuration"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "check_interval": 3600,  # 1 hour
                "report_interval": 86400,  # 24 hours
                "alert_threshold": 5
            },
            "compliance_standards": {
                "gdpr": {
                    "enabled": True,
                    "required_elements": [
                        "privacy_policy",
                        "cookie_policy", 
                        "data_retention_policy",
                        "user_consent_mechanism",
                        "data_subject_rights",
                        "breach_notification_procedure"
                    ]
                },
                "ccpa": {
                    "enabled": True,
                    "required_elements": [
                        "privacy_notice",
                        "opt_out_mechanism",
                        "consumer_rights_disclosure",
                        "data_collection_disclosure"
                    ]
                },
                "ada": {
                    "enabled": True,
                    "required_elements": [
                        "alt_text_images",
                        "keyboard_navigation",
                        "color_contrast",
                        "screen_reader_compatibility",
                        "accessibility_statement"
                    ]
                },
                "general": {
                    "enabled": True,
                    "required_elements": [
                        "terms_of_service",
                        "privacy_policy",
                        "contact_information",
                        "copyright_notice",
                        "disclaimer"
                    ]
                }
            },
            "alerts": {
                "email_enabled": False,
                "email_smtp": "smtp.gmail.com",
                "email_port": 587,
                "email_user": "",
                "email_password": "",
                "alert_recipients": []
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load compliance config: {e}")
        
        return default_config
    
    def start_monitoring(self):
        """Start compliance monitoring"""
        self.running = True
        print("📋 Starting SquashPlot Compliance Monitor")
        print("=" * 50)
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._monitor_compliance, daemon=True),
            threading.Thread(target=self._generate_reports, daemon=True),
            threading.Thread(target=self._check_violations, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print("✅ Compliance monitoring active")
        print("📊 Monitoring GDPR, CCPA, ADA, and general compliance")
        print("🚨 Alerts will be sent for violations")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping compliance monitor...")
            self.running = False
    
    def _monitor_compliance(self):
        """Monitor compliance status"""
        while self.running:
            try:
                # Check all compliance standards
                self._check_gdpr_compliance()
                self._check_ccpa_compliance()
                self._check_ada_compliance()
                self._check_general_compliance()
                
                # Update overall compliance status
                self._update_compliance_status()
                
            except Exception as e:
                print(f"Error monitoring compliance: {e}")
            
            time.sleep(self.config["monitoring"]["check_interval"])
    
    def _check_gdpr_compliance(self):
        """Check GDPR compliance"""
        if not self.config["compliance_standards"]["gdpr"]["enabled"]:
            return
        
        gdpr_status = {
            "standard": "GDPR",
            "status": "COMPLIANT",
            "violations": [],
            "last_checked": datetime.now().isoformat()
        }
        
        required_elements = self.config["compliance_standards"]["gdpr"]["required_elements"]
        
        for element in required_elements:
            if not self._check_element_exists(element):
                gdpr_status["status"] = "NON_COMPLIANT"
                gdpr_status["violations"].append({
                    "element": element,
                    "severity": "HIGH",
                    "description": f"Missing {element}",
                    "recommendation": f"Implement {element} to comply with GDPR"
                })
        
        # Check for GDPR-specific requirements
        if not self._check_privacy_policy_gdpr():
            gdpr_status["status"] = "NON_COMPLIANT"
            gdpr_status["violations"].append({
                "element": "gdpr_privacy_policy",
                "severity": "CRITICAL",
                "description": "Privacy policy does not meet GDPR requirements",
                "recommendation": "Update privacy policy to include GDPR-specific disclosures"
            })
        
        self.compliance_status["gdpr"] = gdpr_status
    
    def _check_ccpa_compliance(self):
        """Check CCPA compliance"""
        if not self.config["compliance_standards"]["ccpa"]["enabled"]:
            return
        
        ccpa_status = {
            "standard": "CCPA",
            "status": "COMPLIANT",
            "violations": [],
            "last_checked": datetime.now().isoformat()
        }
        
        required_elements = self.config["compliance_standards"]["ccpa"]["required_elements"]
        
        for element in required_elements:
            if not self._check_element_exists(element):
                ccpa_status["status"] = "NON_COMPLIANT"
                ccpa_status["violations"].append({
                    "element": element,
                    "severity": "HIGH",
                    "description": f"Missing {element}",
                    "recommendation": f"Implement {element} to comply with CCPA"
                })
        
        # Check for CCPA-specific requirements
        if not self._check_opt_out_mechanism():
            ccpa_status["status"] = "NON_COMPLIANT"
            ccpa_status["violations"].append({
                "element": "opt_out_mechanism",
                "severity": "CRITICAL",
                "description": "No opt-out mechanism for data sale",
                "recommendation": "Implement clear opt-out mechanism for CCPA compliance"
            })
        
        self.compliance_status["ccpa"] = ccpa_status
    
    def _check_ada_compliance(self):
        """Check ADA compliance"""
        if not self.config["compliance_standards"]["ada"]["enabled"]:
            return
        
        ada_status = {
            "standard": "ADA",
            "status": "COMPLIANT",
            "violations": [],
            "last_checked": datetime.now().isoformat()
        }
        
        required_elements = self.config["compliance_standards"]["ada"]["required_elements"]
        
        for element in required_elements:
            if not self._check_element_exists(element):
                ada_status["status"] = "NON_COMPLIANT"
                ada_status["violations"].append({
                    "element": element,
                    "severity": "MEDIUM",
                    "description": f"Missing {element}",
                    "recommendation": f"Implement {element} for ADA compliance"
                })
        
        # Check for ADA-specific requirements
        if not self._check_accessibility_features():
            ada_status["status"] = "NON_COMPLIANT"
            ada_status["violations"].append({
                "element": "accessibility_features",
                "severity": "HIGH",
                "description": "Website lacks accessibility features",
                "recommendation": "Implement WCAG 2.1 accessibility standards"
            })
        
        self.compliance_status["ada"] = ada_status
    
    def _check_general_compliance(self):
        """Check general compliance requirements"""
        if not self.config["compliance_standards"]["general"]["enabled"]:
            return
        
        general_status = {
            "standard": "GENERAL",
            "status": "COMPLIANT",
            "violations": [],
            "last_checked": datetime.now().isoformat()
        }
        
        required_elements = self.config["compliance_standards"]["general"]["required_elements"]
        
        for element in required_elements:
            if not self._check_element_exists(element):
                general_status["status"] = "NON_COMPLIANT"
                general_status["violations"].append({
                    "element": element,
                    "severity": "MEDIUM",
                    "description": f"Missing {element}",
                    "recommendation": f"Implement {element} for general compliance"
                })
        
        self.compliance_status["general"] = general_status
    
    def _check_element_exists(self, element: str) -> bool:
        """Check if a compliance element exists"""
        # Map elements to files or features
        element_map = {
            "privacy_policy": "PRIVACY_POLICY.md",
            "terms_of_service": "TERMS_OF_SERVICE.md",
            "cookie_policy": "COOKIE_POLICY.md",
            "data_retention_policy": "DATA_RETENTION_POLICY.md",
            "accessibility_statement": "ACCESSIBILITY_STATEMENT.md",
            "contact_information": "CONTACT_INFO.md",
            "copyright_notice": "COPYRIGHT_NOTICE.md",
            "disclaimer": "DISCLAIMER.md"
        }
        
        if element in element_map:
            return os.path.exists(element_map[element])
        
        # Check for specific features
        if element == "user_consent_mechanism":
            return self._check_consent_mechanism()
        elif element == "opt_out_mechanism":
            return self._check_opt_out_mechanism()
        elif element == "alt_text_images":
            return self._check_alt_text()
        elif element == "keyboard_navigation":
            return self._check_keyboard_navigation()
        elif element == "color_contrast":
            return self._check_color_contrast()
        elif element == "screen_reader_compatibility":
            return self._check_screen_reader()
        
        return False
    
    def _check_privacy_policy_gdpr(self) -> bool:
        """Check if privacy policy meets GDPR requirements"""
        try:
            if os.path.exists("PRIVACY_POLICY.md"):
                with open("PRIVACY_POLICY.md", 'r') as f:
                    content = f.read().lower()
                    
                # Check for GDPR-specific elements
                gdpr_elements = [
                    "lawful basis",
                    "data subject rights",
                    "data retention",
                    "breach notification",
                    "data protection officer",
                    "consent",
                    "legitimate interest"
                ]
                
                found_elements = sum(1 for element in gdpr_elements if element in content)
                return found_elements >= 5  # At least 5 GDPR elements present
        except Exception:
            pass
        
        return False
    
    def _check_consent_mechanism(self) -> bool:
        """Check if user consent mechanism exists"""
        # Check for consent banner or mechanism
        return True  # Simplified for demo
    
    def _check_opt_out_mechanism(self) -> bool:
        """Check if opt-out mechanism exists"""
        # Check for opt-out functionality
        return True  # Simplified for demo
    
    def _check_accessibility_features(self) -> bool:
        """Check if accessibility features are implemented"""
        # Check for accessibility features
        return True  # Simplified for demo
    
    def _check_alt_text(self) -> bool:
        """Check if images have alt text"""
        return True  # Simplified for demo
    
    def _check_keyboard_navigation(self) -> bool:
        """Check if keyboard navigation is supported"""
        return True  # Simplified for demo
    
    def _check_color_contrast(self) -> bool:
        """Check if color contrast meets standards"""
        return True  # Simplified for demo
    
    def _check_screen_reader(self) -> bool:
        """Check if screen reader compatibility exists"""
        return True  # Simplified for demo
    
    def _update_compliance_status(self):
        """Update overall compliance status"""
        total_standards = len(self.compliance_status)
        compliant_standards = sum(1 for status in self.compliance_status.values() 
                                if status["status"] == "COMPLIANT")
        
        if compliant_standards == total_standards:
            overall_status = "FULLY_COMPLIANT"
        elif compliant_standards > total_standards // 2:
            overall_status = "PARTIALLY_COMPLIANT"
        else:
            overall_status = "NON_COMPLIANT"
        
        self.compliance_status["overall"] = {
            "status": overall_status,
            "compliant_standards": compliant_standards,
            "total_standards": total_standards,
            "compliance_percentage": (compliant_standards / total_standards) * 100,
            "last_updated": datetime.now().isoformat()
        }
    
    def _check_violations(self):
        """Check for compliance violations"""
        while self.running:
            try:
                violations = []
                
                for standard, status in self.compliance_status.items():
                    if standard != "overall" and status["status"] == "NON_COMPLIANT":
                        violations.extend(status["violations"])
                
                if len(violations) >= self.config["monitoring"]["alert_threshold"]:
                    self._send_compliance_alert(violations)
                
            except Exception as e:
                print(f"Error checking violations: {e}")
            
            time.sleep(3600)  # Check every hour
    
    def _send_compliance_alert(self, violations: List[Dict]):
        """Send compliance alert"""
        if not self.config["alerts"]["email_enabled"]:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["alerts"]["email_user"]
            msg['To'] = ", ".join(self.config["alerts"]["alert_recipients"])
            msg['Subject'] = "SquashPlot Compliance Alert - Violations Detected"
            
            body = f"""
Compliance Alert Details:
- Total Violations: {len(violations)}
- Critical Issues: {len([v for v in violations if v['severity'] == 'CRITICAL'])}
- High Priority Issues: {len([v for v in violations if v['severity'] == 'HIGH'])}
- Medium Priority Issues: {len([v for v in violations if v['severity'] == 'MEDIUM'])}

Violations:
{json.dumps(violations, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config["alerts"]["email_smtp"], self.config["alerts"]["email_port"])
            server.starttls()
            server.login(self.config["alerts"]["email_user"], self.config["alerts"]["email_password"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f"Failed to send compliance alert: {e}")
    
    def _generate_reports(self):
        """Generate compliance reports"""
        while self.running:
            try:
                self._create_compliance_report()
            except Exception as e:
                print(f"Error generating reports: {e}")
            
            time.sleep(self.config["monitoring"]["report_interval"])
    
    def _create_compliance_report(self):
        """Create compliance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'compliance_status': self.compliance_status,
            'summary': self._generate_compliance_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_file = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Compliance report generated: {report_file}")
    
    def _generate_compliance_summary(self) -> str:
        """Generate compliance summary"""
        overall = self.compliance_status.get("overall", {})
        status = overall.get("status", "UNKNOWN")
        percentage = overall.get("compliance_percentage", 0)
        
        summary = f"""
Compliance Summary:
- Overall Status: {status}
- Compliance Percentage: {percentage:.1f}%
- Standards Monitored: {len(self.compliance_status) - 1}
- Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return summary.strip()
    
    def _generate_recommendations(self) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for standard, status in self.compliance_status.items():
            if standard != "overall" and status["status"] == "NON_COMPLIANT":
                for violation in status["violations"]:
                    recommendations.append(violation["recommendation"])
        
        return recommendations
    
    def get_compliance_status(self) -> Dict:
        """Get current compliance status"""
        return {
            'compliance_status': self.compliance_status,
            'monitoring_active': self.running,
            'last_checked': datetime.now().isoformat()
        }

def main():
    """Main entry point for compliance monitor"""
    print("📋 SquashPlot Compliance Monitor")
    print("=" * 50)
    
    try:
        monitor = ComplianceMonitor()
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Compliance monitor stopped")
    except Exception as e:
        print(f"❌ Error starting compliance monitor: {e}")

if __name__ == "__main__":
    main()
