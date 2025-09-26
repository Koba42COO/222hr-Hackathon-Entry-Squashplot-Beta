#!/usr/bin/env python3
"""
SquashPlot Deployment Readiness Checker
Comprehensive pre-deployment validation system
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import importlib.util

class DeploymentReadinessChecker:
    """Comprehensive deployment readiness validation"""
    
    def __init__(self):
        self.readiness_checks = {
            "security": {
                "priority": "CRITICAL",
                "checks": [
                    "hardened_bridge_exists",
                    "security_config_exists", 
                    "authentication_implemented",
                    "input_validation_active",
                    "audit_logging_enabled",
                    "rate_limiting_configured"
                ]
            },
            "legal": {
                "priority": "CRITICAL", 
                "checks": [
                    "terms_of_service_exists",
                    "privacy_policy_exists",
                    "cookie_policy_exists",
                    "data_retention_policy_exists",
                    "accessibility_statement_exists",
                    "eula_exists"
                ]
            },
            "compliance": {
                "priority": "HIGH",
                "checks": [
                    "gdpr_compliance",
                    "ccpa_compliance", 
                    "ada_compliance",
                    "legal_review_completed"
                ]
            },
            "operational": {
                "priority": "HIGH",
                "checks": [
                    "backup_system_implemented",
                    "monitoring_system_active",
                    "recovery_procedures_documented",
                    "incident_response_plan_exists"
                ]
            },
            "technical": {
                "priority": "MEDIUM",
                "checks": [
                    "dependencies_installed",
                    "configuration_valid",
                    "testing_completed",
                    "documentation_complete"
                ]
            }
        }
        
        self.required_files = {
            "security": [
                "secure_bridge_hardened.py",
                "security_config.json",
                "security_test_suite.py",
                "penetration_test.py",
                "security_monitor.py"
            ],
            "legal": [
                "TERMS_OF_SERVICE.md",
                "PRIVACY_POLICY.md",
                "COOKIE_POLICY.md", 
                "DATA_RETENTION_POLICY.md",
                "ACCESSIBILITY_STATEMENT.md",
                "EULA.md"
            ],
            "operational": [
                "backup_system.py",
                "recovery_procedures.json",
                "monitoring_config.json",
                "alert_system.py"
            ],
            "compliance": [
                "legal_compliance_checker.py",
                "compliance_monitor.py",
                "user_agreement_manager.py"
            ]
        }
        
        self.check_results = {}
        self.blocking_issues = []
        self.warnings = []
        self.recommendations = []
    
    def run_readiness_check(self) -> Dict:
        """Run comprehensive deployment readiness check"""
        print("🚀 SquashPlot Deployment Readiness Checker")
        print("=" * 70)
        print("Validating system readiness for production deployment...")
        print()
        
        # Run all readiness checks
        for category, config in self.readiness_checks.items():
            print(f"🔍 Checking {category.upper()} readiness...")
            self._check_category(category, config)
        
        # Generate readiness report
        report = self._generate_readiness_report()
        
        return report
    
    def _check_category(self, category: str, config: Dict):
        """Check readiness for a specific category"""
        category_results = {
            "category": category,
            "priority": config["priority"],
            "checks": {},
            "passed": 0,
            "failed": 0,
            "blocking": False
        }
        
        for check in config["checks"]:
            result = self._run_specific_check(category, check)
            category_results["checks"][check] = result
            
            if result["status"] == "PASS":
                category_results["passed"] += 1
            else:
                category_results["failed"] += 1
                if result["status"] == "BLOCKING":
                    category_results["blocking"] = True
                    self.blocking_issues.append({
                        "category": category,
                        "check": check,
                        "issue": result["issue"],
                        "recommendation": result["recommendation"]
                    })
                elif result["status"] == "WARNING":
                    self.warnings.append({
                        "category": category,
                        "check": check,
                        "issue": result["issue"],
                        "recommendation": result["recommendation"]
                    })
        
        self.check_results[category] = category_results
        
        # Print category summary
        total_checks = len(config["checks"])
        passed_checks = category_results["passed"]
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate >= 70 else "❌"
        print(f"   {status_icon} {category.title()}: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
    
    def _run_specific_check(self, category: str, check: str) -> Dict:
        """Run a specific readiness check"""
        try:
            if category == "security":
                return self._check_security_item(check)
            elif category == "legal":
                return self._check_legal_item(check)
            elif category == "compliance":
                return self._check_compliance_item(check)
            elif category == "operational":
                return self._check_operational_item(check)
            elif category == "technical":
                return self._check_technical_item(check)
            else:
                return {"status": "UNKNOWN", "issue": f"Unknown category: {category}"}
        except Exception as e:
            return {"status": "ERROR", "issue": f"Check failed: {str(e)}"}
    
    def _check_security_item(self, check: str) -> Dict:
        """Check security-related items"""
        if check == "hardened_bridge_exists":
            if os.path.exists("secure_bridge_hardened.py"):
                return {"status": "PASS", "message": "Hardened bridge app found"}
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Hardened bridge app not found",
                    "recommendation": "Implement secure_bridge_hardened.py before deployment"
                }
        
        elif check == "security_config_exists":
            if os.path.exists("security_config.json"):
                return {"status": "PASS", "message": "Security configuration found"}
            else:
                return {
                    "status": "BLOCKING", 
                    "issue": "Security configuration missing",
                    "recommendation": "Create security_config.json with security settings"
                }
        
        elif check == "authentication_implemented":
            if os.path.exists("secure_bridge_hardened.py"):
                with open("secure_bridge_hardened.py", 'r') as f:
                    content = f.read()
                    if "SecureAuthentication" in content and "validate_auth" in content:
                        return {"status": "PASS", "message": "Authentication system implemented"}
                    else:
                        return {
                            "status": "BLOCKING",
                            "issue": "Authentication system not properly implemented",
                            "recommendation": "Ensure authentication is fully implemented in hardened bridge"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Hardened bridge app not found",
                    "recommendation": "Implement hardened bridge app with authentication"
                }
        
        elif check == "input_validation_active":
            if os.path.exists("secure_bridge_hardened.py"):
                with open("secure_bridge_hardened.py", 'r') as f:
                    content = f.read()
                    if "CommandSanitizer" in content and "sanitize_command" in content:
                        return {"status": "PASS", "message": "Input validation implemented"}
                    else:
                        return {
                            "status": "BLOCKING",
                            "issue": "Input validation not implemented",
                            "recommendation": "Implement comprehensive input validation"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Hardened bridge app not found",
                    "recommendation": "Implement hardened bridge app with input validation"
                }
        
        elif check == "audit_logging_enabled":
            if os.path.exists("secure_bridge_hardened.py"):
                with open("secure_bridge_hardened.py", 'r') as f:
                    content = f.read()
                    if "SecurityAuditLogger" in content and "log_security_event" in content:
                        return {"status": "PASS", "message": "Audit logging implemented"}
                    else:
                        return {
                            "status": "BLOCKING",
                            "issue": "Audit logging not implemented",
                            "recommendation": "Implement comprehensive audit logging"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Hardened bridge app not found",
                    "recommendation": "Implement hardened bridge app with audit logging"
                }
        
        elif check == "rate_limiting_configured":
            if os.path.exists("secure_bridge_hardened.py"):
                with open("secure_bridge_hardened.py", 'r') as f:
                    content = f.read()
                    if "RateLimiter" in content and "is_allowed" in content:
                        return {"status": "PASS", "message": "Rate limiting implemented"}
                    else:
                        return {
                            "status": "BLOCKING",
                            "issue": "Rate limiting not implemented",
                            "recommendation": "Implement rate limiting to prevent DoS attacks"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Hardened bridge app not found",
                    "recommendation": "Implement hardened bridge app with rate limiting"
                }
        
        return {"status": "UNKNOWN", "issue": f"Unknown security check: {check}"}
    
    def _check_legal_item(self, check: str) -> Dict:
        """Check legal-related items"""
        legal_files = {
            "terms_of_service_exists": "TERMS_OF_SERVICE.md",
            "privacy_policy_exists": "PRIVACY_POLICY.md",
            "cookie_policy_exists": "COOKIE_POLICY.md",
            "data_retention_policy_exists": "DATA_RETENTION_POLICY.md",
            "accessibility_statement_exists": "ACCESSIBILITY_STATEMENT.md",
            "eula_exists": "EULA.md"
        }
        
        if check in legal_files:
            filename = legal_files[check]
            if os.path.exists(filename):
                # Check if file has content
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content) > 100:  # Basic content check
                        return {"status": "PASS", "message": f"{filename} exists with content"}
                    else:
                        return {
                            "status": "WARNING",
                            "issue": f"{filename} exists but appears incomplete",
                            "recommendation": f"Complete {filename} with comprehensive content"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": f"{filename} not found",
                    "recommendation": f"Create {filename} before deployment"
                }
        
        return {"status": "UNKNOWN", "issue": f"Unknown legal check: {check}"}
    
    def _check_compliance_item(self, check: str) -> Dict:
        """Check compliance-related items"""
        if check == "gdpr_compliance":
            if os.path.exists("PRIVACY_POLICY.md"):
                with open("PRIVACY_POLICY.md", 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    gdpr_keywords = ["consent", "data subject rights", "lawful basis", "breach notification"]
                    found_keywords = [kw for kw in gdpr_keywords if kw in content]
                    if len(found_keywords) >= 3:
                        return {"status": "PASS", "message": "GDPR compliance indicators found"}
                    else:
                        return {
                            "status": "WARNING",
                            "issue": "GDPR compliance may be incomplete",
                            "recommendation": "Ensure privacy policy includes all GDPR requirements"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Privacy policy not found",
                    "recommendation": "Create privacy policy with GDPR compliance"
                }
        
        elif check == "ccpa_compliance":
            if os.path.exists("PRIVACY_POLICY.md"):
                with open("PRIVACY_POLICY.md", 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    ccpa_keywords = ["consumer rights", "opt-out", "personal information", "data sale"]
                    found_keywords = [kw for kw in ccpa_keywords if kw in content]
                    if len(found_keywords) >= 2:
                        return {"status": "PASS", "message": "CCPA compliance indicators found"}
                    else:
                        return {
                            "status": "WARNING",
                            "issue": "CCPA compliance may be incomplete",
                            "recommendation": "Ensure privacy policy includes CCPA requirements"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Privacy policy not found",
                    "recommendation": "Create privacy policy with CCPA compliance"
                }
        
        elif check == "ada_compliance":
            if os.path.exists("ACCESSIBILITY_STATEMENT.md"):
                with open("ACCESSIBILITY_STATEMENT.md", 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    ada_keywords = ["wcag", "accessibility", "screen reader", "keyboard navigation"]
                    found_keywords = [kw for kw in ada_keywords if kw in content]
                    if len(found_keywords) >= 2:
                        return {"status": "PASS", "message": "ADA compliance indicators found"}
                    else:
                        return {
                            "status": "WARNING",
                            "issue": "ADA compliance may be incomplete",
                            "recommendation": "Ensure accessibility statement includes WCAG compliance"
                        }
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Accessibility statement not found",
                    "recommendation": "Create accessibility statement for ADA compliance"
                }
        
        elif check == "legal_review_completed":
            # This would normally check for legal review documentation
            return {
                "status": "WARNING",
                "issue": "Legal review status unknown",
                "recommendation": "Ensure all legal documents are reviewed by qualified legal counsel"
            }
        
        return {"status": "UNKNOWN", "issue": f"Unknown compliance check: {check}"}
    
    def _check_operational_item(self, check: str) -> Dict:
        """Check operational-related items"""
        if check == "backup_system_implemented":
            if os.path.exists("backup_system.py"):
                return {"status": "PASS", "message": "Backup system implemented"}
            else:
                return {
                    "status": "BLOCKING",
                    "issue": "Backup system not implemented",
                    "recommendation": "Implement comprehensive backup system before deployment"
                }
        
        elif check == "monitoring_system_active":
            if os.path.exists("monitoring_config.json") and os.path.exists("alert_system.py"):
                return {"status": "PASS", "message": "Monitoring system implemented"}
            else:
                return {
                    "status": "WARNING",
                    "issue": "Monitoring system incomplete",
                    "recommendation": "Implement comprehensive monitoring and alerting system"
                }
        
        elif check == "recovery_procedures_documented":
            if os.path.exists("recovery_procedures.json"):
                return {"status": "PASS", "message": "Recovery procedures documented"}
            else:
                return {
                    "status": "WARNING",
                    "issue": "Recovery procedures not documented",
                    "recommendation": "Document comprehensive recovery procedures"
                }
        
        elif check == "incident_response_plan_exists":
            # This would check for incident response documentation
            return {
                "status": "WARNING",
                "issue": "Incident response plan status unknown",
                "recommendation": "Create comprehensive incident response plan"
            }
        
        return {"status": "UNKNOWN", "issue": f"Unknown operational check: {check}"}
    
    def _check_technical_item(self, check: str) -> Dict:
        """Check technical-related items"""
        if check == "dependencies_installed":
            # Check if required Python packages are available
            required_packages = ["fastapi", "uvicorn", "requests", "cryptography"]
            missing_packages = []
            
            for package in required_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_packages.append(package)
            
            if not missing_packages:
                return {"status": "PASS", "message": "All required dependencies available"}
            else:
                return {
                    "status": "WARNING",
                    "issue": f"Missing dependencies: {', '.join(missing_packages)}",
                    "recommendation": f"Install missing packages: pip install {' '.join(missing_packages)}"
                }
        
        elif check == "configuration_valid":
            config_files = ["security_config.json", "monitoring_config.json"]
            valid_configs = 0
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            json.load(f)
                        valid_configs += 1
                    except json.JSONDecodeError:
                        pass
            
            if valid_configs == len(config_files):
                return {"status": "PASS", "message": "Configuration files are valid"}
            else:
                return {
                    "status": "WARNING",
                    "issue": "Some configuration files are invalid or missing",
                    "recommendation": "Validate all configuration files"
                }
        
        elif check == "testing_completed":
            test_files = ["security_test_suite.py", "penetration_test.py"]
            existing_tests = [f for f in test_files if os.path.exists(f)]
            
            if len(existing_tests) == len(test_files):
                return {"status": "PASS", "message": "Testing frameworks available"}
            else:
                return {
                    "status": "WARNING",
                    "issue": "Some testing frameworks missing",
                    "recommendation": "Implement comprehensive testing before deployment"
                }
        
        elif check == "documentation_complete":
            doc_files = ["README.md", "SECURITY_IMPLEMENTATION_GUIDE.md"]
            existing_docs = [f for f in doc_files if os.path.exists(f)]
            
            if len(existing_docs) == len(doc_files):
                return {"status": "PASS", "message": "Documentation available"}
            else:
                return {
                    "status": "WARNING",
                    "issue": "Some documentation missing",
                    "recommendation": "Complete all documentation before deployment"
                }
        
        return {"status": "UNKNOWN", "issue": f"Unknown technical check: {check}"}
    
    def _generate_readiness_report(self) -> Dict:
        """Generate comprehensive readiness report"""
        print("\n" + "=" * 70)
        print("📊 DEPLOYMENT READINESS REPORT")
        print("=" * 70)
        
        # Calculate overall readiness
        total_checks = sum(len(config["checks"]) for config in self.readiness_checks.values())
        total_passed = sum(result["passed"] for result in self.check_results.values())
        overall_readiness = (total_passed / total_checks * 100) if total_checks > 0 else 0
        
        # Determine deployment status
        if self.blocking_issues:
            deployment_status = "NOT READY - BLOCKING ISSUES"
            status_icon = "🚫"
        elif overall_readiness >= 90:
            deployment_status = "READY FOR DEPLOYMENT"
            status_icon = "✅"
        elif overall_readiness >= 70:
            deployment_status = "MOSTLY READY - MINOR ISSUES"
            status_icon = "⚠️"
        else:
            deployment_status = "NOT READY - MAJOR ISSUES"
            status_icon = "❌"
        
        print(f"{status_icon} Deployment Status: {deployment_status}")
        print(f"Overall Readiness: {overall_readiness:.1f}%")
        print(f"Total Checks: {total_checks}")
        print(f"Passed Checks: {total_passed}")
        print(f"Blocking Issues: {len(self.blocking_issues)}")
        print(f"Warnings: {len(self.warnings)}")
        
        # Print blocking issues
        if self.blocking_issues:
            print(f"\n🚫 BLOCKING ISSUES (Must be resolved before deployment):")
            for i, issue in enumerate(self.blocking_issues, 1):
                print(f"   {i}. [{issue['category'].upper()}] {issue['issue']}")
                print(f"      💡 {issue['recommendation']}")
        
        # Print warnings
        if self.warnings:
            print(f"\n⚠️ WARNINGS (Should be addressed):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. [{warning['category'].upper()}] {warning['issue']}")
                print(f"      💡 {warning['recommendation']}")
        
        # Generate recommendations
        self._generate_recommendations()
        
        if self.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "deployment_status": deployment_status,
            "overall_readiness": overall_readiness,
            "total_checks": total_checks,
            "passed_checks": total_passed,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "check_results": self.check_results
        }
        
        report_file = f"deployment_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        self.recommendations = []
        
        # Security recommendations
        if any(issue["category"] == "security" for issue in self.blocking_issues):
            self.recommendations.append("Implement comprehensive security hardening before deployment")
        
        # Legal recommendations
        if any(issue["category"] == "legal" for issue in self.blocking_issues):
            self.recommendations.append("Complete all legal documents and get professional legal review")
        
        # Compliance recommendations
        if any(issue["category"] == "compliance" for issue in self.blocking_issues):
            self.recommendations.append("Ensure full compliance with applicable regulations (GDPR, CCPA, ADA)")
        
        # Operational recommendations
        if any(issue["category"] == "operational" for issue in self.blocking_issues):
            self.recommendations.append("Implement backup systems and monitoring before deployment")
        
        # General recommendations
        if self.blocking_issues:
            self.recommendations.append("Resolve all blocking issues before attempting deployment")
        
        if self.warnings:
            self.recommendations.append("Address warnings to improve system reliability and compliance")
        
        # Professional review recommendation
        if len(self.blocking_issues) > 0 or len(self.warnings) > 5:
            self.recommendations.append("Consider professional security and legal review before deployment")

def main():
    """Main entry point for deployment readiness checker"""
    print("🚀 SquashPlot Deployment Readiness Checker")
    print("=" * 70)
    
    try:
        checker = DeploymentReadinessChecker()
        report = checker.run_readiness_check()
        
        if report["deployment_status"] == "READY FOR DEPLOYMENT":
            print(f"\n🎉 System is ready for deployment!")
        else:
            print(f"\n⚠️ System is not ready for deployment. Please address the issues above.")
        
    except Exception as e:
        print(f"❌ Error running readiness check: {e}")

if __name__ == "__main__":
    main()
