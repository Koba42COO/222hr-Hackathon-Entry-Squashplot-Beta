#!/usr/bin/env python3
"""
SquashPlot Legal Compliance Checker - OS Compatible Version
Automated legal compliance verification and reporting
"""

import json
import os
import re
import time
import sys
from datetime import datetime
from typing import Dict, List, Tuple

class LegalComplianceChecker:
    """Automated legal compliance verification system - OS Compatible"""
    
    def __init__(self):
        self.legal_documents = {
            "terms_of_service": "TERMS_OF_SERVICE.md",
            "privacy_policy": "PRIVACY_POLICY.md", 
            "cookie_policy": "COOKIE_POLICY.md",
            "data_retention_policy": "DATA_RETENTION_POLICY.md",
            "accessibility_statement": "ACCESSIBILITY_STATEMENT.md",
            "eula": "EULA.md"
        }
        
        self.compliance_requirements = {
            "gdpr": {
                "required_sections": [
                    "lawful_basis",
                    "data_subject_rights", 
                    "data_retention",
                    "breach_notification",
                    "consent_mechanism",
                    "privacy_by_design"
                ],
                "keywords": [
                    "consent", "legitimate interest", "data subject rights",
                    "right to erasure", "data portability", "breach notification",
                    "data protection officer", "privacy impact assessment"
                ]
            },
            "ccpa": {
                "required_sections": [
                    "consumer_rights",
                    "opt_out_mechanism",
                    "data_collection_disclosure",
                    "third_party_sharing",
                    "data_sale_disclosure"
                ],
                "keywords": [
                    "right to know", "right to delete", "right to opt-out",
                    "personal information", "data sale", "third party",
                    "consumer rights", "non-discrimination"
                ]
            },
            "ada": {
                "required_sections": [
                    "accessibility_commitment",
                    "wcag_compliance",
                    "assistive_technologies",
                    "accessibility_testing",
                    "contact_information"
                ],
                "keywords": [
                    "wcag", "accessibility", "screen reader", "keyboard navigation",
                    "alt text", "color contrast", "assistive technology",
                    "disability", "accommodation"
                ]
            }
        }
        
        self.compliance_status = {}
        self.violations = []
        self.recommendations = []
    
    def safe_print(self, message: str):
        """Print message with safe encoding for all OS"""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback for systems with encoding issues
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    def run_compliance_check(self) -> Dict:
        """Run comprehensive legal compliance check"""
        self.safe_print("SquashPlot Legal Compliance Checker")
        self.safe_print("=" * 60)
        
        # Check document existence
        self._check_document_existence()
        
        # Check GDPR compliance
        self._check_gdpr_compliance()
        
        # Check CCPA compliance  
        self._check_ccpa_compliance()
        
        # Check ADA compliance
        self._check_ada_compliance()
        
        # Check general legal requirements
        self._check_general_legal_requirements()
        
        # Generate compliance report
        report = self._generate_compliance_report()
        
        return report
    
    def _check_document_existence(self):
        """Check if all required legal documents exist"""
        self.safe_print("\nChecking Legal Document Existence...")
        
        missing_docs = []
        existing_docs = []
        
        for doc_type, filename in self.legal_documents.items():
            if os.path.exists(filename):
                existing_docs.append(doc_type)
                self.safe_print(f"[OK] {doc_type.replace('_', ' ').title()}: Found")
            else:
                missing_docs.append(doc_type)
                self.safe_print(f"[MISSING] {doc_type.replace('_', ' ').title()}: Missing")
        
        if missing_docs:
            self.violations.append({
                "type": "MISSING_DOCUMENTS",
                "severity": "CRITICAL",
                "description": f"Missing legal documents: {', '.join(missing_docs)}",
                "recommendation": "Create all missing legal documents before deployment"
            })
        
        self.compliance_status["document_existence"] = {
            "existing": existing_docs,
            "missing": missing_docs,
            "compliance_rate": len(existing_docs) / len(self.legal_documents) * 100
        }
    
    def _check_gdpr_compliance(self):
        """Check GDPR compliance requirements"""
        self.safe_print("\nChecking GDPR Compliance...")
        
        gdpr_status = {
            "compliant": True,
            "violations": [],
            "compliance_score": 0
        }
        
        # Check if privacy policy exists and has GDPR content
        if os.path.exists("PRIVACY_POLICY.md"):
            try:
                with open("PRIVACY_POLICY.md", 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                # Check for GDPR keywords
                gdpr_keywords = self.compliance_requirements["gdpr"]["keywords"]
                found_keywords = [kw for kw in gdpr_keywords if kw in content]
                
                gdpr_status["compliance_score"] = (len(found_keywords) / len(gdpr_keywords)) * 100
                
                if gdpr_status["compliance_score"] < 70:
                    gdpr_status["compliant"] = False
                    gdpr_status["violations"].append({
                        "issue": "Insufficient GDPR content in privacy policy",
                        "score": gdpr_status["compliance_score"],
                        "recommendation": "Add more GDPR-specific content to privacy policy"
                    })
            except Exception as e:
                gdpr_status["compliant"] = False
                gdpr_status["violations"].append({
                    "issue": f"Error reading privacy policy: {str(e)}",
                    "recommendation": "Fix privacy policy file"
                })
        else:
            gdpr_status["compliant"] = False
            gdpr_status["violations"].append({
                "issue": "Privacy policy missing",
                "recommendation": "Create comprehensive privacy policy with GDPR compliance"
            })
        
        # Check for data retention policy
        if not os.path.exists("DATA_RETENTION_POLICY.md"):
            gdpr_status["compliant"] = False
            gdpr_status["violations"].append({
                "issue": "Data retention policy missing",
                "recommendation": "Create data retention policy for GDPR compliance"
            })
        
        if not gdpr_status["compliant"]:
            self.violations.extend(gdpr_status["violations"])
        
        self.compliance_status["gdpr"] = gdpr_status
        self.safe_print(f"GDPR Compliance Score: {gdpr_status['compliance_score']:.1f}%")
    
    def _check_ccpa_compliance(self):
        """Check CCPA compliance requirements"""
        self.safe_print("\nChecking CCPA Compliance...")
        
        ccpa_status = {
            "compliant": True,
            "violations": [],
            "compliance_score": 0
        }
        
        # Check if privacy policy exists and has CCPA content
        if os.path.exists("PRIVACY_POLICY.md"):
            try:
                with open("PRIVACY_POLICY.md", 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                # Check for CCPA keywords
                ccpa_keywords = self.compliance_requirements["ccpa"]["keywords"]
                found_keywords = [kw for kw in ccpa_keywords if kw in content]
                
                ccpa_status["compliance_score"] = (len(found_keywords) / len(ccpa_keywords)) * 100
                
                if ccpa_status["compliance_score"] < 60:
                    ccpa_status["compliant"] = False
                    ccpa_status["violations"].append({
                        "issue": "Insufficient CCPA content in privacy policy",
                        "score": ccpa_status["compliance_score"],
                        "recommendation": "Add CCPA-specific consumer rights and opt-out information"
                    })
            except Exception as e:
                ccpa_status["compliant"] = False
                ccpa_status["violations"].append({
                    "issue": f"Error reading privacy policy: {str(e)}",
                    "recommendation": "Fix privacy policy file"
                })
        else:
            ccpa_status["compliant"] = False
            ccpa_status["violations"].append({
                "issue": "Privacy policy missing",
                "recommendation": "Create privacy policy with CCPA compliance"
            })
        
        if not ccpa_status["compliant"]:
            self.violations.extend(ccpa_status["violations"])
        
        self.compliance_status["ccpa"] = ccpa_status
        self.safe_print(f"CCPA Compliance Score: {ccpa_status['compliance_score']:.1f}%")
    
    def _check_ada_compliance(self):
        """Check ADA compliance requirements"""
        self.safe_print("\nChecking ADA Compliance...")
        
        ada_status = {
            "compliant": True,
            "violations": [],
            "compliance_score": 0
        }
        
        # Check if accessibility statement exists
        if os.path.exists("ACCESSIBILITY_STATEMENT.md"):
            try:
                with open("ACCESSIBILITY_STATEMENT.md", 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                # Check for ADA keywords
                ada_keywords = self.compliance_requirements["ada"]["keywords"]
                found_keywords = [kw for kw in ada_keywords if kw in content]
                
                ada_status["compliance_score"] = (len(found_keywords) / len(ada_keywords)) * 100
                
                if ada_status["compliance_score"] < 70:
                    ada_status["compliant"] = False
                    ada_status["violations"].append({
                        "issue": "Insufficient accessibility content in statement",
                        "score": ada_status["compliance_score"],
                        "recommendation": "Enhance accessibility statement with more comprehensive content"
                    })
            except Exception as e:
                ada_status["compliant"] = False
                ada_status["violations"].append({
                    "issue": f"Error reading accessibility statement: {str(e)}",
                    "recommendation": "Fix accessibility statement file"
                })
        else:
            ada_status["compliant"] = False
            ada_status["violations"].append({
                "issue": "Accessibility statement missing",
                "recommendation": "Create comprehensive accessibility statement"
            })
        
        if not ada_status["compliant"]:
            self.violations.extend(ada_status["violations"])
        
        self.compliance_status["ada"] = ada_status
        self.safe_print(f"ADA Compliance Score: {ada_status['compliance_score']:.1f}%")
    
    def _check_general_legal_requirements(self):
        """Check general legal requirements"""
        self.safe_print("\nChecking General Legal Requirements...")
        
        general_status = {
            "compliant": True,
            "violations": [],
            "compliance_score": 0
        }
        
        # Check for essential legal elements
        essential_docs = ["terms_of_service", "privacy_policy", "eula"]
        existing_essential = [doc for doc in essential_docs if os.path.exists(self.legal_documents[doc])]
        
        general_status["compliance_score"] = (len(existing_essential) / len(essential_docs)) * 100
        
        if general_status["compliance_score"] < 100:
            general_status["compliant"] = False
            missing_docs = [doc for doc in essential_docs if doc not in existing_essential]
            general_status["violations"].append({
                "issue": f"Missing essential legal documents: {', '.join(missing_docs)}",
                "recommendation": "Create all essential legal documents"
            })
        
        # Check for contact information
        contact_required = ["email", "phone", "address", "website"]
        contact_found = 0
        
        for doc_type, filename in self.legal_documents.items():
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    for contact_type in contact_required:
                        if contact_type in content:
                            contact_found += 1
                            break
                except Exception:
                    pass
        
        if contact_found < 2:
            general_status["compliant"] = False
            general_status["violations"].append({
                "issue": "Insufficient contact information in legal documents",
                "recommendation": "Add comprehensive contact information to all legal documents"
            })
        
        if not general_status["compliant"]:
            self.violations.extend(general_status["violations"])
        
        self.compliance_status["general"] = general_status
        self.safe_print(f"General Legal Compliance Score: {general_status['compliance_score']:.1f}%")
    
    def _generate_compliance_report(self) -> Dict:
        """Generate comprehensive compliance report"""
        self.safe_print("\n" + "=" * 60)
        self.safe_print("LEGAL COMPLIANCE REPORT")
        self.safe_print("=" * 60)
        
        # Calculate overall compliance
        total_score = 0
        total_standards = 0
        
        for standard, status in self.compliance_status.items():
            if "compliance_score" in status:
                total_score += status["compliance_score"]
                total_standards += 1
        
        overall_compliance = total_score / total_standards if total_standards > 0 else 0
        
        # Generate recommendations
        self._generate_recommendations()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_compliance": overall_compliance,
            "compliance_status": self.compliance_status,
            "violations": self.violations,
            "recommendations": self.recommendations,
            "summary": self._generate_summary()
        }
        
        # Print summary
        self.safe_print(f"Overall Legal Compliance: {overall_compliance:.1f}%")
        self.safe_print(f"Total Violations: {len(self.violations)}")
        self.safe_print(f"Critical Issues: {len([v for v in self.violations if v.get('severity') == 'CRITICAL'])}")
        
        if self.violations:
            self.safe_print(f"\nVIOLATIONS FOUND:")
            for i, violation in enumerate(self.violations, 1):
                severity = violation.get('severity', 'UNKNOWN')
                severity_icon = "[CRITICAL]" if severity == "CRITICAL" else "[HIGH]" if severity == "HIGH" else "[MEDIUM]"
                self.safe_print(f"   {i}. {severity_icon} {violation['description']}")
        
        if self.recommendations:
            self.safe_print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                self.safe_print(f"   {i}. {rec}")
        
        # Save report
        report_file = f"legal_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            self.safe_print(f"\nDetailed report saved to: {report_file}")
        except Exception as e:
            self.safe_print(f"\nError saving report: {str(e)}")
        
        return report
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        self.recommendations = []
        
        # Document creation recommendations
        missing_docs = self.compliance_status.get("document_existence", {}).get("missing", [])
        if missing_docs:
            self.recommendations.append(f"Create missing legal documents: {', '.join(missing_docs)}")
        
        # GDPR recommendations
        gdpr_score = self.compliance_status.get("gdpr", {}).get("compliance_score", 0)
        if gdpr_score < 80:
            self.recommendations.append("Enhance GDPR compliance in privacy policy")
        
        # CCPA recommendations
        ccpa_score = self.compliance_status.get("ccpa", {}).get("compliance_score", 0)
        if ccpa_score < 80:
            self.recommendations.append("Add CCPA consumer rights and opt-out information")
        
        # ADA recommendations
        ada_score = self.compliance_status.get("ada", {}).get("compliance_score", 0)
        if ada_score < 80:
            self.recommendations.append("Enhance accessibility statement with WCAG compliance")
        
        # General recommendations
        general_score = self.compliance_status.get("general", {}).get("compliance_score", 0)
        if general_score < 100:
            self.recommendations.append("Complete all essential legal documents")
        
        # Professional review recommendation
        if len(self.violations) > 0:
            self.recommendations.append("Schedule professional legal review of all documents")
    
    def _generate_summary(self) -> str:
        """Generate compliance summary"""
        total_violations = len(self.violations)
        critical_violations = len([v for v in self.violations if v.get('severity') == 'CRITICAL'])
        
        if critical_violations > 0:
            status = "NON_COMPLIANT - CRITICAL ISSUES"
        elif total_violations > 0:
            status = "PARTIALLY_COMPLIANT - ISSUES FOUND"
        else:
            status = "COMPLIANT - NO ISSUES"
        
        summary = f"""
Legal Compliance Summary:
- Status: {status}
- Total Violations: {total_violations}
- Critical Issues: {critical_violations}
- Recommendations: {len(self.recommendations)}
- Last Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return summary.strip()
    
    def create_missing_documents(self):
        """Create missing legal documents with templates"""
        self.safe_print("\nCreating Missing Legal Documents...")
        
        missing_docs = self.compliance_status.get("document_existence", {}).get("missing", [])
        
        if not missing_docs:
            self.safe_print("All legal documents exist")
            return
        
        for doc_type in missing_docs:
            filename = self.legal_documents[doc_type]
            self.safe_print(f"Creating {doc_type.replace('_', ' ').title()}...")
            
            # Create basic template
            template = self._get_document_template(doc_type)
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(template)
                self.safe_print(f"[OK] Created {filename}")
            except Exception as e:
                self.safe_print(f"[ERROR] Failed to create {filename}: {str(e)}")
    
    def _get_document_template(self, doc_type: str) -> str:
        """Get template for missing document"""
        templates = {
            "terms_of_service": """# Terms of Service

**Effective Date: [Date]**

## 1. Acceptance of Terms
By using this service, you agree to these terms.

## 2. Use License
Permission granted for personal, non-commercial use only.

## 3. Disclaimer
Materials provided "as is" without warranties.

## 4. Limitations
No liability for damages arising from use.

## 5. Contact Information
For questions, contact: legal@squashplot.dev

---
**SquashPlot Terms of Service - Version 1.0**""",
            
            "privacy_policy": """# Privacy Policy

**Effective Date: [Date]**

## 1. Information We Collect
We collect information you provide directly to us.

## 2. How We Use Information
We use information to provide and improve our services.

## 3. Information Sharing
We do not sell, trade, or share your personal information.

## 4. Data Security
We implement appropriate security measures to protect your information.

## 5. Your Rights
You have the right to access, update, or delete your information.

## 6. Contact Us
For privacy questions, contact: privacy@squashplot.dev

---
**SquashPlot Privacy Policy - Version 1.0**""",
            
            "cookie_policy": """# Cookie Policy

**Effective Date: [Date]**

## 1. What Are Cookies
Cookies are small text files placed on your device.

## 2. How We Use Cookies
We use cookies to improve your experience on our website.

## 3. Types of Cookies
- Essential cookies for website functionality
- Analytics cookies for website improvement
- Preference cookies for user settings

## 4. Managing Cookies
You can control cookies through your browser settings.

## 5. Contact Us
For cookie questions, contact: privacy@squashplot.dev

---
**SquashPlot Cookie Policy - Version 1.0**"""
        }
        
        return templates.get(doc_type, f"# {doc_type.replace('_', ' ').title()}\n\n**Content to be added**")

def main():
    """Main entry point for legal compliance checker"""
    print("SquashPlot Legal Compliance Checker - OS Compatible")
    print("=" * 60)
    
    try:
        checker = LegalComplianceChecker()
        
        # Run compliance check
        report = checker.run_compliance_check()
        
        # Create missing documents if needed
        if report["violations"]:
            response = input("\nWould you like to create missing documents? (yes/no): ")
            if response.lower() == 'yes':
                checker.create_missing_documents()
        
        print(f"\nLegal compliance check completed")
        
    except Exception as e:
        print(f"Error running compliance check: {e}")

if __name__ == "__main__":
    main()
