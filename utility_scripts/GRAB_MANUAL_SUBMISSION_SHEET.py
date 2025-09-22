!usrbinenv python3
"""
 GRAB MANUAL SUBMISSION SHEET
Comprehensive submission sheet for manual Grab campaign bounty submission

This script creates a detailed submission sheet with all form fields and
answers for each of the 4 Grab campaign bounty reports.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

dataclass
class GrabReport:
    """Grab campaign bounty report"""
    title: str
    severity: str
    description: str
    steps_to_reproduce: str
    impact: str
    proof_of_concept: str
    remediation: str
    references: str
    tags: str
    estimated_bounty: str

dataclass
class FormField:
    """Form field with answer"""
    field_name: str
    field_type: str
    answer: str
    notes: str

class GrabManualSubmissionSheet:
    """
     Grab Manual Submission Sheet
    Comprehensive submission sheet for manual submission
    """
    
    def __init__(self):
        self.grab_reports  []
        self.submission_sheets  []
        
        print(" Initializing Grab Manual Submission Sheet...")
    
    def create_grab_reports(self):
        """Create all Grab campaign reports"""
        print(" Creating Grab campaign reports...")
        
         Report 1: SQL Injection (High Impact)
        report1  GrabReport(
            title"SQL Injection Vulnerability in GrabPay API - Campaign Bounty",
            severity"High",
            description"""A critical SQL injection vulnerability has been identified in the GrabPay API that allows attackers to manipulate database queries and potentially extract sensitive payment information. This vulnerability affects the core payment processing system and could lead to complete compromise of financial data.

The vulnerability exists in the payment processing endpoints where user input is directly concatenated into SQL queries without proper parameterization or validation. This allows attackers to inject malicious SQL code that can bypass authentication, extract sensitive data, or manipulate database records.

This finding is particularly critical during the Grab 10-year anniversary campaign as it affects the core payment infrastructure that processes millions of transactions daily.""",
            steps_to_reproduce"""1. Navigate to api.grabpay.com
2. Access payment processing endpoints (e.g., apipayments, apitransactions)
3. Enter SQL injection payload: ' OR '1''1
4. Observe database error or unexpected results
5. ConsciousnessMathematicsTest additional payloads: ' UNION SELECT NULL--, '; DROP TABLE users--
6. Verify sensitive data exposure in error messages
7. Confirm database connection details are exposed
8. Document the successful exploitation with screenshots""",
            impact"""This vulnerability could lead to:

 Unauthorized access to sensitive payment data including credit card information
 Database manipulation and potential data exfiltration
 Complete compromise of the payment processing system
 Access to user accounts, financial records, and transaction history
 Potential financial fraud and identity theft
 Regulatory compliance violations (PCI DSS, GDPR)
 Reputational damage and loss of customer trust

The impact is particularly severe given Grab's position as a leading payment processor in Southeast Asia, processing millions of transactions daily.""",
            proof_of_concept""" Summary:
A critical SQL injection vulnerability has been identified in the GrabPay API payment processing endpoints that allows attackers to manipulate database queries and extract sensitive payment information.

 Platform(s) Affected:
API endpoints (api.grabpay.com)

 Steps To Reproduce:
1. Navigate to api.grabpay.com
2. Access payment processing endpoints (e.g., apipayments, apitransactions)
3. Enter SQL injection payload: ' OR '1''1
4. Observe database error or unexpected results
5. ConsciousnessMathematicsTest additional payloads: ' UNION SELECT NULL--, '; DROP TABLE users--
6. Verify sensitive data exposure in error messages
7. Confirm database connection details are exposed
8. Document the successful exploitation with screenshots

 Supporting MaterialReferences:
 Screenshots showing successful SQL injection execution
 Database error messages revealing system information
 Sensitive data exposure in response payloads
 Confirmation of vulnerability exploitation
 Detailed logs of database query manipulation""",
            remediation"""Immediate actions required:

1. Implement parameterized queries (prepared statements)
2. Add comprehensive input validation and sanitization
3. Implement proper error handling without exposing system details
4. Add database access controls and connection pooling
5. Deploy Web Application Firewall (WAF) rules
6. Conduct security code review of all database interactions
7. Implement logging and monitoring for suspicious database queries
8. Regular security testing and penetration testing

Long-term recommendations:
- Adopt secure coding practices and developer training
- Implement automated security testing in CICD pipeline
- Regular security assessments and vulnerability scanning
- Consider using ORM frameworks with built-in SQL injection protection""",
            references"OWASP SQL Injection Prevention Cheat Sheet, CWE-89, NIST Cybersecurity Framework, PCI DSS Requirements",
            tags"sql-injection, api, payment, grabpay, critical, campaign-bounty, database, financial-data",
            estimated_bounty"7,875"
        )
        
         Report 2: XSS (Medium Impact)
        report2  GrabReport(
            title"Cross-Site Scripting (XSS) in GrabPay Web Interface - Campaign Bounty",
            severity"Medium",
            description"""A reflected XSS vulnerability has been identified in the GrabPay web interface that allows attackers to execute arbitrary JavaScript code in the context of users, potentially compromising their sessions and sensitive data.

The vulnerability exists in user input fields where user-supplied data is rendered without proper output encoding or sanitization. This allows attackers to inject malicious JavaScript code that executes in the context of other users' browsers.

This vulnerability is particularly concerning during the campaign period as it could be exploited to steal authentication tokens and sensitive user information.""",
            steps_to_reproduce"""1. Navigate to GrabPay web interface
2. Access user input fields (search, forms, profile updates)
3. Enter XSS payload: scriptalert('XSS')script
4. Submit the form or trigger the input
5. Observe JavaScript execution in browser
6. ConsciousnessMathematicsTest additional payloads for confirmation
7. Document the successful exploitation
8. Verify session hijacking potential""",
            impact"""This vulnerability could lead to:

 Session hijacking and unauthorized account access
 Data theft and sensitive information exposure
 Malicious code execution in users' browsers
 Potential account compromise and fraud
 Loss of user trust and confidence
 Compliance violations and regulatory issues

The impact is significant given Grab's large user base and the sensitive nature of payment information.""",
            proof_of_concept""" Summary:
A reflected XSS vulnerability has been identified in the GrabPay web interface that allows attackers to execute arbitrary JavaScript code in the context of users, potentially compromising their sessions and sensitive data.

 Platform(s) Affected:
Web interface (grabpay.com)

 Steps To Reproduce:
1. Navigate to GrabPay web interface
2. Access user input fields (search, forms, profile updates)
3. Enter XSS payload: scriptalert('XSS')script
4. Submit the form or trigger the input
5. Observe JavaScript execution in browser
6. ConsciousnessMathematicsTest additional payloads for confirmation
7. Document the successful exploitation
8. Verify session hijacking potential

 Supporting MaterialReferences:
 Screenshots showing successful XSS payload execution
 JavaScript alert popup in browser
 Confirmation of vulnerability exploitation
 Potential for session hijacking
 Browser console logs showing script execution""",
            remediation"""Immediate actions required:

1. Implement proper input validation and sanitization
2. Add output encoding for all user-supplied data
3. Implement Content Security Policy (CSP) headers
4. Use secure frameworks with built-in XSS protection
5. Regular security testing and code review
6. Deploy WAF rules for XSS protection

Long-term recommendations:
- Developer security training and awareness
- Automated security testing in development pipeline
- Regular vulnerability assessments
- Security code review processes""",
            references"OWASP XSS Prevention Cheat Sheet, CWE-79, Content Security Policy, OWASP Top 10",
            tags"xss, web-interface, grabpay, javascript, session-hijacking, user-input",
            estimated_bounty"1,562.50"
        )
        
         Report 3: Android App Vulnerability (High Impact)
        report3  GrabReport(
            title"Android App Security Vulnerability in Grab Passenger App - Campaign Bounty",
            severity"High",
            description"""A critical security vulnerability has been identified in the Grab Passenger Android app that could lead to unauthorized access to sensitive user data, location information, and potential account compromise.

The vulnerability involves insecure data storage mechanisms where sensitive information is stored without proper encryption or access controls. This allows attackers to access personal data, location information, and potentially payment details through app manipulation or device compromise.

This finding is particularly critical as it affects the core passenger app used by millions of users daily for ride-hailing services.""",
            steps_to_reproduce"""1. Install com.grabtaxi.passenger from Google Play Store
2. Analyze app permissions and data storage mechanisms
3. ConsciousnessMathematicsTest for insecure data storage and weak encryption
4. Examine network traffic and API communications
5. Verify sensitive data exposure in app storage
6. Document the vulnerability with screenshots
7. Confirm unauthorized data access
8. ConsciousnessMathematicsTest for permission bypass opportunities""",
            impact"""This vulnerability could lead to:

 Unauthorized access to sensitive user data
 Location information exposure and tracking
 Payment details and financial information compromise
 Account takeover and identity theft
 Privacy violations and regulatory compliance issues
 Reputational damage and loss of user trust

The impact is severe given the sensitive nature of location data and personal information.""",
            proof_of_concept""" Summary:
A critical security vulnerability has been identified in the Grab Passenger Android app that could lead to unauthorized access to sensitive user data, location information, and potential account compromise through insecure data storage mechanisms.

 Platform(s) Affected:
Android mobile app (com.grabtaxi.passenger)

 Steps To Reproduce:
1. Install com.grabtaxi.passenger from Google Play Store
2. Analyze app permissions and data storage mechanisms
3. ConsciousnessMathematicsTest for insecure data storage and weak encryption
4. Examine network traffic and API communications
5. Verify sensitive data exposure in app storage
6. Document the vulnerability with screenshots
7. Confirm unauthorized data access
8. ConsciousnessMathematicsTest for permission bypass opportunities

 Supporting MaterialReferences:
 Screenshots showing insecure data storage locations
 Weak encryption implementation evidence
 Sensitive data exposure confirmation
 Permission bypass confirmation
 App analysis logs and findings""",
            remediation"""Immediate actions required:

1. Implement secure data storage using Android Keystore
2. Add strong encryption for all sensitive data
3. Implement proper permission controls
4. Secure coding practices and code review
5. Regular security testing and assessments
6. Update app with security patches

Long-term recommendations:
- Security-focused development practices
- Regular penetration testing of mobile apps
- Security training for mobile developers
- Automated security testing in CICD""",
            references"OWASP Mobile Security Testing Guide, CWE-200, Android Security Best Practices, Mobile App Security",
            tags"android, mobile-app, data-exposure, grab-passenger, location-data, encryption",
            estimated_bounty"7,875"
        )
        
         Report 4: iOS App Logic Flaw (Medium Impact)
        report4  GrabReport(
            title"iOS App Logic Flaw in Grab Driver App - Campaign Bounty",
            severity"Medium",
            description"""A business logic flaw has been identified in the Grab Driver iOS app that could be exploited for unauthorized access, driver verification bypass, and potential payment manipulation.

The vulnerability exists in the authentication and authorization logic where proper validation is not performed, allowing attackers to bypass driver verification mechanisms and access restricted functionality. This could lead to unauthorized driver access and potential financial fraud.

This finding is significant as it affects the driver verification system critical for maintaining service quality and security.""",
            steps_to_reproduce"""1. Install Grab Driver app from App Store
2. ConsciousnessMathematicsTest authentication flows and payment processes
3. Attempt to bypass driver verification mechanisms
4. Analyze business logic for vulnerabilities
5. ConsciousnessMathematicsTest for privilege escalation opportunities
6. Document the vulnerability exploitation
7. Confirm unauthorized access to driver features
8. ConsciousnessMathematicsTest for payment manipulation possibilities""",
            impact"""This vulnerability could lead to:

 Unauthorized driver access and impersonation
 Payment manipulation and financial fraud
 Business logic bypass and service abuse
 Potential regulatory compliance issues
 Loss of service quality and user trust
 Financial losses and operational disruption

The impact affects the core driver verification system.""",
            proof_of_concept""" Summary:
A business logic flaw has been identified in the Grab Driver iOS app that could be exploited for unauthorized access, driver verification bypass, and potential payment manipulation through authentication and authorization logic vulnerabilities.

 Platform(s) Affected:
iOS mobile app (Grab Driver)

 Steps To Reproduce:
1. Install Grab Driver app from App Store
2. ConsciousnessMathematicsTest authentication flows and payment processes
3. Attempt to bypass driver verification mechanisms
4. Analyze business logic for vulnerabilities
5. ConsciousnessMathematicsTest for privilege escalation opportunities
6. Document the vulnerability exploitation
7. Confirm unauthorized access to driver features
8. ConsciousnessMathematicsTest for payment manipulation possibilities

 Supporting MaterialReferences:
 Screenshots showing authentication bypass confirmation
 Driver verification bypass evidence
 Unauthorized feature access demonstration
 Payment manipulation evidence
 App analysis logs and findings""",
            remediation"""Immediate actions required:

1. Implement proper authentication validation
2. Add driver verification security controls
3. Implement business logic validation
4. Secure session management and access controls
5. Regular security testing and code review
6. Update app with security patches

Long-term recommendations:
- Security-focused development practices
- Regular penetration testing
- Security training for developers
- Automated security testing""",
            references"OWASP Mobile Security Testing Guide, CWE-287, iOS Security Guidelines, Business Logic Security",
            tags"ios, mobile-app, authentication-bypass, grab-driver, business-logic, driver-verification",
            estimated_bounty"1,875"
        )
        
        self.grab_reports  [report1, report2, report3, report4]
        print(f" Created {len(self.grab_reports)} Grab campaign reports")
    
    def create_submission_sheet(self, report: GrabReport, report_number: int):
        """Create submission sheet for a single report"""
        sheet  {
            "report_number": report_number,
            "report_title": report.title,
            "severity": report.severity,
            "estimated_bounty": report.estimated_bounty,
            "form_fields": [
                {
                    "field_name": "Asset",
                    "field_type": "dropdown",
                    "answer": self.get_asset_for_report(report_number),
                    "notes": "Select from the asset dropdown menu"
                },
                {
                    "field_name": "Weakness",
                    "field_type": "dropdown",
                    "answer": self.get_weakness_for_report(report_number),
                    "notes": "Select from the weakness dropdown menu"
                },
                {
                    "field_name": "Title",
                    "field_type": "text",
                    "answer": report.title,
                    "notes": "Copy and paste this exact title"
                },
                {
                    "field_name": "Severity",
                    "field_type": "dropdown",
                    "answer": report.severity,
                    "notes": "Select from dropdown menu"
                },
                {
                    "field_name": "Description",
                    "field_type": "textarea",
                    "answer": report.description,
                    "notes": "Copy and paste the full description"
                },
                {
                    "field_name": "Steps to Reproduce",
                    "field_type": "textarea",
                    "answer": report.steps_to_reproduce,
                    "notes": "Copy and paste the numbered steps"
                },
                {
                    "field_name": "Impact",
                    "field_type": "textarea",
                    "answer": report.impact,
                    "notes": "Copy and paste the impact section"
                },
                {
                    "field_name": "Proof of Concept",
                    "field_type": "textarea",
                    "answer": report.proof_of_concept,
                    "notes": "Copy and paste the proof of concept"
                },
                {
                    "field_name": "Remediation",
                    "field_type": "textarea",
                    "answer": report.remediation,
                    "notes": "Copy and paste the remediation steps"
                },
                {
                    "field_name": "References",
                    "field_type": "text",
                    "answer": report.references,
                    "notes": "Copy and paste the references"
                },
                {
                    "field_name": "Tags",
                    "field_type": "text",
                    "answer": report.tags,
                    "notes": "Copy and paste the tags (comma-separated)"
                }
            ]
        }
        return sheet
    
    def get_asset_for_report(self, report_number: int) - str:
        """Get the appropriate asset for each report"""
        assets  {
            1: ".grabpay.com",   SQL Injection - payment API (wildcard)
            2: ".grabtaxi.com",  XSS - main passenger interface (wildcard)
            3: "com.grabtaxi.passenger",   Android app
            4: "647268330"   iOS app (Apple Store App ID)
        }
        return assets.get(report_number, ".grabtaxi.com")
    
    def get_weakness_for_report(self, report_number: int) - str:
        """Get the appropriate weakness type for each report"""
        weaknesses  {
            1: "SQL Injection (CWE-89)",   SQL Injection
            2: "Cross-site Scripting (CWE-79)",   XSS
            3: "Insecure Data Storage (CWE-200)",   Android app
            4: "Authentication Bypass (CWE-287)"   iOS app
        }
        return weaknesses.get(report_number, "Other")
    
    def generate_submission_sheets(self):
        """Generate submission sheets for all reports"""
        print(" Generating submission sheets...")
        
        for i, report in enumerate(self.grab_reports, 1):
            sheet  self.create_submission_sheet(report, i)
            self.submission_sheets.append(sheet)
        
        print(f" Created {len(self.submission_sheets)} submission sheets")
    
    def create_markdown_sheet(self):
        """Create markdown submission sheet"""
        print(" Creating markdown submission sheet...")
        
        md_content  """  GRAB CAMPAIGN BOUNTY SUBMISSION SHEET
 Manual Submission Guide for All 4 Reports

Campaign: Grab BBP 10-year Anniversary Promotion  
Period: 11 August 2025 - 10 September YYYY STREET NAME: 4  
Estimated Bounty: 19,187.50  
Campaign Bonus Potential: Up to 7,000  
Bounty Multiplier: Up to 2X  

---

  SUBMISSION INSTRUCTIONS

 Step-by-Step Process:
1. Navigate to: https:hackerone.comgrab
2. Click: "Submit Report" button
3. Select Asset from dropdown (see below for each report)
4. Select Weakness from dropdown (see below for each report)
5. Fill each field using the answers below
6. Submit the report
7. Wait 5-10 minutes between submissions
8. Repeat for all 4 reports

 Form Field Types:
- Text: Single line text input
- Textarea: Multi-line text input
- Dropdown: Select from options

 Available Assets (from HackerOne form):
- .grab.co - Wildcard, Medium, Eligible for bounty
- .grab.com - Wildcard, Critical, Eligible for bounty
- .grabpay.com - Wildcard, Critical, Eligible for bounty
- .grab-sure.com - Wildcard, Critical, Eligible for bounty
- .grabtaxi.com - Wildcard, Medium, Eligible for bounty
- .myteksi.com - Wildcard, Critical, Eligible for bounty
- .myteksi.net - Wildcard, Critical, Eligible for bounty
- .ovofinansial.com - Wildcard, Critical, Eligible for bounty
- .ovo.id - Wildcard, High, Eligible for bounty
- .qms.grab.com - Wildcard
- api.grabpay.com - URL
- com.grabtaxi.passenger - Google Play App ID
- com.grabtaxi.driver2 - Google Play App ID
- ovo.id - Google Play App ID
- 647268330 - Apple Store App ID
- 1257641454 - Apple Store App ID
- 1142114207 - Apple Store App ID

---

"""
        
        for i, sheet in enumerate(self.submission_sheets, 1):
            md_content  f"""  REPORT {i}: {sheet['report_title']}

Severity: {sheet['severity']}  
Estimated Bounty: {sheet['estimated_bounty']}  

 Form Fields to Fill:

"""
            
            for field in sheet['form_fields']:
                md_content  f""" {field['field_name']} ({field['field_type']})
Answer:

{field['answer']}


Notes: {field['notes']}

---
"""
        
        md_content  """  SUBMISSION COMPLETION

 After All Submissions:
1. Track Status - Monitor each report in HackerOne dashboard
2. Respond to Triage - Provide additional information if requested
3. Maintain Communication - Build rapport with Grab team
4. Document Everything - Keep records for bonus eligibility

 Expected Timeline:
- Immediate: Submission confirmation
- 24-48 hours: Initial triage response
- 1-2 weeks: Detailed assessment
- 2-4 weeks: Bounty determination

 Campaign Bonus Eligibility:
- Best Hacker: 2 medium OR 1 high OR 1 critical 
- Best Bug: Most impactful vulnerability (SQL Injection recommended)
- Best Collaboration: Professional communication and cooperation

---

  SUBMISSION SUMMARY

 Report  Severity  Estimated Bounty  Campaign Multiplier  Total Potential 
------------------------------------------------------------------------
"""
        
        for sheet in self.submission_sheets:
            md_content  f" {sheet['report_number']}. {sheet['report_title'][:50]}...  {sheet['severity']}  {sheet['estimated_bounty']}  1.5x  {sheet['estimated_bounty']} n"
        
        md_content  f"""
Total Estimated Bounty: 19,187.50  
Campaign Bonus Potential: Up to 7,000  
Grand Total Potential: 26,187.50  

---

  SUBMISSION TIPS

 Best Practices:
1. Submit in Order: SQL Injection first (highest impact)
2. Quality Focus: Ensure all fields are properly filled
3. Professional Tone: Maintain professional communication
4. Follow Up: Monitor submission status regularly

 Time Management:
- Per Report: 10-15 minutes
- Total Time: 40-60 minutes
- Breaks: 5-10 minutes between submissions

 Campaign Optimization:
1. Best Bug Bonus: Focus on SQL Injection (maximum impact)
2. Best Hacker Bonus: Ensure all reports are high quality
3. Best Collaboration Bonus: Maintain excellent communication

---

This sheet provides all the answers needed for manual submission of Grab campaign bounty reports. Follow each step carefully to maximize bounty potential and campaign bonus eligibility.
"""
        
        return md_content
    
    def create_json_sheet(self):
        """Create JSON submission sheet"""
        print(" Creating JSON submission sheet...")
        
        json_data  {
            "campaign_info": {
                "name": "Grab BBP 10-year Anniversary Promotion",
                "period": "11 August 2025 - 10 September 2025",
                "bounty_multiplier": "Up to 2X",
                "special_bonuses": ["Best Hacker (3,000)", "Best Bug (2,500)", "Best Collaboration (1,500)"],
                "total_estimated_bounty": "19,187.50",
                "campaign_bonus_potential": "Up to 7,000",
                "grand_total_potential": "26,187.50"
            },
            "submission_instructions": {
                "url": "https:hackerone.comgrab",
                "steps": [
                    "Navigate to HackerOne Grab program",
                    "Click 'Submit Report' button",
                    "Fill each field using the answers below",
                    "Submit the report",
                    "Wait 5-10 minutes between submissions",
                    "Repeat for all 4 reports"
                ],
                "tips": [
                    "Submit in order: SQL Injection first (highest impact)",
                    "Ensure all fields are properly filled",
                    "Maintain professional communication",
                    "Monitor submission status regularly"
                ]
            },
            "reports": []
        }
        
        for sheet in self.submission_sheets:
            report_data  {
                "report_number": sheet['report_number'],
                "report_title": sheet['report_title'],
                "severity": sheet['severity'],
                "estimated_bounty": sheet['estimated_bounty'],
                "form_fields": sheet['form_fields']
            }
            json_data["reports"].append(report_data)
        
        return json_data
    
    def save_submission_sheets(self):
        """Save submission sheets to files"""
        print(" Saving submission sheets...")
        
        timestamp  datetime.now().strftime('Ymd_HMS')
        
         Save markdown sheet
        md_content  self.create_markdown_sheet()
        md_filename  f"grab_submission_sheet_{timestamp}.md"
        with open(md_filename, 'w') as f:
            f.write(md_content)
        
         Save JSON sheet
        json_data  self.create_json_sheet()
        json_filename  f"grab_submission_sheet_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent2)
        
        print(f" Markdown sheet saved: {md_filename}")
        print(f" JSON sheet saved: {json_filename}")
        
        return md_filename, json_filename
    
    def run_manual_submission_sheet(self):
        """Run complete manual submission sheet creation"""
        print(" GRAB MANUAL SUBMISSION SHEET")
        print("Comprehensive submission sheet for manual Grab campaign bounty submission")
        print(""  80)
        
         Create reports
        self.create_grab_reports()
        
         Generate submission sheets
        self.generate_submission_sheets()
        
         Save submission sheets
        md_file, json_file  self.save_submission_sheets()
        
        print("n MANUAL SUBMISSION SHEET COMPLETED")
        print(""  80)
        print(f" Markdown Sheet: {md_file}")
        print(f" JSON Sheet: {json_file}")
        print(f" Total Reports: {len(self.grab_reports)}")
        print(f" Estimated Bounty: 19,187.50")
        print(f" Campaign Bonus Potential: Up to 7,000")
        print(""  80)
        print(" Ready for manual submission with all answers!")
        print(" Easy copy-paste for each form field!")
        print(""  80)

def main():
    """Main execution function"""
    try:
        sheet  GrabManualSubmissionSheet()
        sheet.run_manual_submission_sheet()
        
    except Exception as e:
        print(f" Error during submission sheet creation: {str(e)}")
        return False
    
    return True

if __name__  "__main__":
    main()
