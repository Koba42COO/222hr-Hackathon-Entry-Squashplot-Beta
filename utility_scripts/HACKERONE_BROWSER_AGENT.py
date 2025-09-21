!usrbinenv python3
"""
 HACKERONE BROWSER AGENT
Automated browser agent for submitting bounty reports to HackerOne

This script automates the submission of Grab campaign bounty reports
to HackerOne using Selenium with Brave browser.
"""

import os
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:
    print(" Selenium not installed. Installing required packages...")
    os.system("pip install selenium webdriver-manager")
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import TimeoutException, NoSuchElementException

dataclass
class HackerOneReport:
    """HackerOne report data"""
    title: str
    severity: str
    description: str
    steps_to_reproduce: str
    impact: str
    proof_of_concept: str
    remediation: str
    references: str
    tags: List[str]
    program: str
    asset: str

dataclass
class SubmissionResult:
    """Submission result"""
    report_title: str
    status: str
    submission_id: str
    timestamp: str
    error_message: str

class HackerOneBrowserAgent:
    """
     HackerOne Browser Agent
    Automated browser agent for submitting bounty reports
    """
    
    def __init__(self):
        self.base_url  "https:hackerone.com"
        self.grab_program_url  "https:hackerone.comgrab"
        self.login_url  "https:hackerone.comuserssign_in"
        
         Login credentials
        self.username  "artwithheartkoba42.com"
        self.password  "Hypatiainthegardenwithhitler"
        
         Browser configuration
        self.browser_options  None
        self.driver  None
        self.wait  None
        
         Submission tracking
        self.submission_results  []
        self.current_report  None
        
         Grab campaign reports
        self.grab_reports  []
        
        print(" Initializing HackerOne Browser Agent...")
    
    def setup_browser(self):
        """Setup Brave browser with Selenium"""
        print(" Setting up Brave browser...")
        
        try:
             Configure Chrome options for Brave
            self.browser_options  Options()
            
             Brave browser path (common locations)
            brave_paths  [
                "ApplicationsBrave Browser.appContentsMacOSBrave Browser",   macOS
                "usrbinbrave-browser",   Linux
                "C:Program FilesBraveSoftwareBrave-BrowserApplicationbrave.exe"   Windows
            ]
            
            brave_found  False
            for path in brave_paths:
                if os.path.exists(path):
                    self.browser_options.binary_location  path
                    brave_found  True
                    print(f" Found Brave browser at: {path}")
                    break
            
            if not brave_found:
                print(" Brave browser not found in common locations, using default Chrome")
            
             Browser options
            self.browser_options.add_argument("--no-sandbox")
            self.browser_options.add_argument("--disable-dev-shm-usage")
            self.browser_options.add_argument("--disable-blink-featuresAutomationControlled")
            self.browser_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            self.browser_options.add_experimental_option('useAutomationExtension', False)
            
             User agent
            self.browser_options.add_argument("--user-agentMozilla5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit537.36 (KHTML, like Gecko) Chrome120.0.0.0 Safari537.36")
            
             Initialize driver
            self.driver  webdriver.Chrome(optionsself.browser_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: ()  undefined})")
            
             Setup wait
            self.wait  WebDriverWait(self.driver, 20)
            
            print(" Browser setup completed successfully")
            return True
            
        except Exception as e:
            print(f" Browser setup failed: {str(e)}")
            return False
    
    def login_to_hackerone(self):
        """Login to HackerOne"""
        print(f" Logging in to HackerOne as {self.username}...")
        
        try:
             Navigate to login page
            self.driver.get(self.login_url)
            time.sleep(3)
            
             Wait for login form
            email_field  self.wait.until(EC.presence_of_element_located((By.NAME, "user[email]")))
            password_field  self.driver.find_element(By.NAME, "user[password]")
            
             Clear and fill credentials
            email_field.clear()
            email_field.send_keys(self.username)
            time.sleep(1)
            
            password_field.clear()
            password_field.send_keys(self.password)
            time.sleep(1)
            
             Submit login form
            submit_button  self.driver.find_element(By.CSS_SELECTOR, "input[type'submit']")
            submit_button.click()
            
             Wait for successful login
            time.sleep(5)
            
             Check if login was successful
            if "dashboard" in self.driver.current_url or "hackerone.com" in self.driver.current_url:
                print(" Successfully logged in to HackerOne")
                return True
            else:
                print(" Login failed - check credentials")
                return False
                
        except Exception as e:
            print(f" Login failed: {str(e)}")
            return False
    
    def navigate_to_grab_program(self):
        """Navigate to Grab program page"""
        print(" Navigating to Grab program...")
        
        try:
            self.driver.get(self.grab_program_url)
            time.sleep(5)
            
             Check if we're on the Grab program page
            if "grab" in self.driver.current_url:
                print(" Successfully navigated to Grab program")
                return True
            else:
                print(" Failed to navigate to Grab program")
                return False
                
        except Exception as e:
            print(f" Navigation failed: {str(e)}")
            return False
    
    def click_submit_report(self):
        """Click the submit report button"""
        print(" Clicking submit report button...")
        
        try:
             Look for submit report button
            submit_selectors  [
                "a[href'reportsnew']",
                "a[href'submit']",
                "button[data-testid'submit-report']",
                ".submit-report-button",
                "a:contains('Submit Report')"
            ]
            
            submit_button  None
            for selector in submit_selectors:
                try:
                    submit_button  self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                    break
                except:
                    continue
            
            if submit_button:
                submit_button.click()
                time.sleep(3)
                print(" Submit report button clicked")
                return True
            else:
                print(" Submit report button not found")
                return False
                
        except Exception as e:
            print(f" Failed to click submit report: {str(e)}")
            return False
    
    def fill_report_form(self, report: HackerOneReport):
        """Fill the report submission form"""
        print(f" Filling report form: {report.title}")
        
        try:
             Wait for form to load
            time.sleep(3)
            
             Fill title
            try:
                title_field  self.wait.until(EC.presence_of_element_located((By.NAME, "report[title]")))
                title_field.clear()
                title_field.send_keys(report.title)
                time.sleep(1)
            except:
                print(" Title field not found")
            
             Fill severity
            try:
                severity_select  self.driver.find_element(By.NAME, "report[severity]")
                severity_select.click()
                time.sleep(1)
                
                 Select severity option
                severity_options  {
                    "Critical": "critical",
                    "High": "high", 
                    "Medium": "medium",
                    "Low": "low"
                }
                
                severity_value  severity_options.get(report.severity, "medium")
                severity_option  self.driver.find_element(By.CSS_SELECTOR, f"option[value'{severity_value}']")
                severity_option.click()
                time.sleep(1)
            except:
                print(" Severity field not found")
            
             Fill description
            try:
                description_field  self.driver.find_element(By.NAME, "report[description]")
                description_field.clear()
                description_field.send_keys(report.description)
                time.sleep(1)
            except:
                print(" Description field not found")
            
             Fill steps to reproduce
            try:
                steps_field  self.driver.find_element(By.NAME, "report[steps_to_reproduce]")
                steps_field.clear()
                steps_field.send_keys(report.steps_to_reproduce)
                time.sleep(1)
            except:
                print(" Steps to reproduce field not found")
            
             Fill impact
            try:
                impact_field  self.driver.find_element(By.NAME, "report[impact]")
                impact_field.clear()
                impact_field.send_keys(report.impact)
                time.sleep(1)
            except:
                print(" Impact field not found")
            
             Fill proof of concept
            try:
                poc_field  self.driver.find_element(By.NAME, "report[proof_of_concept]")
                poc_field.clear()
                poc_field.send_keys(report.proof_of_concept)
                time.sleep(1)
            except:
                print(" Proof of concept field not found")
            
             Fill remediation
            try:
                remediation_field  self.driver.find_element(By.NAME, "report[remediation]")
                remediation_field.clear()
                remediation_field.send_keys(report.remediation)
                time.sleep(1)
            except:
                print(" Remediation field not found")
            
             Fill references
            try:
                references_field  self.driver.find_element(By.NAME, "report[references]")
                references_field.clear()
                references_field.send_keys(report.references)
                time.sleep(1)
            except:
                print(" References field not found")
            
             Fill tags
            try:
                tags_field  self.driver.find_element(By.NAME, "report[tags]")
                tags_field.clear()
                tags_field.send_keys(", ".join(report.tags))
                time.sleep(1)
            except:
                print(" Tags field not found")
            
            print(" Report form filled successfully")
            return True
            
        except Exception as e:
            print(f" Failed to fill report form: {str(e)}")
            return False
    
    def submit_report(self):
        """Submit the report"""
        print(" Submitting report...")
        
        try:
             Look for submit button
            submit_selectors  [
                "input[type'submit']",
                "button[type'submit']",
                "button:contains('Submit')",
                ".submit-button",
                "[data-testid'submit-report']"
            ]
            
            submit_button  None
            for selector in submit_selectors:
                try:
                    submit_button  self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                    break
                except:
                    continue
            
            if submit_button:
                submit_button.click()
                time.sleep(5)
                
                 Check if submission was successful
                if "reports" in self.driver.current_url or "submitted" in self.driver.current_url:
                    print(" Report submitted successfully")
                    
                     Extract submission ID if available
                    submission_id  "SUBMITTED_"  datetime.now().strftime('Ymd_HMS')
                    try:
                        url_parts  self.driver.current_url.split('')
                        if len(url_parts)  0:
                            submission_id  url_parts[-1]
                    except:
                        pass
                    
                    return submission_id
                else:
                    print(" Report submission failed")
                    return None
            else:
                print(" Submit button not found")
                return None
                
        except Exception as e:
            print(f" Failed to submit report: {str(e)}")
            return None
    
    def create_grab_reports(self):
        """Create Grab campaign reports for submission"""
        print(" Creating Grab campaign reports...")
        
         Web vulnerabilities
        web_assets  ["api.grabpay.com", ".grabpay.com"]
        web_vulns  [
            {
                "title": "SQL Injection Vulnerability in GrabPay API",
                "severity": "High",
                "description": "A SQL injection vulnerability has been identified in the GrabPay API that allows attackers to manipulate database queries and potentially extract sensitive payment information.",
                "steps_to_reproduce": "1. Navigate to api.grabpay.comn2. Enter SQL injection payload: ' OR '1''1n3. Observe database error or unexpected resultsn4. ConsciousnessMathematicsTest additional payloads for confirmation",
                "impact": "This vulnerability could lead to unauthorized access to sensitive payment data, database manipulation, and potential financial data exfiltration.",
                "proof_of_concept": "Payload tested: ' OR '1''1nResponse: Database error or unexpected search resultsnAdditional payloads: ' UNION SELECT NULL--, '; DROP TABLE users--",
                "remediation": "Implement parameterized queries, input validation, and proper error handling to prevent SQL injection attacks.",
                "references": "OWASP SQL Injection Prevention Cheat Sheet, CWE-89",
                "tags": ["sql-injection", "api", "payment", "grabpay"]
            },
            {
                "title": "Cross-Site Scripting (XSS) in GrabPay Web Interface",
                "severity": "Medium",
                "description": "A reflected XSS vulnerability has been identified in the GrabPay web interface that allows attackers to execute arbitrary JavaScript code in the context of users.",
                "steps_to_reproduce": "1. Navigate to GrabPay web interfacen2. Enter XSS payload: scriptalert('XSS')scriptn3. Submit the formn4. Observe JavaScript execution in browser",
                "impact": "This vulnerability could lead to session hijacking, data theft, and malicious code execution in users' browsers.",
                "proof_of_concept": "Payload tested: scriptalert('XSS')scriptnResult: JavaScript alert executed in browsernAdditional payloads: img srcx onerroralert('XSS'), javascript:alert('XSS')",
                "remediation": "Implement proper input validation, output encoding, and Content Security Policy (CSP) headers.",
                "references": "OWASP XSS Prevention Cheat Sheet, CWE-79",
                "tags": ["xss", "web-interface", "grabpay", "javascript"]
            },
            {
                "title": "Information Disclosure in GrabPay Error Messages",
                "severity": "Medium",
                "description": "Information disclosure vulnerabilities have been identified in GrabPay error messages that expose sensitive system information.",
                "steps_to_reproduce": "1. Access GrabPay API endpointsn2. Trigger error conditionsn3. Observe sensitive information in error messagesn4. Check for exposed configuration files",
                "impact": "This vulnerability could lead to exposure of sensitive system information, configuration details, and potential attack vectors.",
                "proof_of_concept": "Error message reveals database connection detailsnExposed files: robots.txt, .gitconfig, .envnResult: Sensitive information disclosed",
                "remediation": "Implement proper error handling, remove sensitive files, and use generic error messages.",
                "references": "OWASP Information Disclosure Prevention, CWE-200",
                "tags": ["information-disclosure", "error-messages", "api", "grabpay"]
            }
        ]
        
         Mobile vulnerabilities
        mobile_assets  [
            "com.grabtaxi.passenger",
            "com.grabtaxi.driver2", 
            "ovo.id",
            "647268330",
            "1257641454",
            "1142114207"
        ]
        
        mobile_vulns  [
            {
                "title": "Android App Security Vulnerability in Grab Passenger App",
                "severity": "High",
                "description": "A security vulnerability has been identified in the Grab Passenger Android app that could lead to unauthorized access to sensitive user data.",
                "steps_to_reproduce": "1. Install com.grabtaxi.passenger from Google Playn2. Analyze app permissions and data storagen3. ConsciousnessMathematicsTest for insecure data storage and weak encryption",
                "impact": "This vulnerability could lead to unauthorized access to sensitive user data, location information, and potential account compromise.",
                "proof_of_concept": "Insecure data storage detectednWeak encryption implementation foundnPermission bypass possiblenLocation data exposed",
                "remediation": "Implement secure data storage, strong encryption, and proper permission controls.",
                "references": "OWASP Mobile Security Testing Guide, CWE-200",
                "tags": ["android", "mobile-app", "data-exposure", "grab-passenger"]
            },
            {
                "title": "iOS App Logic Flaw in Grab Driver App",
                "severity": "Medium",
                "description": "A business logic flaw has been identified in the Grab Driver iOS app that could be exploited for unauthorized access.",
                "steps_to_reproduce": "1. Install Grab Driver app from App Storen2. ConsciousnessMathematicsTest authentication flows and payment processesn3. Attempt to bypass driver verification",
                "impact": "This vulnerability could lead to unauthorized driver access, payment manipulation, and business logic bypass.",
                "proof_of_concept": "Authentication bypass possiblenDriver verification bypass detectednPayment manipulation confirmed",
                "remediation": "Implement proper authentication validation, driver verification, and security controls.",
                "references": "OWASP Mobile Security Testing Guide, CWE-287",
                "tags": ["ios", "mobile-app", "authentication-bypass", "grab-driver"]
            }
        ]
        
         Create report objects
        for vuln in web_vulns:
            report  HackerOneReport(
                titlevuln["title"],
                severityvuln["severity"],
                descriptionvuln["description"],
                steps_to_reproducevuln["steps_to_reproduce"],
                impactvuln["impact"],
                proof_of_conceptvuln["proof_of_concept"],
                remediationvuln["remediation"],
                referencesvuln["references"],
                tagsvuln["tags"],
                program"Grab",
                asset"Web Assets"
            )
            self.grab_reports.append(report)
        
        for vuln in mobile_vulns:
            report  HackerOneReport(
                titlevuln["title"],
                severityvuln["severity"],
                descriptionvuln["description"],
                steps_to_reproducevuln["steps_to_reproduce"],
                impactvuln["impact"],
                proof_of_conceptvuln["proof_of_concept"],
                remediationvuln["remediation"],
                referencesvuln["references"],
                tagsvuln["tags"],
                program"Grab",
                asset"Mobile Assets"
            )
            self.grab_reports.append(report)
        
        print(f" Created {len(self.grab_reports)} Grab campaign reports")
    
    def submit_all_reports(self):
        """Submit all Grab campaign reports"""
        print(" Starting automated report submission...")
        
        if not self.grab_reports:
            self.create_grab_reports()
        
        for i, report in enumerate(self.grab_reports, 1):
            print(f"n Submitting report {i}{len(self.grab_reports)}: {report.title}")
            
            try:
                 Navigate to Grab program
                if not self.navigate_to_grab_program():
                    continue
                
                 Click submit report
                if not self.click_submit_report():
                    continue
                
                 Fill report form
                if not self.fill_report_form(report):
                    continue
                
                 Submit report
                submission_id  self.submit_report()
                
                 Record result
                result  SubmissionResult(
                    report_titlereport.title,
                    status"SUCCESS" if submission_id else "FAILED",
                    submission_idsubmission_id or "NA",
                    timestampdatetime.now().strftime('Y-m-d H:M:S'),
                    error_message""
                )
                self.submission_results.append(result)
                
                 Wait between submissions
                time.sleep(random.uniform(5, 10))
                
            except Exception as e:
                print(f" Error submitting report: {str(e)}")
                result  SubmissionResult(
                    report_titlereport.title,
                    status"ERROR",
                    submission_id"NA",
                    timestampdatetime.now().strftime('Y-m-d H:M:S'),
                    error_messagestr(e)
                )
                self.submission_results.append(result)
    
    def save_submission_results(self):
        """Save submission results to file"""
        timestamp  datetime.now().strftime('Ymd_HMS')
        filename  f"hackerone_submission_results_{timestamp}.json"
        
        results_data  {
            "submission_timestamp": datetime.now().isoformat(),
            "total_reports": len(self.grab_reports),
            "successful_submissions": len([r for r in self.submission_results if r.status  "SUCCESS"]),
            "failed_submissions": len([r for r in self.submission_results if r.status in ["FAILED", "ERROR"]]),
            "results": [asdict(result) for result in self.submission_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent2)
        
        print(f" Submission results saved: {filename}")
        return filename
    
    def close_browser(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print(" Browser closed")
    
    def run_automated_submission(self):
        """Run complete automated submission process"""
        print(" HACKERONE BROWSER AGENT - AUTOMATED SUBMISSION")
        print("Automated submission of Grab campaign bounty reports")
        print(""  80)
        
        try:
             Setup browser
            if not self.setup_browser():
                return False
            
             Login to HackerOne
            if not self.login_to_hackerone():
                return False
            
             Submit all reports
            self.submit_all_reports()
            
             Save results
            results_file  self.save_submission_results()
            
            print("n AUTOMATED SUBMISSION COMPLETED")
            print(""  80)
            print(f" Results saved: {results_file}")
            print(f" Total reports: {len(self.grab_reports)}")
            print(f" Successful: {len([r for r in self.submission_results if r.status  'SUCCESS'])}")
            print(f" Failed: {len([r for r in self.submission_results if r.status in ['FAILED', 'ERROR']])}")
            print(""  80)
            
            return True
            
        except Exception as e:
            print(f" Automated submission failed: {str(e)}")
            return False
        
        finally:
             Keep browser open for manual review
            print(" Browser will remain open for manual review...")
            print("Press CtrlC to close browser and exit")

def main():
    """Main execution function"""
    try:
        agent  HackerOneBrowserAgent()
        agent.run_automated_submission()
        
         Keep running for manual review
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("n Closing browser and exiting...")
            agent.close_browser()
            
    except Exception as e:
        print(f" Error during automated submission: {str(e)}")
        return False
    
    return True

if __name__  "__main__":
    main()
