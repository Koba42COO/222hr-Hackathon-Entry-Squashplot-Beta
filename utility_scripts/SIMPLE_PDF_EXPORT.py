!usrbinenv python3
"""
 SIMPLE PDF EXPORT FOR XBOW PENETRATION PROOF REPORT
HTML to PDF conversion using system tools

This system converts the hyper-detailed penetration proof report
into a professional PDF document using HTML and system tools.
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

class SimplePDFExporter:
    """
     Simple PDF Exporter using HTML and system tools
    """
    
    def __init__(self):
        self.report_content  self._load_report_content()
        self.remediation_content  self._load_remediation_content()
        
    def _load_report_content(self):
        """Load the latest penetration proof report content"""
        try:
            report_files  list(Path('.').glob('xbow_penetration_proof_report_.txt'))
            if report_files:
                latest_report  max(report_files, keylambda x: x.stat().st_mtime)
                with open(latest_report, 'r', encoding'utf-8') as f:
                    return f.read()
            else:
                return "Report content not found"
        except Exception as e:
            return f"Error loading report: {e}"
    
    def _load_remediation_content(self):
        """Load the latest remediation guide content"""
        try:
            remediation_files  list(Path('.').glob('xbow_remediation_guide_.txt'))
            if remediation_files:
                latest_remediation  max(remediation_files, keylambda x: x.stat().st_mtime)
                with open(latest_remediation, 'r', encoding'utf-8') as f:
                    return f.read()
            else:
                return "Remediation guide not found"
        except Exception as e:
            return f"Error loading remediation guide: {e}"
    
    def _create_html_report(self, content, title, output_filename):
        """Create HTML version of the report"""
        
        html_content  f"""
!DOCTYPE html
html lang"en"
head
    meta charset"UTF-8"
    meta name"viewport" content"widthdevice-width, initial-scale1.0"
    title{title}title
    style
        body {{
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 40px;
            color: 333;
            background-color: f9f9f9;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: 2c3e50;
            text-align: center;
            border-bottom: 3px solid 3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: e74c3c;
            border-left: 4px solid e74c3c;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: 27ae60;
            margin-top: 25px;
        }}
        .warning {{
            background-color: fff3cd;
            border: 1px solid ffeaa7;
            color: 856404;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .critical {{
            background-color: f8d7da;
            border: 1px solid f5c6cb;
            color: 721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .code-block {{
            background-color: f8f9fa;
            border: 1px solid e9ecef;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
            margin: 15px 0;
        }}
        .header {{
            background-color: 2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            margin: -40px -40px 30px -40px;
        }}
        .footer {{
            background-color: 34495e;
            color: white;
            padding: 15px;
            text-align: center;
            margin: 30px -40px -40px -40px;
            font-size: 12px;
        }}
        table {{
            width: 100;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: 3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: f2f2f2;
        }}
        .timestamp {{
            color: 7f8c8d;
            font-size: 12px;
            text-align: center;
            margin: 10px 0;
        }}
    style
head
body
    div class"container"
        div class"header"
            h1XBow Engineeringh1
            h2{title}h2
            pComprehensive Security Assessment  Collaboration Proposalp
        div
        
        div class"timestamp"
            Report Generated: {datetime.now().strftime('Y-m-d H:M:S')}
        div
        
        div class"warning"
            strongCLASSIFICATION:strong CONFIDENTIAL - FOR XBOW LEADERSHIP ONLY
        div
        
        {self._convert_content_to_html(content)}
        
        div class"footer"
            pGenerated by Advanced Security Research Team  {datetime.now().strftime('Y-m-d')}p
            pThis report represents legitimate security research conducted using authorized methodologiesp
        div
    div
body
html
"""
        
        with open(output_filename, 'w', encoding'utf-8') as f:
            f.write(html_content)
        
        return output_filename
    
    def _convert_content_to_html(self, content):
        """Convert text content to HTML with proper formatting"""
        
         Replace markdown-style formatting with HTML
        html_content  content
        
         Convert section headers
        html_content  html_content.replace('', 'h2')
        html_content  html_content.replace('', 'h2')
        html_content  html_content.replace('', 'h2')
        html_content  html_content.replace('', 'h2')
        
         Convert bold headers
        html_content  html_content.replace('', 'strong')
        html_content  html_content.replace('', 'strong')
        
         Convert code blocks
        html_content  html_content.replace('', 'div class"code-block"')
        html_content  html_content.replace('', 'div')
        
         Convert critical sections
        html_content  html_content.replace('CRITICAL', 'span class"critical"CRITICALspan')
        html_content  html_content.replace('WARNING', 'span class"warning"WARNINGspan')
        
         Convert line breaks
        html_content  html_content.replace('nn', 'pp')
        html_content  html_content.replace('n', 'br')
        
         Wrap in paragraphs
        html_content  f'p{html_content}p'
        
        return html_content
    
    def _html_to_pdf(self, html_file, pdf_file):
        """Convert HTML to PDF using system tools"""
        
         Try different methods to convert HTML to PDF
        methods  [
             Method 1: wkhtmltopdf (if available)
            ['wkhtmltopdf', '--page-size', 'Letter', '--margin-top', '0.75in', 
             '--margin-bottom', '0.75in', '--margin-left', '0.75in', '--margin-right', '0.75in',
             html_file, pdf_file],
            
             Method 2: pandoc (if available)
            ['pandoc', html_file, '-o', pdf_file, '--pdf-enginewkhtmltopdf'],
            
             Method 3: weasyprint (if available)
            ['weasyprint', html_file, pdf_file],
        ]
        
        for method in methods:
            try:
                result  subprocess.run(method, capture_outputTrue, textTrue, timeout30)
                if result.returncode  0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return False
    
    def generate_pdf_report(self):
        """Generate PDF report"""
        
         Create HTML version
        html_file  f"xbow_penetration_proof_report_{datetime.now().strftime('Ymd_HMS')}.html"
        pdf_file  html_file.replace('.html', '.pdf')
        
        self._create_html_report(
            self.report_content,
            "Penetration Proof Security Report",
            html_file
        )
        
         Convert to PDF
        if self._html_to_pdf(html_file, pdf_file):
            print(f" PDF generated: {pdf_file}")
            return pdf_file
        else:
            print(f"  PDF conversion failed, HTML file available: {html_file}")
            return html_file
    
    def generate_remediation_pdf(self):
        """Generate PDF remediation guide"""
        
         Create HTML version
        html_file  f"xbow_remediation_guide_{datetime.now().strftime('Ymd_HMS')}.html"
        pdf_file  html_file.replace('.html', '.pdf')
        
        self._create_html_report(
            self.remediation_content,
            "Comprehensive Remediation Guide",
            html_file
        )
        
         Convert to PDF
        if self._html_to_pdf(html_file, pdf_file):
            print(f" PDF generated: {pdf_file}")
            return pdf_file
        else:
            print(f"  PDF conversion failed, HTML file available: {html_file}")
            return html_file

def main():
    """Generate PDF reports"""
    print(" XBOW PENETRATION PROOF REPORT - SIMPLE PDF EXPORT")
    print(""  60)
    print()
    
     Create PDF exporter
    pdf_exporter  SimplePDFExporter()
    
     Generate main penetration proof report
    print(" Generating Penetration Proof Report...")
    penetration_file  pdf_exporter.generate_pdf_report()
    print()
    
     Generate remediation guide
    print(" Generating Remediation Guide...")
    remediation_file  pdf_exporter.generate_remediation_pdf()
    print()
    
     Display summary
    print(" EXPORT SUMMARY:")
    print("-"  25)
    print(f" Penetration Proof: {penetration_file}")
    print(f" Remediation Guide: {remediation_file}")
    print()
    
    print(" FEATURES:")
    print(" Professional HTML formatting")
    print(" Responsive design")
    print(" Color-coded sections")
    print(" Executive summary")
    print(" Technical details")
    print(" Ready for XBow collaboration")
    print()
    
    if penetration_file.endswith('.pdf') and remediation_file.endswith('.pdf'):
        print(" PDFs READY FOR EMAIL ATTACHMENT!")
        print(""  40)
        print("Both PDFs are ready to be attached to your XBow collaboration email.")
    else:
        print(" HTML FILES READY!")
        print(""  40)
        print("HTML files are ready. You can:")
        print(" Open in browser and print to PDF")
        print(" Use online HTML to PDF converters")
        print(" Attach HTML files to email")
    print()
    
    print(" XBOW EXPORT COMPLETE! ")

if __name__  "__main__":
    main()
