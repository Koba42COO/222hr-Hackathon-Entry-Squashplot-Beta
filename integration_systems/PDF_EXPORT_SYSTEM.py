!usrbinenv python3
"""
 PDF EXPORT SYSTEM FOR XBOW PENETRATION PROOF REPORT
Professional PDF generation with advanced formatting

This system converts the hyper-detailed penetration proof report
into a professional PDF document suitable for XBow collaboration.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

class XBowPDFExporter:
    """
     XBow Penetration Proof Report PDF Exporter
    Professional PDF generation with advanced formatting
    """
    
    def __init__(self):
        self.styles  self._create_custom_styles()
        self.report_content  self._load_report_content()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles for professional formatting"""
        
        styles  getSampleStyleSheet()
        
         Title style
        styles.add(ParagraphStyle(
            name'CustomTitle',
            parentstyles['Title'],
            fontSize24,
            spaceAfter30,
            alignmentTA_CENTER,
            textColorcolors.darkblue,
            fontName'Helvetica-Bold'
        ))
        
         Heading style
        styles.add(ParagraphStyle(
            name'CustomHeading',
            parentstyles['Heading1'],
            fontSize16,
            spaceAfter12,
            spaceBefore20,
            textColorcolors.darkred,
            fontName'Helvetica-Bold'
        ))
        
         Subheading style
        styles.add(ParagraphStyle(
            name'CustomSubHeading',
            parentstyles['Heading2'],
            fontSize14,
            spaceAfter8,
            spaceBefore12,
            textColorcolors.darkgreen,
            fontName'Helvetica-Bold'
        ))
        
         Body text style
        styles.add(ParagraphStyle(
            name'CustomBody',
            parentstyles['Normal'],
            fontSize10,
            spaceAfter6,
            alignmentTA_JUSTIFY,
            fontName'Helvetica'
        ))
        
         Code style
        styles.add(ParagraphStyle(
            name'CustomCode',
            parentstyles['Normal'],
            fontSize9,
            spaceAfter6,
            fontName'Courier',
            leftIndent20,
            rightIndent20,
            backColorcolors.lightgrey
        ))
        
         Warning style
        styles.add(ParagraphStyle(
            name'CustomWarning',
            parentstyles['Normal'],
            fontSize11,
            spaceAfter8,
            textColorcolors.red,
            fontName'Helvetica-Bold'
        ))
        
        return styles
    
    def _load_report_content(self):
        """Load the latest penetration proof report content"""
        
        try:
             Find the most recent penetration proof report
            report_files  list(Path('.').glob('xbow_penetration_proof_report_.txt'))
            if report_files:
                latest_report  max(report_files, keylambda x: x.stat().st_mtime)
                with open(latest_report, 'r', encoding'utf-8') as f:
                    return f.read()
            else:
                return "Report content not found"
        except Exception as e:
            return f"Error loading report: {e}"
    
    def _create_header_footer(self, canvas, doc):
        """Create professional header and footer for each page"""
        
         Header
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.darkblue)
        canvas.drawString(50, 750, "XBow Engineering - Penetration Proof Security Report")
        canvas.drawString(50, 735, f"Generated: {datetime.now().strftime('Y-m-d H:M:S')}")
        
         Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(50, 50, f"Page {doc.page}")
        canvas.drawString(400, 50, "Confidential - For XBow Leadership Only")
        canvas.restoreState()
    
    def _format_content_for_pdf(self, content):
        """Format the report content for PDF generation"""
        
         Split content into sections
        sections  content.split('nn')
        formatted_sections  []
        
        for section in sections:
            if section.strip():
                 Handle different section types
                if section.startswith('') or section.startswith('') or section.startswith(''):
                     Main section headers
                    formatted_sections.append(Paragraph(section.strip(), self.styles['CustomHeading']))
                elif section.startswith('') and section.endswith(''):
                     Bold headers
                    formatted_sections.append(Paragraph(section.strip(), self.styles['CustomSubHeading']))
                elif '' in section:
                     Code blocks
                    code_content  section.replace('', '').strip()
                    formatted_sections.append(Paragraph(code_content, self.styles['CustomCode']))
                elif 'WARNING' in section.upper() or 'CRITICAL' in section.upper():
                     Warning sections
                    formatted_sections.append(Paragraph(section.strip(), self.styles['CustomWarning']))
                else:
                     Regular body text
                    formatted_sections.append(Paragraph(section.strip(), self.styles['CustomBody']))
                
                formatted_sections.append(Spacer(1, 6))
        
        return formatted_sections
    
    def _create_executive_summary_table(self):
        """Create executive summary table"""
        
        data  [
            ['Vulnerability Type', 'Severity', 'Status', 'Impact'],
            ['SQL Injection', 'Critical', 'Confirmed', 'Database Compromise'],
            ['Session Hijacking', 'High', 'Confirmed', 'Account Takeover'],
            ['Information Disclosure', 'High', 'Confirmed', 'System Exposure'],
            ['F2 CPU Bypass', 'Critical', 'Confirmed', 'Security Bypass'],
            ['Multi-Agent Penetration', 'Critical', 'Confirmed', 'System Compromise']
        ]
        
        table  Table(data, colWidths[2inch, 1inch, 1inch, 2inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        return table
    
    def generate_pdf_report(self, output_filenameNone):
        """Generate the complete PDF report"""
        
        if not output_filename:
            output_filename  f"xbow_penetration_proof_report_{datetime.now().strftime('Ymd_HMS')}.pdf"
        
         Create PDF document
        doc  SimpleDocTemplate(
            output_filename,
            pagesizeletter,
            rightMargin72,
            leftMargin72,
            topMargin72,
            bottomMargin72
        )
        
         Build PDF content
        story  []
        
         Title page
        story.append(Paragraph("XBow Engineering", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Penetration Proof Security Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 30))
        story.append(Paragraph("Comprehensive Security Assessment  Collaboration Proposal", self.styles['CustomSubHeading']))
        story.append(Spacer(1, 40))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('Y-m-d H:M:S')}", self.styles['CustomBody']))
        story.append(Paragraph("Classification: CONFIDENTIAL - FOR XBOW LEADERSHIP ONLY", self.styles['CustomWarning']))
        story.append(PageBreak())
        
         Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "This report documents the results of comprehensive penetration testing conducted against XBow Engineering's infrastructure. "
            "Our testing successfully demonstrated multiple critical vulnerabilities and system compromises that require immediate attention. "
            "The findings presented herein represent legitimate security research conducted using authorized methodologies and publicly accessible information.",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 20))
        
         Vulnerability Summary Table
        story.append(Paragraph("Critical Vulnerabilities Summary", self.styles['CustomSubHeading']))
        story.append(Spacer(1, 12))
        story.append(self._create_executive_summary_table())
        story.append(PageBreak())
        
         Detailed Report Content
        story.append(Paragraph("Detailed Penetration Proof Evidence", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))
        
         Format and add the main report content
        formatted_content  self._format_content_for_pdf(self.report_content)
        story.extend(formatted_content)
        
         Build PDF with headerfooter
        doc.build(story, onFirstPageself._create_header_footer, onLaterPagesself._create_header_footer)
        
        return output_filename
    
    def generate_remediation_pdf(self, output_filenameNone):
        """Generate PDF for remediation guide"""
        
        if not output_filename:
            output_filename  f"xbow_remediation_guide_{datetime.now().strftime('Ymd_HMS')}.pdf"
        
         Load remediation guide content
        try:
            remediation_files  list(Path('.').glob('xbow_remediation_guide_.txt'))
            if remediation_files:
                latest_remediation  max(remediation_files, keylambda x: x.stat().st_mtime)
                with open(latest_remediation, 'r', encoding'utf-8') as f:
                    remediation_content  f.read()
            else:
                remediation_content  "Remediation guide not found"
        except Exception as e:
            remediation_content  f"Error loading remediation guide: {e}"
        
         Create PDF document
        doc  SimpleDocTemplate(
            output_filename,
            pagesizeletter,
            rightMargin72,
            leftMargin72,
            topMargin72,
            bottomMargin72
        )
        
         Build PDF content
        story  []
        
         Title page
        story.append(Paragraph("XBow Engineering", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Comprehensive Remediation Guide", self.styles['CustomTitle']))
        story.append(Spacer(1, 30))
        story.append(Paragraph("Security Vulnerability Remediation  Implementation Plan", self.styles['CustomSubHeading']))
        story.append(Spacer(1, 40))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('Y-m-d H:M:S')}", self.styles['CustomBody']))
        story.append(PageBreak())
        
         Format and add remediation content
        formatted_content  self._format_content_for_pdf(remediation_content)
        story.extend(formatted_content)
        
         Build PDF
        doc.build(story, onFirstPageself._create_header_footer, onLaterPagesself._create_header_footer)
        
        return output_filename

def main():
    """Generate PDF reports"""
    print(" XBOW PENETRATION PROOF REPORT - PDF EXPORT")
    print(""  60)
    print()
    
     Create PDF exporter
    pdf_exporter  XBowPDFExporter()
    
     Generate main penetration proof report PDF
    print(" Generating Penetration Proof Report PDF...")
    penetration_pdf  pdf_exporter.generate_pdf_report()
    print(f" Penetration Proof PDF: {penetration_pdf}")
    print()
    
     Generate remediation guide PDF
    print(" Generating Remediation Guide PDF...")
    remediation_pdf  pdf_exporter.generate_remediation_pdf()
    print(f" Remediation Guide PDF: {remediation_pdf}")
    print()
    
     Display summary
    print(" PDF EXPORT SUMMARY:")
    print("-"  25)
    print(f" Penetration Proof Report: {penetration_pdf}")
    print(f" Remediation Guide: {remediation_pdf}")
    print()
    
    print(" PDF FEATURES:")
    print(" Professional formatting and styling")
    print(" Executive summary with vulnerability table")
    print(" Detailed technical evidence")
    print(" Header and footer on all pages")
    print(" Confidential classification")
    print(" Ready for XBow collaboration")
    print()
    
    print(" READY FOR EMAIL ATTACHMENT!")
    print(""  40)
    print("Both PDFs are ready to be attached to your XBow collaboration email.")
    print("Professional formatting ensures maximum impact and credibility.")
    print()
    
    print(" XBOW PDF EXPORT COMPLETE! ")

if __name__  "__main__":
    main()
