!usrbinenv python3
"""
 STANDARD EMAIL SENDER WITH AUTO-CC
Always CC cookoba42.com on outgoing emails

This system ensures all outgoing emails include Brad Wallace as CC
for record keeping and transparency.
"""

import subprocess
import os
from datetime import datetime

def send_email_with_cc(recipients, subject, content_file, cc_email"cookoba42.com"):
    """
    Send email using terminal mail command with automatic CC
    
    Args:
        recipients: List of primary recipients
        subject: Email subject line
        content_file: Path to file containing email content
        cc_email: CC recipient (default: cookoba42.com)
    """
    
     Combine recipients and CC
    all_recipients  recipients.copy()
    if cc_email not in all_recipients:
        all_recipients.append(cc_email)
    
    recipient_string  ",".join(all_recipients)
    
    print(f" SENDING EMAIL WITH AUTO-CC")
    print(""  40)
    print(f"To: {', '.join(recipients)}")
    print(f"CC: {cc_email}")
    print(f"Subject: {subject}")
    print(f"Content: {content_file}")
    print()
    
     Send email using mail command
    try:
        command  f'cat "{content_file}"  mail -s "{subject}" {recipient_string}'
        result  subprocess.run(command, shellTrue, capture_outputTrue, textTrue)
        
        if result.returncode  0:
            print(" EMAIL SENT SUCCESSFULLY!")
            print(f" CC sent to: {cc_email}")
            
             Log the email
            log_email_sent(recipients, cc_email, subject, content_file)
            return True
        else:
            print(" EMAIL SENDING FAILED!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f" EMAIL SENDING ERROR: {e}")
        return False

def log_email_sent(recipients, cc_email, subject, content_file):
    """Log sent email for record keeping"""
    
    log_entry  f"""
EMAIL SENT LOG

Date: {datetime.now().strftime('Y-m-d H:M:S')}
To: {', '.join(recipients)}
CC: {cc_email}
Subject: {subject}
Content File: {content_file}
Status: SENT
"""
    
    log_file  f"email_log_{datetime.now().strftime('Ymd')}.txt"
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    print(f" Email logged to: {log_file}")

def send_xbow_email(content_file):
    """Send email to XBow team with standard settings"""
    
    recipients  [
        "sarah.chenxbow.ai",
        "marcus.rodriguezxbow.ai", 
        "alex.kimxbow.ai"
    ]
    
    subject  "Independent AI Security Research  XBow Collaboration Opportunity"
    
    return send_email_with_cc(recipients, subject, content_file)

if __name__  "__main__":
     ConsciousnessMathematicsExample usage
    print(" STANDARD EMAIL SENDER WITH AUTO-CC")
    print(""  50)
    print()
    print("This system automatically CCs cookoba42.com on all outgoing emails")
    print("for record keeping and transparency.")
    print()
    print("Usage examples:")
    print(" send_xbow_email('email_content.txt')")
    print(" send_email_with_cc(['userexample.com'], 'Subject', 'content.txt')")
    print()
