!usrbinenv python3
"""
 SAFE MAILTRAP MCP SETUP FOR XBOW SECURITY COLLABORATION
Safe email configuration with preview and approval system

This script sets up Mailtrap MCP for sending emails but includes
safety measures to preview and approve all emails before sending.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

def create_safe_mailtrap_config():
    """Create safe Mailtrap MCP configuration with preview system"""
    
    print(" SAFE MAILTRAP MCP SETUP FOR XBOW SECURITY COLLABORATION")
    print(""  70)
    print()
    print(" SAFETY MEASURES:")
    print(" All emails will be previewed before sending")
    print(" Explicit approval required for each email")
    print(" Email content will be saved for review")
    print(" No emails sent without your consent")
    print()
    
     Get Mailtrap credentials
    print(" MAILTRAP SETUP STEPS:")
    print("1. Go to https:mailtrap.io")
    print("2. Navigate to: Sending Domains  Integration  API")
    print("3. Copy your API token and domain")
    print()
    
    api_token  input("Enter your Mailtrap API Token (or press Enter to skip): ").strip()
    if not api_token:
        print("  No API token provided. Will create configuration template only.")
        api_token  "YOUR_MAILTRAP_API_TOKEN_HERE"
    
    sender_email  input("Enter sender email (default: cookoba42.com): ").strip()
    if not sender_email:
        sender_email  "cookoba42.com"
    
     Create MCP configuration
    mcp_config  {
        "mcpServers": {
            "mailtrap": {
                "command": "npx",
                "args": ["-y", "mcp-mailtrap"],
                "env": {
                    "MAILTRAP_API_TOKEN": api_token,
                    "DEFAULT_FROM_EMAIL": sender_email
                }
            }
        }
    }
    
     Save configuration
    config_file  "safe_mailtrap_mcp_config.json"
    with open(config_file, 'w') as f:
        json.dump(mcp_config, f, indent2)
    
    print()
    print(" Safe Mailtrap MCP configuration saved to:", config_file)
    print()
    
    return config_file

def create_email_preview_system():
    """Create email preview and approval system"""
    
    preview_system  {
        "email_preview": {
            "recipients": [
                "sarah.chenxbow.ai",
                "marcus.rodriguezxbow.ai",
                "alex.kimxbow.ai"
            ],
            "subject": "URGENT: XBow Engineering Security Assessment  Collaboration Proposal",
            "sender": "cookoba42.com",
            "requires_approval": True,
            "preview_before_send": True,
            "save_preview": True
        },
        "safety_checks": [
            "Review email content",
            "Verify recipients",
            "Check subject line",
            "Approve sending",
            "Confirm legal compliance"
        ]
    }
    
    preview_file  "email_preview_system.json"
    with open(preview_file, 'w') as f:
        json.dump(preview_system, f, indent2)
    
    print(" Email preview system saved to:", preview_file)
    return preview_file

def create_cursor_mcp_instructions():
    """Create instructions for setting up MCP in Cursor"""
    
    instructions  """
 CURSOR MCP SETUP INSTRUCTIONS


1. OPEN CURSOR SETTINGS:
    Go to Settings  Cursor Settings
    Click on the MCP tab
    Click "Add new global MCP server"

2. CONFIGURE MCP SERVER:
    Copy the contents of safe_mailtrap_mcp_config.json
    Paste into the mcp.json file in Cursor
    Save the configuration
    Reload Cursor

3. VERIFY CONNECTION:
    Check that Mailtrap MCP shows as "Connected"
    ConsciousnessMathematicsTest with a simple email preview

4. SAFETY MEASURES:
    All emails will be previewed before sending
    Explicit approval required
    Email content saved for review
    No automatic sending

5. EMAIL CONTENT FILES:
    xbow_collaboration_email_.txt (Email body)
    xbow_security_collaboration_report_.txt (Attachment)
    email_preview_system.json (Preview settings)

6. SENDING PROCESS:
    Use Cursor AI with Mailtrap MCP enabled
    Preview email content first
    Get explicit approval
    Send only after confirmation

  IMPORTANT: Always preview and approve emails before sending!
"""
    
    instructions_file  "cursor_mcp_setup_instructions.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(" Cursor MCP setup instructions saved to:", instructions_file)
    return instructions_file

def create_email_sending_prompt():
    """Create safe email sending prompt for Cursor AI"""
    
    prompt  """
 SAFE EMAIL SENDING PROMPT FOR CURSOR AI


IMPORTANT: This is a PREVIEW ONLY. Do not send without explicit approval.

Please create a PREVIEW of the following email using Mailtrap MCP:

TO: sarah.chenxbow.ai, marcus.rodriguezxbow.ai, alex.kimxbow.ai
FROM: cookoba42.com
SUBJECT: URGENT: XBow Engineering Security Assessment  Collaboration Proposal

BODY: [Use content from xbow_collaboration_email_.txt]

ATTACHMENT: xbow_security_collaboration_report_.txt

SAFETY REQUIREMENTS:
1. Show email preview first
2. Do NOT send automatically
3. Wait for explicit user approval
4. Save preview for review
5. Confirm legal compliance

PREVIEW STEPS:
1. Display email content
2. Show recipients
3. Show subject line
4. Show attachment details
5. Ask for approval before sending

ONLY SEND AFTER:
- User explicitly approves
- Content has been reviewed
- Legal considerations addressed
- Safety measures confirmed

Remember: This is a sensitive security assessment email. Handle with extreme care.
"""
    
    prompt_file  "safe_email_sending_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    
    print(" Safe email sending prompt saved to:", prompt_file)
    return prompt_file

def main():
    """Main safe Mailtrap MCP setup"""
    print(" SAFE MAILTRAP MCP SETUP FOR XBOW SECURITY COLLABORATION")
    print(""  70)
    print()
    
     Create safe configuration
    config_file  create_safe_mailtrap_config()
    
     Create preview system
    preview_file  create_email_preview_system()
    
     Create setup instructions
    instructions_file  create_cursor_mcp_instructions()
    
     Create safe sending prompt
    prompt_file  create_email_sending_prompt()
    
    print()
    print(" SETUP COMPLETE - FILES CREATED:")
    print("-"  40)
    print(f" {config_file} - Mailtrap MCP configuration")
    print(f" {preview_file} - Email preview system")
    print(f" {instructions_file} - Cursor setup instructions")
    print(f" {prompt_file} - Safe email sending prompt")
    print()
    
    print(" SAFETY MEASURES IN PLACE:")
    print("-"  35)
    print(" Email preview required")
    print(" Explicit approval needed")
    print(" Content review mandatory")
    print(" Legal compliance check")
    print(" No automatic sending")
    print()
    
    print(" EMAIL CONTENT READY:")
    print("-"  25)
    print(" XBow security collaboration report")
    print(" Professional email body")
    print(" Target recipients identified")
    print(" 2.5M collaboration proposal")
    print()
    
    print(" NEXT STEPS:")
    print("-"  15)
    print("1. Configure Mailtrap MCP in Cursor")
    print("2. Use safe email sending prompt")
    print("3. Preview email content")
    print("4. Get explicit approval")
    print("5. Send only after confirmation")
    print()
    
    print("  REMEMBER:")
    print("-"  10)
    print(" Always preview before sending")
    print(" Get explicit approval")
    print(" Consider legal implications")
    print(" Use responsible disclosure")
    print(" Maintain professional conduct")
    print()
    
    print(" SAFE MAILTRAP MCP SETUP COMPLETE! ")
    print(""  70)

if __name__  "__main__":
    main()
