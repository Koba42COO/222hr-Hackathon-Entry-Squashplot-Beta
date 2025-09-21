GPT Project Exporter - README
============================

A Playwright-based tool that extracts conversations directly from the ChatGPT web app,
bypassing the official export pipeline by using the same API endpoints as the web interface.

Why This Works When Export Doesn't
==================================

• Uses your live web session (no password scraping, no tokens needed)
• Calls the same endpoints the app calls to populate your chat list and messages
• Saves to Markdown with front matter (title, project, timestamps)
• Groups conversations by Project (e.g., "Structured chaos")

Installation
===========

1. Create a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

2. Install Playwright:
   pip install playwright

3. Install browser:
   playwright install chromium

Usage
=====

Basic Export (All Conversations)
-------------------------------
python gpt_project_exporter.py --dst "$HOME/dev/gpt_export" --headful

This will:
• Open a Chrome window
• Navigate to ChatGPT
• Wait for you to log in (if needed)
• Export ALL conversations to ~/dev/gpt_export/

Project-Specific Export
----------------------
python gpt_project_exporter.py --dst "$HOME/dev/gpt_export" --headful --project "Structured chaos"

This will:
• Export ONLY conversations from the "Structured chaos" project
• Create a clean folder structure: ~/dev/gpt_export/Structured_chaos/*.md

Options
=======

--dst PATH          Destination directory (required)
--project NAME      Filter by project name (case-insensitive)
--headful           Show browser window (recommended for login)
--verbose           Enable detailed logging

Examples
========

# Export everything to ~/dev/gpt_export
python gpt_project_exporter.py --dst "$HOME/dev/gpt_export" --headful

# Export only "Structured chaos" project
python gpt_project_exporter.py --dst "$HOME/dev/gpt_export" --headful --project "Structured chaos"

# Export to current directory
python gpt_project_exporter.py --dst ./gpt_export --headful

# Export with verbose logging
python gpt_project_exporter.py --dst "$HOME/dev/gpt_export" --headful --verbose

Output Structure
===============

The tool creates a clean folder structure:

~/dev/gpt_export/
├── Structured_chaos/
│   ├── Conversation_Title_abc12345.md
│   ├── Another_Conversation_def67890.md
│   └── ...
├── Other_Project/
│   ├── Some_Conversation_ghi11111.md
│   └── ...
└── default/
    └── ...

Each .md file contains:
• YAML front matter with metadata
• Formatted conversation with user/assistant messages
• Timestamps and conversation IDs
• Code blocks preserved

Markdown Format
==============

Each exported file includes:

---
title: "Conversation Title"
project: "Structured chaos"
created_at: "2025-01-15 10:30:00 UTC"
updated_at: "2025-01-15 11:45:00 UTC"
conversation_id: "abc12345-def6-7890-ghij-klmnopqrstuv"
exported_at: "2025-08-27 21:50:00"
---

# Conversation Title

**Project:** Structured chaos  
**Created:** 2025-01-15 10:30:00 UTC  
**Updated:** 2025-01-15 11:45:00 UTC  
**Conversation ID:** abc12345-def6-7890-ghij-klmnopqrstuv

---

## 👤 User

Your message here...

## 🤖 Assistant

Assistant's response here...

---

Integration with Cursor
=======================

1. Run the export:
   python gpt_project_exporter.py --dst "$HOME/dev/gpt_export" --headful

2. Open in Cursor:
   Cursor → File → Open Folder… → ~/dev/gpt_export

3. Optional: Rebuild index:
   ⌘⇧P → Developer: Rebuild Index

Continuous Sync Workflow
========================

For ongoing sync, you can:

1. Export to ~/dev/gpt_export
2. Use the gpt_sync.sh script to copy to ~/dev/gpt
3. Open ~/dev/gpt in Cursor

Or create a combined script that:
• Exports from ChatGPT
• Copies to ~/dev/gpt
• Optionally commits to git

Troubleshooting
==============

Common Issues:

1. "Playwright not installed"
   Solution: pip install playwright && playwright install chromium

2. "Login timeout"
   Solution: Make sure to log in within 5 minutes when browser opens

3. "No conversations found"
   Solution: Check that you're logged into the correct ChatGPT account

4. "Permission denied"
   Solution: Check write permissions for destination directory

5. "Browser won't start"
   Solution: Try running with --headful to see what's happening

Error Handling
=============

The tool includes comprehensive error handling:
• Graceful timeout handling
• Detailed error messages
• Statistics tracking
• Clean browser cleanup

Security Notes
=============

• No passwords or tokens are stored
• Uses your existing ChatGPT session
• Temporary files are cleaned up
• No data is sent to external servers

Limitations
==========

• Requires manual login (for security)
• Depends on ChatGPT's internal API structure
• May need updates if ChatGPT changes their API
• Rate limited by ChatGPT's servers

Future Enhancements
==================

Potential improvements:
• Automatic login (if you provide credentials)
• Incremental sync (only new conversations)
• Git integration
• Cloud storage sync
• Web interface
• Scheduled exports

Support
=======

If the tool stops working:
1. Check if ChatGPT has changed their API
2. Update the script with new endpoints
3. The tool is designed to be easily maintainable

The script uses the same API calls as the web interface, so it should continue working
as long as the web interface works.
