#!/usr/bin/env python3
"""
GPT Project Exporter
Extracts conversations directly from ChatGPT web app using Playwright
Bypasses export pipeline by using the same API endpoints as the web app
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import logging
from urllib.parse import urlparse, parse_qs

try:
    from playwright.async_api import async_playwright, Browser, Page
except ImportError:
    print("❌ Playwright not installed. Please run: pip install playwright")
    print("Then run: playwright install chromium")
    sys.exit(1)

class GPTProjectExporter:
    """Exports GPT conversations directly from web app"""
    
    def __init__(self, destination: str, project_filter: Optional[str] = None, headful: bool = False):
        self.destination = Path(destination).expanduser()
        self.project_filter = project_filter
        self.headful = headful
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # ChatGPT API endpoints
        self.chatgpt_base = "https://chat.openai.com"
        self.conversations_endpoint = "/backend-api/conversations"
        self.messages_endpoint = "/backend-api/conversation/"
        
        # Create destination directory
        self.destination.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_conversations': 0,
            'exported_conversations': 0,
            'total_projects': 0,
            'exported_projects': 0,
            'errors': 0
        }
    
    async def setup_browser(self):
        """Initialize browser and page"""
        self.logger.info("🚀 Starting browser...")
        
        playwright = await async_playwright().start()
        
        # Launch browser
        browser_args = [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor'
        ]
        
        self.browser = await playwright.chromium.launch(
            headless=not self.headful,
            args=browser_args
        )
        
        # Create page
        self.page = await self.browser.new_page()
        
        # Set user agent
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        self.logger.info("✅ Browser ready")
    
    async def login_to_chatgpt(self):
        """Navigate to ChatGPT and wait for login"""
        self.logger.info("🔐 Navigating to ChatGPT...")
        
        await self.page.goto(self.chatgpt_base)
        
        # Wait for page to load
        await self.page.wait_for_load_state('networkidle')
        
        # Check if we need to log in
        login_button = await self.page.query_selector('button[data-testid="login-button"]')
        if login_button:
            self.logger.info("⚠️  Please log in to ChatGPT in the browser window")
            self.logger.info("   The script will wait for you to complete login...")
            
            # Wait for login to complete (URL changes to main chat)
            await self.page.wait_for_url(lambda url: "chat.openai.com" in url and "auth" not in url, timeout=300000)
            self.logger.info("✅ Login detected")
        else:
            self.logger.info("✅ Already logged in")
        
        # Wait for main chat interface
        await self.page.wait_for_selector('[data-testid="conversation-turn-2"]', timeout=10000)
        self.logger.info("✅ ChatGPT interface loaded")
    
    async def get_conversations(self) -> List[Dict]:
        """Fetch conversations from ChatGPT API"""
        self.logger.info("📋 Fetching conversations...")
        
        # Get conversations using the same API the web app uses
        response = await self.page.evaluate("""
            async () => {
                const response = await fetch('/backend-api/conversations', {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                return await response.json();
            }
        """)
        
        if 'items' not in response:
            self.logger.error("❌ Failed to fetch conversations")
            return []
        
        conversations = response['items']
        self.logger.info(f"📊 Found {len(conversations)} conversations")
        
        return conversations
    
    async def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """Fetch messages for a specific conversation"""
        try:
            response = await self.page.evaluate(f"""
                async () => {{
                    const response = await fetch('/backend-api/conversation/{conversation_id}', {{
                        method: 'GET',
                        headers: {{
                            'Content-Type': 'application/json',
                        }}
                    }});
                    return await response.json();
                }}
            """)
            
            if 'items' not in response:
                return []
            
            return response['items']
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching messages for conversation {conversation_id}: {e}")
            return []
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename.strip('._')
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename
    
    def format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            return timestamp
    
    def create_markdown_content(self, conversation: Dict, messages: List[Dict]) -> str:
        """Create markdown content from conversation and messages"""
        title = conversation.get('title', 'Untitled Conversation')
        created_at = conversation.get('create_time', '')
        updated_at = conversation.get('update_time', '')
        
        # Get project info
        project = conversation.get('current_node', {}).get('parent_id', 'default')
        
        # Create front matter
        front_matter = f"""---
title: "{title}"
project: "{project}"
created_at: "{self.format_timestamp(created_at)}"
updated_at: "{self.format_timestamp(updated_at)}"
conversation_id: "{conversation.get('id', '')}"
exported_at: "{datetime.now().isoformat()}"
---

# {title}

**Project:** {project}  
**Created:** {self.format_timestamp(created_at)}  
**Updated:** {self.format_timestamp(updated_at)}  
**Conversation ID:** {conversation.get('id', '')}

---

"""
        
        # Add messages
        content = front_matter
        
        for message in messages:
            role = message.get('message', {}).get('author', {}).get('role', 'unknown')
            content_parts = message.get('message', {}).get('content', {}).get('parts', [])
            
            if role == 'user':
                content += f"## 👤 User\n\n"
            elif role == 'assistant':
                content += f"## 🤖 Assistant\n\n"
            else:
                content += f"## {role.title()}\n\n"
            
            for part in content_parts:
                if isinstance(part, str):
                    content += f"{part}\n\n"
                elif isinstance(part, dict):
                    # Handle different content types
                    if part.get('type') == 'text':
                        content += f"{part.get('text', '')}\n\n"
                    elif part.get('type') == 'code':
                        content += f"```{part.get('language', '')}\n{part.get('text', '')}\n```\n\n"
            
            content += "---\n\n"
        
        return content
    
    async def export_conversation(self, conversation: Dict) -> bool:
        """Export a single conversation to markdown"""
        try:
            conversation_id = conversation.get('id')
            if not conversation_id:
                return False
            
            # Get messages
            messages = await self.get_conversation_messages(conversation_id)
            if not messages:
                self.logger.warning(f"⚠️  No messages found for conversation {conversation_id}")
                return False
            
            # Get project info
            project = conversation.get('current_node', {}).get('parent_id', 'default')
            
            # Apply project filter if specified
            if self.project_filter and self.project_filter.lower() not in project.lower():
                return False
            
            # Create project directory
            project_dir = self.destination / self.sanitize_filename(project)
            project_dir.mkdir(exist_ok=True)
            
            # Create markdown content
            title = conversation.get('title', 'Untitled Conversation')
            filename = self.sanitize_filename(f"{title}_{conversation_id[:8]}.md")
            filepath = project_dir / filename
            
            content = self.create_markdown_content(conversation, messages)
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"✅ Exported: {project}/{filename}")
            self.stats['exported_conversations'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting conversation {conversation.get('id', 'unknown')}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def export_all_conversations(self):
        """Export all conversations"""
        self.logger.info("🚀 Starting export process...")
        
        # Get conversations
        conversations = await self.get_conversations()
        self.stats['total_conversations'] = len(conversations)
        
        if not conversations:
            self.logger.warning("⚠️  No conversations found")
            return
        
        # Group by project
        projects = {}
        for conv in conversations:
            project = conv.get('current_node', {}).get('parent_id', 'default')
            if project not in projects:
                projects[project] = []
            projects[project].append(conv)
        
        self.stats['total_projects'] = len(projects)
        
        # Filter projects if specified
        if self.project_filter:
            filtered_projects = {
                k: v for k, v in projects.items() 
                if self.project_filter.lower() in k.lower()
            }
            projects = filtered_projects
            self.logger.info(f"🔍 Filtering for project: {self.project_filter}")
        
        self.logger.info(f"📁 Found {len(projects)} projects")
        
        # Export conversations
        for project_name, project_conversations in projects.items():
            self.logger.info(f"📂 Processing project: {project_name} ({len(project_conversations)} conversations)")
            
            project_exported = 0
            for conversation in project_conversations:
                if await self.export_conversation(conversation):
                    project_exported += 1
                
                # Small delay to be respectful
                await asyncio.sleep(0.5)
            
            if project_exported > 0:
                self.stats['exported_projects'] += 1
                self.logger.info(f"✅ Project '{project_name}': {project_exported}/{len(project_conversations)} conversations exported")
    
    def print_summary(self):
        """Print export summary"""
        print("\n" + "="*60)
        print("📊 EXPORT SUMMARY")
        print("="*60)
        print(f"📁 Destination: {self.destination}")
        print(f"📋 Total conversations found: {self.stats['total_conversations']}")
        print(f"✅ Conversations exported: {self.stats['exported_conversations']}")
        print(f"📂 Total projects found: {self.stats['total_projects']}")
        print(f"✅ Projects exported: {self.stats['exported_projects']}")
        print(f"❌ Errors: {self.stats['errors']}")
        
        if self.project_filter:
            print(f"🔍 Project filter: {self.project_filter}")
        
        print(f"\n📁 Files saved to: {self.destination}")
        print("🎉 Export complete!")
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
    
    async def run(self):
        """Main execution method"""
        try:
            await self.setup_browser()
            await self.login_to_chatgpt()
            await self.export_all_conversations()
            self.print_summary()
        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
            raise
        finally:
            await self.cleanup()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Export GPT conversations directly from ChatGPT web app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all conversations
  python gpt_project_exporter.py --dst ~/dev/gpt_export --headful
  
  # Export only "Structured chaos" project
  python gpt_project_exporter.py --dst ~/dev/gpt_export --headful --project "Structured chaos"
  
  # Export to current directory
  python gpt_project_exporter.py --dst ./gpt_export --headful
        """
    )
    
    parser.add_argument(
        '--dst',
        required=True,
        help='Destination directory for exported files'
    )
    
    parser.add_argument(
        '--project',
        help='Filter conversations by project name (case-insensitive)'
    )
    
    parser.add_argument(
        '--headful',
        action='store_true',
        help='Run browser in headful mode (show browser window)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create exporter and run
    exporter = GPTProjectExporter(
        destination=args.dst,
        project_filter=args.project,
        headful=args.headful
    )
    
    # Run async
    asyncio.run(exporter.run())

if __name__ == "__main__":
    main()
