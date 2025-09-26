#!/usr/bin/env python3
"""
Simple GPT Scraper
A reliable ChatGPT conversation exporter using Playwright
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import re

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available. Install with: pip install playwright")

class SimpleGPTScraper:
    """Simple and reliable ChatGPT conversation scraper"""
    
    def __init__(self, output_dir: str = "~/dev/gpt_exports"):
        self.output_dir = Path(os.path.expanduser(output_dir))
        self.conversations_dir = self.output_dir / "conversations"
        self.markdown_dir = self.output_dir / "markdown"
        
        # Create directories
        for dir_path in [self.conversations_dir, self.markdown_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def scrape_conversations(self):
        """Scrape ChatGPT conversations using Playwright"""
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not available. Please install: pip install playwright")
            return
        
        print("🚀 Starting Simple GPT Scraper...")
        print("📁 Output directory:", self.output_dir)
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            
            try:
                # Navigate to ChatGPT
                print("🌐 Navigating to ChatGPT...")
                await page.goto("https://chat.openai.com/")
                
                # Wait for user to login
                print("🔐 Please log in to ChatGPT manually...")
                print("⏳ Waiting for login...")
                
                # Wait for login to complete (look for conversation elements)
                await page.wait_for_selector('a[href*="/c/"]', timeout=300000)  # 5 minutes
                print("✅ Login detected!")
                
                # Get all conversation links
                print("📋 Scanning for conversations...")
                conversation_links = await page.query_selector_all('a[href*="/c/"]')
                
                if not conversation_links:
                    print("❌ No conversations found. Make sure you have conversations in your history.")
                    return
                
                print(f"📄 Found {len(conversation_links)} conversations")
                
                conversations = []
                
                # Process each conversation
                for i, link in enumerate(conversation_links[:10]):  # Limit to first 10 for testing
                    try:
                        href = await link.get_attribute('href')
                        title = await link.text_content()
                        
                        if href and title:
                            conversation_id = href.split('/c/')[1].split('?')[0]
                            conversations.append({
                                'id': conversation_id,
                                'title': title.strip(),
                                'href': href
                            })
                            print(f"📝 {i+1}. {title.strip()}")
                    except Exception as e:
                        print(f"⚠️ Error processing conversation {i+1}: {e}")
                
                # Export conversations
                if conversations:
                    await self.export_conversations(page, conversations)
                else:
                    print("❌ No conversations to export")
                    
            except Exception as e:
                print(f"❌ Error during scraping: {e}")
            finally:
                await browser.close()
    
    async def export_conversations(self, page, conversations: List[Dict]):
        """Export conversations to JSON and Markdown"""
        print(f"\n📤 Exporting {len(conversations)} conversations...")
        
        exported_count = 0
        
        for i, conv in enumerate(conversations):
            try:
                print(f"📄 Exporting {i+1}/{len(conversations)}: {conv['title']}")
                
                # Navigate to conversation
                await page.goto(f"https://chat.openai.com{conv['href']}")
                await page.wait_for_load_state('networkidle')
                
                # Extract conversation content
                messages = await self.extract_messages(page)
                
                if messages:
                    # Save as JSON
                    json_data = {
                        'id': conv['id'],
                        'title': conv['title'],
                        'url': f"https://chat.openai.com{conv['href']}",
                        'exported_at': datetime.now().isoformat(),
                        'messages': messages
                    }
                    
                    json_file = self.conversations_dir / f"{conv['id']}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    # Convert to Markdown
                    markdown_content = self.convert_to_markdown(json_data)
                    markdown_file = self.markdown_dir / f"{conv['id']}.md"
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    
                    exported_count += 1
                    print(f"✅ Exported: {conv['title']}")
                else:
                    print(f"⚠️ No messages found in: {conv['title']}")
                    
            except Exception as e:
                print(f"❌ Error exporting {conv['title']}: {e}")
        
        print(f"\n🎉 Export complete! {exported_count}/{len(conversations)} conversations exported")
        print(f"📁 JSON files: {self.conversations_dir}")
        print(f"📁 Markdown files: {self.markdown_dir}")
    
    async def extract_messages(self, page) -> List[Dict]:
        """Extract messages from the current conversation page"""
        try:
            # Wait for messages to load
            await page.wait_for_selector('[data-message-author-role]', timeout=10000)
            
            # Get all message elements
            message_elements = await page.query_selector_all('[data-message-author-role]')
            
            messages = []
            for element in message_elements:
                try:
                    role = await element.get_attribute('data-message-author-role')
                    content_element = await element.query_selector('.markdown')
                    
                    if content_element:
                        content = await content_element.text_content()
                        if content and content.strip():
                            messages.append({
                                'role': role,
                                'content': content.strip(),
                                'timestamp': datetime.now().isoformat()
                            })
                except Exception as e:
                    print(f"⚠️ Error extracting message: {e}")
            
            return messages
            
        except Exception as e:
            print(f"⚠️ Error extracting messages: {e}")
            return []
    
    def convert_to_markdown(self, conversation_data: Dict) -> str:
        """Convert conversation data to Markdown format"""
        markdown = f"# {conversation_data['title']}\n\n"
        markdown += f"**Conversation ID:** {conversation_data['id']}\n"
        markdown += f"**URL:** {conversation_data['url']}\n"
        markdown += f"**Exported:** {conversation_data['exported_at']}\n\n"
        markdown += "---\n\n"
        
        for i, message in enumerate(conversation_data['messages']):
            role_emoji = "👤" if message['role'] == 'user' else "🤖"
            markdown += f"## {role_emoji} {message['role'].title()}\n\n"
            markdown += f"{message['content']}\n\n"
            markdown += "---\n\n"
        
        return markdown

async def main():
    """Main function"""
    scraper = SimpleGPTScraper()
    await scraper.scrape_conversations()

if __name__ == "__main__":
    print("🤖 Simple GPT Scraper")
    print("=" * 50)
    print("This scraper will:")
    print("1. Open ChatGPT in a browser")
    print("2. Wait for you to log in manually")
    print("3. Scan for your conversations")
    print("4. Export them to JSON and Markdown")
    print("=" * 50)
    
    asyncio.run(main())
