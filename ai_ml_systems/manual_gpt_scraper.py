#!/usr/bin/env python3
"""
Manual GPT Scraper
Step-by-step ChatGPT conversation exporter with manual control
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available. Install with: pip install playwright")

class ManualGPTScraper:
    """Manual ChatGPT conversation scraper with step-by-step control"""
    
    def __init__(self, output_dir: str = "~/dev/gpt_exports"):
        self.output_dir = Path(os.path.expanduser(output_dir))
        self.conversations_dir = self.output_dir / "conversations"
        self.markdown_dir = self.output_dir / "markdown"
        
        # Create directories
        for dir_path in [self.conversations_dir, self.markdown_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def run(self):
        """Run the manual scraper with step-by-step control"""
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not available. Please install: pip install playwright")
            return
        
        print("🤖 Manual GPT Scraper")
        print("=" * 50)
        print("This scraper will guide you through the process step by step.")
        print("=" * 50)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            
            try:
                # Step 1: Navigate to ChatGPT
                print("\n📋 Step 1: Opening ChatGPT...")
                await page.goto("https://chat.openai.com/")
                print("✅ ChatGPT opened in browser")
                
                # Step 2: Manual login
                print("\n📋 Step 2: Manual Login")
                print("🔐 Please log in to ChatGPT manually in the browser window.")
                print("⏳ Press Enter when you're logged in and can see your conversations...")
                input()
                
                # Step 3: Check for conversations
                print("\n📋 Step 3: Scanning for conversations...")
                await page.wait_for_load_state('networkidle')
                
                # Try different selectors for conversations
                selectors = [
                    'a[href*="/c/"]',
                    '[data-testid*="conversation"]',
                    'nav a[href*="/c/"]',
                    'a[href*="conversation"]'
                ]
                
                conversations = []
                for selector in selectors:
                    try:
                        links = await page.query_selector_all(selector)
                        print(f"🔍 Found {len(links)} elements with selector: {selector}")
                        
                        for link in links:
                            try:
                                href = await link.get_attribute('href')
                                title = await link.text_content()
                                
                                if href and title and '/c/' in href:
                                    conversation_id = href.split('/c/')[1].split('?')[0]
                                    conversations.append({
                                        'id': conversation_id,
                                        'title': title.strip(),
                                        'href': href
                                    })
                            except Exception as e:
                                continue
                    except Exception as e:
                        print(f"⚠️ Error with selector {selector}: {e}")
                
                # Remove duplicates
                unique_conversations = []
                seen_ids = set()
                for conv in conversations:
                    if conv['id'] not in seen_ids:
                        unique_conversations.append(conv)
                        seen_ids.add(conv['id'])
                
                print(f"\n📄 Found {len(unique_conversations)} unique conversations:")
                for i, conv in enumerate(unique_conversations[:10]):  # Show first 10
                    print(f"  {i+1}. {conv['title']}")
                
                if len(unique_conversations) > 10:
                    print(f"  ... and {len(unique_conversations) - 10} more")
                
                if not unique_conversations:
                    print("❌ No conversations found. Make sure you're logged in and have conversations.")
                    return
                
                # Step 4: Export conversations
                print(f"\n📋 Step 4: Exporting conversations...")
                print(f"📤 Will export {len(unique_conversations)} conversations")
                print("⏳ Press Enter to start export...")
                input()
                
                exported_count = 0
                for i, conv in enumerate(unique_conversations):
                    try:
                        print(f"\n📄 Exporting {i+1}/{len(unique_conversations)}: {conv['title']}")
                        
                        # Navigate to conversation
                        full_url = f"https://chat.openai.com{conv['href']}"
                        await page.goto(full_url)
                        await page.wait_for_load_state('networkidle')
                        
                        # Extract messages
                        messages = await self.extract_messages(page)
                        
                        if messages:
                            # Save JSON
                            json_data = {
                                'id': conv['id'],
                                'title': conv['title'],
                                'url': full_url,
                                'exported_at': datetime.now().isoformat(),
                                'message_count': len(messages),
                                'messages': messages
                            }
                            
                            json_file = self.conversations_dir / f"{conv['id']}.json"
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)
                            
                            # Save Markdown
                            markdown_content = self.convert_to_markdown(json_data)
                            markdown_file = self.markdown_dir / f"{conv['id']}.md"
                            with open(markdown_file, 'w', encoding='utf-8') as f:
                                f.write(markdown_content)
                            
                            exported_count += 1
                            print(f"✅ Exported: {conv['title']} ({len(messages)} messages)")
                        else:
                            print(f"⚠️ No messages found in: {conv['title']}")
                            
                    except Exception as e:
                        print(f"❌ Error exporting {conv['title']}: {e}")
                
                # Summary
                print(f"\n🎉 Export complete!")
                print(f"📊 Exported: {exported_count}/{len(unique_conversations)} conversations")
                print(f"📁 JSON files: {self.conversations_dir}")
                print(f"📁 Markdown files: {self.markdown_dir}")
                
            except Exception as e:
                print(f"❌ Error during scraping: {e}")
            finally:
                print("\n🔒 Closing browser...")
                await browser.close()
    
    async def extract_messages(self, page) -> List[Dict]:
        """Extract messages from the current conversation page"""
        try:
            # Wait a bit for content to load
            await asyncio.sleep(2)
            
            # Try different selectors for messages
            message_selectors = [
                '[data-message-author-role]',
                '.markdown',
                '[data-testid*="message"]',
                '.text-base'
            ]
            
            messages = []
            for selector in message_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        print(f"  🔍 Found {len(elements)} elements with selector: {selector}")
                        
                        for element in elements:
                            try:
                                # Try to get role
                                role = await element.get_attribute('data-message-author-role')
                                if not role:
                                    # Try to determine role from context
                                    parent = await element.query_selector('xpath=..')
                                    if parent:
                                        role_attr = await parent.get_attribute('data-message-author-role')
                                        if role_attr:
                                            role = role_attr
                                
                                # Get content
                                content = await element.text_content()
                                
                                if content and content.strip() and len(content.strip()) > 10:
                                    messages.append({
                                        'role': role or 'unknown',
                                        'content': content.strip(),
                                        'timestamp': datetime.now().isoformat()
                                    })
                            except Exception as e:
                                continue
                        
                        if messages:
                            break  # Found messages, stop trying other selectors
                            
                except Exception as e:
                    continue
            
            return messages
            
        except Exception as e:
            print(f"⚠️ Error extracting messages: {e}")
            return []
    
    def convert_to_markdown(self, conversation_data: Dict) -> str:
        """Convert conversation data to Markdown format"""
        markdown = f"# {conversation_data['title']}\n\n"
        markdown += f"**Conversation ID:** {conversation_data['id']}\n"
        markdown += f"**URL:** {conversation_data['url']}\n"
        markdown += f"**Exported:** {conversation_data['exported_at']}\n"
        markdown += f"**Message Count:** {conversation_data['message_count']}\n\n"
        markdown += "---\n\n"
        
        for i, message in enumerate(conversation_data['messages']):
            role_emoji = "👤" if message['role'] == 'user' else "🤖"
            role_text = message['role'].title() if message['role'] != 'unknown' else 'Message'
            markdown += f"## {role_emoji} {role_text}\n\n"
            markdown += f"{message['content']}\n\n"
            markdown += "---\n\n"
        
        return markdown

async def main():
    """Main function"""
    scraper = ManualGPTScraper()
    await scraper.run()

if __name__ == "__main__":
    asyncio.run(main())
