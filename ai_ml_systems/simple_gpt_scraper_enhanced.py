
import time
from collections import defaultdict
from typing import Dict, Tuple

class RateLimiter:
    """Intelligent rate limiting system"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        client_requests = self.requests[client_id]

        # Remove old requests outside the time window
        window_start = now - 60  # 1 minute window
        client_requests[:] = [req for req in client_requests if req > window_start]

        # Check if under limit
        if len(client_requests) < self.requests_per_minute:
            client_requests.append(now)
            return True

        return False

    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        now = time.time()
        client_requests = self.requests[client_id]
        window_start = now - 60
        client_requests[:] = [req for req in client_requests if req > window_start]

        return max(0, self.requests_per_minute - len(client_requests))

    def get_reset_time(self, client_id: str) -> float:
        """Get time until rate limit resets"""
        client_requests = self.requests[client_id]
        if not client_requests:
            return 0

        oldest_request = min(client_requests)
        return max(0, 60 - (time.time() - oldest_request))


# Enhanced with rate limiting

import logging
import json
from datetime import datetime
from typing import Dict, Any

class SecurityLogger:
    """Security event logging system"""

    def __init__(self, log_file: str = 'security.log'):
        self.logger = logging.getLogger('security')
        self.logger.setLevel(logging.INFO)

        # Create secure log format
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )

        # File handler with secure permissions
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          severity: str = 'INFO'):
        """Log security-related events"""
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details,
            'severity': severity,
            'source': 'enhanced_system'
        }

        if severity == 'CRITICAL':
            self.logger.critical(json.dumps(event_data))
        elif severity == 'ERROR':
            self.logger.error(json.dumps(event_data))
        elif severity == 'WARNING':
            self.logger.warning(json.dumps(event_data))
        else:
            self.logger.info(json.dumps(event_data))

    def log_access_attempt(self, resource: str, user_id: str = None,
                          success: bool = True):
        """Log access attempts"""
        self.log_security_event(
            'ACCESS_ATTEMPT',
            {
                'resource': resource,
                'user_id': user_id or 'anonymous',
                'success': success,
                'ip_address': 'logged_ip'  # Would get real IP in production
            },
            'WARNING' if not success else 'INFO'
        )

    def log_suspicious_activity(self, activity: str, details: Dict[str, Any]):
        """Log suspicious activities"""
        self.log_security_event(
            'SUSPICIOUS_ACTIVITY',
            {'activity': activity, 'details': details},
            'WARNING'
        )


# Enhanced with security logging

import logging
from contextlib import contextmanager
from typing import Any, Optional

class SecureErrorHandler:
    """Security-focused error handling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @contextmanager
    def secure_context(self, operation: str):
        """Context manager for secure operations"""
        try:
            yield
        except Exception as e:
            # Log error without exposing sensitive information
            self.logger.error(f"Secure operation '{operation}' failed: {type(e).__name__}")
            # Don't re-raise to prevent information leakage
            raise RuntimeError(f"Operation '{operation}' failed securely")

    def validate_and_execute(self, func: callable, *args, **kwargs) -> Any:
        """Execute function with security validation"""
        with self.secure_context(func.__name__):
            # Validate inputs before execution
            validated_args = [self._validate_arg(arg) for arg in args]
            validated_kwargs = {k: self._validate_arg(v) for k, v in kwargs.items()}

            return func(*validated_args, **validated_kwargs)

    def _validate_arg(self, arg: Any) -> Any:
        """Validate individual argument"""
        # Implement argument validation logic
        if isinstance(arg, str) and len(arg) > 10000:
            raise ValueError("Input too large")
        if isinstance(arg, (dict, list)) and len(str(arg)) > 50000:
            raise ValueError("Complex input too large")
        return arg


# Enhanced with secure error handling

import re
from typing import Union, Any

class InputValidator:
    """Comprehensive input validation system"""

    @staticmethod
    def validate_string(input_str: str, max_length: int = 1000,
                       pattern: str = None) -> bool:
        """Validate string input"""
        if not isinstance(input_str, str):
            return False
        if len(input_str) > max_length:
            return False
        if pattern and not re.match(pattern, input_str):
            return False
        return True

    @staticmethod
    def sanitize_input(input_data: Any) -> Any:
        """Sanitize input to prevent injection attacks"""
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            return re.sub(r'[^\w\s\-_.]', '', input_data)
        elif isinstance(input_data, dict):
            return {k: InputValidator.sanitize_input(v)
                   for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [InputValidator.sanitize_input(item) for item in input_data]
        return input_data

    @staticmethod
    def validate_numeric(value: Any, min_val: float = None,
                        max_val: float = None) -> Union[float, None]:
        """Validate numeric input"""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return None
            if max_val is not None and num > max_val:
                return None
            return num
        except (ValueError, TypeError):
            return None


# Enhanced with input validation

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import os

class ConcurrencyManager:
    """Intelligent concurrency management"""

    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)

    def get_optimal_workers(self, task_type: str = 'cpu') -> int:
        """Determine optimal number of workers"""
        if task_type == 'cpu':
            return max(1, self.cpu_count - 1)
        elif task_type == 'io':
            return min(32, self.cpu_count * 2)
        else:
            return self.cpu_count

    def parallel_process(self, items: List[Any], process_func: callable,
                        task_type: str = 'cpu') -> List[Any]:
        """Process items in parallel"""
        num_workers = self.get_optimal_workers(task_type)

        if task_type == 'cpu' and len(items) > 100:
            # Use process pool for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_func, items))
        else:
            # Use thread pool for I/O or small tasks
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_func, items))

        return results


# Enhanced with intelligent concurrency

import asyncio
from typing import Coroutine, Any

class AsyncEnhancer:
    """Async enhancement wrapper"""

    @staticmethod
    async def run_async(func: Callable[..., Any], *args, **kwargs) -> Any:
        """Run function asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    @staticmethod
    def make_async(func: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Convert sync function to async"""
        async def wrapper(*args, **kwargs):
            return await AsyncEnhancer.run_async(func, *args, **kwargs)
        return wrapper


# Enhanced with async support
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
    print('⚠️ Playwright not available. Install with: pip install playwright')

class SimpleGPTScraper:
    """Simple and reliable ChatGPT conversation scraper"""

    def __init__(self, output_dir: str='~/dev/gpt_exports'):
        self.output_dir = Path(os.path.expanduser(output_dir))
        self.conversations_dir = self.output_dir / 'conversations'
        self.markdown_dir = self.output_dir / 'markdown'
        for dir_path in [self.conversations_dir, self.markdown_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    async def scrape_conversations(self):
        """Scrape ChatGPT conversations using Playwright"""
        if not PLAYWRIGHT_AVAILABLE:
            print('❌ Playwright not available. Please install: pip install playwright')
            return
        print('🚀 Starting Simple GPT Scraper...')
        print('📁 Output directory:', self.output_dir)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            try:
                print('🌐 Navigating to ChatGPT...')
                await page.goto('https://chat.openai.com/')
                print('🔐 Please log in to ChatGPT manually...')
                print('⏳ Waiting for login...')
                await page.wait_for_selector('a[href*="/c/"]', timeout=300000)
                print('✅ Login detected!')
                print('📋 Scanning for conversations...')
                conversation_links = await page.query_selector_all('a[href*="/c/"]')
                if not conversation_links:
                    print('❌ No conversations found. Make sure you have conversations in your history.')
                    return
                print(f'📄 Found {len(conversation_links)} conversations')
                conversations = []
                for (i, link) in enumerate(conversation_links[:10]):
                    try:
                        href = await link.get_attribute('href')
                        title = await link.text_content()
                        if href and title:
                            conversation_id = href.split('/c/')[1].split('?')[0]
                            conversations.append({'id': conversation_id, 'title': title.strip(), 'href': href})
                            print(f'📝 {i + 1}. {title.strip()}')
                    except Exception as e:
                        print(f'⚠️ Error processing conversation {i + 1}: {e}')
                if conversations:
                    await self.export_conversations(page, conversations)
                else:
                    print('❌ No conversations to export')
            except Exception as e:
                print(f'❌ Error during scraping: {e}')
            finally:
                await browser.close()

    async def export_conversations(self, page, conversations: List[Dict]):
        """Export conversations to JSON and Markdown"""
        print(f'\n📤 Exporting {len(conversations)} conversations...')
        exported_count = 0
        for (i, conv) in enumerate(conversations):
            try:
                print(f"📄 Exporting {i + 1}/{len(conversations)}: {conv['title']}")
                await page.goto(f"https://chat.openai.com{conv['href']}")
                await page.wait_for_load_state('networkidle')
                messages = await self.extract_messages(page)
                if messages:
                    json_data = {'id': conv['id'], 'title': conv['title'], 'url': f"https://chat.openai.com{conv['href']}", 'exported_at': datetime.now().isoformat(), 'messages': messages}
                    json_file = self.conversations_dir / f"{conv['id']}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
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
        print(f'\n🎉 Export complete! {exported_count}/{len(conversations)} conversations exported')
        print(f'📁 JSON files: {self.conversations_dir}')
        print(f'📁 Markdown files: {self.markdown_dir}')

    async def extract_messages(self, page) -> List[Dict]:
        """Extract messages from the current conversation page"""
        try:
            await page.wait_for_selector('[data-message-author-role]', timeout=10000)
            message_elements = await page.query_selector_all('[data-message-author-role]')
            messages = []
            for element in message_elements:
                try:
                    role = await element.get_attribute('data-message-author-role')
                    content_element = await element.query_selector('.markdown')
                    if content_element:
                        content = await content_element.text_content()
                        if content and content.strip():
                            messages.append({'role': role, 'content': content.strip(), 'timestamp': datetime.now().isoformat()})
                except Exception as e:
                    print(f'⚠️ Error extracting message: {e}')
            return messages
        except Exception as e:
            print(f'⚠️ Error extracting messages: {e}')
            return []

    def convert_to_markdown(self, conversation_data: Dict) -> str:
        """Convert conversation data to Markdown format"""
        markdown = f"# {conversation_data['title']}\n\n"
        markdown += f"**Conversation ID:** {conversation_data['id']}\n"
        markdown += f"**URL:** {conversation_data['url']}\n"
        markdown += f"**Exported:** {conversation_data['exported_at']}\n\n"
        markdown += '---\n\n'
        for (i, message) in enumerate(conversation_data['messages']):
            role_emoji = '👤' if message['role'] == 'user' else '🤖'
            markdown += f"## {role_emoji} {message['role'].title()}\n\n"
            markdown += f"{message['content']}\n\n"
            markdown += '---\n\n'
        return markdown

async def main():
    """Main function"""
    scraper = SimpleGPTScraper()
    await scraper.scrape_conversations()
if __name__ == '__main__':
    print('🤖 Simple GPT Scraper')
    print('=' * 50)
    print('This scraper will:')
    print('1. Open ChatGPT in a browser')
    print('2. Wait for you to log in manually')
    print('3. Scan for your conversations')
    print('4. Export them to JSON and Markdown')
    print('=' * 50)
    asyncio.run(main())