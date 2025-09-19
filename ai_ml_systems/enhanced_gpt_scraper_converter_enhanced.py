
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
Enhanced GPT Scraper & Converter
Combines GPT_scraper concepts with markdown conversion capabilities

Based on: https://github.com/rodolflying/GPT_scraper
Enhanced with markdown conversion and modern approaches
"""
import json
import os
import time
import random
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print('⚠️ Selenium not available. Install with: pip install selenium')

class EnhancedGPTScraper:
    """Enhanced GPT Scraper with multiple methods and markdown conversion"""

    def __init__(self, output_dir: str='~/dev/gpt'):
        self.output_dir = Path(os.path.expanduser(output_dir))
        self.conversations_dir = self.output_dir / 'conversations'
        self.markdown_dir = self.output_dir / 'markdown'
        self.csv_dir = self.output_dir / 'csv'
        for dir_path in [self.conversations_dir, self.markdown_dir, self.csv_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36', 'Accept': 'application/json, text/plain, */*', 'Accept-Language': 'en-US,en;q=0.9', 'Accept-Encoding': 'gzip, deflate, br', 'Connection': 'keep-alive', 'Upgrade-Insecure-Requests': '1'}

    def scrape_with_api(self, headers_file: str=None) -> List[Dict]:
        """
        Scrape conversations using ChatGPT's hidden API endpoints
        
        Args:
            headers_file: Path to headers file with authentication
            
        Returns:
            List of conversation data
        """
        print('🔗 Attempting API-based scraping...')
        if headers_file and os.path.exists(headers_file):
            with open(headers_file, 'r') as f:
                custom_headers = json.load(f)
            self.headers.update(custom_headers)
        conversations = []
        try:
            conversations_url = 'https://chat.openai.com/backend-api/conversations'
            params = {'offset': 0, 'limit': 20}
            response = requests.get(conversations_url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                conversation_list = data.get('items', [])
                print(f'📄 Found {len(conversation_list)} conversations')
                for conv in conversation_list:
                    conv_id = conv.get('id')
                    title = conv.get('title', 'Untitled')
                    created_time = conv.get('create_time')
                    print(f'🔄 Scraping conversation: {title}')
                    conv_url = f'https://chat.openai.com/backend-api/conversation/{conv_id}'
                    conv_response = requests.get(conv_url, headers=self.headers)
                    if conv_response.status_code == 200:
                        conv_data = conv_response.json()
                        conversations.append({'id': conv_id, 'title': title, 'created_time': created_time, 'data': conv_data})
                    time.sleep(random.uniform(2, 5))
            else:
                print(f'❌ API request failed: {response.status_code}')
                print('💡 This might be due to Cloudflare protection or updated API endpoints')
        except Exception as e:
            print(f'❌ API scraping error: {e}')
        return conversations

    def scrape_with_selenium(self, headless: bool=True) -> List[Dict]:
        """
        Scrape conversations using Selenium browser automation
        
        Args:
            headless: Run browser in headless mode
            
        Returns:
            List of conversation data
        """
        if not SELENIUM_AVAILABLE:
            print('❌ Selenium not available. Install with: pip install selenium')
            return []
        print('🌐 Attempting Selenium-based scraping...')
        conversations = []
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            driver = webdriver.Chrome(options=chrome_options)
            driver.get('https://chat.openai.com')
            print('🔐 Please log in to ChatGPT manually...')
            print('⏳ Waiting for login...')
            input("Press Enter after you've logged in to ChatGPT...")
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='conversation-turn-2']")))
            conversation_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid^='conversation-turn-']")
            for element in conversation_elements:
                try:
                    conv_data = self.extract_conversation_from_element(element)
                    if conv_data:
                        conversations.append(conv_data)
                except Exception as e:
                    print(f'⚠️ Error extracting conversation: {e}')
            driver.quit()
        except Exception as e:
            print(f'❌ Selenium scraping error: {e}')
        return conversations

    def extract_conversation_from_element(self, element) -> Optional[Dict]:
        """Extract conversation data from Selenium element"""
        try:
            text = element.text
            return {'content': text, 'timestamp': datetime.now().isoformat(), 'source': 'selenium'}
        except:
            return None

    def convert_to_markdown(self, conversations: List[Dict]) -> None:
        """
        Convert scraped conversations to markdown format
        
        Args:
            conversations: List of conversation data
        """
        print('📝 Converting conversations to markdown...')
        for (i, conv) in enumerate(conversations):
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                conv_id = conv.get('id', f'conv_{i}')
                title = conv.get('title', 'Untitled').replace(' ', '_')[:50]
                filename = f'{timestamp}_{title}_{conv_id}'
                json_file = self.conversations_dir / f'{filename}.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(conv, f, indent=2, ensure_ascii=False)
                markdown_content = self.convert_conversation_to_markdown(conv)
                markdown_file = self.markdown_dir / f'{filename}.md'
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f'✅ Converted: {title} → {markdown_file.name}')
            except Exception as e:
                print(f'❌ Error converting conversation {i}: {e}')

    def convert_conversation_to_markdown(self, conversation: Dict) -> str:
        """
        Convert a single conversation to markdown format
        
        Args:
            conversation: Conversation data dictionary
            
        Returns:
            Markdown formatted string
        """
        title = conversation.get('title', 'Untitled Conversation')
        created_time = conversation.get('created_time')
        data = conversation.get('data', {})
        markdown = f'# {title}\n\n'
        if created_time:
            markdown += f'**Created:** {created_time}\n\n'
        markdown += '---\n\n'
        messages = []
        if 'mapping' in data:
            for (msg_id, msg_data) in data['mapping'].items():
                if 'content' in msg_data and 'parts' in msg_data['content']:
                    for part in msg_data['content']['parts']:
                        if isinstance(part, str) and part.strip():
                            messages.append(part.strip())
        elif 'messages' in data:
            for msg in data['messages']:
                if 'content' in msg and isinstance(msg['content'], str):
                    messages.append(msg['content'].strip())
        elif 'content' in conversation:
            messages.append(conversation['content'])
        for (i, message) in enumerate(messages, 1):
            markdown += f'## Message {i}\n\n{message}\n\n---\n\n'
        return markdown

    def save_to_csv(self, conversations: List[Dict]) -> None:
        """
        Save conversations to CSV format
        
        Args:
            conversations: List of conversation data
        """
        print('📊 Saving conversations to CSV...')
        csv_data = []
        for conv in conversations:
            csv_data.append({'id': conv.get('id', ''), 'title': conv.get('title', ''), 'created_time': conv.get('created_time', ''), 'source': conv.get('source', 'api')})
        if csv_data:
            df = pd.DataFrame(csv_data)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            csv_file = self.csv_dir / f'conversations_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            print(f'✅ Saved CSV: {csv_file}')

    def create_sample_headers_file(self) -> None:
        """Create a sample headers file for API authentication"""
        sample_headers = {'authorization': 'Bearer YOUR_SESSION_TOKEN_HERE', 'cookie': 'YOUR_COOKIE_STRING_HERE', 'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36', 'accept': 'application/json, text/plain, */*', 'accept-language': 'en-US,en;q=0.9', 'accept-encoding': 'gzip, deflate, br', 'connection': 'keep-alive'}
        headers_file = self.output_dir / 'sample_headers.json'
        with open(headers_file, 'w') as f:
            json.dump(sample_headers, f, indent=2)
        print(f'📝 Created sample headers file: {headers_file}')
        print('💡 Instructions:')
        print('1. Open ChatGPT in your browser')
        print('2. Open Developer Tools (F12)')
        print('3. Go to Network tab')
        print('4. Refresh the page and look for API requests')
        print('5. Copy the headers from a successful request')
        print('6. Update the headers file with your actual values')

    def run_comprehensive_scrape(self, method: str='both') -> None:
        """
        Run comprehensive scraping with specified method
        
        Args:
            method: "api", "selenium", or "both"
        """
        print('🚀 Starting comprehensive GPT conversation scraping...')
        print(f'📁 Output directory: {self.output_dir}')
        all_conversations = []
        if method in ['api', 'both']:
            print('\n🔗 Method 1: API-based scraping')
            api_conversations = self.scrape_with_api()
            all_conversations.extend(api_conversations)
        if method in ['selenium', 'both']:
            print('\n🌐 Method 2: Selenium-based scraping')
            selenium_conversations = self.scrape_with_selenium()
            all_conversations.extend(selenium_conversations)
        if all_conversations:
            print(f'\n📊 Total conversations scraped: {len(all_conversations)}')
            self.convert_to_markdown(all_conversations)
            self.save_to_csv(all_conversations)
            print(f'\n🎉 Scraping completed successfully!')
            print(f'📁 Check the following directories:')
            print(f'   JSON: {self.conversations_dir}')
            print(f'   Markdown: {self.markdown_dir}')
            print(f'   CSV: {self.csv_dir}')
        else:
            print('\n❌ No conversations were scraped successfully')
            print('💡 This might be due to:')
            print("   - ChatGPT's anti-scraping measures")
            print('   - Outdated API endpoints')
            print('   - Missing authentication headers')
            print('   - Cloudflare protection')

def main():
    """Main function"""
    print('🔍 Enhanced GPT Scraper & Converter')
    print('=' * 50)
    print('Based on: https://github.com/rodolflying/GPT_scraper')
    print('Enhanced with modern approaches and markdown conversion')
    print('=' * 50)
    scraper = EnhancedGPTScraper()
    print('\n📋 Available options:')
    print('1. API-based scraping (requires headers)')
    print('2. Selenium-based scraping (requires manual login)')
    print('3. Both methods')
    print('4. Create sample headers file')
    print('5. Convert existing JSON files to markdown')
    choice = input('\n🎯 Choose an option (1-5): ').strip()
    if choice == '1':
        headers_file = input('📁 Headers file path (optional): ').strip()
        if headers_file and (not os.path.exists(headers_file)):
            print('❌ Headers file not found')
            return
        scraper.run_comprehensive_scrape('api')
    elif choice == '2':
        scraper.run_comprehensive_scrape('selenium')
    elif choice == '3':
        scraper.run_comprehensive_scrape('both')
    elif choice == '4':
        scraper.create_sample_headers_file()
    elif choice == '5':
        json_files = list(scraper.conversations_dir.glob('*.json'))
        if json_files:
            conversations = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    conv_data = json.load(f)
                    conversations.append(conv_data)
            scraper.convert_to_markdown(conversations)
        else:
            print('❌ No JSON files found in conversations directory')
    else:
        print('❌ Invalid choice')
if __name__ == '__main__':
    main()