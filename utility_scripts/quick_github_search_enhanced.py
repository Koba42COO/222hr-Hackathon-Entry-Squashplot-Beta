
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
Quick GitHub AI Repository Search
Fast discovery of interesting AI and programming repositories
"""
import requests
import json
import time
from datetime import datetime

def search_github_repos(query, max_results=20):
    """Search GitHub repositories"""
    print(f'🔍 Searching: {query}')
    url = 'https://api.github.com/search/repositories'
    headers = {'User-Agent': 'GitHub-AI-Search/1.0', 'Accept': 'application/vnd.github.v3+json'}
    params = {'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': min(max_results, 30)}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            repos = []
            for repo in data.get('items', []):
                repo_info = {'name': repo['full_name'], 'description': repo.get('description', ''), 'language': repo.get('language', ''), 'stars': repo['stargazers_count'], 'forks': repo['forks_count'], 'topics': repo.get('topics', []), 'url': repo['html_url'], 'created_at': repo['created_at'], 'pushed_at': repo['pushed_at']}
                repos.append(repo_info)
            print(f'✅ Found {len(repos)} repositories')
            return repos
        else:
            print(f'❌ API request failed: {response.status_code}')
            return []
    except Exception as e:
        print(f'❌ Error: {e}')
        return []

def main():
    """Main function"""
    print('🔍 Quick GitHub AI Repository Search')
    print('=' * 50)
    search_patterns = ['language:python topic:artificial-intelligence', 'language:python topic:machine-learning', 'language:python topic:quantum-computing', 'language:python topic:consciousness', 'language:rust topic:ai', 'language:rust topic:quantum', 'language:go topic:ai', 'topic:web3', 'topic:blockchain', 'stars:>1000 language:python created:>2024-01-01']
    all_repos = []
    seen_repos = set()
    print('🚀 Starting quick search...')
    for pattern in search_patterns:
        repos = search_github_repos(pattern, 15)
        for repo in repos:
            if repo['name'] not in seen_repos:
                seen_repos.add(repo['name'])
                all_repos.append(repo)
        time.sleep(2)
    all_repos.sort(key=lambda x: x['stars'], reverse=True)
    print(f'\n🎉 Search complete! Found {len(all_repos)} unique repositories')
    print('\n🏆 Top 20 Most Interesting Repositories:')
    print('=' * 80)
    for (i, repo) in enumerate(all_repos[:20], 1):
        print(f"{i:2d}. {repo['name']}")
        print(f"    ⭐ {repo['stars']:,} stars | 🍴 {repo['forks']:,} forks | {repo['language']}")
        print(f"    📝 {(repo['description'][:100] if repo['description'] else 'No description')}...")
        print(f"    🏷️  Topics: {', '.join(repo['topics'][:5])}")
        print(f"    🔗 {repo['url']}")
        print()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'github_search_results_{timestamp}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_repos, f, indent=2, ensure_ascii=False)
    print(f'💾 Results saved to: {filename}')
    print('\n📊 Summary:')
    print(f'   Total repositories: {len(all_repos)}')
    print(f"   Languages found: {set((repo['language'] for repo in all_repos if repo['language']))}")
    all_topics = []
    for repo in all_repos:
        all_topics.extend(repo['topics'])
    topic_counts = {}
    for topic in all_topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f'   Top topics: {[topic for (topic, count) in top_topics[:5]]}')
if __name__ == '__main__':
    main()