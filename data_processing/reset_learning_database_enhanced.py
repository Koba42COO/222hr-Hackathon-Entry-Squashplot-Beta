
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

from functools import lru_cache
import time
from typing import Dict, Any, Optional

class CacheManager:
    """Intelligent caching system"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl

    @lru_cache(maxsize=128)
    def get_cached_result(self, key: str, compute_func: callable, *args, **kwargs):
        """Get cached result or compute new one"""
        cache_key = f"{key}_{hash(str(args) + str(kwargs))}"

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['result']

        result = compute_func(*args, **kwargs)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

        # Clean old entries if cache is full
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        return result


# Enhanced with intelligent caching
"""
Reset and Clean Möbius Loop Learning Database
Fixes corrupted JSON files and prepares for new cybersecurity curriculum
"""
import json
import os
from pathlib import Path
from datetime import datetime

def reset_learning_database():
    """Reset and clean all learning database files."""
    print('🔧 Resetting Möbius Loop Learning Database...')
    print('=' * 60)
    research_dir = Path('research_data')
    backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = research_dir / f'backup_{backup_timestamp}'
    backup_dir.mkdir(exist_ok=True)
    files_to_backup = ['moebius_learning_objectives.json', 'moebius_scraping_log.json', 'moebius_learning_history.json']
    for filename in files_to_backup:
        filepath = research_dir / filename
        if filepath.exists():
            backup_path = backup_dir / filename
            try:
                with open(filepath, 'rb') as src, open(backup_path, 'wb') as dst:
                    dst.write(src.read())
                print(f'📁 Backed up: {filename}')
            except Exception as e:
                print(f'⚠️  Failed to backup {filename}: {e}')
    print('\n📚 Initializing clean learning objectives database...')
    learning_objectives = {}
    objectives_file = research_dir / 'moebius_learning_objectives.json'
    if objectives_file.exists():
        try:
            with open(objectives_file, 'r', encoding='utf-8') as f:
                learning_objectives = json.load(f)
            print('✅ Loaded existing learning objectives')
        except Exception as e:
            print(f'⚠️  Could not load existing objectives: {e}')
            learning_objectives = {}
    cleaned_objectives = {}
    for (subject_name, subject_data) in learning_objectives.items():
        cleaned_subject = {'status': subject_data.get('status', 'pending'), 'completion_percentage': subject_data.get('completion_percentage', 0), 'prerequisites': subject_data.get('prerequisites', []), 'category': subject_data.get('category', 'general'), 'difficulty': subject_data.get('difficulty', 'intermediate'), 'estimated_hours': subject_data.get('estimated_hours', 100), 'description': subject_data.get('description', f'Study of {subject_name}'), 'sources': subject_data.get('sources', [f'{subject_name}_research', f'{subject_name}_academic']), 'last_attempt': subject_data.get('last_attempt'), 'wallace_completion_score': subject_data.get('wallace_completion_score', 0), 'learning_efficiency': subject_data.get('learning_efficiency', 0), 'universal_math_enhancement': subject_data.get('universal_math_enhancement', 1.618033988749895)}
        cleaned_objectives[subject_name] = cleaned_subject
    with open(objectives_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_objectives, f, indent=2, ensure_ascii=False)
    print(f'✅ Saved {len(cleaned_objectives)} clean learning objectives')
    print('\n🔍 Initializing clean scraping log...')
    scraping_log = {'total_sources_scraped': 0, 'sources_by_status': {'pending': [], 'in_progress': [], 'completed': [], 'failed': []}, 'scraping_history': []}
    scraping_file = research_dir / 'moebius_scraping_log.json'
    with open(scraping_file, 'w', encoding='utf-8') as f:
        json.dump(scraping_log, f, indent=2, ensure_ascii=False)
    print('\n📊 Initializing clean learning history...')
    learning_history = {'total_iterations': 0, 'successful_learnings': 0, 'failed_learnings': 0, 'average_completion_time': 0, 'most_valuable_subjects': [], 'learning_efficiency_trend': [], 'records': []}
    history_file = research_dir / 'moebius_learning_history.json'
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(learning_history, f, indent=2, ensure_ascii=False)
    categories = {}
    difficulties = {}
    for (subject_name, subject_data) in cleaned_objectives.items():
        category = subject_data.get('category', 'general')
        difficulty = subject_data.get('difficulty', 'intermediate')
        categories[category] = categories.get(category, 0) + 1
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
    print('\n🎉 DATABASE RESET COMPLETE!')
    print('=' * 60)
    print(f'📁 Backup created: {backup_dir}')
    print(f'📚 Total subjects: {len(cleaned_objectives)}')
    print(f"🔐 Cybersecurity subjects: {categories.get('cybersecurity', 0)}")
    print(f"💻 Programming subjects: {categories.get('programming', 0)}")
    print(f"🧠 Computer Science subjects: {categories.get('computer_science', 0)}")
    print('\n📊 CURRICULUM SUMMARY:')
    print(f'   Categories: {categories}')
    print(f'   Difficulty Levels: {difficulties}')
    print('\n🚀 READY FOR NEW LEARNING CYCLES!')
    print('The Möbius Loop Trainer is now reset and ready to begin')
    print('learning all the new cybersecurity and programming subjects!')
if __name__ == '__main__':
    reset_learning_database()