
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
"""
Live System Monitor for Revolutionary Continuous Learning System
Shows real-time activity from the enhanced academic learning system
"""
import time
import os
import json
from datetime import datetime
from pathlib import Path

def print_header():
    print('\n' + '=' * 80)
    print('🌌 REVOLUTIONARY CONTINUOUS LEARNING SYSTEM - LIVE MONITOR')
    print('=' * 80)
    print('🎓 Enhanced with Premium Academic Sources')
    print('📚 Learning from: arXiv, MIT OCW, Stanford, Harvard, Nature, Science')
    print('=' * 80)

def check_system_status():
    """Check if main system components are running"""
    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - System Status:")
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        components = {'orchestrator': False, 'knowledge_manager': False, 'scraper_system': False, 'backend': False, 'frontend': False}
        for component in components:
            if component in processes:
                components[component] = True
                print(f'✅ {component}: RUNNING')
            else:
                print(f'❌ {component}: NOT FOUND')
        return components
    except Exception as e:
        print(f'❌ Error checking processes: {e}')
        return {}

def check_learning_cycles():
    """Check for new learning cycle reports"""
    reports_dir = Path('learning_cycle_reports')
    if not reports_dir.exists():
        print('📊 No learning cycle reports directory found')
        return 0
    reports = list(reports_dir.glob('cycle_report_*.json'))
    print(f'📊 Total learning cycles completed: {len(reports)}')
    if reports:
        latest_report = max(reports, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_report, 'r') as f:
                data = json.load(f)
            print(f"🔄 Latest cycle: {data.get('cycle_id', 'Unknown')}")
            print(f"🎯 Objectives: {len(data.get('objectives', []))}")
            print(f"🤖 Systems involved: {len(data.get('systems_involved', []))}")
        except Exception as e:
            print(f'❌ Error reading latest report: {e}')
    return len(reports)

def check_scraping_activity():
    """Check for scraping activity indicators"""
    print('\n🌐 Academic Scraping Activity:')
    research_db = Path('research_data/scraped_content.db')
    if research_db.exists():
        size = research_db.stat().st_size
        print('.1f')
    else:
        print('📊 No scraped content database found yet')
    knowledge_db = Path('research_data/knowledge_fragments.db')
    if knowledge_db.exists():
        size = knowledge_db.stat().st_size
        print('.1f')
    else:
        print('🧠 No knowledge fragments database found yet')

def check_ml_training():
    """Check for ML training activity"""
    print('\n🧠 ML F2 Training Status:')
    training_logs = list(Path('.').glob('*training*.log'))
    if training_logs:
        latest_training = max(training_logs, key=lambda x: x.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest_training.stat().st_mtime)
        print(f'🚀 Latest training log: {latest_training.name}')
        print(f"⏰ Last modified: {mtime.strftime('%H:%M:%S')}")
    else:
        print('📝 No training logs found yet')

def check_academic_sources():
    """Show configured academic sources"""
    print('\n🎓 Configured Academic Sources:')
    sources = ['arXiv (Quantum, AI, Physics, Math)', 'MIT OpenCourseWare (EECS, Physics, Math)', 'Stanford University (AI, CS Research)', 'Harvard University (Physics, Sciences)', 'Nature Journal (Breakthrough Research)', 'Science Magazine (Scientific Reviews)', 'Phys.org (Physics News)', 'Coursera (ML Courses)', 'edX (AI Education)', 'Google AI Research', 'OpenAI Research', 'DeepMind']
    for (i, source) in enumerate(sources, 1):
        print('2d')

def main():
    """Main monitoring loop"""
    print_header()
    cycle_count = 0
    while True:
        try:
            components = check_system_status()
            current_cycles = check_learning_cycles()
            if current_cycles > cycle_count:
                print('\n🎉 NEW LEARNING CYCLE DETECTED!')
                print('🔍 System is actively learning from academic sources...')
                cycle_count = current_cycles
            check_scraping_activity()
            check_ml_training()
            if cycle_count == 0:
                check_academic_sources()
            print('\n' + '-' * 50)
            print('🔄 Monitoring... (Press Ctrl+C to stop)')
            time.sleep(30)
        except KeyboardInterrupt:
            print('\n👋 Monitoring stopped by user')
            break
        except Exception as e:
            print(f'\n❌ Error in monitoring loop: {e}')
            time.sleep(10)
if __name__ == '__main__':
    main()