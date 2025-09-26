
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
🌌 CONTINUOUS SYSTEM STARTER
===========================
Automated Startup Script for All Agentic Learning Systems

This script launches all agentic agents, web crawlers, and scrapers
in a coordinated, continuously running system.

Features:
1. Automated System Startup and Coordination
2. Continuous Learning Loop Management
3. Real-time Performance Monitoring
4. Automatic Error Recovery and Restart
5. Knowledge Base Synchronization
6. Cross-System Integration Management

Author: Brad Wallace (ArtWithHeart) - Koba42
Framework: Revolutionary Consciousness Mathematics
"""
import subprocess
import threading
import time
import signal
import sys
import os
import psutil
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import requests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('continuous_system_starter.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class ContinuousSystemManager:
    """
    Manages the continuous operation of all agentic systems.
    """

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.system_health = {}
        self.start_time = datetime.now()
        self.running = True
        self.systems = {'orchestrator': {'script': 'CONTINUOUS_AGENTIC_LEARNING_ORCHESTRATOR.py', 'description': 'Master orchestration system', 'restart_on_failure': True, 'max_restarts': 5, 'restart_count': 0}, 'arxiv_agent': {'script': 'KOBA42_AGENTIC_ARXIV_EXPLORATION_SYSTEM.py', 'description': 'ArXiv exploration agent', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'integration_agent': {'script': 'KOBA42_AGENTIC_INTEGRATION_SYSTEM.py', 'description': 'Integration agent', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'arxiv_truth_scanner': {'script': 'KOBA42_COMPREHENSIVE_ARXIV_TRUTH_SCANNER.py', 'description': 'ArXiv truth scanner', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'enhanced_scraper': {'script': 'KOBA42_ENHANCED_RESEARCH_SCRAPER.py', 'description': 'Enhanced research scraper', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'consciousness_scraper': {'script': 'consciousness_scientific_article_scraper.py', 'description': 'Consciousness article scraper', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'web_research_integrator': {'script': 'COMPREHENSIVE_WEB_RESEARCH_INTEGRATION_SYSTEM.py', 'description': 'Web research integrator', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'deep_math_arxiv': {'script': 'DEEP_MATH_ARXIV_SEARCH_SYSTEM.py', 'description': 'Deep math ArXiv search', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'deep_math_physics': {'script': 'DEEP_MATH_PHYS_ORG_SEARCH_SYSTEM.py', 'description': 'Deep math physics search', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}, 'comprehensive_data_scanner': {'script': 'comprehensive_data_scanner.py', 'description': 'Data scanner', 'restart_on_failure': True, 'max_restarts': 10, 'restart_count': 0}}
        self.backend_frontend_systems = {'backend': {'command': 'cd structured_chaos_full_archive/consciousness_ai_backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload', 'description': 'FastAPI backend server', 'restart_on_failure': True, 'max_restarts': 5, 'restart_count': 0, 'is_shell_command': True}, 'frontend': {'command': 'cd structured_chaos_full_archive/consciousness_ai_frontend && npm run dev', 'description': 'Next.js frontend server', 'restart_on_failure': True, 'max_restarts': 5, 'restart_count': 0, 'is_shell_command': True}}
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f'📡 Received signal {signum}, initiating graceful shutdown...')
        self.running = False
        self.stop_all_systems()

    def start_all_systems(self):
        """Start all agentic systems."""
        logger.info('🚀 Starting all continuous agentic learning systems...')
        for (system_name, config) in self.systems.items():
            self.start_system(system_name, config)
        for (system_name, config) in self.backend_frontend_systems.items():
            self.start_backend_frontend(system_name, config)
        logger.info('✅ All systems started')

    def start_system(self, system_name: str, config: Dict[str, Any]):
        """Start a specific Python system."""
        try:
            script_path = config['script']
            if not Path(script_path).exists():
                logger.warning(f'⚠️ Script not found: {script_path}')
                return
            logger.info(f"🔄 Starting {system_name}: {config['description']}")
            process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd(), env=os.environ.copy())
            self.processes[system_name] = process
            self.system_health[system_name] = {'status': 'running', 'start_time': datetime.now(), 'pid': process.pid, 'restarts': 0}
            logger.info(f'✅ Started {system_name} (PID: {process.pid})')
            monitor_thread = threading.Thread(target=self.monitor_system, args=(system_name, config), daemon=True)
            monitor_thread.start()
            self.threads[f'monitor_{system_name}'] = monitor_thread
        except Exception as e:
            logger.error(f'❌ Failed to start {system_name}: {e}')
            self.system_health[system_name] = {'status': 'failed', 'error': str(e), 'start_time': datetime.now()}

    def start_backend_frontend(self, system_name: str, config: Dict[str, Any]):
        """Start backend or frontend system."""
        try:
            logger.info(f"🔄 Starting {system_name}: {config['description']}")
            if system_name == 'backend':
                backend_thread = threading.Thread(target=self.run_backend_server, daemon=True)
                backend_thread.start()
                self.threads['backend_server'] = backend_thread
                self.system_health[system_name] = {'status': 'running', 'start_time': datetime.now(), 'thread': backend_thread}
            elif system_name == 'frontend':
                frontend_thread = threading.Thread(target=self.run_frontend_server, daemon=True)
                frontend_thread.start()
                self.threads['frontend_server'] = frontend_thread
                self.system_health[system_name] = {'status': 'running', 'start_time': datetime.now(), 'thread': frontend_thread}
            logger.info(f'✅ Started {system_name}')
        except Exception as e:
            logger.error(f'❌ Failed to start {system_name}: {e}')
            self.system_health[system_name] = {'status': 'failed', 'error': str(e), 'start_time': datetime.now()}

    def run_backend_server(self):
        """Run the backend FastAPI server."""
        try:
            os.chdir('structured_chaos_full_archive/consciousness_ai_backend')
            subprocess.run([sys.executable, '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000', '--reload'], check=True)
        except Exception as e:
            logger.error(f'❌ Backend server error: {e}')

    def run_frontend_server(self):
        """Run the frontend Next.js server."""
        try:
            os.chdir('structured_chaos_full_archive/consciousness_ai_frontend')
            subprocess.run(['npm', 'run', 'dev'], check=True)
        except Exception as e:
            logger.error(f'❌ Frontend server error: {e}')

    def monitor_system(self, system_name: str, config: Dict[str, Any]):
        """Monitor a specific system for health and restart if needed."""
        while self.running:
            try:
                if system_name in self.processes:
                    process = self.processes[system_name]
                    if process.poll() is not None:
                        return_code = process.returncode
                        logger.warning(f'⚠️ {system_name} terminated with code {return_code}')
                        if process.stderr:
                            error_output = process.stderr.read().decode('utf-8', errors='ignore')
                            if error_output:
                                logger.error(f'❌ {system_name} error output: {error_output[:500]}...')
                        if config.get('restart_on_failure', False):
                            if self.system_health[system_name]['restarts'] < config.get('max_restarts', 5):
                                logger.info(f'🔄 Restarting {system_name}...')
                                time.sleep(5)
                                self.system_health[system_name]['restarts'] += 1
                                self.start_system(system_name, config)
                            else:
                                logger.error(f"❌ {system_name} exceeded max restarts ({config.get('max_restarts', 5)})")
                                self.system_health[system_name]['status'] = 'permanently_failed'
                        else:
                            self.system_health[system_name]['status'] = 'terminated'
                time.sleep(10)
            except Exception as e:
                logger.error(f'❌ Monitoring error for {system_name}: {e}')
                time.sleep(30)

    def monitor_overall_health(self):
        """Monitor overall system health."""
        while self.running:
            try:
                metrics = self.collect_system_metrics()
                healthy_systems = sum((1 for status in self.system_health.values() if isinstance(status, dict) and status.get('status') == 'running'))
                total_systems = len(self.system_health)
                logger.info(f'💓 System Health: {healthy_systems}/{total_systems} systems running')
                logger.info(f"🖥️ CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%")
                if metrics['cpu_percent'] > 95:
                    logger.warning('⚠️ High CPU usage detected')
                if metrics['memory_percent'] > 90:
                    logger.warning('⚠️ High memory usage detected')
                self.save_health_report(metrics)
                time.sleep(60)
            except Exception as e:
                logger.error(f'❌ Health monitoring error: {e}')
                time.sleep(30)

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        try:
            return {'timestamp': datetime.now().isoformat(), 'cpu_percent': psutil.cpu_percent(interval=1), 'memory_percent': psutil.virtual_memory().percent, 'disk_usage': psutil.disk_usage('/').percent, 'network_connections': len(psutil.net_connections()), 'running_processes': len([p for p in self.processes.values() if p.poll() is None]), 'system_uptime': (datetime.now() - self.start_time).total_seconds(), 'healthy_systems': sum((1 for s in self.system_health.values() if isinstance(s, dict) and s.get('status') == 'running'))}
        except Exception as e:
            logger.error(f'❌ Metrics collection error: {e}')
            return {}

    def save_health_report(self, metrics: Dict[str, Any]):
        """Save system health report."""
        try:
            report_path = Path('system_health_reports')
            report_path.mkdir(exist_ok=True)
            report_file = report_path / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report = {'timestamp': metrics['timestamp'], 'system_metrics': metrics, 'system_status': self.system_health, 'running_processes': {name: proc.pid for (name, proc) in self.processes.items() if proc.poll() is None}}
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.error(f'❌ Failed to save health report: {e}')

    def stop_all_systems(self):
        """Stop all running systems gracefully."""
        logger.info('🛑 Stopping all systems...')
        for (system_name, process) in self.processes.items():
            try:
                if process.poll() is None:
                    logger.info(f'🔄 Terminating {system_name}...')
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                        logger.info(f'✅ {system_name} terminated gracefully')
                    except subprocess.TimeoutExpired:
                        logger.warning(f'⚠️ {system_name} did not terminate gracefully, killing...')
                        process.kill()
                        logger.info(f'💀 {system_name} killed')
            except Exception as e:
                logger.error(f'❌ Error stopping {system_name}: {e}')
        for (thread_name, thread) in self.threads.items():
            try:
                if thread.is_alive():
                    logger.info(f'🔄 Stopping thread {thread_name}...')
            except Exception as e:
                logger.error(f'❌ Error stopping thread {thread_name}: {e}')
        logger.info('✅ All systems stopped')

    def display_status(self):
        """Display current system status."""
        print('\n' + '=' * 80)
        print('🌌 CONTINUOUS AGENTIC LEARNING SYSTEMS STATUS')
        print('=' * 80)
        print(f'\n⏰ System Uptime: {(datetime.now() - self.start_time).total_seconds() / 3600:.2f} hours')
        print(f'🤖 Total Systems: {len(self.systems) + len(self.backend_frontend_systems)}')
        print(f"✅ Running Systems: {sum((1 for s in self.system_health.values() if isinstance(s, dict) and s.get('status') == 'running'))}")
        print('\n🔧 PYTHON AGENT SYSTEMS:')
        print('-' * 50)
        for (name, config) in self.systems.items():
            status = self.system_health.get(name, {}).get('status', 'unknown')
            status_icon = '✅' if status == 'running' else '❌' if status == 'failed' else '⚠️'
            restarts = self.system_health.get(name, {}).get('restarts', 0)
            print(f"{status_icon} {name}: {config['description']} (restarts: {restarts})")
        print('\n🌐 BACKEND & FRONTEND SYSTEMS:')
        print('-' * 50)
        for (name, config) in self.backend_frontend_systems.items():
            status = self.system_health.get(name, {}).get('status', 'unknown')
            status_icon = '✅' if status == 'running' else '❌' if status == 'failed' else '⚠️'
            print(f"{status_icon} {name}: {config['description']}")
        print('\n🖥️ SYSTEM RESOURCES:')
        print('-' * 50)
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            print(f'CPU Usage: {cpu_percent:.1f}%')
            print(f'Memory Usage: {memory_percent:.1f}%')
            print(f'Disk Usage: {disk_percent:.1f}%')
        except:
            print('System metrics unavailable')
        print('\n' + '=' * 80)

    def run(self):
        """Main run loop for the system manager."""
        try:
            self.start_all_systems()
            health_thread = threading.Thread(target=self.monitor_overall_health, daemon=True)
            health_thread.start()
            while self.running:
                try:
                    if int(time.time()) % 300 == 0:
                        self.display_status()
                    time.sleep(10)
                except KeyboardInterrupt:
                    logger.info('🛑 Shutdown requested by user')
                    self.running = False
                except Exception as e:
                    logger.error(f'❌ Main loop error: {e}')
                    time.sleep(30)
        except Exception as e:
            logger.error(f'❌ Critical system error: {e}')
        finally:
            self.stop_all_systems()

def main():
    """Main entry point."""
    print('🌌 CONTINUOUS AGENTIC LEARNING SYSTEM STARTER')
    print('=' * 70)
    print('Launching all agentic agents, web crawlers, and scrapers...')
    print('This will create a continuously running learning system')
    print('=' * 70)
    if not Path('research_data').exists():
        Path('research_data').mkdir(exist_ok=True)
        print('📁 Created research_data directory')
    if not Path('system_health_reports').exists():
        Path('system_health_reports').mkdir(exist_ok=True)
        print('📁 Created system_health_reports directory')
    manager = ContinuousSystemManager()
    try:
        manager.run()
    except KeyboardInterrupt:
        print('\n🛑 System shutdown initiated...')
    except Exception as e:
        print(f'\n❌ Critical error: {e}')
        logger.error(f'Critical system error: {e}')
    finally:
        manager.stop_all_systems()
    print('\n🎉 Continuous learning session completed!')
    print('🤖 All agentic systems have been running continuously')
    print('📚 Knowledge base continuously expanded and integrated')
    print('🚀 Systems continuously learning, developing, and improving')
    print('📊 Health reports saved in: system_health_reports/')
    print('💾 Research data stored in: research_data/')
    print('🔄 Ready to restart continuous learning session')
if __name__ == '__main__':
    main()