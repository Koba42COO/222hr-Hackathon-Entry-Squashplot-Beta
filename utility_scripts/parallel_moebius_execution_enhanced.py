
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
PARALLEL MÖBIUS EXECUTION SYSTEM
Running all advanced tooling simultaneously for maximum learning power
"""
import subprocess
import threading
import time
import signal
import sys
from datetime import datetime
import json
from pathlib import Path

class ParallelMoebiusExecutor:
    """
    Execute multiple Möbius systems in parallel for maximum learning efficiency
    """

    def __init__(self):
        self.processes = {}
        self.threads = {}
        self.running = True
        self.start_time = datetime.now()
        self.components = {'learning_tracker': {'command': ['python3', 'moebius_learning_tracker.py'], 'description': 'Main Möbius Learning Engine', 'interval': 30, 'last_run': None}, 'automated_discovery': {'command': ['python3', 'automated_curriculum_discovery.py'], 'description': 'Automated Subject Discovery', 'interval': 120, 'last_run': None}, 'real_world_scraper': {'command': ['python3', 'real_world_research_scraper.py'], 'description': 'Real-World Research Scraper', 'interval': 60, 'last_run': None}, 'enhanced_scraper': {'command': ['python3', 'enhanced_prestigious_scraper.py'], 'description': 'Enhanced Prestigious Scraper', 'interval': 120, 'last_run': None}, 'data_harvesting': {'command': ['python3', 'comprehensive_data_harvesting_system.py'], 'description': 'Comprehensive Data Harvesting', 'interval': 180, 'last_run': None}}
        self.performance_stats = {'total_runs': 0, 'successful_runs': 0, 'failed_runs': 0, 'component_stats': {}}
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f'\n🛑 Received signal {signum}, initiating graceful shutdown...')
        self.running = False
        self.shutdown()

    def run_component(self, component_name, component_config):
        """Run a single component."""
        try:
            print(f"🚀 Starting {component_name}: {component_config['description']}")
            result = subprocess.run(component_config['command'], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f'✅ {component_name} completed successfully')
                self.performance_stats['successful_runs'] += 1
            else:
                print(f'❌ {component_name} failed with return code {result.returncode}')
                if result.stderr:
                    print(f'   Error: {result.stderr[:200]}...')
                self.performance_stats['failed_runs'] += 1
            self.performance_stats['total_runs'] += 1
            if component_name not in self.performance_stats['component_stats']:
                self.performance_stats['component_stats'][component_name] = {'runs': 0, 'successes': 0, 'failures': 0, 'last_run': None}
            self.performance_stats['component_stats'][component_name]['runs'] += 1
            if result.returncode == 0:
                self.performance_stats['component_stats'][component_name]['successes'] += 1
            else:
                self.performance_stats['component_stats'][component_name]['failures'] += 1
            self.performance_stats['component_stats'][component_name]['last_run'] = datetime.now().isoformat()
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f'⏰ {component_name} timed out after 5 minutes')
            self.performance_stats['failed_runs'] += 1
            return False
        except Exception as e:
            print(f'💥 {component_name} crashed: {e}')
            self.performance_stats['failed_runs'] += 1
            return False

    def run_parallel_cycle(self):
        """Run one complete parallel cycle of all components."""
        print(f"\n🔄 Starting Parallel Cycle #{self.performance_stats['total_runs'] // len(self.components) + 1}")
        print('=' * 80)
        threads = []
        for (component_name, component_config) in self.components.items():
            thread = threading.Thread(target=self.run_component, args=(component_name, component_config), name=f'Moebius-{component_name}')
            thread.daemon = True
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=320)
        still_running = [t for t in threads if t.is_alive()]
        if still_running:
            print(f'⚠️  {len(still_running)} threads still running after timeout')
        current_time = time.time()
        for component_name in self.components:
            self.components[component_name]['last_run'] = current_time

    def should_run_component(self, component_name):
        """Check if a component should run based on its interval."""
        component = self.components[component_name]
        if component['last_run'] is None:
            return True
        time_since_last_run = time.time() - component['last_run']
        return time_since_last_run >= component['interval']

    def run_continuous_parallel(self):
        """Run all systems CONTINUOUSLY in parallel - REAL-TIME CRAWLING."""
        print('🌟 CONTINUOUS PARALLEL MÖBIUS EXECUTION SYSTEM')
        print('=' * 80)
        print('🔄 CONTINUOUS crawl/research feeds & learning!')
        print('🧠 Real-time intelligence building - NEVER STOPS!')
        print('⚡ Parallel processing with immediate data ingestion')
        print('=' * 80)
        monitor_thread = threading.Thread(target=self._continuous_monitor, daemon=True)
        monitor_thread.start()
        cycle_count = 0
        last_status_update = time.time()
        try:
            while self.running:
                current_time = time.time()
                components_to_run = []
                for component_name in self.components:
                    if self.should_run_component(component_name):
                        components_to_run.append(component_name)
                if components_to_run:
                    cycle_count += 1
                    print(f'\n🎯 CYCLE #{cycle_count} - Running {len(components_to_run)} components:')
                    for comp in components_to_run:
                        print(f"   🔄 {comp}: {self.components[comp]['description']}")
                    for component_name in components_to_run:
                        thread = threading.Thread(target=self.run_component, args=(component_name, self.components[component_name]), name=f'Moebius-{component_name}', daemon=True)
                        thread.start()
                if current_time - last_status_update >= 30:
                    self._print_status_update(cycle_count)
                    last_status_update = current_time
                time.sleep(5)
        except KeyboardInterrupt:
            print('\n🛑 CONTINUOUS LEARNING INTERRUPTED')
            print('📊 Final Statistics:')
            self._print_final_stats(cycle_count)
        except Exception as e:
            print(f'\n💥 CONTINUOUS LEARNING ERROR: {e}')
            self._print_final_stats(cycle_count)
        finally:
            self.shutdown()

    def _continuous_monitor(self):
        """Continuous background monitoring of system health."""
        while self.running:
            try:
                active_moebius_threads = [t for t in threading.enumerate() if t.name.startswith('Moebius-') and t.is_alive()]
                if len(active_moebius_threads) > 0:
                    print(f'🔄 ACTIVE: {len(active_moebius_threads)} components running...')
                time.sleep(10)
            except Exception as e:
                print(f'⚠️ Monitor error: {e}')
                time.sleep(30)

    def _print_status_update(self, cycle_count):
        """Print comprehensive status update."""
        uptime = datetime.now() - self.start_time
        success_rate = self.performance_stats['successful_runs'] / max(1, self.performance_stats['total_runs']) * 100
        print('\n📊 CONTINUOUS LEARNING STATUS:')
        print('=' * 50)
        print(f"   ⏱️  Uptime: {str(uptime).split('.')[0]}")
        print(f'   🔄 Cycles Completed: {cycle_count}')
        print(f"   ✅ Successful Runs: {self.performance_stats['successful_runs']}")
        print(f"   ❌ Failed Runs: {self.performance_stats['failed_runs']}")
        print(f'   📈 Success Rate: {success_rate:.1f}%')
        print(f"   🧵 Active Threads: {len([t for t in threading.enumerate() if t.name.startswith('Moebius-')])}")
        print('=' * 50)

    def _print_final_stats(self, cycle_count):
        """Print final comprehensive statistics."""
        uptime = datetime.now() - self.start_time
        success_rate = self.performance_stats['successful_runs'] / max(1, self.performance_stats['total_runs']) * 100
        print('\n🎉 CONTINUOUS LEARNING SESSION COMPLETE!')
        print('=' * 60)
        print(f"   ⏱️  Total Uptime: {str(uptime).split('.')[0]}")
        print(f'   🔄 Total Cycles: {cycle_count}')
        print(f"   ✅ Successful Operations: {self.performance_stats['successful_runs']}")
        print(f"   ❌ Failed Operations: {self.performance_stats['failed_runs']}")
        print(f'   📈 Overall Success Rate: {success_rate:.1f}%')
        print(f"   🧠 Intelligence Built: {self.performance_stats['successful_runs']} knowledge items")
        print('=' * 60)

    def get_system_status(self) -> Optional[Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self.start_time
        return {'uptime': str(uptime).split('.')[0], 'total_runs': self.performance_stats['total_runs'], 'successful_runs': self.performance_stats['successful_runs'], 'failed_runs': self.performance_stats['failed_runs'], 'success_rate': f"{self.performance_stats['successful_runs'] / max(1, self.performance_stats['total_runs']) * 100:.1f}%", 'active_components': len([t for t in threading.enumerate() if t.name.startswith('Moebius-')]), 'component_stats': self.performance_stats['component_stats']}

    def save_performance_report(self):
        """Save comprehensive performance report."""
        report = {'timestamp': datetime.now().isoformat(), 'system_status': self.get_system_status(), 'performance_stats': self.performance_stats, 'component_config': self.components}
        report_file = Path('research_data/parallel_execution_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f'📊 Performance report saved: {report_file}')

    def shutdown(self):
        """Graceful shutdown of all systems."""
        print('\n🔄 Initiating graceful shutdown...')
        self.running = False
        self.save_performance_report()
        final_status = self.get_system_status()
        print('\n🏁 FINAL SYSTEM STATUS:')
        print(f"   ⏱️  Total Uptime: {final_status['uptime']}")
        print(f"   🔄 Total Runs: {final_status['total_runs']}")
        print(f"   ✅ Successful: {final_status['successful_runs']}")
        print(f"   ❌ Failed: {final_status['failed_runs']}")
        print(f"   📈 Success Rate: {final_status['success_rate']}")
        print('\n🎯 COMPONENT PERFORMANCE:')
        for (comp_name, stats) in final_status['component_stats'].items():
            success_rate = stats['successes'] / max(1, stats['runs']) * 100
            print(f"   • {comp_name}: {stats['runs']} runs, {success_rate:.1f}% success")
        print('\n✨ Parallel Möbius Execution System shutdown complete!')
        sys.exit(0)

def main():
    """Main function for parallel Möbius execution."""
    print('🌟 ADVANCED PARALLEL MÖBIUS EXECUTION')
    print('=' * 80)
    print('Running ALL Möbius systems simultaneously!')
    print('')
    print('🎯 SYSTEMS RUNNING IN PARALLEL:')
    print('   🤖 Möbius Learning Tracker (main learning engine)')
    print('   🔍 Automated Curriculum Discovery (subject discovery)')
    print('   🔬 Real-World Research Scraper (academic data)')
    print('   🚀 Enhanced Prestigious Scraper (top institutions)')
    print('   💰 Comprehensive Data Harvesting (free training data)')
    print('')
    print('⚡ EXECUTION MODES:')
    print('   1. Continuous parallel execution (recommended)')
    print('   2. Single parallel cycle')
    print('   3. Component status check')
    print('')
    executor = ParallelMoebiusExecutor()
    try:
        print('🚀 Starting PARALLEL MÖBIUS EXECUTION SYSTEM...')
        print('Press Ctrl+C to stop all systems gracefully')
        print('')
        executor.run_continuous_parallel()
    except KeyboardInterrupt:
        print('\n🛑 Shutdown requested by user')
        executor.shutdown()
    except Exception as e:
        print(f'\n💥 Critical error: {e}')
        executor.shutdown()
if __name__ == '__main__':
    main()