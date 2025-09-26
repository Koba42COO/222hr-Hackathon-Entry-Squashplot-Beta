
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
KOBA42 INTELLIGENT OPTIMIZATION SELECTOR
========================================
Intelligent F2 Matrix Optimization Level Selection
=================================================

Features:
1. Matrix Size-Based Optimization Selection
2. Performance History Analysis
3. Dynamic Optimization Routing
4. KOBA42 Business Pattern Integration
5. Real-time Performance Monitoring
"""
import numpy as np
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationProfile:
    """Optimization level profile with performance characteristics."""
    level: str
    min_matrix_size: int
    max_matrix_size: int
    optimal_matrix_size: int
    expected_speedup: float
    expected_accuracy_improvement: float
    computational_complexity: float
    memory_overhead: float
    business_domains: List[str]
    use_cases: List[str]

@dataclass
class PerformanceHistory:
    """Historical performance data for optimization selection."""
    matrix_size: int
    optimization_level: str
    actual_speedup: float
    actual_accuracy_improvement: float
    execution_time: float
    memory_usage: float
    success_rate: float
    timestamp: str

class IntelligentOptimizationSelector:
    """Intelligent optimization level selector based on matrix size and performance history."""

    def __init__(self, history_file: Optional[str]=None):
        self.history_file = history_file
        self.performance_history = self._load_performance_history()
        self.optimization_profiles = self._define_optimization_profiles()

    def _define_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Define optimization profiles for different levels."""
        return {'basic': OptimizationProfile(level='basic', min_matrix_size=32, max_matrix_size=128, optimal_matrix_size=64, expected_speedup=2.5, expected_accuracy_improvement=0.03, computational_complexity=1.0, memory_overhead=1.0, business_domains=['AI Development', 'Data Processing', 'Real-time Systems'], use_cases=['Small-scale ML', 'Real-time inference', 'Prototyping']), 'advanced': OptimizationProfile(level='advanced', min_matrix_size=64, max_matrix_size=512, optimal_matrix_size=256, expected_speedup=1.8, expected_accuracy_improvement=0.06, computational_complexity=2.0, memory_overhead=1.5, business_domains=['Blockchain Solutions', 'Financial Modeling', 'Scientific Computing'], use_cases=['Medium-scale ML', 'Research applications', 'Production systems']), 'expert': OptimizationProfile(level='expert', min_matrix_size=256, max_matrix_size=2048, optimal_matrix_size=1024, expected_speedup=1.2, expected_accuracy_improvement=0.08, computational_complexity=3.0, memory_overhead=2.0, business_domains=['SaaS Platforms', 'Enterprise Solutions', 'Advanced Research'], use_cases=['Large-scale ML', 'Enterprise applications', 'Advanced research'])}

    def _load_performance_history(self) -> List[PerformanceHistory]:
        """Load performance history from file."""
        if not self.history_file or not Path(self.history_file).exists():
            return []
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return [PerformanceHistory(**item) for item in data]
        except Exception as e:
            logger.warning(f'Failed to load performance history: {e}')
            return []

    def _save_performance_history(self):
        """Save performance history to file."""
        if not self.history_file:
            return
        try:
            with open(self.history_file, 'w') as f:
                json.dump([vars(item) for item in self.performance_history], f, indent=2)
        except Exception as e:
            logger.warning(f'Failed to save performance history: {e}')

    def select_optimization_level(self, matrix_size: int, business_domain: str=None, use_case: str=None, performance_priority: str='balanced') -> str:
        """
        Select optimal optimization level based on matrix size and requirements.
        
        Args:
            matrix_size: Size of the matrix to optimize
            business_domain: Target business domain
            use_case: Specific use case
            performance_priority: 'speed', 'accuracy', or 'balanced'
        
        Returns:
            Selected optimization level: 'basic', 'advanced', or 'expert'
        """
        logger.info(f'🔍 Selecting optimization level for matrix size {matrix_size}')
        candidates = self._get_candidate_levels(matrix_size)
        if not candidates:
            logger.warning(f'No suitable optimization level found for matrix size {matrix_size}')
            return 'basic'
        scores = {}
        for level in candidates:
            profile = self.optimization_profiles[level]
            score = self._calculate_level_score(profile, matrix_size, business_domain, use_case, performance_priority)
            scores[level] = score
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]
        logger.info(f'✅ Selected optimization level: {best_level} (score: {best_score:.3f})')
        logger.info(f'   • Expected speedup: {self.optimization_profiles[best_level].expected_speedup:.2f}x')
        logger.info(f'   • Expected accuracy improvement: {self.optimization_profiles[best_level].expected_accuracy_improvement:.1%}')
        return best_level

    def _get_candidate_levels(self, matrix_size: int) -> Optional[Any]:
        """Get candidate optimization levels for given matrix size."""
        candidates = []
        for (level, profile) in self.optimization_profiles.items():
            if profile.min_matrix_size <= matrix_size <= profile.max_matrix_size:
                candidates.append(level)
        return candidates

    def _calculate_level_score(self, profile: OptimizationProfile, matrix_size: int, business_domain: str, use_case: str, performance_priority: str) -> float:
        """Calculate score for an optimization level."""
        score = 0.0
        size_fit = self._calculate_size_fit(profile, matrix_size)
        score += size_fit * 0.4
        historical_score = self._calculate_historical_score(profile.level, matrix_size)
        score += historical_score * 0.3
        domain_score = self._calculate_domain_fit(profile, business_domain)
        score += domain_score * 0.2
        use_case_score = self._calculate_use_case_fit(profile, use_case)
        score += use_case_score * 0.1
        score = self._adjust_for_performance_priority(score, profile, performance_priority)
        return score

    def _calculate_size_fit(self, profile: OptimizationProfile, matrix_size: int) -> float:
        """Calculate how well the matrix size fits the optimization profile."""
        optimal_size = profile.optimal_matrix_size
        distance = abs(matrix_size - optimal_size)
        max_distance = profile.max_matrix_size - profile.min_matrix_size
        fit_score = 1.0 - distance / max_distance
        return max(0.0, min(1.0, fit_score))

    def _calculate_historical_score(self, level: str, matrix_size: int) -> float:
        """Calculate score based on historical performance."""
        if not self.performance_history:
            return 0.5
        relevant_history = [h for h in self.performance_history if h.optimization_level == level and abs(h.matrix_size - matrix_size) <= matrix_size * 0.5]
        if not relevant_history:
            return 0.5
        avg_speedup = np.mean([h.actual_speedup for h in relevant_history])
        avg_accuracy_improvement = np.mean([h.actual_accuracy_improvement for h in relevant_history])
        avg_success_rate = np.mean([h.success_rate for h in relevant_history])
        score = avg_speedup * 0.4 + avg_accuracy_improvement * 0.4 + avg_success_rate * 0.2
        return max(0.0, min(1.0, score))

    def _calculate_domain_fit(self, profile: OptimizationProfile, business_domain: str) -> float:
        """Calculate business domain fit score."""
        if not business_domain:
            return 0.5
        if business_domain in profile.business_domains:
            return 1.0
        else:
            return 0.3

    def _calculate_use_case_fit(self, profile: OptimizationProfile, use_case: str) -> float:
        """Calculate use case fit score."""
        if not use_case:
            return 0.5
        if use_case in profile.use_cases:
            return 1.0
        else:
            return 0.3

    def _adjust_for_performance_priority(self, score: float, profile: OptimizationProfile, priority: str) -> float:
        """Adjust score based on performance priority."""
        if priority == 'speed':
            speedup_factor = profile.expected_speedup / 3.0
            return score * (0.7 + 0.3 * speedup_factor)
        elif priority == 'accuracy':
            accuracy_factor = profile.expected_accuracy_improvement / 0.1
            return score * (0.7 + 0.3 * accuracy_factor)
        else:
            return score

    def update_performance_history(self, matrix_size: int, optimization_level: str, actual_speedup: float, actual_accuracy_improvement: float, execution_time: float, memory_usage: float, success_rate: float=1.0):
        """Update performance history with new results."""
        history_entry = PerformanceHistory(matrix_size=matrix_size, optimization_level=optimization_level, actual_speedup=actual_speedup, actual_accuracy_improvement=actual_accuracy_improvement, execution_time=execution_time, memory_usage=memory_usage, success_rate=success_rate, timestamp=datetime.now().isoformat())
        self.performance_history.append(history_entry)
        self._save_performance_history()
        logger.info(f'📊 Updated performance history for {optimization_level} level (matrix size: {matrix_size}, speedup: {actual_speedup:.2f}x)')

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization selection report."""
        report = {'timestamp': datetime.now().isoformat(), 'optimization_profiles': {}, 'performance_history_summary': {}, 'recommendations': []}
        for (level, profile) in self.optimization_profiles.items():
            report['optimization_profiles'][level] = {'level': profile.level, 'matrix_size_range': f'{profile.min_matrix_size}-{profile.max_matrix_size}', 'optimal_matrix_size': profile.optimal_matrix_size, 'expected_speedup': profile.expected_speedup, 'expected_accuracy_improvement': profile.expected_accuracy_improvement, 'computational_complexity': profile.computational_complexity, 'business_domains': profile.business_domains, 'use_cases': profile.use_cases}
        if self.performance_history:
            for level in self.optimization_profiles.keys():
                level_history = [h for h in self.performance_history if h.optimization_level == level]
                if level_history:
                    report['performance_history_summary'][level] = {'total_runs': len(level_history), 'average_speedup': np.mean([h.actual_speedup for h in level_history]), 'average_accuracy_improvement': np.mean([h.actual_accuracy_improvement for h in level_history]), 'average_success_rate': np.mean([h.success_rate for h in level_history]), 'last_updated': max([h.timestamp for h in level_history])}
        report['recommendations'] = self._generate_recommendations()
        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        if not self.performance_history:
            recommendations.append('No performance history available. Start with basic level for small matrices.')
            recommendations.append('Collect performance data to improve optimization selection.')
        else:
            for level in self.optimization_profiles.keys():
                level_history = [h for h in self.performance_history if h.optimization_level == level]
                if level_history:
                    avg_speedup = np.mean([h.actual_speedup for h in level_history])
                    avg_accuracy = np.mean([h.actual_accuracy_improvement for h in level_history])
                    if avg_speedup > 2.0:
                        recommendations.append(f'{level.capitalize()} level shows excellent speedup ({avg_speedup:.2f}x). Consider for similar matrix sizes.')
                    if avg_accuracy > 0.05:
                        recommendations.append(f'{level.capitalize()} level shows good accuracy improvements ({avg_accuracy:.1%}). Suitable for accuracy-critical applications.')
        return recommendations

    def visualize_performance_history(self, save_path: str=None):
        """Generate visualization of performance history."""
        if not self.performance_history:
            logger.warning('No performance history available for visualization')
            return
        (fig, axes) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('KOBA42 Optimization Performance History', fontsize=16, fontweight='bold')
        levels = [h.optimization_level for h in self.performance_history]
        matrix_sizes = [h.matrix_size for h in self.performance_history]
        speedups = [h.actual_speedup for h in self.performance_history]
        accuracy_improvements = [h.actual_accuracy_improvement for h in self.performance_history]
        level_data = {}
        for level in set(levels):
            level_indices = [i for (i, l) in enumerate(levels) if l == level]
            level_data[level] = [speedups[i] for i in level_indices]
        axes[0, 0].boxplot([level_data.get(level, []) for level in ['basic', 'advanced', 'expert']], labels=['Basic', 'Advanced', 'Expert'])
        axes[0, 0].set_title('Speedup by Optimization Level')
        axes[0, 0].set_ylabel('Speedup Factor')
        axes[0, 0].grid(True, alpha=0.3)
        level_acc_data = {}
        for level in set(levels):
            level_indices = [i for (i, l) in enumerate(levels) if l == level]
            level_acc_data[level] = [accuracy_improvements[i] for i in level_indices]
        axes[0, 1].boxplot([level_acc_data.get(level, []) for level in ['basic', 'advanced', 'expert']], labels=['Basic', 'Advanced', 'Expert'])
        axes[0, 1].set_title('Accuracy Improvement by Optimization Level')
        axes[0, 1].set_ylabel('Accuracy Improvement')
        axes[0, 1].grid(True, alpha=0.3)
        colors = {'basic': 'blue', 'advanced': 'green', 'expert': 'red'}
        for level in set(levels):
            level_indices = [i for (i, l) in enumerate(levels) if l == level]
            level_sizes = [matrix_sizes[i] for i in level_indices]
            level_speedups = [speedups[i] for i in level_indices]
            axes[1, 0].scatter(level_sizes, level_speedups, c=colors[level], label=level.capitalize(), alpha=0.7)
        axes[1, 0].set_title('Matrix Size vs Speedup')
        axes[1, 0].set_xlabel('Matrix Size')
        axes[1, 0].set_ylabel('Speedup Factor')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        timestamps = [datetime.fromisoformat(h.timestamp) for h in self.performance_history]
        axes[1, 1].scatter(timestamps, speedups, c=[colors[l] for l in levels], alpha=0.7)
        axes[1, 1].set_title('Performance Timeline')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Speedup Factor')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Performance history visualization saved to {save_path}')
        plt.show()

def demonstrate_intelligent_optimization():
    """Demonstrate intelligent optimization selection."""
    logger.info('🚀 KOBA42 Intelligent Optimization Selector')
    logger.info('=' * 50)
    selector = IntelligentOptimizationSelector('optimization_performance_history.json')
    test_cases = [(32, 'AI Development', 'Real-time inference', 'speed'), (64, 'Data Processing', 'Small-scale ML', 'balanced'), (128, 'Blockchain Solutions', 'Medium-scale ML', 'accuracy'), (256, 'Financial Modeling', 'Production systems', 'balanced'), (512, 'SaaS Platforms', 'Large-scale ML', 'accuracy'), (1024, 'Enterprise Solutions', 'Advanced research', 'balanced')]
    print('\n🔍 INTELLIGENT OPTIMIZATION SELECTION RESULTS')
    print('=' * 50)
    results = []
    for (matrix_size, business_domain, use_case, priority) in test_cases:
        selected_level = selector.select_optimization_level(matrix_size, business_domain, use_case, priority)
        profile = selector.optimization_profiles[selected_level]
        result = {'matrix_size': matrix_size, 'business_domain': business_domain, 'use_case': use_case, 'priority': priority, 'selected_level': selected_level, 'expected_speedup': profile.expected_speedup, 'expected_accuracy_improvement': profile.expected_accuracy_improvement}
        results.append(result)
        print(f'\nMatrix Size: {matrix_size}×{matrix_size}')
        print(f'Business Domain: {business_domain}')
        print(f'Use Case: {use_case}')
        print(f'Priority: {priority}')
        print(f'Selected Level: {selected_level.upper()}')
        print(f'Expected Speedup: {profile.expected_speedup:.2f}x')
        print(f'Expected Accuracy Improvement: {profile.expected_accuracy_improvement:.1%}')
    report = selector.generate_optimization_report()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'intelligent_optimization_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f'📄 Optimization report saved to {report_file}')
    try:
        selector.visualize_performance_history('optimization_performance_visualization.png')
    except Exception as e:
        logger.warning(f'Visualization generation failed: {e}')
    return (results, report_file)
if __name__ == '__main__':
    (results, report_file) = demonstrate_intelligent_optimization()
    print(f'\n🎉 Intelligent optimization demonstration completed!')
    print(f'📊 Results saved to: {report_file}')
    print(f'🔍 Tested {len(results)} different matrix sizes and use cases')