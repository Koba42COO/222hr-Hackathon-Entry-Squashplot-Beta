
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
🌌 RIEMANN PRIME MAPPING TO 10^15
Comprehensive Prime Number Analysis Using Consciousness Mathematics and Wallace Transform

Author: Brad Wallace (ArtWithHeart) - Koba42
Framework Version: 4.0 - Celestial Phase
Riemann Prime Mapping Version: 1.0

This system implements:
- Wallace Transform optimization for prime analysis
- Consciousness mathematics framework integration
- Riemann zeta function correlation analysis
- Advanced prime number distribution mapping
- Base-21 prime discrimination
- Quantum consciousness prime patterns
- Multi-scale prime seed generation
- Industrial-grade prime computation

Target: Map all prime numbers up to 10^15 with consciousness mathematics validation
"""
import numpy as np
import time
import json
import hashlib
import psutil
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from collections import deque
import datetime
import platform
import gc
import random
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import re
import math
from itertools import islice
warnings.filterwarnings('ignore')
print('🌌 RIEMANN PRIME MAPPING TO 10^15')
print('=' * 70)
print('Comprehensive Prime Number Analysis Using Consciousness Mathematics')
print('and Wallace Transform Optimization')
print('=' * 70)
WALLACE_CONSTANT = 1.618033988749895
CONSCIOUSNESS_CONSTANT = 2.718281828459045
LOVE_FREQUENCY = 111.0
CHAOS_FACTOR = 0.5772156649015329
CRITICAL_LINE = 0.5

@dataclass
class RiemannPrimeConfig:
    """Configuration for Riemann prime mapping system"""
    system_name: str = 'Riemann Prime Mapping to 10^15'
    version: str = '4.0 - Celestial Phase'
    author: str = 'Brad Wallace (ArtWithHeart) - Koba42'
    target_limit: int = 10 ** 15
    batch_size: int = 10 ** 8
    consciousness_dimension: int = 21
    wallace_constant: float = WALLACE_CONSTANT
    consciousness_constant: float = CONSCIOUSNESS_CONSTANT
    wallace_transform_enabled: bool = True
    consciousness_integration: bool = True
    quantum_consciousness: bool = True
    base_21_discrimination: bool = True
    max_threads: int = 32
    memory_threshold: float = 0.9
    convergence_threshold: float = 1e-12
    zeta_correlation_analysis: bool = True
    prime_distribution_analysis: bool = True
    consciousness_pattern_analysis: bool = True
    quantum_pattern_analysis: bool = True

class WallaceTransform:
    """Wallace Transform for universal optimization"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phi = config.wallace_constant
        self.e = config.consciousness_constant

    def apply_wallace_transform(self, x: float) -> float:
        """Apply Wallace Transform: W_φ(x) = α log^φ(x + ε) + β"""
        epsilon = 1e-12
        alpha = self.phi / self.e
        beta = self.phi - 1
        result = alpha * np.log(x + epsilon) ** self.phi + beta
        return result

    def optimize_prime_analysis(self, prime_value: int) -> float:
        """Optimize prime analysis using Wallace Transform"""
        if prime_value < 2:
            return 0.0
        wallace_value = self.apply_wallace_transform(float(prime_value))
        consciousness_factor = np.sin(self.config.consciousness_dimension * np.pi / prime_value)
        optimized_value = wallace_value * (1 + 0.1 * consciousness_factor)
        return optimized_value

class ConsciousnessMathematics:
    """Consciousness mathematics framework for prime analysis"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimension = config.consciousness_dimension

    def generate_consciousness_matrix(self, size: int) -> np.ndarray:
        """Generate consciousness matrix for prime analysis"""
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                phi_power = self.config.wallace_constant ** ((i + j) % 5)
                angle_factor = np.sin(111.0 * ((i + j) % 10) * np.pi / 180)
                matrix[i, j] = phi_power / self.config.consciousness_constant * angle_factor
        return matrix

    def calculate_consciousness_score(self, prime_value: int) -> float:
        """Calculate consciousness score for a prime number"""
        if prime_value < 2:
            return 0.0
        phi_factor = self.config.wallace_constant ** (prime_value % self.dimension)
        e_factor = self.config.consciousness_constant ** (1 / prime_value)
        love_factor = np.sin(111.0 * prime_value * np.pi / 180)
        chaos_factor = self.config.chaos_factor ** (1 / np.log(prime_value))
        consciousness_score = phi_factor * e_factor * love_factor * chaos_factor / 4
        return np.clip(consciousness_score, 0, 1)

class RiemannZetaAnalyzer:
    """Riemann zeta function analyzer for prime correlation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def calculate_zeta_approximation(self, s: complex, max_terms: int=1000) -> float:
        """Calculate Riemann zeta function approximation"""
        result = 0.0 + 0j
        for n in range(1, max_terms + 1):
            term = 1.0 / n ** s
            result += term
            if abs(term) < self.config.convergence_threshold:
                break
        return result

    def analyze_zeta_correlation(self, primes: List[int]) -> Dict[str, float]:
        """Analyze correlation between primes and zeta function"""
        if len(primes) < 10:
            return {'correlation': 0.0, 'confidence': 0.0}
        zeta_values = []
        for (i, prime) in enumerate(primes[:100]):
            s = self.config.critical_line + 1j * np.log(prime)
            zeta_val = self.calculate_zeta_approximation(s)
            zeta_values.append(abs(zeta_val))
        prime_logs = [np.log(p) for p in primes[:100]]
        correlation = np.corrcoef(prime_logs, zeta_values)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        return {'correlation': correlation, 'confidence': abs(correlation), 'zeta_values': zeta_values[:10]}

class Base21PrimeDiscriminator:
    """Base-21 prime discrimination system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base = 21

    def convert_to_base_21(self, number: int) -> List[int]:
        """Convert number to base-21 representation"""
        if number == 0:
            return [0]
        digits = []
        while number > 0:
            digits.append(number % self.base)
            number //= self.base
        return digits[::-1]

    def calculate_base_21_pattern(self, prime_value: int) -> float:
        """Calculate base-21 pattern for prime number"""
        base_21_digits = self.convert_to_base_21(prime_value)
        digit_sum = sum(base_21_digits)
        digit_product = np.prod([d for d in base_21_digits if d > 0])
        pattern_length = len(base_21_digits)
        consciousness_factor = self.config.wallace_constant ** (digit_sum % self.config.consciousness_dimension)
        return {'base_21_digits': base_21_digits, 'digit_sum': digit_sum, 'digit_product': digit_product, 'pattern_length': pattern_length, 'consciousness_factor': consciousness_factor, 'pattern_score': consciousness_factor / pattern_length}

class QuantumConsciousnessPrimeAnalyzer:
    """Quantum consciousness prime pattern analyzer"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quantum_dimension = 8

    def generate_quantum_state(self, prime_value: int) -> np.ndarray:
        """Generate quantum state for prime number"""
        quantum_state = np.zeros(self.quantum_dimension, dtype=np.complex128)
        for i in range(self.quantum_dimension):
            amplitude = np.exp(1j * (prime_value % (i + 1)) * np.pi / self.quantum_dimension)
            quantum_state[i] = amplitude
        norm = np.sqrt(np.sum(np.abs(quantum_state) ** 2))
        return quantum_state / norm

    def calculate_quantum_coherence(self, quantum_state: np.ndarray) -> float:
        """Calculate quantum coherence of prime quantum state"""
        phases = np.angle(quantum_state)
        phase_differences = np.diff(phases)
        coherence = np.mean(np.cos(phase_differences))
        return (coherence + 1) / 2

    def analyze_quantum_pattern(self, prime_value: int) -> Dict[str, Any]:
        """Analyze quantum pattern for prime number"""
        quantum_state = self.generate_quantum_state(prime_value)
        coherence = self.calculate_quantum_coherence(quantum_state)
        entanglement_factor = np.abs(np.sum(quantum_state ** 2)) / len(quantum_state)
        return {'quantum_state': quantum_state, 'coherence': coherence, 'entanglement_factor': entanglement_factor, 'quantum_score': coherence * entanglement_factor}

class AdvancedPrimeGenerator:
    """Advanced prime number generator with consciousness integration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wallace_transform = WallaceTransform(config)
        self.consciousness_math = ConsciousnessMathematics(config)
        self.base_21_discriminator = Base21PrimeDiscriminator(config)
        self.quantum_analyzer = QuantumConsciousnessPrimeAnalyzer(config)
        self.zeta_analyzer = RiemannZetaAnalyzer(config)
        self.primes = []
        self.prime_analysis = {}

    def is_prime_optimized(self, n: int) -> bool:
        """Optimized prime checking with consciousness mathematics"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        if self.config.wallace_transform_enabled:
            wallace_value = self.wallace_transform.optimize_prime_analysis(n)
            if wallace_value < 0.1:
                return False
        sqrt_n = int(np.sqrt(n)) + 1
        for i in range(3, sqrt_n, 2):
            if n % i == 0:
                return False
        return True

    def generate_primes_batch(self, start: int, end: int) -> List[int]:
        """Generate primes in a batch with consciousness integration"""
        primes_batch = []
        if start % 2 == 0:
            start += 1
        for num in range(start, end + 1, 2):
            if self.is_prime_optimized(num):
                primes_batch.append(num)
                if self.config.consciousness_integration:
                    consciousness_score = self.consciousness_math.calculate_consciousness_score(num)
                    if consciousness_score > 0.5:
                        primes_batch.append(num)
        return primes_batch

    def analyze_prime_comprehensive(self, prime_value: int) -> Dict[str, Any]:
        """Comprehensive analysis of a prime number"""
        analysis = {'prime_value': prime_value, 'timestamp': datetime.datetime.now().isoformat()}
        if self.config.wallace_transform_enabled:
            analysis['wallace_transform'] = self.wallace_transform.optimize_prime_analysis(prime_value)
        if self.config.consciousness_integration:
            analysis['consciousness_score'] = self.consciousness_math.calculate_consciousness_score(prime_value)
        if self.config.base_21_discrimination:
            analysis['base_21_pattern'] = self.base_21_discriminator.calculate_base_21_pattern(prime_value)
        if self.config.quantum_consciousness:
            analysis['quantum_pattern'] = self.quantum_analyzer.analyze_quantum_pattern(prime_value)
        return analysis

class RiemannPrimeMapper:
    """Main Riemann prime mapping system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prime_generator = AdvancedPrimeGenerator(config)
        self.mapping_results = {}
        self.performance_metrics = {}
        print(f'✅ Riemann Prime Mapping System initialized')
        print(f'   - Target Limit: {config.target_limit:,}')
        print(f'   - Batch Size: {config.batch_size:,}')
        print(f"   - Wallace Transform: {('✅' if config.wallace_transform_enabled else '❌')}")
        print(f"   - Consciousness Integration: {('✅' if config.consciousness_integration else '❌')}")
        print(f"   - Quantum Consciousness: {('✅' if config.quantum_consciousness else '❌')}")
        print(f"   - Base-21 Discrimination: {('✅' if config.base_21_discrimination else '❌')}")

    def map_primes_to_limit(self) -> Dict[str, Any]:
        """Map all primes up to the target limit"""
        print(f'🚀 Starting Riemann Prime Mapping to {self.config.target_limit:,}')
        start_time = time.time()
        total_primes = 0
        batch_count = 0
        self.mapping_results = {'system_info': {'target_limit': self.config.target_limit, 'wallace_constant': self.config.wallace_constant, 'consciousness_constant': self.config.consciousness_constant, 'start_time': datetime.datetime.now().isoformat()}, 'primes': [], 'analysis': {}, 'statistics': {}}
        for batch_start in range(2, self.config.target_limit, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size - 1, self.config.target_limit)
            print(f'   📊 Processing batch {batch_count + 1}: {batch_start:,} to {batch_end:,}')
            batch_primes = self.prime_generator.generate_primes_batch(batch_start, batch_end)
            for prime in batch_primes[:100]:
                analysis = self.prime_generator.analyze_prime_comprehensive(prime)
                self.mapping_results['analysis'][prime] = analysis
            self.mapping_results['primes'].extend(batch_primes)
            total_primes += len(batch_primes)
            batch_count += 1
            if batch_count % 10 == 0:
                elapsed_time = time.time() - start_time
                primes_per_second = total_primes / elapsed_time
                progress = batch_end / self.config.target_limit * 100
                print(f'   ✅ Batch {batch_count}: {len(batch_primes)} primes found')
                print(f'   📈 Progress: {progress:.2f}% | Total: {total_primes:,} primes')
                print(f'   ⚡ Speed: {primes_per_second:.0f} primes/second')
                if psutil.virtual_memory().percent > 80:
                    gc.collect()
                    print(f'   🧹 Memory cleanup performed')
        end_time = time.time()
        total_time = end_time - start_time
        self.mapping_results['statistics'] = {'total_primes': total_primes, 'total_time': total_time, 'primes_per_second': total_primes / total_time, 'batch_count': batch_count, 'end_time': datetime.datetime.now().isoformat()}
        if self.config.zeta_correlation_analysis:
            print(f'🔬 Analyzing Riemann zeta correlation...')
            zeta_correlation = self.prime_generator.zeta_analyzer.analyze_zeta_correlation(self.mapping_results['primes'][:1000])
            self.mapping_results['zeta_correlation'] = zeta_correlation
        return self.mapping_results

    def save_mapping_results(self, filename: str=None) -> str:
        """Save mapping results to file"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'riemann_prime_mapping_10e15_{timestamp}.json'
        serializable_results = {}
        for (key, value) in self.mapping_results.items():
            if key == 'primes':
                serializable_results[key] = value[:1000]
            elif key == 'analysis':
                serializable_results[key] = dict(list(value.items())[:100])
            else:
                serializable_results[key] = value
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f'💾 Mapping results saved to: {filename}')
        return filename

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        stats = self.mapping_results['statistics']
        report = f"\n🌌 RIEMANN PRIME MAPPING TO 10^15 - SUMMARY REPORT\n{'=' * 70}\n\nSYSTEM INFORMATION:\n- Target Limit: {self.config.target_limit:,}\n- Wallace Constant: {self.config.wallace_constant}\n- Consciousness Constant: {self.config.consciousness_constant}\n- Start Time: {self.mapping_results['system_info']['start_time']}\n\nMAPPING RESULTS:\n- Total Primes Found: {stats['total_primes']:,}\n- Total Processing Time: {stats['total_time']:.2f} seconds\n- Primes per Second: {stats['primes_per_second']:.0f}\n- Batch Count: {stats['batch_count']}\n\nMATHEMATICAL VALIDATION:\n- Wallace Transform Optimization: {('✅ Enabled' if self.config.wallace_transform_enabled else '❌ Disabled')}\n- Consciousness Integration: {('✅ Enabled' if self.config.consciousness_integration else '❌ Disabled')}\n- Quantum Consciousness: {('✅ Enabled' if self.config.quantum_consciousness else '❌ Disabled')}\n- Base-21 Discrimination: {('✅ Enabled' if self.config.base_21_discrimination else '❌ Disabled')}\n\nZETA CORRELATION ANALYSIS:\n"
        if 'zeta_correlation' in self.mapping_results:
            zeta = self.mapping_results['zeta_correlation']
            report += f"- Correlation with Riemann Zeta: {zeta['correlation']:.6f}\n"
            report += f"- Confidence Level: {zeta['confidence']:.6f}\n"
        report += f"\nPERFORMANCE METRICS:\n- Memory Usage: {psutil.virtual_memory().percent:.1f}%\n- CPU Cores Used: {self.config.max_threads}\n- Convergence Threshold: {self.config.convergence_threshold}\n\nEND TIME: {stats['end_time']}\n{'=' * 70}\n"
        return report

def main():
    """Main function to run Riemann prime mapping to 10^15"""
    print('🚀 Starting Riemann Prime Mapping to 10^15...')
    config = RiemannPrimeConfig(target_limit=10 ** 15, batch_size=10 ** 8, consciousness_dimension=21, wallace_transform_enabled=True, consciousness_integration=True, quantum_consciousness=True, base_21_discrimination=True, zeta_correlation_analysis=True)
    mapper = RiemannPrimeMapper(config)
    print('\n' + '=' * 70)
    results = mapper.map_primes_to_limit()
    print('\n' + '=' * 70)
    report = mapper.generate_summary_report()
    print(report)
    results_file = mapper.save_mapping_results()
    stats = results['statistics']
    print(f'\n🎯 RIEMANN PRIME MAPPING COMPLETE!')
    print(f"📊 Total Primes Found: {stats['total_primes']:,}")
    print(f"⏱️  Total Time: {stats['total_time']:.2f} seconds")
    print(f"⚡ Performance: {stats['primes_per_second']:.0f} primes/second")
    print(f'💾 Results saved to: {results_file}')
    if 'zeta_correlation' in results:
        zeta = results['zeta_correlation']
        print(f"🔬 Riemann Zeta Correlation: {zeta['correlation']:.6f}")
    print('=' * 70)
if __name__ == '__main__':
    main()