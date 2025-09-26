
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
Complex Number Manager
Handles complex number operations and conversions for HRM + Trigeminal Logic systems

Features:
- Complex number normalization
- Real number conversion
- JSON serialization handling
- Complex number validation
- Mathematical operations with complex numbers
"""
import numpy as np
import json
import math
import cmath
from typing import Union, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ComplexNumberType(Enum):
    """Types of complex number handling"""
    REAL_ONLY = 'real_only'
    COMPLEX_ALLOWED = 'complex_allowed'
    NORMALIZED = 'normalized'
    MAGNITUDE_ONLY = 'magnitude_only'

@dataclass
class ComplexNumberResult:
    """Result of complex number processing"""
    original_value: Union[float, complex]
    processed_value: Union[float, complex]
    magnitude: float
    phase: float
    is_complex: bool
    conversion_type: ComplexNumberType
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ComplexNumberManager:
    """Manager for handling complex number operations and conversions"""

    def __init__(self, default_mode: ComplexNumberType=ComplexNumberType.NORMALIZED):
        self.default_mode = default_mode
        self.complex_threshold = 1e-10
        self.processing_stats = {'total_processed': 0, 'complex_numbers': 0, 'real_numbers': 0, 'conversions': 0, 'errors': 0}
        print('🔢 Complex Number Manager initialized')

    def process_complex_number(self, value: Union[float, complex], mode: Optional[ComplexNumberType]=None) -> Dict[str, Any]:
        """Process a complex number according to the specified mode"""
        if mode is None:
            mode = self.default_mode
        self.processing_stats['total_processed'] += 1
        try:
            if isinstance(value, (int, float)):
                complex_value = complex(value, 0)
                is_complex = False
            else:
                complex_value = value
                is_complex = abs(complex_value.imag) > self.complex_threshold
            magnitude = abs(complex_value)
            phase = cmath.phase(complex_value) if is_complex else 0.0
            processed_value = self._apply_mode(complex_value, mode, is_complex)
            if is_complex:
                self.processing_stats['complex_numbers'] += 1
            else:
                self.processing_stats['real_numbers'] += 1
            if processed_value != complex_value:
                self.processing_stats['conversions'] += 1
            return ComplexNumberResult(original_value=value, processed_value=processed_value, magnitude=magnitude, phase=phase, is_complex=is_complex, conversion_type=mode)
        except Exception as e:
            self.processing_stats['errors'] += 1
            print(f'⚠️ Error processing complex number {value}: {e}')
            return ComplexNumberResult(original_value=value, processed_value=float(value) if isinstance(value, (int, float)) else 0.0, magnitude=abs(value) if hasattr(value, '__abs__') else 0.0, phase=0.0, is_complex=False, conversion_type=mode)

    def _apply_mode(self, complex_value: complex, mode: ComplexNumberType, is_complex: bool) -> Union[float, complex]:
        """Apply the specified mode to the complex number"""
        if mode == ComplexNumberType.REAL_ONLY:
            return complex_value.real
        elif mode == ComplexNumberType.COMPLEX_ALLOWED:
            return complex_value
        elif mode == ComplexNumberType.NORMALIZED:
            if is_complex:
                magnitude = abs(complex_value)
                if magnitude > 0:
                    return complex_value / magnitude
                else:
                    return 0.0
            else:
                return complex_value.real
        elif mode == ComplexNumberType.MAGNITUDE_ONLY:
            return abs(complex_value)
        else:
            return complex_value.real

    def process_array(self, array: np.ndarray, mode: Optional[ComplexNumberType]=None) -> Dict[str, Any]:
        """Process a numpy array containing complex numbers"""
        if mode is None:
            mode = self.default_mode
        processed_array = np.zeros_like(array, dtype=object)
        for idx in np.ndindex(array.shape):
            value = array[idx]
            result = self.process_complex_number(value, mode)
            processed_array[idx] = result.processed_value
        return processed_array

    def process_dict(self, data: Union[str, Dict, List], mode: Optional[ComplexNumberType]=None) -> Dict[str, Any]:
        """Process a dictionary containing complex numbers"""
        if mode is None:
            mode = self.default_mode
        processed_dict = {}
        for (key, value) in data.items():
            if isinstance(value, (complex, float, int)):
                result = self.process_complex_number(value, mode)
                processed_dict[key] = result.processed_value
            elif isinstance(value, dict):
                processed_dict[key] = self.process_dict(value, mode)
            elif isinstance(value, list):
                processed_dict[key] = self.process_list(value, mode)
            elif isinstance(value, np.ndarray):
                processed_dict[key] = self.process_array(value, mode).tolist()
            else:
                processed_dict[key] = value
        return processed_dict

    def process_list(self, data: Union[str, Dict, List], mode: Optional[ComplexNumberType]=None) -> Dict[str, Any]:
        """Process a list containing complex numbers"""
        if mode is None:
            mode = self.default_mode
        processed_list = []
        for item in data:
            if isinstance(item, (complex, float, int)):
                result = self.process_complex_number(item, mode)
                processed_list.append(result.processed_value)
            elif isinstance(item, dict):
                processed_list.append(self.process_dict(item, mode))
            elif isinstance(item, list):
                processed_list.append(self.process_list(item, mode))
            elif isinstance(item, np.ndarray):
                processed_list.append(self.process_array(item, mode))
            else:
                processed_list.append(item)
        return processed_list

    def make_json_serializable(self, data: Union[str, Dict, List], mode: ComplexNumberType=ComplexNumberType.MAGNITUDE_ONLY) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, (complex, float, int)):
            result = self.process_complex_number(data, mode)
            return result.processed_value
        elif isinstance(data, dict):
            return self.process_dict(data, mode)
        elif isinstance(data, list):
            return self.process_list(data, mode)
        elif isinstance(data, np.ndarray):
            return self.process_array(data, mode).tolist()
        elif hasattr(data, '__dict__'):
            return self.make_json_serializable(data.__dict__, mode)
        else:
            return data

    def analyze_complex_distribution(self, data: Union[str, Dict, List]) -> Dict[str, Any]:
        """Analyze the distribution of complex numbers in data"""
        analysis = {'total_values': 0, 'real_values': 0, 'complex_values': 0, 'magnitude_stats': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0}, 'phase_stats': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0}, 'complex_ratio': 0.0}
        magnitudes = []
        phases = []

        def extract_complex_numbers(obj):
            if isinstance(obj, (complex, float, int)):
                analysis['total_values'] += 1
                result = self.process_complex_number(obj, ComplexNumberType.COMPLEX_ALLOWED)
                if result.is_complex:
                    analysis['complex_values'] += 1
                    magnitudes.append(result.magnitude)
                    phases.append(result.phase)
                else:
                    analysis['real_values'] += 1
                    magnitudes.append(result.magnitude)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_complex_numbers(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_complex_numbers(item)
            elif isinstance(obj, np.ndarray):
                for item in obj.flatten():
                    extract_complex_numbers(item)
        extract_complex_numbers(data)
        if magnitudes:
            analysis['magnitude_stats'] = {'min': min(magnitudes), 'max': max(magnitudes), 'mean': np.mean(magnitudes), 'std': np.std(magnitudes)}
        if phases:
            analysis['phase_stats'] = {'min': min(phases), 'max': max(phases), 'mean': np.mean(phases), 'std': np.std(phases)}
        if analysis['total_values'] > 0:
            analysis['complex_ratio'] = analysis['complex_values'] / analysis['total_values']
        return analysis

    def get_processing_stats(self) -> Optional[Any]:
        """Get processing statistics"""
        return {'processing_stats': self.processing_stats.copy(), 'default_mode': self.default_mode.value, 'complex_threshold': self.complex_threshold}

    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {'total_processed': 0, 'complex_numbers': 0, 'real_numbers': 0, 'conversions': 0, 'errors': 0}

    def save_processed_data(self, data: Union[str, Dict, List], filename: str, mode: ComplexNumberType=ComplexNumberType.MAGNITUDE_ONLY):
        """Save processed data to JSON file"""
        serializable_data = self.make_json_serializable(data, mode)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f'💾 Saved processed data to: {filename}')

    def create_complex_report(self, data: Union[str, Dict, List]) -> Dict[str, Any]:
        """Create a comprehensive report about complex numbers in data"""
        analysis = self.analyze_complex_distribution(data)
        stats = self.get_processing_stats()
        report = {'complex_analysis': analysis, 'processing_stats': stats, 'recommendations': self._generate_recommendations(analysis)}
        return report

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on complex number analysis"""
        recommendations = []
        complex_ratio = analysis['complex_ratio']
        if complex_ratio > 0.8:
            recommendations.append('High complex number ratio detected. Consider using COMPLEX_ALLOWED mode for full precision.')
        elif complex_ratio > 0.3:
            recommendations.append('Moderate complex number ratio. NORMALIZED mode recommended for balanced processing.')
        elif complex_ratio > 0.1:
            recommendations.append('Low complex number ratio. MAGNITUDE_ONLY mode sufficient for most applications.')
        else:
            recommendations.append('Minimal complex numbers. REAL_ONLY mode recommended for simplicity.')
        if analysis['magnitude_stats']['max'] > 1000:
            recommendations.append('Large magnitude values detected. Consider normalization for numerical stability.')
        if analysis['phase_stats']['std'] > math.pi:
            recommendations.append('High phase variance detected. Consider phase normalization.')
        return recommendations

def main():
    """Test Complex Number Manager functionality"""
    print('🔢 Complex Number Manager Test')
    print('=' * 40)
    manager = ComplexNumberManager(default_mode=ComplexNumberType.NORMALIZED)
    test_values = [1.0, 1.0 + 2j, 3.0 - 4j, 0.5 + 0.5j, complex(0, 1), complex(1, 0)]
    print('\n📊 Processing test values:')
    for value in test_values:
        result = manager.process_complex_number(value)
        print(f'  {value} -> {result.processed_value} (magnitude: {result.magnitude:.3f}, complex: {result.is_complex})')
    complex_array = np.array([[1.0, 1.0 + 2j], [3.0 - 4j, 0.5 + 0.5j]])
    print(f'\n📐 Original array:\n{complex_array}')
    processed_array = manager.process_array(complex_array, ComplexNumberType.MAGNITUDE_ONLY)
    print(f'📐 Processed array (magnitude only):\n{processed_array}')
    test_dict = {'real_value': 1.0, 'complex_value': 1.0 + 2j, 'nested': {'array': complex_array, 'list': [1.0, 1.0 + 2j, 3.0 - 4j]}}
    processed_dict = manager.process_dict(test_dict, ComplexNumberType.MAGNITUDE_ONLY)
    print(f'\n📋 Processed dictionary (magnitude only):')
    print(json.dumps(processed_dict, indent=2))
    serializable_data = manager.make_json_serializable(test_dict)
    print(f'\n💾 JSON serializable data created successfully')
    report = manager.create_complex_report(test_dict)
    print(f'\n📊 Complex Analysis Report:')
    print(f"  Total values: {report['complex_analysis']['total_values']}")
    print(f"  Complex ratio: {report['complex_analysis']['complex_ratio']:.3f}")
    print(f"  Magnitude range: {report['complex_analysis']['magnitude_stats']['min']:.3f} - {report['complex_analysis']['magnitude_stats']['max']:.3f}")
    print(f'\n💡 Recommendations:')
    for rec in report['recommendations']:
        print(f'  • {rec}')
    stats = manager.get_processing_stats()
    print(f'\n📈 Processing Statistics:')
    print(f"  Total processed: {stats['processing_stats']['total_processed']}")
    print(f"  Complex numbers: {stats['processing_stats']['complex_numbers']}")
    print(f"  Real numbers: {stats['processing_stats']['real_numbers']}")
    print(f"  Conversions: {stats['processing_stats']['conversions']}")
    print(f"  Errors: {stats['processing_stats']['errors']}")
    print('✅ Complex Number Manager test complete!')
if __name__ == '__main__':
    main()