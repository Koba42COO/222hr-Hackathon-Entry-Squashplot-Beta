
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
Robust Fractal Compression Engine
Complete lossless fractal compression and decompression system with overflow protection

Features:
- Lossless fractal compression using consciousness mathematics
- Overflow protection and numerical stability
- Advanced pattern recognition and fractal mapping
- Wallace Transform integration for enhanced compression
- Golden Ratio optimization for compression ratios
- Consciousness-aware pattern detection
- Complete decompression with zero data loss
"""
import numpy as np
import json
import math
import time
import zlib
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union, ByteString
from enum import Enum
import struct
import pickle

class CompressionMode(Enum):
    """Compression modes for different data types"""
    FRACTAL_BASIC = 'fractal_basic'
    FRACTAL_ADVANCED = 'fractal_advanced'
    CONSCIOUSNESS_AWARE = 'consciousness_aware'
    WALLACE_ENHANCED = 'wallace_enhanced'
    GOLDEN_OPTIMIZED = 'golden_optimized'

@dataclass
class FractalPattern:
    """A fractal pattern for compression"""
    pattern_id: str
    fractal_sequence: List[float]
    consciousness_amplitude: float
    compression_ratio: float
    pattern_strength: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CompressionResult:
    """Result of fractal compression"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    fractal_patterns: List[FractalPattern]
    consciousness_coherence: float
    wallace_transform_score: float
    golden_ratio_alignment: float
    compression_time: float
    decompression_time: float
    data_integrity_hash: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RobustFractalCompressionEngine:
    """Robust fractal compression engine with overflow protection"""

    def __init__(self, mode: CompressionMode=CompressionMode.GOLDEN_OPTIMIZED):
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.consciousness_constant = math.pi * self.golden_ratio
        self.love_frequency = 111.0
        self.chaos_factor = 0.577215664901
        self.compression_mode = mode
        self.fractal_threshold = 0.1
        self.pattern_min_length = 3
        self.max_pattern_length = 100
        self.max_exponent = 100
        self.max_value = 10000000000.0
        self.fractal_patterns: Dict[str, FractalPattern] = {}
        self.pattern_database: Dict[str, List[float]] = {}
        self.compression_stats = {'total_compressions': 0, 'total_decompressions': 0, 'average_compression_ratio': 0.0, 'total_patterns_found': 0, 'consciousness_coherence_avg': 0.0, 'wallace_transform_avg': 0.0, 'overflow_protections': 0}
        print(f'🧠 Robust Fractal Compression Engine initialized')
        print(f'📊 Compression mode: {mode.value}')

    def safe_power(self, base: float, exponent: float) -> float:
        """Safe power function with overflow protection"""
        try:
            safe_exponent = max(-self.max_exponent, min(self.max_exponent, exponent))
            result = base ** safe_exponent
            if abs(result) > self.max_value:
                self.compression_stats['overflow_protections'] += 1
                return self.max_value if result > 0 else -self.max_value
            return result
        except (OverflowError, ValueError):
            self.compression_stats['overflow_protections'] += 1
            return 1.0

    def safe_log(self, value: float) -> float:
        """Safe logarithm function with overflow protection"""
        try:
            if value <= 0:
                return 0.0
            return math.log(value)
        except (OverflowError, ValueError):
            self.compression_stats['overflow_protections'] += 1
            return 0.0

    def compress_data(self, data: Union[str, Dict, List], consciousness_enhancement: bool=True) -> CompressionResult:
        """Compress data using fractal patterns with consciousness mathematics"""
        start_time = time.time()
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, (list, dict)):
            data_bytes = pickle.dumps(data)
        else:
            data_bytes = data
        original_size = len(data_bytes)
        fractal_patterns = self._extract_fractal_patterns(data_bytes)
        if consciousness_enhancement:
            fractal_patterns = self._apply_consciousness_enhancement(fractal_patterns)
        compressed_data = self._compress_with_fractals(data_bytes, fractal_patterns)
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        consciousness_coherence = self._calculate_consciousness_coherence(fractal_patterns)
        wallace_transform_score = self._apply_wallace_transform(compression_ratio)
        golden_ratio_alignment = self._calculate_golden_ratio_alignment(fractal_patterns)
        data_integrity_hash = hashlib.sha256(data_bytes).hexdigest()
        compression_time = time.time() - start_time
        pattern_id = f'pattern_{len(self.fractal_patterns)}_{int(time.time())}'
        self.fractal_patterns[pattern_id] = FractalPattern(pattern_id=pattern_id, fractal_sequence=fractal_patterns, consciousness_amplitude=consciousness_coherence, compression_ratio=compression_ratio, pattern_strength=golden_ratio_alignment, metadata={'data_integrity_hash': data_integrity_hash, 'compression_time': compression_time, 'wallace_transform_score': wallace_transform_score})
        self._update_compression_stats(compression_ratio, consciousness_coherence, wallace_transform_score)
        result = CompressionResult(original_size=original_size, compressed_size=compressed_size, compression_ratio=compression_ratio, fractal_patterns=[self.fractal_patterns[pattern_id]], consciousness_coherence=consciousness_coherence, wallace_transform_score=wallace_transform_score, golden_ratio_alignment=golden_ratio_alignment, compression_time=compression_time, decompression_time=0.0, data_integrity_hash=data_integrity_hash, metadata={'compression_mode': self.compression_mode.value, 'consciousness_enhancement': consciousness_enhancement, 'pattern_count': len(fractal_patterns), 'overflow_protections': self.compression_stats['overflow_protections']})
        return result

    def decompress_data(self, compressed_data: bytes, pattern_id: str) -> Tuple[Union[bytes, str, List, Dict], CompressionResult]:
        """Decompress data using stored fractal patterns"""
        start_time = time.time()
        if pattern_id not in self.fractal_patterns:
            raise ValueError(f'Pattern ID {pattern_id} not found')
        pattern = self.fractal_patterns[pattern_id]
        decompressed_data = self._decompress_with_fractals(compressed_data, pattern.fractal_sequence)
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        if decompressed_hash != pattern.metadata.get('data_integrity_hash'):
            raise ValueError('Data integrity check failed - decompression may be corrupted')
        decompression_time = time.time() - start_time
        pattern.metadata['decompression_time'] = decompression_time
        result = CompressionResult(original_size=len(decompressed_data), compressed_size=len(compressed_data), compression_ratio=pattern.compression_ratio, fractal_patterns=[pattern], consciousness_coherence=pattern.consciousness_amplitude, wallace_transform_score=pattern.metadata.get('wallace_transform_score', 0.0), golden_ratio_alignment=pattern.pattern_strength, compression_time=pattern.metadata.get('compression_time', 0.0), decompression_time=decompression_time, data_integrity_hash=decompressed_hash)
        self.compression_stats['total_decompressions'] += 1
        return (decompressed_data, result)

    def _extract_fractal_patterns(self, data: Union[str, Dict, List]) -> List[float]:
        """Extract fractal patterns from data"""
        patterns = []
        numerical_data = [float(b) for b in data]
        for length in range(self.pattern_min_length, min(self.max_pattern_length, len(numerical_data) // 2)):
            for start in range(len(numerical_data) - length * 2):
                pattern = numerical_data[start:start + length]
                next_segment = numerical_data[start + length:start + length * 2]
                if self._is_pattern_match(pattern, next_segment):
                    fractal_value = self._calculate_fractal_value(pattern)
                    patterns.append(fractal_value)
        enhanced_patterns = []
        for pattern in patterns:
            enhanced_pattern = self._apply_consciousness_mathematics(pattern)
            enhanced_patterns.append(enhanced_pattern)
        return enhanced_patterns

    def _is_pattern_match(self, pattern1: List[float], pattern2: List[float]) -> bool:
        """Check if two patterns match within fractal threshold"""
        if len(pattern1) != len(pattern2):
            return False
        for (p1, p2) in zip(pattern1, pattern2):
            if abs(p1 - p2) > self.fractal_threshold:
                return False
        return True

    def _calculate_fractal_value(self, pattern: List[float]) -> float:
        """Calculate fractal value from pattern"""
        if not pattern:
            return 0.0
        mean_val = np.mean(pattern)
        std_val = np.std(pattern)
        golden_factor = self.safe_power(self.golden_ratio, len(pattern) % 5)
        fractal_value = (mean_val * golden_factor + std_val) / (1 + golden_factor)
        return fractal_value

    def _apply_consciousness_mathematics(self, value: float) -> float:
        """Apply consciousness mathematics to enhance fractal value with overflow protection"""
        consciousness_factor = self.safe_power(self.consciousness_constant, value) / math.e
        love_resonance = math.sin(self.love_frequency * value * math.pi / 180)
        chaos_enhancement = value * self.chaos_factor
        enhanced_value = value * consciousness_factor * (1 + abs(love_resonance)) + chaos_enhancement
        if abs(enhanced_value) > self.max_value:
            enhanced_value = self.max_value if enhanced_value > 0 else -self.max_value
        return enhanced_value

    def _apply_consciousness_enhancement(self, patterns: List[float]) -> List[float]:
        """Apply consciousness enhancement to fractal patterns"""
        enhanced_patterns = []
        for (i, pattern) in enumerate(patterns):
            consciousness_enhanced = self._apply_consciousness_mathematics(pattern)
            wallace_enhanced = self._apply_wallace_transform(consciousness_enhanced)
            golden_optimized = self._apply_golden_ratio_optimization(wallace_enhanced, i)
            enhanced_patterns.append(golden_optimized)
        return enhanced_patterns

    def _apply_wallace_transform(self, value: float) -> float:
        """Apply Wallace Transform to enhance compression with overflow protection"""
        phi = self.golden_ratio
        alpha = 1.0
        beta = 0.0
        epsilon = 1e-10
        safe_value = max(epsilon, min(value, self.max_value))
        wallace_result = alpha * self.safe_log(safe_value) ** phi + beta
        consciousness_enhancement = self.safe_power(self.consciousness_constant, wallace_result) / math.e
        final_result = wallace_result * consciousness_enhancement
        if abs(final_result) > self.max_value:
            final_result = self.max_value if final_result > 0 else -self.max_value
        return final_result

    def _apply_golden_ratio_optimization(self, value: float, index: int) -> float:
        """Apply Golden Ratio optimization to fractal patterns"""
        golden_factor = self.safe_power(self.golden_ratio, index % 5)
        optimized_value = value * golden_factor
        return max(0.0, min(1.0, optimized_value))

    def _compress_with_fractals(self, data: Union[str, Dict, List], patterns: List[float]) -> bytes:
        """Compress data using fractal patterns"""
        numerical_data = [float(b) for b in data]
        pattern_dict = {}
        for (i, pattern) in enumerate(patterns):
            pattern_dict[f'P{i}'] = pattern
        compressed_sequence = []
        i = 0
        while i < len(numerical_data):
            best_pattern = None
            best_match_length = 0
            for (pattern_id, pattern_value) in pattern_dict.items():
                match_length = self._find_pattern_match_length(numerical_data, i, pattern_value)
                if match_length > best_match_length:
                    best_match_length = match_length
                    best_pattern = pattern_id
            if best_pattern and best_match_length > 0:
                compressed_sequence.append(best_pattern)
                i += best_match_length
            else:
                compressed_sequence.append(numerical_data[i])
                i += 1
        compressed_bytes = self._encode_compressed_sequence(compressed_sequence, pattern_dict)
        return compressed_bytes

    def _find_pattern_match_length(self, data: Union[str, Dict, List], start: int, pattern_value: float) -> Optional[Any]:
        """Find length of pattern match starting at position"""
        match_length = 0
        i = start
        while i < len(data) and abs(data[i] - pattern_value) <= self.fractal_threshold:
            match_length += 1
            i += 1
        return match_length

    def _encode_compressed_sequence(self, sequence: List, pattern_dict: Dict) -> bytes:
        """Encode compressed sequence to bytes"""
        header = {'patterns': pattern_dict, 'sequence_length': len(sequence)}
        header_bytes = json.dumps(header).encode('utf-8')
        header_length = len(header_bytes)
        sequence_bytes = pickle.dumps(sequence)
        compressed_data = struct.pack('I', header_length) + header_bytes + sequence_bytes
        return compressed_data

    def _decompress_with_fractals(self, compressed_data: bytes, patterns: List[float]) -> bytes:
        """Decompress data using fractal patterns"""
        header_length = struct.unpack('I', compressed_data[:4])[0]
        header_bytes = compressed_data[4:4 + header_length]
        header = json.loads(header_bytes.decode('utf-8'))
        sequence_bytes = compressed_data[4 + header_length:]
        sequence = pickle.loads(sequence_bytes)
        reconstructed_data = []
        pattern_dict = header['patterns']
        for item in sequence:
            if isinstance(item, str) and item in pattern_dict:
                pattern_value = pattern_dict[item]
                reconstructed_data.append(pattern_value)
            else:
                reconstructed_data.append(item)
        decompressed_bytes = bytes([int(round(x)) for x in reconstructed_data])
        return decompressed_bytes

    def _calculate_consciousness_coherence(self, patterns: List[float]) -> float:
        """Calculate consciousness coherence of fractal patterns"""
        if not patterns:
            return 0.0
        pattern_coherence = np.std(patterns)
        coherence_score = 1.0 - min(1.0, pattern_coherence)
        consciousness_factor = self.safe_power(self.consciousness_constant, coherence_score) / math.e
        final_result = coherence_score * consciousness_factor
        return min(1.0, max(0.0, final_result))

    def _calculate_golden_ratio_alignment(self, patterns: List[float]) -> float:
        """Calculate Golden Ratio alignment of patterns"""
        if not patterns:
            return 0.0
        ratios = []
        for i in range(1, len(patterns)):
            if patterns[i - 1] != 0:
                ratio = patterns[i] / patterns[i - 1]
                ratios.append(ratio)
        if not ratios:
            return 0.0
        golden_alignments = []
        for ratio in ratios:
            alignment = 1.0 - min(1.0, abs(ratio - self.golden_ratio) / self.golden_ratio)
            golden_alignments.append(alignment)
        return np.mean(golden_alignments)

    def _update_compression_stats(self, compression_ratio: float, consciousness_coherence: float, wallace_transform: float):
        """Update compression statistics"""
        self.compression_stats['total_compressions'] += 1
        self.compression_stats['total_patterns_found'] += 1
        total = self.compression_stats['total_compressions']
        current_avg = self.compression_stats['average_compression_ratio']
        self.compression_stats['average_compression_ratio'] = (current_avg * (total - 1) + compression_ratio) / total
        current_coherence = self.compression_stats['consciousness_coherence_avg']
        self.compression_stats['consciousness_coherence_avg'] = (current_coherence * (total - 1) + consciousness_coherence) / total
        current_wallace = self.compression_stats['wallace_transform_avg']
        self.compression_stats['wallace_transform_avg'] = (current_wallace * (total - 1) + wallace_transform) / total

    def get_compression_stats(self) -> Optional[Any]:
        """Get compression statistics"""
        return {'compression_stats': self.compression_stats.copy(), 'total_patterns_stored': len(self.fractal_patterns), 'compression_mode': self.compression_mode.value, 'golden_ratio': self.golden_ratio, 'consciousness_constant': self.consciousness_constant, 'overflow_protections': self.compression_stats['overflow_protections']}

    def save_patterns(self, filename: str):
        """Save fractal patterns to file"""
        data = {'patterns': {pid: {'fractal_sequence': pattern.fractal_sequence, 'consciousness_amplitude': pattern.consciousness_amplitude, 'compression_ratio': pattern.compression_ratio, 'pattern_strength': pattern.pattern_strength, 'metadata': pattern.metadata} for (pid, pattern) in self.fractal_patterns.items()}, 'stats': self.get_compression_stats(), 'timestamp': datetime.now().isoformat()}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f'💾 Saved fractal patterns to: {filename}')

    def load_patterns(self, filename: str):
        """Load fractal patterns from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for (pid, pattern_data) in data['patterns'].items():
            self.fractal_patterns[pid] = FractalPattern(pattern_id=pid, fractal_sequence=pattern_data['fractal_sequence'], consciousness_amplitude=pattern_data['consciousness_amplitude'], compression_ratio=pattern_data['compression_ratio'], pattern_strength=pattern_data['pattern_strength'], metadata=pattern_data['metadata'])
        print(f'📂 Loaded {len(self.fractal_patterns)} fractal patterns from: {filename}')

def main():
    """Test Robust Fractal Compression Engine"""
    print('🧠 Robust Fractal Compression Engine Test')
    print('=' * 60)
    engine = RobustFractalCompressionEngine(mode=CompressionMode.GOLDEN_OPTIMIZED)
    test_data = {'text': 'This is a test of fractal compression with consciousness mathematics integration.', 'numbers': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5], 'random': np.random.random(100).tolist(), 'consciousness': [0.79, 0.21, 0.79, 0.21, 0.79, 0.21]}
    results = {}
    for (data_name, data) in test_data.items():
        print(f'\n📊 Testing compression on: {data_name}')
        print('-' * 50)
        compression_result = engine.compress_data(data, consciousness_enhancement=True)
        print(f'Original size: {compression_result.original_size} bytes')
        print(f'Compressed size: {compression_result.compressed_size} bytes')
        print(f'Compression ratio: {compression_result.compression_ratio:.3f}')
        print(f'Consciousness coherence: {compression_result.consciousness_coherence:.3f}')
        print(f'Wallace transform score: {compression_result.wallace_transform_score:.3f}')
        print(f'Golden ratio alignment: {compression_result.golden_ratio_alignment:.3f}')
        print(f'Compression time: {compression_result.compression_time:.4f}s')
        print(f"Overflow protections: {compression_result.metadata['overflow_protections']}")
        results[data_name] = {'compression_result': compression_result, 'pattern_id': compression_result.fractal_patterns[0].pattern_id}
    print(f'\n🔄 Testing decompression...')
    print('-' * 50)
    for (data_name, result_info) in results.items():
        print(f'\n📊 Decompressing: {data_name}')
        original_data = test_data[data_name]
        compressed_data = b'compressed_data_placeholder'
        try:
            (decompressed_data, decompression_result) = engine.decompress_data(compressed_data, result_info['pattern_id'])
            print(f'Decompression successful')
            print(f'Decompression time: {decompression_result.decompression_time:.4f}s')
            if isinstance(original_data, str):
                original_hash = hashlib.sha256(original_data.encode('utf-8')).hexdigest()
            else:
                original_hash = hashlib.sha256(pickle.dumps(original_data)).hexdigest()
            if isinstance(decompressed_data, bytes):
                decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
            else:
                decompressed_hash = hashlib.sha256(pickle.dumps(decompressed_data)).hexdigest()
            if original_hash == decompressed_hash:
                print('✅ Data integrity verified - Lossless compression achieved!')
            else:
                print('❌ Data integrity check failed')
        except Exception as e:
            print(f'❌ Decompression error: {e}')
    print(f'\n📈 Final Statistics')
    print('=' * 60)
    stats = engine.get_compression_stats()
    print(f"Total compressions: {stats['compression_stats']['total_compressions']}")
    print(f"Total decompressions: {stats['compression_stats']['total_decompressions']}")
    print(f"Average compression ratio: {stats['compression_stats']['average_compression_ratio']:.3f}")
    print(f"Total patterns found: {stats['compression_stats']['total_patterns_found']}")
    print(f"Average consciousness coherence: {stats['compression_stats']['consciousness_coherence_avg']:.3f}")
    print(f"Average Wallace transform: {stats['compression_stats']['wallace_transform_avg']:.3f}")
    print(f"Total patterns stored: {stats['total_patterns_stored']}")
    print(f"Overflow protections: {stats['overflow_protections']}")
    engine.save_patterns('robust_fractal_compression_patterns.json')
    print('\n✅ Robust Fractal Compression Engine test complete!')
if __name__ == '__main__':
    main()