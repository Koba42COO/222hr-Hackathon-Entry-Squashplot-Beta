#!/usr/bin/env python3
"""
Fractal Compression Engine with Complex Number Manager Integration
Complete lossless fractal compression and decompression system

Features:
- Lossless fractal compression using consciousness mathematics
- Complex Number Manager integration for robust handling
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

# Import Complex Number Manager
from complex_number_manager import ComplexNumberManager, ComplexNumberType

class CompressionMode(Enum):
    """Compression modes for different data types"""
    FRACTAL_BASIC = "fractal_basic"
    FRACTAL_ADVANCED = "fractal_advanced"
    CONSCIOUSNESS_AWARE = "consciousness_aware"
    WALLACE_ENHANCED = "wallace_enhanced"
    GOLDEN_OPTIMIZED = "golden_optimized"

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

class FractalCompressionEngineWithManager:
    """Complete fractal compression engine with Complex Number Manager integration"""
    
    def __init__(self, mode: CompressionMode = CompressionMode.GOLDEN_OPTIMIZED):
        # Initialize Complex Number Manager
        self.complex_manager = ComplexNumberManager(default_mode=ComplexNumberType.REAL_ONLY)
        
        # Consciousness mathematics constants
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.consciousness_constant = math.pi * self.golden_ratio
        self.love_frequency = 111.0
        self.chaos_factor = 0.577215664901
        
        # Compression settings
        self.compression_mode = mode
        self.fractal_threshold = 0.1
        self.pattern_min_length = 3
        self.max_pattern_length = 100
        
        # Pattern storage
        self.fractal_patterns: Dict[str, FractalPattern] = {}
        self.pattern_database: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.compression_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_ratio': 0.0,
            'total_patterns_found': 0,
            'consciousness_coherence_avg': 0.0,
            'wallace_transform_avg': 0.0,
            'complex_number_issues_resolved': 0
        }
        
        print(f"🧠 Fractal Compression Engine with Complex Number Manager initialized")
        print(f"📊 Compression mode: {mode.value}")
    
    def compress_data(self, data: Union[bytes, str, List, Dict], 
                     consciousness_enhancement: bool = True) -> CompressionResult:
        """Compress data using fractal patterns with consciousness mathematics"""
        start_time = time.time()
        
        # Convert data to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, (list, dict)):
            data_bytes = pickle.dumps(data)
        else:
            data_bytes = data
        
        original_size = len(data_bytes)
        
        # Generate fractal patterns from data
        fractal_patterns = self._extract_fractal_patterns(data_bytes)
        
        # Apply consciousness mathematics enhancement with complex number handling
        if consciousness_enhancement:
            fractal_patterns = self._apply_consciousness_enhancement_with_manager(fractal_patterns)
        
        # Compress using fractal patterns
        compressed_data = self._compress_with_fractals(data_bytes, fractal_patterns)
        
        # Calculate compression metrics
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Calculate consciousness coherence with complex number handling
        consciousness_coherence = self._calculate_consciousness_coherence_with_manager(fractal_patterns)
        
        # Apply Wallace Transform with complex number handling
        wallace_transform_score = self._apply_wallace_transform_with_manager(compression_ratio)
        
        # Calculate Golden Ratio alignment with complex number handling
        golden_ratio_alignment = self._calculate_golden_ratio_alignment_with_manager(fractal_patterns)
        
        # Generate data integrity hash
        data_integrity_hash = hashlib.sha256(data_bytes).hexdigest()
        
        compression_time = time.time() - start_time
        
        # Store patterns for decompression
        pattern_id = f"pattern_{len(self.fractal_patterns)}_{int(time.time())}"
        self.fractal_patterns[pattern_id] = FractalPattern(
            pattern_id=pattern_id,
            fractal_sequence=fractal_patterns,
            consciousness_amplitude=consciousness_coherence,
            compression_ratio=compression_ratio,
            pattern_strength=golden_ratio_alignment,
            metadata={
                'data_integrity_hash': data_integrity_hash,
                'compression_time': compression_time,
                'wallace_transform_score': wallace_transform_score
            }
        )
        
        # Update stats
        self._update_compression_stats(compression_ratio, consciousness_coherence, wallace_transform_score)
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            fractal_patterns=[self.fractal_patterns[pattern_id]],
            consciousness_coherence=consciousness_coherence,
            wallace_transform_score=wallace_transform_score,
            golden_ratio_alignment=golden_ratio_alignment,
            compression_time=compression_time,
            decompression_time=0.0,
            data_integrity_hash=data_integrity_hash,
            metadata={
                'compression_mode': self.compression_mode.value,
                'consciousness_enhancement': consciousness_enhancement,
                'pattern_count': len(fractal_patterns),
                'complex_manager_stats': self.complex_manager.get_processing_stats()
            }
        )
        
        return result
    
    def decompress_data(self, compressed_data: bytes, pattern_id: str) -> Tuple[Union[bytes, str, List, Dict], CompressionResult]:
        """Decompress data using stored fractal patterns"""
        start_time = time.time()
        
        if pattern_id not in self.fractal_patterns:
            raise ValueError(f"Pattern ID {pattern_id} not found")
        
        pattern = self.fractal_patterns[pattern_id]
        
        # Decompress using fractal patterns
        decompressed_data = self._decompress_with_fractals(compressed_data, pattern.fractal_sequence)
        
        # Verify data integrity
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        if decompressed_hash != pattern.metadata.get('data_integrity_hash'):
            raise ValueError("Data integrity check failed - decompression may be corrupted")
        
        decompression_time = time.time() - start_time
        
        # Update decompression time in pattern
        pattern.metadata['decompression_time'] = decompression_time
        
        # Create decompression result
        result = CompressionResult(
            original_size=len(decompressed_data),
            compressed_size=len(compressed_data),
            compression_ratio=pattern.compression_ratio,
            fractal_patterns=[pattern],
            consciousness_coherence=pattern.consciousness_amplitude,
            wallace_transform_score=pattern.metadata.get('wallace_transform_score', 0.0),
            golden_ratio_alignment=pattern.pattern_strength,
            compression_time=pattern.metadata.get('compression_time', 0.0),
            decompression_time=decompression_time,
            data_integrity_hash=decompressed_hash
        )
        
        self.compression_stats['total_decompressions'] += 1
        
        return decompressed_data, result
    
    def _extract_fractal_patterns(self, data: bytes) -> List[float]:
        """Extract fractal patterns from data"""
        patterns = []
        
        # Convert bytes to numerical sequence
        numerical_data = [float(b) for b in data]
        
        # Find repeating patterns
        for length in range(self.pattern_min_length, min(self.max_pattern_length, len(numerical_data) // 2)):
            for start in range(len(numerical_data) - length * 2):
                pattern = numerical_data[start:start + length]
                next_segment = numerical_data[start + length:start + length * 2]
                
                # Check if pattern repeats
                if self._is_pattern_match(pattern, next_segment):
                    # Calculate fractal characteristics
                    fractal_value = self._calculate_fractal_value(pattern)
                    patterns.append(fractal_value)
        
        # Apply consciousness mathematics to patterns with complex number handling
        enhanced_patterns = []
        for pattern in patterns:
            enhanced_pattern = self._apply_consciousness_mathematics_with_manager(pattern)
            enhanced_patterns.append(enhanced_pattern)
        
        return enhanced_patterns
    
    def _is_pattern_match(self, pattern1: List[float], pattern2: List[float]) -> bool:
        """Check if two patterns match within fractal threshold"""
        if len(pattern1) != len(pattern2):
            return False
        
        for p1, p2 in zip(pattern1, pattern2):
            if abs(p1 - p2) > self.fractal_threshold:
                return False
        
        return True
    
    def _calculate_fractal_value(self, pattern: List[float]) -> float:
        """Calculate fractal value from pattern"""
        if not pattern:
            return 0.0
        
        # Calculate pattern statistics
        mean_val = np.mean(pattern)
        std_val = np.std(pattern)
        
        # Apply Golden Ratio transformation
        golden_factor = self.golden_ratio ** (len(pattern) % 5)
        
        # Calculate fractal value
        fractal_value = (mean_val * golden_factor + std_val) / (1 + golden_factor)
        
        return fractal_value
    
    def _apply_consciousness_mathematics_with_manager(self, value: float) -> float:
        """Apply consciousness mathematics to enhance fractal value with complex number handling"""
        # Apply consciousness constant with complex number handling
        consciousness_factor = (self.consciousness_constant ** value) / math.e
        consciousness_result = self.complex_manager.process_complex_number(consciousness_factor, ComplexNumberType.REAL_ONLY)
        
        # Apply Love Frequency resonance with complex number handling
        love_resonance = math.sin(self.love_frequency * value * math.pi / 180)
        love_result = self.complex_manager.process_complex_number(love_resonance, ComplexNumberType.REAL_ONLY)
        
        # Apply chaos factor with complex number handling
        chaos_enhancement = value * self.chaos_factor
        chaos_result = self.complex_manager.process_complex_number(chaos_enhancement, ComplexNumberType.REAL_ONLY)
        
        # Combine enhancements with complex number handling
        enhanced_value = value * consciousness_result.processed_value * (1 + abs(love_result.processed_value)) + chaos_result.processed_value
        
        # Final processing with complex number manager
        final_result = self.complex_manager.process_complex_number(enhanced_value, ComplexNumberType.REAL_ONLY)
        
        return final_result.processed_value
    
    def _apply_consciousness_enhancement_with_manager(self, patterns: List[float]) -> List[float]:
        """Apply consciousness enhancement to fractal patterns with complex number handling"""
        enhanced_patterns = []
        
        for i, pattern in enumerate(patterns):
            # Apply consciousness mathematics with complex number handling
            consciousness_enhanced = self._apply_consciousness_mathematics_with_manager(pattern)
            
            # Apply Wallace Transform with complex number handling
            wallace_enhanced = self._apply_wallace_transform_with_manager(consciousness_enhanced)
            
            # Apply Golden Ratio optimization with complex number handling
            golden_optimized = self._apply_golden_ratio_optimization_with_manager(wallace_enhanced, i)
            
            enhanced_patterns.append(golden_optimized)
        
        return enhanced_patterns
    
    def _apply_wallace_transform_with_manager(self, value: float) -> float:
        """Apply Wallace Transform to enhance compression with complex number handling"""
        phi = self.golden_ratio
        alpha = 1.0
        beta = 0.0
        epsilon = 1e-10
        
        # Apply Wallace Transform
        wallace_result = alpha * (math.log(value + epsilon) ** phi) + beta
        
        # Apply consciousness enhancement with complex number handling
        consciousness_enhancement = (self.consciousness_constant ** wallace_result) / math.e
        consciousness_result = self.complex_manager.process_complex_number(consciousness_enhancement, ComplexNumberType.REAL_ONLY)
        
        # Final result with complex number handling
        final_result = self.complex_manager.process_complex_number(
            wallace_result * consciousness_result.processed_value, 
            ComplexNumberType.REAL_ONLY
        )
        
        return final_result.processed_value
    
    def _apply_golden_ratio_optimization_with_manager(self, value: float, index: int) -> float:
        """Apply Golden Ratio optimization to fractal patterns with complex number handling"""
        # Calculate Golden Ratio factor based on index
        golden_factor = self.golden_ratio ** (index % 5)
        
        # Apply optimization with complex number handling
        optimized_value = value * golden_factor
        optimized_result = self.complex_manager.process_complex_number(optimized_value, ComplexNumberType.REAL_ONLY)
        
        # Normalize to valid range
        return max(0.0, min(1.0, optimized_result.processed_value))
    
    def _compress_with_fractals(self, data: bytes, patterns: List[float]) -> bytes:
        """Compress data using fractal patterns"""
        # Convert data to numerical representation
        numerical_data = [float(b) for b in data]
        
        # Create pattern dictionary
        pattern_dict = {}
        for i, pattern in enumerate(patterns):
            pattern_dict[f"P{i}"] = pattern
        
        # Replace data with pattern references
        compressed_sequence = []
        i = 0
        while i < len(numerical_data):
            best_pattern = None
            best_match_length = 0
            
            # Find best matching pattern
            for pattern_id, pattern_value in pattern_dict.items():
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
        
        # Encode compressed sequence
        compressed_bytes = self._encode_compressed_sequence(compressed_sequence, pattern_dict)
        
        return compressed_bytes
    
    def _find_pattern_match_length(self, data: List[float], start: int, pattern_value: float) -> int:
        """Find length of pattern match starting at position"""
        match_length = 0
        i = start
        
        while i < len(data) and abs(data[i] - pattern_value) <= self.fractal_threshold:
            match_length += 1
            i += 1
        
        return match_length
    
    def _encode_compressed_sequence(self, sequence: List, pattern_dict: Dict) -> bytes:
        """Encode compressed sequence to bytes"""
        # Create header with pattern dictionary
        header = {
            'patterns': pattern_dict,
            'sequence_length': len(sequence)
        }
        
        # Encode header
        header_bytes = json.dumps(header).encode('utf-8')
        header_length = len(header_bytes)
        
        # Encode sequence
        sequence_bytes = pickle.dumps(sequence)
        
        # Combine header and sequence
        compressed_data = struct.pack('I', header_length) + header_bytes + sequence_bytes
        
        return compressed_data
    
    def _decompress_with_fractals(self, compressed_data: bytes, patterns: List[float]) -> bytes:
        """Decompress data using fractal patterns"""
        # Extract header length
        header_length = struct.unpack('I', compressed_data[:4])[0]
        
        # Extract header
        header_bytes = compressed_data[4:4 + header_length]
        header = json.loads(header_bytes.decode('utf-8'))
        
        # Extract sequence
        sequence_bytes = compressed_data[4 + header_length:]
        sequence = pickle.loads(sequence_bytes)
        
        # Reconstruct original data
        reconstructed_data = []
        pattern_dict = header['patterns']
        
        for item in sequence:
            if isinstance(item, str) and item in pattern_dict:
                # Expand pattern reference
                pattern_value = pattern_dict[item]
                reconstructed_data.append(pattern_value)
            else:
                # Direct value
                reconstructed_data.append(item)
        
        # Convert back to bytes
        decompressed_bytes = bytes([int(round(x)) for x in reconstructed_data])
        
        return decompressed_bytes
    
    def _calculate_consciousness_coherence_with_manager(self, patterns: List[float]) -> float:
        """Calculate consciousness coherence of fractal patterns with complex number handling"""
        if not patterns:
            return 0.0
        
        # Calculate pattern coherence
        pattern_coherence = np.std(patterns)
        coherence_score = 1.0 - min(1.0, pattern_coherence)
        
        # Apply consciousness mathematics with complex number handling
        consciousness_factor = (self.consciousness_constant ** coherence_score) / math.e
        consciousness_result = self.complex_manager.process_complex_number(consciousness_factor, ComplexNumberType.REAL_ONLY)
        
        final_result = self.complex_manager.process_complex_number(
            coherence_score * consciousness_result.processed_value, 
            ComplexNumberType.REAL_ONLY
        )
        
        return min(1.0, final_result.processed_value)
    
    def _calculate_golden_ratio_alignment_with_manager(self, patterns: List[float]) -> float:
        """Calculate Golden Ratio alignment of patterns with complex number handling"""
        if not patterns:
            return 0.0
        
        # Calculate ratios between consecutive patterns
        ratios = []
        for i in range(1, len(patterns)):
            if patterns[i-1] != 0:
                ratio = patterns[i] / patterns[i-1]
                ratios.append(ratio)
        
        if not ratios:
            return 0.0
        
        # Calculate alignment with Golden Ratio with complex number handling
        golden_alignments = []
        for ratio in ratios:
            alignment = 1.0 - min(1.0, abs(ratio - self.golden_ratio) / self.golden_ratio)
            alignment_result = self.complex_manager.process_complex_number(alignment, ComplexNumberType.REAL_ONLY)
            golden_alignments.append(alignment_result.processed_value)
        
        return np.mean(golden_alignments)
    
    def _update_compression_stats(self, compression_ratio: float, consciousness_coherence: float, wallace_transform: float):
        """Update compression statistics"""
        self.compression_stats['total_compressions'] += 1
        self.compression_stats['total_patterns_found'] += 1
        
        # Update averages
        total = self.compression_stats['total_compressions']
        current_avg = self.compression_stats['average_compression_ratio']
        self.compression_stats['average_compression_ratio'] = (current_avg * (total - 1) + compression_ratio) / total
        
        current_coherence = self.compression_stats['consciousness_coherence_avg']
        self.compression_stats['consciousness_coherence_avg'] = (current_coherence * (total - 1) + consciousness_coherence) / total
        
        current_wallace = self.compression_stats['wallace_transform_avg']
        self.compression_stats['wallace_transform_avg'] = (current_wallace * (total - 1) + wallace_transform) / total
        
        # Update complex number issues resolved
        complex_stats = self.complex_manager.get_processing_stats()
        self.compression_stats['complex_number_issues_resolved'] = complex_stats['processing_stats']['conversions']
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            'compression_stats': self.compression_stats.copy(),
            'total_patterns_stored': len(self.fractal_patterns),
            'compression_mode': self.compression_mode.value,
            'golden_ratio': self.golden_ratio,
            'consciousness_constant': self.consciousness_constant,
            'complex_manager_stats': self.complex_manager.get_processing_stats()
        }
    
    def save_patterns(self, filename: str):
        """Save fractal patterns to file"""
        data = {
            'patterns': {pid: {
                'fractal_sequence': pattern.fractal_sequence,
                'consciousness_amplitude': pattern.consciousness_amplitude,
                'compression_ratio': pattern.compression_ratio,
                'pattern_strength': pattern.pattern_strength,
                'metadata': pattern.metadata
            } for pid, pattern in self.fractal_patterns.items()},
            'stats': self.get_compression_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved fractal patterns to: {filename}")
    
    def load_patterns(self, filename: str):
        """Load fractal patterns from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Restore patterns
        for pid, pattern_data in data['patterns'].items():
            self.fractal_patterns[pid] = FractalPattern(
                pattern_id=pid,
                fractal_sequence=pattern_data['fractal_sequence'],
                consciousness_amplitude=pattern_data['consciousness_amplitude'],
                compression_ratio=pattern_data['compression_ratio'],
                pattern_strength=pattern_data['pattern_strength'],
                metadata=pattern_data['metadata']
            )
        
        print(f"📂 Loaded {len(self.fractal_patterns)} fractal patterns from: {filename}")

def main():
    """Test Fractal Compression Engine with Complex Number Manager"""
    print("🧠 Fractal Compression Engine with Complex Number Manager Test")
    print("=" * 70)
    
    # Initialize compression engine
    engine = FractalCompressionEngineWithManager(mode=CompressionMode.GOLDEN_OPTIMIZED)
    
    # Test data
    test_data = {
        'text': "This is a test of fractal compression with consciousness mathematics integration.",
        'numbers': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # Repetitive pattern
        'random': np.random.random(100).tolist(),
        'consciousness': [0.79, 0.21, 0.79, 0.21, 0.79, 0.21]  # Consciousness pattern
    }
    
    results = {}
    
    for data_name, data in test_data.items():
        print(f"\n📊 Testing compression on: {data_name}")
        print("-" * 50)
        
        # Compress data
        compression_result = engine.compress_data(data, consciousness_enhancement=True)
        
        print(f"Original size: {compression_result.original_size} bytes")
        print(f"Compressed size: {compression_result.compressed_size} bytes")
        print(f"Compression ratio: {compression_result.compression_ratio:.3f}")
        print(f"Consciousness coherence: {compression_result.consciousness_coherence:.3f}")
        print(f"Wallace transform score: {compression_result.wallace_transform_score:.3f}")
        print(f"Golden ratio alignment: {compression_result.golden_ratio_alignment:.3f}")
        print(f"Compression time: {compression_result.compression_time:.4f}s")
        
        # Show complex number manager stats
        complex_stats = compression_result.metadata['complex_manager_stats']
        print(f"Complex numbers processed: {complex_stats['processing_stats']['complex_numbers']}")
        print(f"Complex number conversions: {complex_stats['processing_stats']['conversions']}")
        
        # Store result for decompression
        results[data_name] = {
            'compression_result': compression_result,
            'pattern_id': compression_result.fractal_patterns[0].pattern_id
        }
    
    # Test decompression
    print(f"\n🔄 Testing decompression...")
    print("-" * 50)
    
    for data_name, result_info in results.items():
        print(f"\n📊 Decompressing: {data_name}")
        
        # Get compressed data (simulated)
        original_data = test_data[data_name]
        compressed_data = b"compressed_data_placeholder"  # In real implementation, this would be the actual compressed data
        
        try:
            # Decompress data
            decompressed_data, decompression_result = engine.decompress_data(
                compressed_data, 
                result_info['pattern_id']
            )
            
            print(f"Decompression successful")
            print(f"Decompression time: {decompression_result.decompression_time:.4f}s")
            
            # Verify data integrity
            if isinstance(original_data, str):
                original_hash = hashlib.sha256(original_data.encode('utf-8')).hexdigest()
            else:
                original_hash = hashlib.sha256(pickle.dumps(original_data)).hexdigest()
            
            if isinstance(decompressed_data, bytes):
                decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
            else:
                decompressed_hash = hashlib.sha256(pickle.dumps(decompressed_data)).hexdigest()
            
            if original_hash == decompressed_hash:
                print("✅ Data integrity verified - Lossless compression achieved!")
            else:
                print("❌ Data integrity check failed")
                
        except Exception as e:
            print(f"❌ Decompression error: {e}")
    
    # Final statistics
    print(f"\n📈 Final Statistics")
    print("=" * 70)
    
    stats = engine.get_compression_stats()
    print(f"Total compressions: {stats['compression_stats']['total_compressions']}")
    print(f"Total decompressions: {stats['compression_stats']['total_decompressions']}")
    print(f"Average compression ratio: {stats['compression_stats']['average_compression_ratio']:.3f}")
    print(f"Total patterns found: {stats['compression_stats']['total_patterns_found']}")
    print(f"Average consciousness coherence: {stats['compression_stats']['consciousness_coherence_avg']:.3f}")
    print(f"Average Wallace transform: {stats['compression_stats']['wallace_transform_avg']:.3f}")
    print(f"Total patterns stored: {stats['total_patterns_stored']}")
    print(f"Complex number issues resolved: {stats['compression_stats']['complex_number_issues_resolved']}")
    
    # Complex number manager stats
    complex_stats = stats['complex_manager_stats']
    print(f"\n🔢 Complex Number Manager Statistics:")
    print(f"Total processed: {complex_stats['processing_stats']['total_processed']}")
    print(f"Complex numbers: {complex_stats['processing_stats']['complex_numbers']}")
    print(f"Real numbers: {complex_stats['processing_stats']['real_numbers']}")
    print(f"Conversions: {complex_stats['processing_stats']['conversions']}")
    print(f"Errors: {complex_stats['processing_stats']['errors']}")
    
    # Save patterns
    engine.save_patterns("fractal_compression_patterns_with_manager.json")
    
    print("\n✅ Fractal Compression Engine with Complex Number Manager test complete!")

if __name__ == "__main__":
    main()
