#!/usr/bin/env python3
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
    REAL_ONLY = "real_only"
    COMPLEX_ALLOWED = "complex_allowed"
    NORMALIZED = "normalized"
    MAGNITUDE_ONLY = "magnitude_only"

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
    
    def __init__(self, default_mode: ComplexNumberType = ComplexNumberType.NORMALIZED):
        self.default_mode = default_mode
        self.complex_threshold = 1e-10  # Threshold for considering a number "complex"
        self.processing_stats = {
            'total_processed': 0,
            'complex_numbers': 0,
            'real_numbers': 0,
            'conversions': 0,
            'errors': 0
        }
        
        print("🔢 Complex Number Manager initialized")
    
    def process_complex_number(self, value: Union[float, complex], 
                             mode: Optional[ComplexNumberType] = None) -> ComplexNumberResult:
        """Process a complex number according to the specified mode"""
        if mode is None:
            mode = self.default_mode
        
        self.processing_stats['total_processed'] += 1
        
        try:
            # Convert to complex if it's not already
            if isinstance(value, (int, float)):
                complex_value = complex(value, 0)
                is_complex = False
            else:
                complex_value = value
                is_complex = abs(complex_value.imag) > self.complex_threshold
            
            # Calculate magnitude and phase
            magnitude = abs(complex_value)
            phase = cmath.phase(complex_value) if is_complex else 0.0
            
            # Process according to mode
            processed_value = self._apply_mode(complex_value, mode, is_complex)
            
            # Update stats
            if is_complex:
                self.processing_stats['complex_numbers'] += 1
            else:
                self.processing_stats['real_numbers'] += 1
            
            if processed_value != complex_value:
                self.processing_stats['conversions'] += 1
            
            return ComplexNumberResult(
                original_value=value,
                processed_value=processed_value,
                magnitude=magnitude,
                phase=phase,
                is_complex=is_complex,
                conversion_type=mode
            )
        
        except Exception as e:
            self.processing_stats['errors'] += 1
            print(f"⚠️ Error processing complex number {value}: {e}")
            return ComplexNumberResult(
                original_value=value,
                processed_value=float(value) if isinstance(value, (int, float)) else 0.0,
                magnitude=abs(value) if hasattr(value, '__abs__') else 0.0,
                phase=0.0,
                is_complex=False,
                conversion_type=mode
            )
    
    def _apply_mode(self, complex_value: complex, mode: ComplexNumberType, is_complex: bool) -> Union[float, complex]:
        """Apply the specified mode to the complex number"""
        if mode == ComplexNumberType.REAL_ONLY:
            return complex_value.real
        
        elif mode == ComplexNumberType.COMPLEX_ALLOWED:
            return complex_value
        
        elif mode == ComplexNumberType.NORMALIZED:
            if is_complex:
                # Normalize complex number to unit circle
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
    
    def process_array(self, array: np.ndarray, mode: Optional[ComplexNumberType] = None) -> np.ndarray:
        """Process a numpy array containing complex numbers"""
        if mode is None:
            mode = self.default_mode
        
        processed_array = np.zeros_like(array, dtype=object)
        
        for idx in np.ndindex(array.shape):
            value = array[idx]
            result = self.process_complex_number(value, mode)
            processed_array[idx] = result.processed_value
        
        return processed_array
    
    def process_dict(self, data: Dict[str, Any], mode: Optional[ComplexNumberType] = None) -> Dict[str, Any]:
        """Process a dictionary containing complex numbers"""
        if mode is None:
            mode = self.default_mode
        
        processed_dict = {}
        
        for key, value in data.items():
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
    
    def process_list(self, data: List[Any], mode: Optional[ComplexNumberType] = None) -> List[Any]:
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
    
    def make_json_serializable(self, data: Any, mode: ComplexNumberType = ComplexNumberType.MAGNITUDE_ONLY) -> Any:
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
            # Handle custom objects
            return self.make_json_serializable(data.__dict__, mode)
        
        else:
            return data
    
    def analyze_complex_distribution(self, data: Any) -> Dict[str, Any]:
        """Analyze the distribution of complex numbers in data"""
        analysis = {
            'total_values': 0,
            'real_values': 0,
            'complex_values': 0,
            'magnitude_stats': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0},
            'phase_stats': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0},
            'complex_ratio': 0.0
        }
        
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
        
        # Calculate statistics
        if magnitudes:
            analysis['magnitude_stats'] = {
                'min': min(magnitudes),
                'max': max(magnitudes),
                'mean': np.mean(magnitudes),
                'std': np.std(magnitudes)
            }
        
        if phases:
            analysis['phase_stats'] = {
                'min': min(phases),
                'max': max(phases),
                'mean': np.mean(phases),
                'std': np.std(phases)
            }
        
        if analysis['total_values'] > 0:
            analysis['complex_ratio'] = analysis['complex_values'] / analysis['total_values']
        
        return analysis
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'processing_stats': self.processing_stats.copy(),
            'default_mode': self.default_mode.value,
            'complex_threshold': self.complex_threshold
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'complex_numbers': 0,
            'real_numbers': 0,
            'conversions': 0,
            'errors': 0
        }
    
    def save_processed_data(self, data: Any, filename: str, mode: ComplexNumberType = ComplexNumberType.MAGNITUDE_ONLY):
        """Save processed data to JSON file"""
        serializable_data = self.make_json_serializable(data, mode)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved processed data to: {filename}")
    
    def create_complex_report(self, data: Any) -> Dict[str, Any]:
        """Create a comprehensive report about complex numbers in data"""
        analysis = self.analyze_complex_distribution(data)
        stats = self.get_processing_stats()
        
        report = {
            'complex_analysis': analysis,
            'processing_stats': stats,
            'recommendations': self._generate_recommendations(analysis)
        }
        
        return report
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on complex number analysis"""
        recommendations = []
        
        complex_ratio = analysis['complex_ratio']
        
        if complex_ratio > 0.8:
            recommendations.append("High complex number ratio detected. Consider using COMPLEX_ALLOWED mode for full precision.")
        elif complex_ratio > 0.3:
            recommendations.append("Moderate complex number ratio. NORMALIZED mode recommended for balanced processing.")
        elif complex_ratio > 0.1:
            recommendations.append("Low complex number ratio. MAGNITUDE_ONLY mode sufficient for most applications.")
        else:
            recommendations.append("Minimal complex numbers. REAL_ONLY mode recommended for simplicity.")
        
        if analysis['magnitude_stats']['max'] > 1000:
            recommendations.append("Large magnitude values detected. Consider normalization for numerical stability.")
        
        if analysis['phase_stats']['std'] > math.pi:
            recommendations.append("High phase variance detected. Consider phase normalization.")
        
        return recommendations

def main():
    """Test Complex Number Manager functionality"""
    print("🔢 Complex Number Manager Test")
    print("=" * 40)
    
    # Initialize manager
    manager = ComplexNumberManager(default_mode=ComplexNumberType.NORMALIZED)
    
    # Test complex numbers
    test_values = [
        1.0,
        1.0 + 2.0j,
        3.0 - 4.0j,
        0.5 + 0.5j,
        complex(0, 1),
        complex(1, 0)
    ]
    
    print("\n📊 Processing test values:")
    for value in test_values:
        result = manager.process_complex_number(value)
        print(f"  {value} -> {result.processed_value} (magnitude: {result.magnitude:.3f}, complex: {result.is_complex})")
    
    # Test array processing
    complex_array = np.array([[1.0, 1.0+2.0j], [3.0-4.0j, 0.5+0.5j]])
    print(f"\n📐 Original array:\n{complex_array}")
    
    processed_array = manager.process_array(complex_array, ComplexNumberType.MAGNITUDE_ONLY)
    print(f"📐 Processed array (magnitude only):\n{processed_array}")
    
    # Test dictionary processing
    test_dict = {
        'real_value': 1.0,
        'complex_value': 1.0 + 2.0j,
        'nested': {
            'array': complex_array,
            'list': [1.0, 1.0+2.0j, 3.0-4.0j]
        }
    }
    
    processed_dict = manager.process_dict(test_dict, ComplexNumberType.MAGNITUDE_ONLY)
    print(f"\n📋 Processed dictionary (magnitude only):")
    print(json.dumps(processed_dict, indent=2))
    
    # Test JSON serialization
    serializable_data = manager.make_json_serializable(test_dict)
    print(f"\n💾 JSON serializable data created successfully")
    
    # Create complex report
    report = manager.create_complex_report(test_dict)
    print(f"\n📊 Complex Analysis Report:")
    print(f"  Total values: {report['complex_analysis']['total_values']}")
    print(f"  Complex ratio: {report['complex_analysis']['complex_ratio']:.3f}")
    print(f"  Magnitude range: {report['complex_analysis']['magnitude_stats']['min']:.3f} - {report['complex_analysis']['magnitude_stats']['max']:.3f}")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Get processing stats
    stats = manager.get_processing_stats()
    print(f"\n📈 Processing Statistics:")
    print(f"  Total processed: {stats['processing_stats']['total_processed']}")
    print(f"  Complex numbers: {stats['processing_stats']['complex_numbers']}")
    print(f"  Real numbers: {stats['processing_stats']['real_numbers']}")
    print(f"  Conversions: {stats['processing_stats']['conversions']}")
    print(f"  Errors: {stats['processing_stats']['errors']}")
    
    print("✅ Complex Number Manager test complete!")

if __name__ == "__main__":
    main()
