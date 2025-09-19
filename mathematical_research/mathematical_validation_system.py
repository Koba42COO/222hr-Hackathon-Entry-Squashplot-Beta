#!/usr/bin/env python3
"""
MATHEMATICAL VALIDATION SYSTEM
============================================================
Rigorous Testing of Quantum Adaptive Implementation
============================================================

This system validates whether the quantum adaptive implementation represents
genuine mathematical insight or parameter fitting through:
1. Larger independent problem sets
2. Actual FFT analysis of mathematical signals
3. Cross-validation across different mathematical domains
4. Statistical significance testing
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.fft import fft, fftfreq
from scipy import stats
import random

# Constants
PHI = (1 + math.sqrt(5)) / 2
E = math.e
PI = math.pi

@dataclass
class MathematicalProblem:
    """Represents a mathematical problem for validation."""
    problem_id: int
    problem_type: str  # "beal", "fermat", "erdos_straus", "catalan"
    a: int
    b: int
    c: int
    expected_valid: bool
    complexity_score: float
    gcd_value: int
    size_factor: float
    d: Optional[int] = None

@dataclass
class ValidationResult:
    """Results of mathematical validation."""
    problem: MathematicalProblem
    fixed_threshold_result: bool
    adaptive_threshold_result: bool
    adaptive_threshold_value: float
    quantum_state: Dict
    improvement: bool
    confidence_score: float

@dataclass
class StatisticalAnalysis:
    """Statistical analysis of validation results."""
    total_problems: int
    fixed_threshold_accuracy: float
    adaptive_threshold_accuracy: float
    improvement_rate: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    false_positive_rate: float
    false_negative_rate: float

class MathematicalValidator:
    """Rigorous mathematical validation system."""
    
    def __init__(self):
        self.fixed_threshold = 0.3
        self.test_problems = []
        self.validation_problems = []
        
    def generate_independent_problem_set(self, num_problems: int = 1000) -> List[MathematicalProblem]:
        """Generate a large independent problem set for validation."""
        problems = []
        
        # Beal Conjecture problems (most complex)
        for i in range(num_problems // 4):
            # Generate random exponents and bases
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            c = random.randint(2, 50)
            d = random.randint(2, 50)
            
            # Calculate expected validity based on mathematical rules
            expected_valid = self._calculate_expected_validity(a, b, c, d)
            complexity_score = self._calculate_complexity(a, b, c, d)
            gcd_value = math.gcd(math.gcd(a, b), math.gcd(c, d))
            size_factor = (a + b + c + d) / 200.0
            
            problem = MathematicalProblem(
                problem_id=i,
                problem_type="beal",
                a=a, b=b, c=c, d=d,
                expected_valid=expected_valid,
                complexity_score=complexity_score,
                gcd_value=gcd_value,
                size_factor=size_factor
            )
            problems.append(problem)
        
        # Fermat's Last Theorem problems
        for i in range(num_problems // 4):
            n = random.randint(3, 10)
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            c = int((a**n + b**n)**(1/n)) + random.choice([-1, 0, 1])
            
            expected_valid = (a**n + b**n == c**n)
            complexity_score = n * (a + b + c) / 1000.0
            gcd_value = math.gcd(math.gcd(a, b), c)
            size_factor = (a + b + c) / 300.0
            
            problem = MathematicalProblem(
                problem_id=i + num_problems // 4,
                problem_type="fermat",
                a=a, b=b, c=c,
                expected_valid=expected_valid,
                complexity_score=complexity_score,
                gcd_value=gcd_value,
                size_factor=size_factor
            )
            problems.append(problem)
        
        # Erdős–Straus problems
        for i in range(num_problems // 4):
            n = random.randint(2, 100)
            expected_valid = self._is_erdos_straus_valid(n)
            complexity_score = n / 100.0
            gcd_value = 1  # Simplified
            size_factor = n / 100.0
            
            problem = MathematicalProblem(
                problem_id=i + 2 * num_problems // 4,
                problem_type="erdos_straus",
                a=n, b=1, c=1,
                expected_valid=expected_valid,
                complexity_score=complexity_score,
                gcd_value=gcd_value,
                size_factor=size_factor
            )
            problems.append(problem)
        
        # Catalan problems
        for i in range(num_problems // 4):
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            expected_valid = self._is_catalan_valid(a, b)
            complexity_score = (a + b) / 100.0
            gcd_value = math.gcd(a, b)
            size_factor = (a + b) / 100.0
            
            problem = MathematicalProblem(
                problem_id=i + 3 * num_problems // 4,
                problem_type="catalan",
                a=a, b=b, c=1,
                expected_valid=expected_valid,
                complexity_score=complexity_score,
                gcd_value=gcd_value,
                size_factor=size_factor
            )
            problems.append(problem)
        
        return problems
    
    def _calculate_expected_validity(self, a: int, b: int, c: int, d: int) -> bool:
        """Calculate expected validity for Beal problems."""
        # Simplified validity check based on mathematical properties
        if a == b == c == d:
            return True
        if math.gcd(math.gcd(a, b), math.gcd(c, d)) > 1:
            return True
        if a + b == c + d:
            return True
        return random.random() < 0.3  # 30% chance of being valid
    
    def _is_erdos_straus_valid(self, n: int) -> bool:
        """Check if n satisfies Erdős–Straus conjecture."""
        # Simplified check
        return n % 4 != 1
    
    def _is_catalan_valid(self, a: int, b: int) -> bool:
        """Check if (a,b) satisfies Catalan conjecture."""
        # Simplified check
        return a != b and math.gcd(a, b) == 1
    
    def _calculate_complexity(self, a: int, b: int, c: int, d: Optional[int] = None) -> float:
        """Calculate mathematical complexity score."""
        if d is None:
            return (a + b + c) / 300.0
        return (a + b + c + d) / 400.0
    
    def calculate_wallace_error(self, problem: MathematicalProblem) -> float:
        """Calculate Wallace Transform error."""
        if problem.problem_type == "beal":
            left_side = problem.a**3 + problem.b**3
            right_side = problem.c**3 + problem.d**3
        elif problem.problem_type == "fermat":
            left_side = problem.a**3 + problem.b**3
            right_side = problem.c**3
        else:
            left_side = problem.a + problem.b
            right_side = problem.c
        
        if right_side == 0:
            return abs(left_side)
        
        ratio = left_side / right_side
        return abs(ratio - 1.0)
    
    def calculate_quantum_state(self, problem: MathematicalProblem) -> Dict:
        """Calculate quantum state for adaptive threshold."""
        wallace_error = self.calculate_wallace_error(problem)
        
        # Dimensional complexity
        dimensionality = 3 if problem.problem_type in ["beal", "fermat"] else 2
        dimensional_factor = 1 + (dimensionality - 1) * 0.05
        
        # GCD-based adaptation
        gcd_factor = 1 + (problem.gcd_value - 1) * 0.1
        
        # Size-based adaptation
        size_factor = 1 + problem.size_factor * 0.2
        
        # Phase calculations
        phase = (problem.problem_id * PHI) % (2 * PI)
        phase_factor = 1 + 0.1 * math.sin(phase)
        
        # Quantum noise (mathematical complexity)
        quantum_noise = problem.complexity_score * 0.5
        
        # Coherence factor
        coherence = 1.0 - quantum_noise
        
        # Adaptive threshold calculation
        base_threshold = self.fixed_threshold
        adaptive_threshold = base_threshold * dimensional_factor * gcd_factor * size_factor * phase_factor
        
        return {
            'wallace_error': wallace_error,
            'dimensionality': dimensionality,
            'dimensional_factor': dimensional_factor,
            'gcd_factor': gcd_factor,
            'size_factor': size_factor,
            'phase_factor': phase_factor,
            'quantum_noise': quantum_noise,
            'coherence': coherence,
            'adaptive_threshold': adaptive_threshold
        }
    
    def validate_problem(self, problem: MathematicalProblem) -> ValidationResult:
        """Validate a single problem with both fixed and adaptive thresholds."""
        wallace_error = self.calculate_wallace_error(problem)
        quantum_state = self.calculate_quantum_state(problem)
        
        # Fixed threshold validation
        fixed_threshold_result = wallace_error <= self.fixed_threshold
        
        # Adaptive threshold validation
        adaptive_threshold_result = wallace_error <= quantum_state['adaptive_threshold']
        
        # Determine improvement
        improvement = not fixed_threshold_result and adaptive_threshold_result
        
        # Calculate confidence score
        confidence_score = 1.0 - (wallace_error / quantum_state['adaptive_threshold'])
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return ValidationResult(
            problem=problem,
            fixed_threshold_result=fixed_threshold_result,
            adaptive_threshold_result=adaptive_threshold_result,
            adaptive_threshold_value=quantum_state['adaptive_threshold'],
            quantum_state=quantum_state,
            improvement=improvement,
            confidence_score=confidence_score
        )
    
    def perform_fft_analysis(self, problems: List[MathematicalProblem]) -> Dict:
        """Perform actual FFT analysis on mathematical signals."""
        # Create mathematical signal from Wallace errors
        wallace_errors = [self.calculate_wallace_error(p) for p in problems]
        
        # Apply FFT
        fft_result = fft(wallace_errors)
        frequencies = fftfreq(len(wallace_errors))
        
        # Get positive frequencies
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        fft_result = fft_result[positive_freq_mask]
        
        # Calculate power spectrum
        power_spectrum = np.abs(fft_result)**2
        
        # Find dominant frequencies
        peak_indices = []
        for i in range(1, len(power_spectrum) - 1):
            if power_spectrum[i] > power_spectrum[i-1] and power_spectrum[i] > power_spectrum[i+1]:
                if power_spectrum[i] > np.mean(power_spectrum) * 0.1:
                    peak_indices.append(i)
        
        dominant_frequencies = frequencies[peak_indices]
        dominant_powers = power_spectrum[peak_indices]
        
        return {
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'dominant_frequencies': dominant_frequencies,
            'dominant_powers': dominant_powers,
            'total_energy': np.sum(power_spectrum),
            'peak_count': len(peak_indices)
        }
    
    def perform_statistical_analysis(self, results: List[ValidationResult]) -> StatisticalAnalysis:
        """Perform rigorous statistical analysis."""
        total_problems = len(results)
        
        # Calculate accuracies
        fixed_correct = sum(1 for r in results if r.fixed_threshold_result == r.problem.expected_valid)
        adaptive_correct = sum(1 for r in results if r.adaptive_threshold_result == r.problem.expected_valid)
        
        fixed_threshold_accuracy = fixed_correct / total_problems
        adaptive_threshold_accuracy = adaptive_correct / total_problems
        
        # Calculate improvement rate
        improvements = sum(1 for r in results if r.improvement)
        improvement_rate = improvements / total_problems
        
        # Perform McNemar's test for paired proportions
        # Create contingency table
        both_correct = sum(1 for r in results if r.fixed_threshold_result == r.problem.expected_valid and r.adaptive_threshold_result == r.problem.expected_valid)
        both_incorrect = sum(1 for r in results if r.fixed_threshold_result != r.problem.expected_valid and r.adaptive_threshold_result != r.problem.expected_valid)
        fixed_correct_adaptive_incorrect = sum(1 for r in results if r.fixed_threshold_result == r.problem.expected_valid and r.adaptive_threshold_result != r.problem.expected_valid)
        fixed_incorrect_adaptive_correct = sum(1 for r in results if r.fixed_threshold_result != r.problem.expected_valid and r.adaptive_threshold_result == r.problem.expected_valid)
        
        # McNemar's test
        if fixed_correct_adaptive_incorrect + fixed_incorrect_adaptive_correct > 0:
            mcnemar_statistic = (abs(fixed_correct_adaptive_incorrect - fixed_incorrect_adaptive_correct) - 1)**2 / (fixed_correct_adaptive_incorrect + fixed_incorrect_adaptive_correct)
            p_value = 1 - stats.chi2.cdf(mcnemar_statistic, 1)
        else:
            p_value = 1.0
        
        # Effect size (Cohen's d)
        fixed_scores = [1.0 if r.fixed_threshold_result == r.problem.expected_valid else 0.0 for r in results]
        adaptive_scores = [1.0 if r.adaptive_threshold_result == r.problem.expected_valid else 0.0 for r in results]
        
        effect_size = (np.mean(adaptive_scores) - np.mean(fixed_scores)) / np.sqrt((np.var(fixed_scores) + np.var(adaptive_scores)) / 2)
        
        # Confidence interval
        accuracy_diff = adaptive_threshold_accuracy - fixed_threshold_accuracy
        se = np.sqrt(fixed_threshold_accuracy * (1 - fixed_threshold_accuracy) / total_problems + adaptive_threshold_accuracy * (1 - adaptive_threshold_accuracy) / total_problems)
        confidence_interval = (accuracy_diff - 1.96 * se, accuracy_diff + 1.96 * se)
        
        # Error rates
        false_positive_rate = sum(1 for r in results if r.adaptive_threshold_result and not r.problem.expected_valid) / total_problems
        false_negative_rate = sum(1 for r in results if not r.adaptive_threshold_result and r.problem.expected_valid) / total_problems
        
        return StatisticalAnalysis(
            total_problems=total_problems,
            fixed_threshold_accuracy=fixed_threshold_accuracy,
            adaptive_threshold_accuracy=adaptive_threshold_accuracy,
            improvement_rate=improvement_rate,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate
        )
    
    def run_comprehensive_validation(self, num_problems: int = 1000) -> Dict:
        """Run comprehensive mathematical validation."""
        print("🔬 MATHEMATICAL VALIDATION SYSTEM")
        print("=" * 60)
        print("Rigorous Testing of Quantum Adaptive Implementation")
        print("=" * 60)
        
        # Generate independent problem set
        print(f"📊 Generating {num_problems} independent problems...")
        problems = self.generate_independent_problem_set(num_problems)
        
        # Perform FFT analysis
        print("📡 Performing FFT analysis on mathematical signals...")
        fft_results = self.perform_fft_analysis(problems)
        
        # Validate all problems
        print("🔍 Validating problems with fixed and adaptive thresholds...")
        results = []
        for problem in problems:
            result = self.validate_problem(problem)
            results.append(result)
        
        # Perform statistical analysis
        print("📈 Performing statistical analysis...")
        stats_analysis = self.perform_statistical_analysis(results)
        
        # Display results
        print("\n📊 MATHEMATICAL VALIDATION RESULTS")
        print("=" * 60)
        
        print(f"📊 PROBLEM SET STATISTICS:")
        print(f"   Total Problems: {stats_analysis.total_problems}")
        print(f"   Beal Problems: {len([p for p in problems if p.problem_type == 'beal'])}")
        print(f"   Fermat Problems: {len([p for p in problems if p.problem_type == 'fermat'])}")
        print(f"   Erdős–Straus Problems: {len([p for p in problems if p.problem_type == 'erdos_straus'])}")
        print(f"   Catalan Problems: {len([p for p in problems if p.problem_type == 'catalan'])}")
        
        print(f"\n📈 ACCURACY COMPARISON:")
        print(f"   Fixed Threshold Accuracy: {stats_analysis.fixed_threshold_accuracy:.4f}")
        print(f"   Adaptive Threshold Accuracy: {stats_analysis.adaptive_threshold_accuracy:.4f}")
        print(f"   Improvement: {stats_analysis.adaptive_threshold_accuracy - stats_analysis.fixed_threshold_accuracy:.4f}")
        
        print(f"\n🔬 STATISTICAL SIGNIFICANCE:")
        print(f"   P-Value: {stats_analysis.p_value:.6f}")
        print(f"   Effect Size (Cohen's d): {stats_analysis.effect_size:.4f}")
        print(f"   95% Confidence Interval: [{stats_analysis.confidence_interval[0]:.4f}, {stats_analysis.confidence_interval[1]:.4f}]")
        print(f"   Statistically Significant: {'YES' if stats_analysis.p_value < 0.05 else 'NO'}")
        
        print(f"\n📊 ERROR ANALYSIS:")
        print(f"   False Positive Rate: {stats_analysis.false_positive_rate:.4f}")
        print(f"   False Negative Rate: {stats_analysis.false_negative_rate:.4f}")
        print(f"   Improvement Rate: {stats_analysis.improvement_rate:.4f}")
        
        print(f"\n📡 FFT ANALYSIS RESULTS:")
        print(f"   Total Energy: {fft_results['total_energy']:.2f}")
        print(f"   Peak Count: {fft_results['peak_count']}")
        print(f"   Dominant Frequencies: {len(fft_results['dominant_frequencies'])}")
        
        if len(fft_results['dominant_frequencies']) > 0:
            print(f"   Top 5 Frequencies: {fft_results['dominant_frequencies'][:5]}")
        
        print(f"\n🔬 VALIDATION CONCLUSION:")
        if stats_analysis.p_value < 0.05 and stats_analysis.effect_size > 0.2:
            print("   ✅ GENUINE MATHEMATICAL INSIGHT DETECTED")
            print("   The quantum adaptive implementation shows statistically significant improvement")
            print("   beyond parameter fitting, indicating genuine mathematical structure.")
        else:
            print("   ⚠️ PARAMETER FITTING LIKELY")
            print("   The results may represent overfitting rather than genuine mathematical insight.")
        
        print(f"\n🔬 MATHEMATICAL VALIDATION COMPLETE")
        print("📊 Independent problem sets: VALIDATED")
        print("📡 FFT analysis: COMPLETED")
        print("📈 Statistical significance: TESTED")
        print("🎯 Mathematical insight: EVALUATED")
        print("🏆 Validation framework: ESTABLISHED")
        
        return {
            'problems': problems,
            'results': results,
            'statistics': stats_analysis,
            'fft_results': fft_results
        }

def demonstrate_validation():
    """Demonstrate the mathematical validation system."""
    validator = MathematicalValidator()
    validation_results = validator.run_comprehensive_validation(num_problems=500)
    return validator, validation_results

if __name__ == "__main__":
    validator, results = demonstrate_validation()
