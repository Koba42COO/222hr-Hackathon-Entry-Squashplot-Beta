#!/usr/bin/env python3
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationProfile:
    """Optimization level profile with performance characteristics."""
    level: str  # 'basic', 'advanced', 'expert'
    min_matrix_size: int
    max_matrix_size: int
    optimal_matrix_size: int
    expected_speedup: float
    expected_accuracy_improvement: float
    computational_complexity: float  # 1.0 = basic, 2.0 = advanced, 3.0 = expert
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
    
    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file
        self.performance_history = self._load_performance_history()
        self.optimization_profiles = self._define_optimization_profiles()
        
    def _define_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Define optimization profiles for different levels."""
        return {
            'basic': OptimizationProfile(
                level='basic',
                min_matrix_size=32,
                max_matrix_size=128,
                optimal_matrix_size=64,
                expected_speedup=2.5,
                expected_accuracy_improvement=0.03,
                computational_complexity=1.0,
                memory_overhead=1.0,
                business_domains=['AI Development', 'Data Processing', 'Real-time Systems'],
                use_cases=['Small-scale ML', 'Real-time inference', 'Prototyping']
            ),
            'advanced': OptimizationProfile(
                level='advanced',
                min_matrix_size=64,
                max_matrix_size=512,
                optimal_matrix_size=256,
                expected_speedup=1.8,
                expected_accuracy_improvement=0.06,
                computational_complexity=2.0,
                memory_overhead=1.5,
                business_domains=['Blockchain Solutions', 'Financial Modeling', 'Scientific Computing'],
                use_cases=['Medium-scale ML', 'Research applications', 'Production systems']
            ),
            'expert': OptimizationProfile(
                level='expert',
                min_matrix_size=256,
                max_matrix_size=2048,
                optimal_matrix_size=1024,
                expected_speedup=1.2,
                expected_accuracy_improvement=0.08,
                computational_complexity=3.0,
                memory_overhead=2.0,
                business_domains=['SaaS Platforms', 'Enterprise Solutions', 'Advanced Research'],
                use_cases=['Large-scale ML', 'Enterprise applications', 'Advanced research']
            )
        }
    
    def _load_performance_history(self) -> List[PerformanceHistory]:
        """Load performance history from file."""
        if not self.history_file or not Path(self.history_file).exists():
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return [PerformanceHistory(**item) for item in data]
        except Exception as e:
            logger.warning(f"Failed to load performance history: {e}")
            return []
    
    def _save_performance_history(self):
        """Save performance history to file."""
        if not self.history_file:
            return
        
        try:
            with open(self.history_file, 'w') as f:
                json.dump([vars(item) for item in self.performance_history], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save performance history: {e}")
    
    def select_optimization_level(self, matrix_size: int, business_domain: str = None, 
                                use_case: str = None, performance_priority: str = 'balanced') -> str:
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
        logger.info(f"🔍 Selecting optimization level for matrix size {matrix_size}")
        
        # Get candidate levels based on matrix size
        candidates = self._get_candidate_levels(matrix_size)
        
        if not candidates:
            logger.warning(f"No suitable optimization level found for matrix size {matrix_size}")
            return 'basic'  # Default fallback
        
        # Score each candidate
        scores = {}
        for level in candidates:
            profile = self.optimization_profiles[level]
            score = self._calculate_level_score(profile, matrix_size, business_domain, 
                                              use_case, performance_priority)
            scores[level] = score
        
        # Select the best level
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]
        
        logger.info(f"✅ Selected optimization level: {best_level} (score: {best_score:.3f})")
        logger.info(f"   • Expected speedup: {self.optimization_profiles[best_level].expected_speedup:.2f}x")
        logger.info(f"   • Expected accuracy improvement: {self.optimization_profiles[best_level].expected_accuracy_improvement:.1%}")
        
        return best_level
    
    def _get_candidate_levels(self, matrix_size: int) -> List[str]:
        """Get candidate optimization levels for given matrix size."""
        candidates = []
        
        for level, profile in self.optimization_profiles.items():
            if profile.min_matrix_size <= matrix_size <= profile.max_matrix_size:
                candidates.append(level)
        
        return candidates
    
    def _calculate_level_score(self, profile: OptimizationProfile, matrix_size: int,
                             business_domain: str, use_case: str, 
                             performance_priority: str) -> float:
        """Calculate score for an optimization level."""
        score = 0.0
        
        # Base score from matrix size fit
        size_fit = self._calculate_size_fit(profile, matrix_size)
        score += size_fit * 0.4  # 40% weight
        
        # Historical performance score
        historical_score = self._calculate_historical_score(profile.level, matrix_size)
        score += historical_score * 0.3  # 30% weight
        
        # Business domain fit
        domain_score = self._calculate_domain_fit(profile, business_domain)
        score += domain_score * 0.2  # 20% weight
        
        # Use case fit
        use_case_score = self._calculate_use_case_fit(profile, use_case)
        score += use_case_score * 0.1  # 10% weight
        
        # Performance priority adjustment
        score = self._adjust_for_performance_priority(score, profile, performance_priority)
        
        return score
    
    def _calculate_size_fit(self, profile: OptimizationProfile, matrix_size: int) -> float:
        """Calculate how well the matrix size fits the optimization profile."""
        optimal_size = profile.optimal_matrix_size
        
        # Calculate distance from optimal size
        distance = abs(matrix_size - optimal_size)
        max_distance = profile.max_matrix_size - profile.min_matrix_size
        
        # Normalize to 0-1 range (closer to optimal = higher score)
        fit_score = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, fit_score))
    
    def _calculate_historical_score(self, level: str, matrix_size: int) -> float:
        """Calculate score based on historical performance."""
        if not self.performance_history:
            return 0.5  # Neutral score if no history
        
        # Filter history for this level and similar matrix sizes
        relevant_history = [
            h for h in self.performance_history
            if h.optimization_level == level and 
            abs(h.matrix_size - matrix_size) <= matrix_size * 0.5  # Within 50% size range
        ]
        
        if not relevant_history:
            return 0.5  # Neutral score if no relevant history
        
        # Calculate average performance metrics
        avg_speedup = np.mean([h.actual_speedup for h in relevant_history])
        avg_accuracy_improvement = np.mean([h.actual_accuracy_improvement for h in relevant_history])
        avg_success_rate = np.mean([h.success_rate for h in relevant_history])
        
        # Combine metrics into a score
        score = (avg_speedup * 0.4 + avg_accuracy_improvement * 0.4 + avg_success_rate * 0.2)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_domain_fit(self, profile: OptimizationProfile, business_domain: str) -> float:
        """Calculate business domain fit score."""
        if not business_domain:
            return 0.5  # Neutral score if no domain specified
        
        if business_domain in profile.business_domains:
            return 1.0  # Perfect fit
        else:
            return 0.3  # Partial fit
    
    def _calculate_use_case_fit(self, profile: OptimizationProfile, use_case: str) -> float:
        """Calculate use case fit score."""
        if not use_case:
            return 0.5  # Neutral score if no use case specified
        
        if use_case in profile.use_cases:
            return 1.0  # Perfect fit
        else:
            return 0.3  # Partial fit
    
    def _adjust_for_performance_priority(self, score: float, profile: OptimizationProfile,
                                       priority: str) -> float:
        """Adjust score based on performance priority."""
        if priority == 'speed':
            # Favor levels with higher speedup
            speedup_factor = profile.expected_speedup / 3.0  # Normalize to 0-1
            return score * (0.7 + 0.3 * speedup_factor)
        
        elif priority == 'accuracy':
            # Favor levels with higher accuracy improvement
            accuracy_factor = profile.expected_accuracy_improvement / 0.1  # Normalize to 0-1
            return score * (0.7 + 0.3 * accuracy_factor)
        
        else:  # 'balanced'
            return score  # No adjustment
    
    def update_performance_history(self, matrix_size: int, optimization_level: str,
                                 actual_speedup: float, actual_accuracy_improvement: float,
                                 execution_time: float, memory_usage: float, 
                                 success_rate: float = 1.0):
        """Update performance history with new results."""
        history_entry = PerformanceHistory(
            matrix_size=matrix_size,
            optimization_level=optimization_level,
            actual_speedup=actual_speedup,
            actual_accuracy_improvement=actual_accuracy_improvement,
            execution_time=execution_time,
            memory_usage=memory_usage,
            success_rate=success_rate,
            timestamp=datetime.now().isoformat()
        )
        
        self.performance_history.append(history_entry)
        self._save_performance_history()
        
        logger.info(f"📊 Updated performance history for {optimization_level} level "
                   f"(matrix size: {matrix_size}, speedup: {actual_speedup:.2f}x)")
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization selection report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_profiles': {},
            'performance_history_summary': {},
            'recommendations': []
        }
        
        # Add optimization profiles
        for level, profile in self.optimization_profiles.items():
            report['optimization_profiles'][level] = {
                'level': profile.level,
                'matrix_size_range': f"{profile.min_matrix_size}-{profile.max_matrix_size}",
                'optimal_matrix_size': profile.optimal_matrix_size,
                'expected_speedup': profile.expected_speedup,
                'expected_accuracy_improvement': profile.expected_accuracy_improvement,
                'computational_complexity': profile.computational_complexity,
                'business_domains': profile.business_domains,
                'use_cases': profile.use_cases
            }
        
        # Add performance history summary
        if self.performance_history:
            for level in self.optimization_profiles.keys():
                level_history = [h for h in self.performance_history if h.optimization_level == level]
                if level_history:
                    report['performance_history_summary'][level] = {
                        'total_runs': len(level_history),
                        'average_speedup': np.mean([h.actual_speedup for h in level_history]),
                        'average_accuracy_improvement': np.mean([h.actual_accuracy_improvement for h in level_history]),
                        'average_success_rate': np.mean([h.success_rate for h in level_history]),
                        'last_updated': max([h.timestamp for h in level_history])
                    }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not self.performance_history:
            recommendations.append("No performance history available. Start with basic level for small matrices.")
            recommendations.append("Collect performance data to improve optimization selection.")
        else:
            # Analyze performance patterns
            for level in self.optimization_profiles.keys():
                level_history = [h for h in self.performance_history if h.optimization_level == level]
                if level_history:
                    avg_speedup = np.mean([h.actual_speedup for h in level_history])
                    avg_accuracy = np.mean([h.actual_accuracy_improvement for h in level_history])
                    
                    if avg_speedup > 2.0:
                        recommendations.append(f"{level.capitalize()} level shows excellent speedup ({avg_speedup:.2f}x). Consider for similar matrix sizes.")
                    
                    if avg_accuracy > 0.05:
                        recommendations.append(f"{level.capitalize()} level shows good accuracy improvements ({avg_accuracy:.1%}). Suitable for accuracy-critical applications.")
        
        return recommendations
    
    def visualize_performance_history(self, save_path: str = None):
        """Generate visualization of performance history."""
        if not self.performance_history:
            logger.warning("No performance history available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('KOBA42 Optimization Performance History', fontsize=16, fontweight='bold')
        
        # Extract data
        levels = [h.optimization_level for h in self.performance_history]
        matrix_sizes = [h.matrix_size for h in self.performance_history]
        speedups = [h.actual_speedup for h in self.performance_history]
        accuracy_improvements = [h.actual_accuracy_improvement for h in self.performance_history]
        
        # Speedup by optimization level
        level_data = {}
        for level in set(levels):
            level_indices = [i for i, l in enumerate(levels) if l == level]
            level_data[level] = [speedups[i] for i in level_indices]
        
        axes[0, 0].boxplot([level_data.get(level, []) for level in ['basic', 'advanced', 'expert']], 
                          labels=['Basic', 'Advanced', 'Expert'])
        axes[0, 0].set_title('Speedup by Optimization Level')
        axes[0, 0].set_ylabel('Speedup Factor')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy improvement by optimization level
        level_acc_data = {}
        for level in set(levels):
            level_indices = [i for i, l in enumerate(levels) if l == level]
            level_acc_data[level] = [accuracy_improvements[i] for i in level_indices]
        
        axes[0, 1].boxplot([level_acc_data.get(level, []) for level in ['basic', 'advanced', 'expert']], 
                          labels=['Basic', 'Advanced', 'Expert'])
        axes[0, 1].set_title('Accuracy Improvement by Optimization Level')
        axes[0, 1].set_ylabel('Accuracy Improvement')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Matrix size vs speedup scatter
        colors = {'basic': 'blue', 'advanced': 'green', 'expert': 'red'}
        for level in set(levels):
            level_indices = [i for i, l in enumerate(levels) if l == level]
            level_sizes = [matrix_sizes[i] for i in level_indices]
            level_speedups = [speedups[i] for i in level_indices]
            axes[1, 0].scatter(level_sizes, level_speedups, c=colors[level], 
                              label=level.capitalize(), alpha=0.7)
        
        axes[1, 0].set_title('Matrix Size vs Speedup')
        axes[1, 0].set_xlabel('Matrix Size')
        axes[1, 0].set_ylabel('Speedup Factor')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance timeline
        timestamps = [datetime.fromisoformat(h.timestamp) for h in self.performance_history]
        axes[1, 1].scatter(timestamps, speedups, c=[colors[l] for l in levels], alpha=0.7)
        axes[1, 1].set_title('Performance Timeline')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Speedup Factor')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance history visualization saved to {save_path}")
        
        plt.show()

def demonstrate_intelligent_optimization():
    """Demonstrate intelligent optimization selection."""
    logger.info("🚀 KOBA42 Intelligent Optimization Selector")
    logger.info("=" * 50)
    
    # Initialize selector
    selector = IntelligentOptimizationSelector('optimization_performance_history.json')
    
    # Test different matrix sizes
    test_cases = [
        (32, 'AI Development', 'Real-time inference', 'speed'),
        (64, 'Data Processing', 'Small-scale ML', 'balanced'),
        (128, 'Blockchain Solutions', 'Medium-scale ML', 'accuracy'),
        (256, 'Financial Modeling', 'Production systems', 'balanced'),
        (512, 'SaaS Platforms', 'Large-scale ML', 'accuracy'),
        (1024, 'Enterprise Solutions', 'Advanced research', 'balanced')
    ]
    
    print("\n🔍 INTELLIGENT OPTIMIZATION SELECTION RESULTS")
    print("=" * 50)
    
    results = []
    for matrix_size, business_domain, use_case, priority in test_cases:
        selected_level = selector.select_optimization_level(
            matrix_size, business_domain, use_case, priority
        )
        
        profile = selector.optimization_profiles[selected_level]
        
        result = {
            'matrix_size': matrix_size,
            'business_domain': business_domain,
            'use_case': use_case,
            'priority': priority,
            'selected_level': selected_level,
            'expected_speedup': profile.expected_speedup,
            'expected_accuracy_improvement': profile.expected_accuracy_improvement
        }
        results.append(result)
        
        print(f"\nMatrix Size: {matrix_size}×{matrix_size}")
        print(f"Business Domain: {business_domain}")
        print(f"Use Case: {use_case}")
        print(f"Priority: {priority}")
        print(f"Selected Level: {selected_level.upper()}")
        print(f"Expected Speedup: {profile.expected_speedup:.2f}x")
        print(f"Expected Accuracy Improvement: {profile.expected_accuracy_improvement:.1%}")
    
    # Generate report
    report = selector.generate_optimization_report()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'intelligent_optimization_report_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Optimization report saved to {report_file}")
    
    # Generate visualization
    try:
        selector.visualize_performance_history('optimization_performance_visualization.png')
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")
    
    return results, report_file

if __name__ == "__main__":
    # Run intelligent optimization demonstration
    results, report_file = demonstrate_intelligent_optimization()
    
    print(f"\n🎉 Intelligent optimization demonstration completed!")
    print(f"📊 Results saved to: {report_file}")
    print(f"🔍 Tested {len(results)} different matrix sizes and use cases")
