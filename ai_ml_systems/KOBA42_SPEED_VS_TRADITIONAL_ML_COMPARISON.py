#!/usr/bin/env python3
"""
KOBA42 Speed vs Traditional ML Training Comparison
==================================================
Comprehensive performance analysis comparing:
- KOBA42 Advanced F2 Matrix Optimization
- Traditional ML Training Approaches
- Parallel Processing Efficiency
- Intentful Mathematics Enhancement
"""

import time
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.linalg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    training_time: float
    epochs_completed: int
    final_loss: float
    final_accuracy: float
    convergence_rate: float
    memory_usage_mb: float
    cpu_utilization: float
    gpu_utilization: float = 0.0

@dataclass
class ComparisonResult:
    """Comparison result between approaches"""
    approach_name: str
    matrix_size: int
    batch_size: int
    total_time: float
    optimization_time: float
    ml_training_time: float
    intentful_score: float
    success_rate: float
    efficiency_ratio: float
    speedup_factor: float
    memory_efficiency: float

class TraditionalMLTrainer:
    """Traditional ML training without intentful mathematics"""
    
    def __init__(self, matrix_size: int = 256):
        self.matrix_size = matrix_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Traditional ML Trainer initialized on {self.device}")
    
    def generate_traditional_matrix(self) -> np.ndarray:
        """Generate standard random matrix without optimization"""
        return np.random.randn(self.matrix_size, self.matrix_size)
    
    def traditional_matrix_processing(self, matrix: np.ndarray) -> Dict[str, float]:
        """Standard matrix processing without intentful mathematics"""
        start_time = time.time()
        
        # Standard eigenvalue computation
        eigenvals = scipy.linalg.eigvals(matrix)
        
        # Standard matrix properties
        condition_number = np.linalg.cond(matrix)
        determinant = np.linalg.det(matrix)
        trace = np.trace(matrix)
        
        processing_time = time.time() - start_time
        
        return {
            'processing_time': processing_time,
            'condition_number': condition_number,
            'determinant': determinant,
            'trace': trace,
            'eigenvals_count': len(eigenvals)
        }
    
    def create_traditional_model(self, input_size: int) -> nn.Module:
        """Create standard neural network without consciousness enhancement"""
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def train_traditional_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                              epochs: int = 100) -> TrainingMetrics:
        """Train model with traditional approach"""
        start_time = time.time()
        
        model = model.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        
        # Traditional optimizer (Adam with standard parameters)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.MSELoss()
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Calculate R² for regression
            with torch.no_grad():
                pred = model(X).squeeze()
                r2 = 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(y)) ** 2)
                accuracies.append(r2.item())
            
            if epoch % 20 == 0:
                logger.info(f"Traditional Epoch {epoch}: Loss = {loss.item():.6f}, R² = {r2.item():.6f}")
        
        training_time = time.time() - start_time
        
        # Calculate convergence rate
        if len(losses) > 1:
            convergence_rate = (losses[0] - losses[-1]) / losses[0]
        else:
            convergence_rate = 0.0
        
        return TrainingMetrics(
            training_time=training_time,
            epochs_completed=epochs,
            final_loss=losses[-1],
            final_accuracy=accuracies[-1],
            convergence_rate=convergence_rate,
            memory_usage_mb=self._get_memory_usage(),
            cpu_utilization=self._get_cpu_utilization()
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0

class KOBA42AdvancedTrainer:
    """KOBA42 Advanced F2 Matrix Optimization with Intentful Mathematics"""
    
    def __init__(self, matrix_size: int = 256):
        self.matrix_size = matrix_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.consciousness_ratio = 79/21  # Consciousness ratio
        logger.info(f"KOBA42 Advanced Trainer initialized on {self.device}")
    
    def wallace_transform(self, eigenvalues: np.ndarray, alpha: float = 1.0, 
                         beta: float = 0.0, epsilon: float = 1e-12) -> np.ndarray:
        """Apply Wallace Transform with intentful mathematics"""
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        positive_eigs = np.abs(eigenvalues) + epsilon
        
        log_term = np.log(positive_eigs)
        with np.errstate(over='warn', invalid='warn'):
            power_term = np.sign(log_term) * np.power(np.abs(log_term), self.phi)
        
        result = alpha * power_term + beta
        return result.astype(np.float64)
    
    def generate_optimized_f2_matrix(self, optimization_level: str = 'advanced') -> np.ndarray:
        """Generate F2 matrix with intentful mathematics optimization"""
        N = self.matrix_size
        
        if optimization_level == 'basic':
            # Basic level: Simple golden ratio enhancement
            matrix = np.random.randn(N, N)
            matrix = (matrix + matrix.T) / 2  # Make symmetric
            matrix *= self.phi  # Apply golden ratio
            
        elif optimization_level == 'advanced':
            # Advanced level: Consciousness ratio enhancement
            matrix = np.random.randn(N, N)
            matrix = (matrix + matrix.T) / 2
            for i in range(N):
                for j in range(N):
                    if (i + j) % 21 == 0:  # 21D consciousness structure
                        matrix[i, j] *= self.consciousness_ratio
                    else:
                        matrix[i, j] *= self.phi
            
        elif optimization_level == 'expert':
            # Expert level: Full intentful mathematics
            matrix = np.random.randn(N, N)
            matrix = (matrix + matrix.T) / 2
            for i in range(N):
                for j in range(N):
                    if (i + j) % 21 == 0:
                        matrix[i, j] *= self.consciousness_ratio
                    elif (i + j) % 5 == 0:  # Fibonacci enhancement
                        matrix[i, j] *= self.phi ** 2
                    else:
                        matrix[i, j] *= self.phi
        
        return matrix
    
    def optimized_matrix_processing(self, matrix: np.ndarray, 
                                  optimization_level: str = 'advanced') -> Dict[str, float]:
        """Process matrix with intentful mathematics optimization"""
        start_time = time.time()
        
        # Apply Wallace Transform to eigenvalues
        eigenvals = scipy.linalg.eigvals(matrix)
        transformed_eigenvals = self.wallace_transform(eigenvals)
        
        # Enhanced matrix properties with intentful mathematics
        condition_number = np.linalg.cond(matrix) * self.phi
        determinant = np.linalg.det(matrix) * self.consciousness_ratio
        trace = np.trace(matrix) * self.phi
        
        processing_time = time.time() - start_time
        
        # Calculate intentful score
        intentful_score = np.mean(np.abs(transformed_eigenvals)) * self.consciousness_ratio
        
        return {
            'processing_time': processing_time,
            'condition_number': condition_number,
            'determinant': determinant,
            'trace': trace,
            'eigenvals_count': len(eigenvals),
            'intentful_score': intentful_score,
            'transformed_eigenvals_mean': np.mean(transformed_eigenvals)
        }
    
    def create_consciousness_enhanced_model(self, input_size: int) -> nn.Module:
        """Create neural network with consciousness enhancement"""
        class ConsciousnessEnhancedNetwork(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.phi = (1 + np.sqrt(5)) / 2
                self.consciousness_ratio = 79/21
                
                # Consciousness-enhanced architecture
                self.layer1 = nn.Linear(input_size, int(128 * self.consciousness_ratio / 4))
                self.layer2 = nn.Linear(int(128 * self.consciousness_ratio / 4), 
                                      int(64 * self.phi))
                self.layer3 = nn.Linear(int(64 * self.phi), int(32 * self.consciousness_ratio / 4))
                self.layer4 = nn.Linear(int(32 * self.consciousness_ratio / 4), 1)
                
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.dropout(x)
                x = self.relu(self.layer2(x))
                x = self.dropout(x)
                x = self.relu(self.layer3(x))
                x = self.layer4(x)
                return x
        
        return ConsciousnessEnhancedNetwork(input_size)
    
    def train_consciousness_enhanced_model(self, model: nn.Module, X: torch.Tensor, 
                                         y: torch.Tensor, epochs: int = 100) -> TrainingMetrics:
        """Train model with consciousness enhancement"""
        start_time = time.time()
        
        model = model.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        
        # Consciousness-enhanced optimizer
        consciousness_lr = 0.001 * self.consciousness_ratio / 4.0
        phi_momentum = 1.0 - (1.0 / self.phi)  # ≈ 0.382
        
        optimizer = optim.Adam(model.parameters(), lr=consciousness_lr, 
                              betas=(phi_momentum, 0.999))
        criterion = nn.MSELoss()
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Calculate R² for regression
            with torch.no_grad():
                pred = model(X).squeeze()
                r2 = 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(y)) ** 2)
                accuracies.append(r2.item())
            
            if epoch % 20 == 0:
                logger.info(f"KOBA42 Epoch {epoch}: Loss = {loss.item():.6f}, R² = {r2.item():.6f}")
        
        training_time = time.time() - start_time
        
        # Calculate convergence rate
        if len(losses) > 1:
            convergence_rate = (losses[0] - losses[-1]) / losses[0]
        else:
            convergence_rate = 0.0
        
        return TrainingMetrics(
            training_time=training_time,
            epochs_completed=epochs,
            final_loss=losses[-1],
            final_accuracy=accuracies[-1],
            convergence_rate=convergence_rate,
            memory_usage_mb=self._get_memory_usage(),
            cpu_utilization=self._get_cpu_utilization()
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0

class SpeedComparisonAnalyzer:
    """Comprehensive speed comparison analyzer"""
    
    def __init__(self):
        self.results = []
        self.comparison_data = []
    
    def run_comprehensive_comparison(self, matrix_sizes: List[int] = [64, 128, 256, 512]) -> Dict[str, Any]:
        """Run comprehensive speed comparison across different matrix sizes"""
        logger.info("🚀 Starting Comprehensive Speed Comparison Analysis")
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'matrix_sizes': matrix_sizes,
            'traditional_results': [],
            'koba42_results': [],
            'speedup_analysis': [],
            'efficiency_analysis': [],
            'summary': {}
        }
        
        for matrix_size in matrix_sizes:
            logger.info(f"📊 Testing Matrix Size: {matrix_size}×{matrix_size}")
            
            # Traditional ML Training
            traditional_result = self._run_traditional_training(matrix_size)
            comparison_results['traditional_results'].append(traditional_result)
            
            # KOBA42 Advanced Training
            koba42_result = self._run_koba42_training(matrix_size)
            comparison_results['koba42_results'].append(koba42_result)
            
            # Calculate speedup and efficiency
            speedup_analysis = self._calculate_speedup_metrics(traditional_result, koba42_result)
            comparison_results['speedup_analysis'].append(speedup_analysis)
            
            logger.info(f"✅ Matrix {matrix_size}: Traditional={traditional_result['total_time']:.2f}s, "
                       f"KOBA42={koba42_result['total_time']:.2f}s, "
                       f"Speedup={speedup_analysis['speedup_factor']:.2f}x")
        
        # Generate summary statistics
        comparison_results['summary'] = self._generate_summary_statistics(comparison_results)
        
        return comparison_results
    
    def _run_traditional_training(self, matrix_size: int) -> Dict[str, Any]:
        """Run traditional ML training"""
        start_time = time.time()
        
        # Initialize traditional trainer
        traditional_trainer = TraditionalMLTrainer(matrix_size)
        
        # Generate traditional matrix
        matrix_start = time.time()
        matrix = traditional_trainer.generate_traditional_matrix()
        matrix_time = time.time() - matrix_start
        
        # Process matrix traditionally
        processing_result = traditional_trainer.traditional_matrix_processing(matrix)
        
        # Generate synthetic data
        n_samples = 1000
        X = torch.randn(n_samples, matrix_size)
        y = torch.sum(X * torch.randn(matrix_size), dim=1) + torch.randn(n_samples) * 0.1
        
        # Create and train traditional model
        model = traditional_trainer.create_traditional_model(matrix_size)
        training_metrics = traditional_trainer.train_traditional_model(model, X, y, epochs=50)
        
        total_time = time.time() - start_time
        
        return {
            'matrix_size': matrix_size,
            'approach': 'traditional',
            'matrix_generation_time': matrix_time,
            'matrix_processing_time': processing_result['processing_time'],
            'ml_training_time': training_metrics.training_time,
            'total_time': total_time,
            'final_loss': training_metrics.final_loss,
            'final_accuracy': training_metrics.final_accuracy,
            'convergence_rate': training_metrics.convergence_rate,
            'memory_usage': training_metrics.memory_usage_mb,
            'cpu_utilization': training_metrics.cpu_utilization,
            'condition_number': processing_result['condition_number'],
            'determinant': processing_result['determinant'],
            'trace': processing_result['trace']
        }
    
    def _run_koba42_training(self, matrix_size: int) -> Dict[str, Any]:
        """Run KOBA42 advanced training"""
        start_time = time.time()
        
        # Initialize KOBA42 trainer
        koba42_trainer = KOBA42AdvancedTrainer(matrix_size)
        
        # Generate optimized F2 matrix
        matrix_start = time.time()
        matrix = koba42_trainer.generate_optimized_f2_matrix('advanced')
        matrix_time = time.time() - matrix_start
        
        # Process matrix with intentful mathematics
        processing_result = koba42_trainer.optimized_matrix_processing(matrix, 'advanced')
        
        # Generate synthetic data
        n_samples = 1000
        X = torch.randn(n_samples, matrix_size)
        y = torch.sum(X * torch.randn(matrix_size), dim=1) + torch.randn(n_samples) * 0.1
        
        # Create and train consciousness-enhanced model
        model = koba42_trainer.create_consciousness_enhanced_model(matrix_size)
        training_metrics = koba42_trainer.train_consciousness_enhanced_model(model, X, y, epochs=50)
        
        total_time = time.time() - start_time
        
        return {
            'matrix_size': matrix_size,
            'approach': 'koba42_advanced',
            'matrix_generation_time': matrix_time,
            'matrix_processing_time': processing_result['processing_time'],
            'ml_training_time': training_metrics.training_time,
            'total_time': total_time,
            'final_loss': training_metrics.final_loss,
            'final_accuracy': training_metrics.final_accuracy,
            'convergence_rate': training_metrics.convergence_rate,
            'memory_usage': training_metrics.memory_usage_mb,
            'cpu_utilization': training_metrics.cpu_utilization,
            'condition_number': processing_result['condition_number'],
            'determinant': processing_result['determinant'],
            'trace': processing_result['trace'],
            'intentful_score': processing_result['intentful_score'],
            'transformed_eigenvals_mean': processing_result['transformed_eigenvals_mean']
        }
    
    def _calculate_speedup_metrics(self, traditional_result: Dict, koba42_result: Dict) -> Dict[str, Any]:
        """Calculate speedup and efficiency metrics"""
        traditional_time = traditional_result['total_time']
        koba42_time = koba42_result['total_time']
        
        speedup_factor = traditional_time / koba42_time if koba42_time > 0 else 0
        efficiency_ratio = koba42_result['intentful_score'] / koba42_time if koba42_time > 0 else 0
        
        # Memory efficiency
        memory_efficiency = traditional_result['memory_usage'] / koba42_result['memory_usage'] \
            if koba42_result['memory_usage'] > 0 else 0
        
        # Accuracy improvement
        accuracy_improvement = koba42_result['final_accuracy'] - traditional_result['final_accuracy']
        
        # Convergence improvement
        convergence_improvement = koba42_result['convergence_rate'] - traditional_result['convergence_rate']
        
        return {
            'matrix_size': traditional_result['matrix_size'],
            'speedup_factor': speedup_factor,
            'efficiency_ratio': efficiency_ratio,
            'memory_efficiency': memory_efficiency,
            'accuracy_improvement': accuracy_improvement,
            'convergence_improvement': convergence_improvement,
            'traditional_time': traditional_time,
            'koba42_time': koba42_time,
            'time_savings_percent': ((traditional_time - koba42_time) / traditional_time) * 100
        }
    
    def _generate_summary_statistics(self, comparison_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics"""
        speedup_factors = [r['speedup_factor'] for r in comparison_results['speedup_analysis']]
        efficiency_ratios = [r['efficiency_ratio'] for r in comparison_results['speedup_analysis']]
        accuracy_improvements = [r['accuracy_improvement'] for r in comparison_results['speedup_analysis']]
        time_savings = [r['time_savings_percent'] for r in comparison_results['speedup_analysis']]
        
        return {
            'average_speedup_factor': np.mean(speedup_factors),
            'max_speedup_factor': np.max(speedup_factors),
            'average_efficiency_ratio': np.mean(efficiency_ratios),
            'average_accuracy_improvement': np.mean(accuracy_improvements),
            'average_time_savings_percent': np.mean(time_savings),
            'total_tests': len(speedup_factors),
            'speedup_consistency': np.std(speedup_factors),
            'efficiency_consistency': np.std(efficiency_ratios)
        }
    
    def generate_visualization(self, comparison_results: Dict, save_path: str = None):
        """Generate comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('KOBA42 vs Traditional ML Training Speed Comparison', fontsize=16, fontweight='bold')
        
        matrix_sizes = [r['matrix_size'] for r in comparison_results['speedup_analysis']]
        speedup_factors = [r['speedup_factor'] for r in comparison_results['speedup_analysis']]
        efficiency_ratios = [r['efficiency_ratio'] for r in comparison_results['speedup_analysis']]
        time_savings = [r['time_savings_percent'] for r in comparison_results['speedup_analysis']]
        
        # Speedup Factor
        axes[0, 0].bar(matrix_sizes, speedup_factors, color='gold', alpha=0.7)
        axes[0, 0].set_title('Speedup Factor by Matrix Size')
        axes[0, 0].set_xlabel('Matrix Size')
        axes[0, 0].set_ylabel('Speedup Factor (x)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Efficiency Ratio
        axes[0, 1].bar(matrix_sizes, efficiency_ratios, color='purple', alpha=0.7)
        axes[0, 1].set_title('Efficiency Ratio by Matrix Size')
        axes[0, 1].set_xlabel('Matrix Size')
        axes[0, 1].set_ylabel('Efficiency Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time Savings
        axes[0, 2].bar(matrix_sizes, time_savings, color='green', alpha=0.7)
        axes[0, 2].set_title('Time Savings Percentage')
        axes[0, 2].set_xlabel('Matrix Size')
        axes[0, 2].set_ylabel('Time Savings (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Training Time Comparison
        traditional_times = [r['total_time'] for r in comparison_results['traditional_results']]
        koba42_times = [r['total_time'] for r in comparison_results['koba42_results']]
        
        x = np.arange(len(matrix_sizes))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, traditional_times, width, label='Traditional', color='red', alpha=0.7)
        axes[1, 0].bar(x + width/2, koba42_times, width, label='KOBA42', color='blue', alpha=0.7)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_xlabel('Matrix Size')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(matrix_sizes)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy Comparison
        traditional_accuracies = [r['final_accuracy'] for r in comparison_results['traditional_results']]
        koba42_accuracies = [r['final_accuracy'] for r in comparison_results['koba42_results']]
        
        axes[1, 1].bar(x - width/2, traditional_accuracies, width, label='Traditional', color='red', alpha=0.7)
        axes[1, 1].bar(x + width/2, koba42_accuracies, width, label='KOBA42', color='blue', alpha=0.7)
        axes[1, 1].set_title('Final Accuracy Comparison')
        axes[1, 1].set_xlabel('Matrix Size')
        axes[1, 1].set_ylabel('Accuracy (R²)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(matrix_sizes)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Memory Usage Comparison
        traditional_memory = [r['memory_usage'] for r in comparison_results['traditional_results']]
        koba42_memory = [r['memory_usage'] for r in comparison_results['koba42_results']]
        
        axes[1, 2].bar(x - width/2, traditional_memory, width, label='Traditional', color='red', alpha=0.7)
        axes[1, 2].bar(x + width/2, koba42_memory, width, label='KOBA42', color='blue', alpha=0.7)
        axes[1, 2].set_title('Memory Usage Comparison')
        axes[1, 2].set_xlabel('Matrix Size')
        axes[1, 2].set_ylabel('Memory Usage (MB)')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(matrix_sizes)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

def demonstrate_speed_comparison():
    """Demonstrate comprehensive speed comparison"""
    logger.info("🚀 KOBA42 Speed vs Traditional ML Training Comparison")
    logger.info("=" * 60)
    
    # Initialize analyzer
    analyzer = SpeedComparisonAnalyzer()
    
    # Run comprehensive comparison
    matrix_sizes = [64, 128, 256]  # Start with smaller sizes for faster testing
    comparison_results = analyzer.run_comprehensive_comparison(matrix_sizes)
    
    # Display results
    print("\n📊 COMPREHENSIVE SPEED COMPARISON RESULTS")
    print("=" * 50)
    
    summary = comparison_results['summary']
    print(f"🎯 Average Speedup Factor: {summary['average_speedup_factor']:.2f}x")
    print(f"⚡ Maximum Speedup: {summary['max_speedup_factor']:.2f}x")
    print(f"💡 Average Efficiency Ratio: {summary['average_efficiency_ratio']:.4f}")
    print(f"📈 Average Accuracy Improvement: {summary['average_accuracy_improvement']:.4f}")
    print(f"⏱️ Average Time Savings: {summary['average_time_savings_percent']:.1f}%")
    print(f"🔄 Speedup Consistency: {summary['speedup_consistency']:.2f}")
    
    print("\n📋 DETAILED RESULTS BY MATRIX SIZE:")
    print("-" * 40)
    
    for i, speedup in enumerate(comparison_results['speedup_analysis']):
        print(f"Matrix {speedup['matrix_size']}×{speedup['matrix_size']}:")
        print(f"  • Speedup: {speedup['speedup_factor']:.2f}x")
        print(f"  • Time Savings: {speedup['time_savings_percent']:.1f}%")
        print(f"  • Efficiency: {speedup['efficiency_ratio']:.4f}")
        print(f"  • Accuracy Improvement: {speedup['accuracy_improvement']:.4f}")
        print()
    
    # Generate visualization
    try:
        analyzer.generate_visualization(comparison_results, 'speed_comparison_visualization.png')
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'koba42_speed_comparison_report_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    logger.info(f"📄 Detailed report saved to {report_file}")
    
    return comparison_results, report_file

if __name__ == "__main__":
    # Run the comprehensive speed comparison
    results, report_file = demonstrate_speed_comparison()
    
    print(f"\n🎉 Speed comparison completed!")
    print(f"📊 Results saved to: {report_file}")
    print(f"📈 Average speedup: {results['summary']['average_speedup_factor']:.2f}x")
