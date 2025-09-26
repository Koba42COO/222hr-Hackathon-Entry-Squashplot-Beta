#!/usr/bin/env python3
"""
KOBA42 ENHANCED SPEED COMPARISON
================================
Enhanced Speed Comparison with Intelligent Optimization Selection
===============================================================

Features:
1. Intelligent Optimization Level Selection
2. Matrix Size-Based Routing
3. Performance History Integration
4. Dynamic Optimization Profiles
5. Comprehensive Speed Analysis
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
import scipy.linalg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Import our intelligent optimization selector
from KOBA42_INTELLIGENT_OPTIMIZATION_SELECTOR import IntelligentOptimizationSelector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedTrainingMetrics:
    """Enhanced training performance metrics with optimization selection."""
    training_time: float
    epochs_completed: int
    final_loss: float
    final_accuracy: float
    convergence_rate: float
    memory_usage_mb: float
    cpu_utilization: float
    selected_optimization_level: str
    optimization_score: float
    expected_speedup: float
    actual_speedup: float
    speedup_accuracy: float  # How close actual is to expected

@dataclass
class EnhancedComparisonResult:
    """Enhanced comparison result with intelligent optimization."""
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
    selected_optimization_level: str
    optimization_score: float
    expected_vs_actual_speedup: float

class EnhancedTraditionalMLTrainer:
    """Enhanced traditional ML training with optimization selection."""
    
    def __init__(self, matrix_size: int = 256):
        self.matrix_size = matrix_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimization_selector = IntelligentOptimizationSelector()
        logger.info(f"Enhanced Traditional ML Trainer initialized on {self.device}")
    
    def generate_traditional_matrix(self) -> np.ndarray:
        """Generate standard random matrix without optimization."""
        return np.random.randn(self.matrix_size, self.matrix_size)
    
    def traditional_matrix_processing(self, matrix: np.ndarray) -> Dict[str, float]:
        """Standard matrix processing without intentful mathematics."""
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
        """Create standard neural network without consciousness enhancement."""
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
                              epochs: int = 100) -> EnhancedTrainingMetrics:
        """Train model with traditional approach."""
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
        
        # Get optimization selection info (for comparison)
        selected_level = self.optimization_selector.select_optimization_level(
            self.matrix_size, 'Traditional ML', 'Standard Training', 'balanced'
        )
        profile = self.optimization_selector.optimization_profiles[selected_level]
        
        return EnhancedTrainingMetrics(
            training_time=training_time,
            epochs_completed=epochs,
            final_loss=losses[-1],
            final_accuracy=accuracies[-1],
            convergence_rate=convergence_rate,
            memory_usage_mb=self._get_memory_usage(),
            cpu_utilization=self._get_cpu_utilization(),
            selected_optimization_level=selected_level,
            optimization_score=0.5,  # Neutral for traditional
            expected_speedup=1.0,  # Baseline
            actual_speedup=1.0,  # Baseline
            speedup_accuracy=1.0  # Perfect baseline
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization."""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0

class EnhancedKOBA42Trainer:
    """Enhanced KOBA42 Advanced F2 Matrix Optimization with Intelligent Selection."""
    
    def __init__(self, matrix_size: int = 256):
        self.matrix_size = matrix_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.consciousness_ratio = 79/21  # Consciousness ratio
        self.optimization_selector = IntelligentOptimizationSelector()
        logger.info(f"Enhanced KOBA42 Trainer initialized on {self.device}")
    
    def wallace_transform(self, eigenvalues: np.ndarray, alpha: float = 1.0, 
                         beta: float = 0.0, epsilon: float = 1e-12) -> np.ndarray:
        """Apply Wallace Transform with intentful mathematics."""
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        positive_eigs = np.abs(eigenvalues) + epsilon
        
        log_term = np.log(positive_eigs)
        with np.errstate(over='warn', invalid='warn'):
            power_term = np.sign(log_term) * np.power(np.abs(log_term), self.phi)
        
        result = alpha * power_term + beta
        return result.astype(np.float64)
    
    def generate_optimized_f2_matrix(self, optimization_level: str = None) -> np.ndarray:
        """Generate F2 matrix with intelligent optimization level selection."""
        # Use intelligent selector if no level specified
        if optimization_level is None:
            optimization_level = self.optimization_selector.select_optimization_level(
                self.matrix_size, 'KOBA42 Advanced', 'Intentful Mathematics', 'balanced'
            )
        
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
                                  optimization_level: str = None) -> Dict[str, float]:
        """Process matrix with intelligent optimization level selection."""
        if optimization_level is None:
            optimization_level = self.optimization_selector.select_optimization_level(
                self.matrix_size, 'KOBA42 Advanced', 'Intentful Mathematics', 'balanced'
            )
        
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
            'transformed_eigenvals_mean': np.mean(transformed_eigenvals),
            'optimization_level': optimization_level
        }
    
    def create_consciousness_enhanced_model(self, input_size: int, optimization_level: str = None) -> nn.Module:
        """Create neural network with intelligent consciousness enhancement."""
        if optimization_level is None:
            optimization_level = self.optimization_selector.select_optimization_level(
                self.matrix_size, 'KOBA42 Advanced', 'Neural Network Training', 'balanced'
            )
        
        class ConsciousnessEnhancedNetwork(nn.Module):
            def __init__(self, input_size: int, optimization_level: str):
                super().__init__()
                self.phi = (1 + np.sqrt(5)) / 2
                self.consciousness_ratio = 79/21
                self.optimization_level = optimization_level
                
                # Adjust architecture based on optimization level
                if optimization_level == 'basic':
                    layer1_size = int(128 * self.consciousness_ratio / 4)
                    layer2_size = int(64 * self.phi)
                    layer3_size = int(32 * self.consciousness_ratio / 4)
                elif optimization_level == 'advanced':
                    layer1_size = int(256 * self.consciousness_ratio / 4)
                    layer2_size = int(128 * self.phi)
                    layer3_size = int(64 * self.consciousness_ratio / 4)
                else:  # expert
                    layer1_size = int(512 * self.consciousness_ratio / 4)
                    layer2_size = int(256 * self.phi)
                    layer3_size = int(128 * self.consciousness_ratio / 4)
                
                # Consciousness-enhanced architecture
                self.layer1 = nn.Linear(input_size, layer1_size)
                self.layer2 = nn.Linear(layer1_size, layer2_size)
                self.layer3 = nn.Linear(layer2_size, layer3_size)
                self.layer4 = nn.Linear(layer3_size, 1)
                
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
        
        return ConsciousnessEnhancedNetwork(input_size, optimization_level)
    
    def train_consciousness_enhanced_model(self, model: nn.Module, X: torch.Tensor, 
                                         y: torch.Tensor, epochs: int = 100, 
                                         optimization_level: str = None) -> EnhancedTrainingMetrics:
        """Train model with intelligent consciousness enhancement."""
        if optimization_level is None:
            optimization_level = self.optimization_selector.select_optimization_level(
                self.matrix_size, 'KOBA42 Advanced', 'Neural Network Training', 'balanced'
            )
        
        start_time = time.time()
        
        model = model.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        
        # Consciousness-enhanced optimizer based on optimization level
        if optimization_level == 'basic':
            consciousness_lr = 0.001 * self.consciousness_ratio / 4.0
        elif optimization_level == 'advanced':
            consciousness_lr = 0.001 * self.consciousness_ratio / 3.0
        else:  # expert
            consciousness_lr = 0.001 * self.consciousness_ratio / 2.0
        
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
                logger.info(f"KOBA42 {optimization_level.capitalize()} Epoch {epoch}: Loss = {loss.item():.6f}, R² = {r2.item():.6f}")
        
        training_time = time.time() - start_time
        
        # Calculate convergence rate
        if len(losses) > 1:
            convergence_rate = (losses[0] - losses[-1]) / losses[0]
        else:
            convergence_rate = 0.0
        
        # Get optimization selection info
        profile = self.optimization_selector.optimization_profiles[optimization_level]
        expected_speedup = profile.expected_speedup
        
        # Calculate actual speedup (compared to baseline)
        actual_speedup = 1.0  # Will be calculated in comparison
        speedup_accuracy = 1.0  # Will be calculated in comparison
        
        return EnhancedTrainingMetrics(
            training_time=training_time,
            epochs_completed=epochs,
            final_loss=losses[-1],
            final_accuracy=accuracies[-1],
            convergence_rate=convergence_rate,
            memory_usage_mb=self._get_memory_usage(),
            cpu_utilization=self._get_cpu_utilization(),
            selected_optimization_level=optimization_level,
            optimization_score=0.8,  # High score for KOBA42
            expected_speedup=expected_speedup,
            actual_speedup=actual_speedup,
            speedup_accuracy=speedup_accuracy
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization."""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0

class EnhancedSpeedComparisonAnalyzer:
    """Enhanced speed comparison analyzer with intelligent optimization selection."""
    
    def __init__(self):
        self.results = []
        self.comparison_data = []
        self.optimization_selector = IntelligentOptimizationSelector()
    
    def run_enhanced_comparison(self, matrix_sizes: List[int] = [64, 128, 256, 512]) -> Dict[str, Any]:
        """Run enhanced speed comparison with intelligent optimization selection."""
        logger.info("🚀 Starting Enhanced Speed Comparison Analysis")
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'matrix_sizes': matrix_sizes,
            'traditional_results': [],
            'koba42_results': [],
            'enhanced_speedup_analysis': [],
            'optimization_selection_analysis': [],
            'summary': {}
        }
        
        for matrix_size in matrix_sizes:
            logger.info(f"📊 Testing Matrix Size: {matrix_size}×{matrix_size}")
            
            # Get intelligent optimization selection
            selected_level = self.optimization_selector.select_optimization_level(
                matrix_size, 'KOBA42 Advanced', 'Speed Comparison', 'balanced'
            )
            profile = self.optimization_selector.optimization_profiles[selected_level]
            
            logger.info(f"🎯 Selected optimization level: {selected_level} for matrix size {matrix_size}")
            
            # Traditional ML Training
            traditional_result = self._run_enhanced_traditional_training(matrix_size)
            comparison_results['traditional_results'].append(traditional_result)
            
            # KOBA42 Advanced Training with intelligent selection
            koba42_result = self._run_enhanced_koba42_training(matrix_size, selected_level)
            comparison_results['koba42_results'].append(koba42_result)
            
            # Calculate enhanced speedup and efficiency
            enhanced_speedup_analysis = self._calculate_enhanced_speedup_metrics(
                traditional_result, koba42_result, selected_level, profile
            )
            comparison_results['enhanced_speedup_analysis'].append(enhanced_speedup_analysis)
            
            # Optimization selection analysis
            optimization_analysis = {
                'matrix_size': matrix_size,
                'selected_level': selected_level,
                'expected_speedup': profile.expected_speedup,
                'expected_accuracy_improvement': profile.expected_accuracy_improvement,
                'actual_speedup': enhanced_speedup_analysis['speedup_factor'],
                'actual_accuracy_improvement': enhanced_speedup_analysis['accuracy_improvement'],
                'speedup_prediction_accuracy': enhanced_speedup_analysis['speedup_prediction_accuracy'],
                'optimization_score': enhanced_speedup_analysis['optimization_score']
            }
            comparison_results['optimization_selection_analysis'].append(optimization_analysis)
            
            logger.info(f"✅ Matrix {matrix_size}: Traditional={traditional_result['total_time']:.2f}s, "
                       f"KOBA42={koba42_result['total_time']:.2f}s, "
                       f"Speedup={enhanced_speedup_analysis['speedup_factor']:.2f}x, "
                       f"Level={selected_level}")
        
        # Generate enhanced summary statistics
        comparison_results['summary'] = self._generate_enhanced_summary_statistics(comparison_results)
        
        return comparison_results
    
    def _run_enhanced_traditional_training(self, matrix_size: int) -> Dict[str, Any]:
        """Run enhanced traditional ML training."""
        start_time = time.time()
        
        # Initialize traditional trainer
        traditional_trainer = EnhancedTraditionalMLTrainer(matrix_size)
        
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
            'trace': processing_result['trace'],
            'selected_optimization_level': training_metrics.selected_optimization_level,
            'optimization_score': training_metrics.optimization_score
        }
    
    def _run_enhanced_koba42_training(self, matrix_size: int, optimization_level: str) -> Dict[str, Any]:
        """Run enhanced KOBA42 training with intelligent optimization selection."""
        start_time = time.time()
        
        # Initialize KOBA42 trainer
        koba42_trainer = EnhancedKOBA42Trainer(matrix_size)
        
        # Generate optimized F2 matrix with selected level
        matrix_start = time.time()
        matrix = koba42_trainer.generate_optimized_f2_matrix(optimization_level)
        matrix_time = time.time() - matrix_start
        
        # Process matrix with selected optimization level
        processing_result = koba42_trainer.optimized_matrix_processing(matrix, optimization_level)
        
        # Generate synthetic data
        n_samples = 1000
        X = torch.randn(n_samples, matrix_size)
        y = torch.sum(X * torch.randn(matrix_size), dim=1) + torch.randn(n_samples) * 0.1
        
        # Create and train consciousness-enhanced model with selected level
        model = koba42_trainer.create_consciousness_enhanced_model(matrix_size, optimization_level)
        training_metrics = koba42_trainer.train_consciousness_enhanced_model(
            model, X, y, epochs=50, optimization_level=optimization_level
        )
        
        total_time = time.time() - start_time
        
        return {
            'matrix_size': matrix_size,
            'approach': 'koba42_enhanced',
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
            'transformed_eigenvals_mean': processing_result['transformed_eigenvals_mean'],
            'selected_optimization_level': optimization_level,
            'optimization_score': training_metrics.optimization_score,
            'expected_speedup': training_metrics.expected_speedup
        }
    
    def _calculate_enhanced_speedup_metrics(self, traditional_result: Dict, koba42_result: Dict,
                                          selected_level: str, profile) -> Dict[str, Any]:
        """Calculate enhanced speedup and efficiency metrics."""
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
        
        # Speedup prediction accuracy
        expected_speedup = profile.expected_speedup
        speedup_prediction_accuracy = 1.0 - abs(speedup_factor - expected_speedup) / expected_speedup \
            if expected_speedup > 0 else 0
        
        # Optimization score
        optimization_score = koba42_result['optimization_score']
        
        return {
            'matrix_size': traditional_result['matrix_size'],
            'selected_optimization_level': selected_level,
            'speedup_factor': speedup_factor,
            'efficiency_ratio': efficiency_ratio,
            'memory_efficiency': memory_efficiency,
            'accuracy_improvement': accuracy_improvement,
            'convergence_improvement': convergence_improvement,
            'traditional_time': traditional_time,
            'koba42_time': koba42_time,
            'time_savings_percent': ((traditional_time - koba42_time) / traditional_time) * 100,
            'expected_speedup': expected_speedup,
            'speedup_prediction_accuracy': speedup_prediction_accuracy,
            'optimization_score': optimization_score
        }
    
    def _generate_enhanced_summary_statistics(self, comparison_results: Dict) -> Dict[str, Any]:
        """Generate enhanced summary statistics."""
        speedup_factors = [r['speedup_factor'] for r in comparison_results['enhanced_speedup_analysis']]
        efficiency_ratios = [r['efficiency_ratio'] for r in comparison_results['enhanced_speedup_analysis']]
        accuracy_improvements = [r['accuracy_improvement'] for r in comparison_results['enhanced_speedup_analysis']]
        time_savings = [r['time_savings_percent'] for r in comparison_results['enhanced_speedup_analysis']]
        prediction_accuracies = [r['speedup_prediction_accuracy'] for r in comparison_results['enhanced_speedup_analysis']]
        optimization_scores = [r['optimization_score'] for r in comparison_results['enhanced_speedup_analysis']]
        
        # Optimization level distribution
        level_distribution = {}
        for analysis in comparison_results['optimization_selection_analysis']:
            level = analysis['selected_level']
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        return {
            'average_speedup_factor': np.mean(speedup_factors),
            'max_speedup_factor': np.max(speedup_factors),
            'average_efficiency_ratio': np.mean(efficiency_ratios),
            'average_accuracy_improvement': np.mean(accuracy_improvements),
            'average_time_savings_percent': np.mean(time_savings),
            'average_prediction_accuracy': np.mean(prediction_accuracies),
            'average_optimization_score': np.mean(optimization_scores),
            'total_tests': len(speedup_factors),
            'speedup_consistency': np.std(speedup_factors),
            'efficiency_consistency': np.std(efficiency_ratios),
            'optimization_level_distribution': level_distribution,
            'intelligent_selection_effectiveness': np.mean(prediction_accuracies)
        }

def demonstrate_enhanced_speed_comparison():
    """Demonstrate enhanced speed comparison with intelligent optimization selection."""
    logger.info("🚀 KOBA42 Enhanced Speed Comparison with Intelligent Optimization")
    logger.info("=" * 70)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSpeedComparisonAnalyzer()
    
    # Run enhanced comparison
    matrix_sizes = [64, 128, 256]  # Test with different sizes
    comparison_results = analyzer.run_enhanced_comparison(matrix_sizes)
    
    # Display results
    print("\n📊 ENHANCED SPEED COMPARISON RESULTS")
    print("=" * 50)
    
    summary = comparison_results['summary']
    print(f"🎯 Average Speedup Factor: {summary['average_speedup_factor']:.2f}x")
    print(f"⚡ Maximum Speedup: {summary['max_speedup_factor']:.2f}x")
    print(f"💡 Average Efficiency Ratio: {summary['average_efficiency_ratio']:.4f}")
    print(f"📈 Average Accuracy Improvement: {summary['average_accuracy_improvement']:.4f}")
    print(f"⏱️ Average Time Savings: {summary['average_time_savings_percent']:.1f}%")
    print(f"🎯 Intelligent Selection Effectiveness: {summary['intelligent_selection_effectiveness']:.1%}")
    print(f"🔄 Speedup Consistency: {summary['speedup_consistency']:.2f}")
    
    print("\n📋 OPTIMIZATION LEVEL DISTRIBUTION:")
    for level, count in summary['optimization_level_distribution'].items():
        print(f"   • {level.capitalize()}: {count} tests")
    
    print("\n📋 DETAILED RESULTS BY MATRIX SIZE:")
    print("-" * 40)
    
    for i, analysis in enumerate(comparison_results['enhanced_speedup_analysis']):
        print(f"Matrix {analysis['matrix_size']}×{analysis['matrix_size']}:")
        print(f"  • Selected Level: {analysis['selected_optimization_level'].upper()}")
        print(f"  • Speedup: {analysis['speedup_factor']:.2f}x")
        print(f"  • Expected Speedup: {analysis['expected_speedup']:.2f}x")
        print(f"  • Prediction Accuracy: {analysis['speedup_prediction_accuracy']:.1%}")
        print(f"  • Time Savings: {analysis['time_savings_percent']:.1f}%")
        print(f"  • Efficiency: {analysis['efficiency_ratio']:.4f}")
        print(f"  • Accuracy Improvement: {analysis['accuracy_improvement']:.4f}")
        print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'enhanced_speed_comparison_report_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    logger.info(f"📄 Enhanced comparison report saved to {report_file}")
    
    return comparison_results, report_file

if __name__ == "__main__":
    # Run the enhanced speed comparison
    results, report_file = demonstrate_enhanced_speed_comparison()
    
    print(f"\n🎉 Enhanced speed comparison completed!")
    print(f"📊 Results saved to: {report_file}")
    print(f"📈 Average speedup: {results['summary']['average_speedup_factor']:.2f}x")
    print(f"🎯 Intelligent selection effectiveness: {results['summary']['intelligent_selection_effectiveness']:.1%}")
