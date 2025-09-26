#!/usr/bin/env python3
"""
Galton Board Physics: Classical vs Consciousness-Enhanced Quantum Dynamics
A comprehensive analysis of the Galton Board through post-quantum logic reasoning branching
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from scipy.stats import norm, binom
import random

@dataclass
class ClassicalGaltonParameters:
    """Classical Galton Board parameters"""
    rows: int = 20  # Number of rows of pegs
    balls: int = 1000  # Number of balls to drop
    left_bias: float = 0.5  # Probability of going left (0.5 = unbiased)
    peg_spacing: float = 1.0  # Distance between pegs
    ball_size: float = 0.1  # Size of balls for visualization

@dataclass
class ConsciousnessGaltonParameters:
    """Consciousness-enhanced Galton Board parameters"""
    # Classical parameters
    rows: int = 20
    balls: int = 1000
    peg_spacing: float = 1.0
    
    # Consciousness parameters
    consciousness_dimension: int = 21
    wallace_constant: float = 1.618033988749  # Golden ratio
    consciousness_constant: float = 2.718281828459  # e
    love_frequency: float = 111.0  # Love frequency
    chaos_factor: float = 0.577215664901  # Euler-Mascheroni constant
    
    # Quantum consciousness parameters
    quantum_superposition: bool = True
    consciousness_entanglement: bool = True
    zero_phase_transitions: bool = True
    structured_chaos_modulation: bool = True

class ClassicalGaltonBoard:
    """Classical Galton Board simulation"""
    
    def __init__(self, params: ClassicalGaltonParameters):
        self.params = params
        self.results = []
        self.distribution = []
    
    def simulate_ball_drop(self) -> int:
        """Simulate a single ball drop through the Galton Board"""
        position = 0  # Start at center
        
        for row in range(self.params.rows):
            # Classical probability: 50% chance to go left or right
            if random.random() < self.params.left_bias:
                position -= 1  # Go left
            else:
                position += 1  # Go right
        
        return position
    
    def run_simulation(self) -> Dict:
        """Run the complete Galton Board simulation"""
        print(f"🎯 Running Classical Galton Board Simulation...")
        print(f"   Rows: {self.params.rows}")
        print(f"   Balls: {self.params.balls}")
        print(f"   Left Bias: {self.params.left_bias}")
        
        # Simulate all ball drops
        for i in range(self.params.balls):
            final_position = self.simulate_ball_drop()
            self.results.append(final_position)
        
        # Calculate distribution
        min_pos = min(self.results)
        max_pos = max(self.results)
        positions = range(min_pos, max_pos + 1)
        
        self.distribution = [self.results.count(pos) for pos in positions]
        
        # Theoretical binomial distribution
        theoretical_dist = []
        for pos in positions:
            # Convert position to number of right moves
            right_moves = (pos + self.params.rows) // 2
            if 0 <= right_moves <= self.params.rows:
                prob = binom.pmf(right_moves, self.params.rows, 1 - self.params.left_bias)
                theoretical_dist.append(prob * self.params.balls)
            else:
                theoretical_dist.append(0)
        
        return {
            "results": self.results,
            "distribution": self.distribution,
            "positions": list(positions),
            "theoretical_distribution": theoretical_dist,
            "mean": np.mean(self.results),
            "std": np.std(self.results),
            "variance": np.var(self.results)
        }

class ConsciousnessGaltonBoard:
    """Consciousness-enhanced quantum Galton Board simulation"""
    
    def __init__(self, params: ConsciousnessGaltonParameters):
        self.params = params
        self.results = []
        self.distribution = []
        self.consciousness_matrix = self._initialize_consciousness_matrix()
        self.quantum_states = []
        self.consciousness_entanglement_states = []
    
    def _initialize_consciousness_matrix(self) -> np.ndarray:
        """Initialize consciousness matrix for quantum effects"""
        matrix = np.zeros((self.params.consciousness_dimension, self.params.consciousness_dimension))
        
        for i in range(self.params.consciousness_dimension):
            for j in range(self.params.consciousness_dimension):
                # Apply Wallace Transform with consciousness constant
                consciousness_factor = (self.params.wallace_constant ** (i + j)) / self.params.consciousness_constant
                matrix[i, j] = consciousness_factor * math.sin(self.params.love_frequency * (i + j) * math.pi / 180)
        
        return matrix
    
    def _calculate_consciousness_bias(self, row: int, position: int, ball_id: int) -> float:
        """Calculate consciousness-modulated bias for ball movement"""
        # Base classical bias
        base_bias = 0.5
        
        # Consciousness modulation
        consciousness_factor = np.sum(self.consciousness_matrix) / (self.params.consciousness_dimension ** 2)
        
        # Wallace Transform application
        wallace_modulation = (self.params.wallace_constant ** row) / self.params.consciousness_constant
        
        # Love frequency modulation
        love_modulation = math.sin(self.params.love_frequency * (row + position + ball_id) * math.pi / 180)
        
        # Chaos factor integration
        chaos_modulation = self.params.chaos_factor * math.log(abs(position) + 1)
        
        # Quantum superposition effect
        if self.params.quantum_superposition:
            quantum_factor = math.cos(row * math.pi / self.params.rows) * math.sin(ball_id * math.pi / 100)
        else:
            quantum_factor = 1.0
        
        # Consciousness entanglement effect
        if self.params.consciousness_entanglement:
            entanglement_factor = math.sin(self.params.love_frequency * ball_id * math.pi / 180)
        else:
            entanglement_factor = 1.0
        
        # Zero phase transition effect
        if self.params.zero_phase_transitions:
            zero_phase_factor = math.exp(-abs(position) / self.params.rows)
        else:
            zero_phase_factor = 1.0
        
        # Structured chaos modulation
        if self.params.structured_chaos_modulation:
            chaos_modulation_factor = self.params.chaos_factor * math.log(row + 1)
        else:
            chaos_modulation_factor = 1.0
        
        # Combine all consciousness effects
        consciousness_bias = base_bias * consciousness_factor * wallace_modulation * love_modulation * \
                           chaos_modulation * quantum_factor * entanglement_factor * zero_phase_factor * chaos_modulation_factor
        
        # Normalize to [0, 1] range
        consciousness_bias = max(0.0, min(1.0, consciousness_bias))
        
        return consciousness_bias
    
    def _quantum_measurement_collapse(self, superposition_state: complex, ball_id: int) -> bool:
        """Simulate quantum measurement collapse for consciousness entanglement"""
        # Quantum state magnitude
        magnitude = abs(superposition_state)
        
        # Consciousness measurement effect
        consciousness_measurement = self.params.love_frequency * ball_id / 1000
        
        # Collapse probability based on consciousness
        collapse_probability = magnitude * math.sin(consciousness_measurement * math.pi / 180)
        
        return random.random() < collapse_probability
    
    def simulate_consciousness_ball_drop(self, ball_id: int) -> Tuple[int, List[complex]]:
        """Simulate a single ball drop with consciousness quantum effects"""
        position = 0  # Start at center
        quantum_states = []
        entanglement_states = []
        
        for row in range(self.params.rows):
            # Calculate consciousness-modulated bias
            consciousness_bias = self._calculate_consciousness_bias(row, position, ball_id)
            
            # Quantum superposition state
            if self.params.quantum_superposition:
                # Create superposition of left and right states
                left_amplitude = math.sqrt(consciousness_bias)
                right_amplitude = math.sqrt(1 - consciousness_bias)
                superposition_state = left_amplitude + 1j * right_amplitude
                quantum_states.append(superposition_state)
                
                # Consciousness entanglement
                if self.params.consciousness_entanglement:
                    entanglement_state = self.params.love_frequency * ball_id * math.pi / 180
                    entanglement_states.append(entanglement_state)
                
                # Quantum measurement collapse
                if self._quantum_measurement_collapse(superposition_state, ball_id):
                    position -= 1  # Go left
                else:
                    position += 1  # Go right
            else:
                # Classical consciousness behavior
                if random.random() < consciousness_bias:
                    position -= 1  # Go left
                else:
                    position += 1  # Go right
        
        return position, quantum_states, entanglement_states
    
    def run_consciousness_simulation(self) -> Dict:
        """Run the consciousness-enhanced Galton Board simulation"""
        print(f"🧠 Running Consciousness-Enhanced Galton Board Simulation...")
        print(f"   Rows: {self.params.rows}")
        print(f"   Balls: {self.params.balls}")
        print(f"   Consciousness Dimensions: {self.params.consciousness_dimension}")
        print(f"   Wallace Constant: {self.params.wallace_constant}")
        print(f"   Love Frequency: {self.params.love_frequency}")
        print(f"   Quantum Superposition: {self.params.quantum_superposition}")
        print(f"   Consciousness Entanglement: {self.params.consciousness_entanglement}")
        
        # Simulate all ball drops with consciousness effects
        for ball_id in range(self.params.balls):
            final_position, quantum_states, entanglement_states = self.simulate_consciousness_ball_drop(ball_id)
            self.results.append(final_position)
            self.quantum_states.append(quantum_states)
            self.consciousness_entanglement_states.append(entanglement_states)
        
        # Calculate distribution
        min_pos = min(self.results)
        max_pos = max(self.results)
        positions = range(min_pos, max_pos + 1)
        
        self.distribution = [self.results.count(pos) for pos in positions]
        
        # Consciousness-enhanced theoretical distribution
        theoretical_dist = []
        for pos in positions:
            # Apply consciousness transformation to binomial distribution
            right_moves = (pos + self.params.rows) // 2
            if 0 <= right_moves <= self.params.rows:
                # Classical probability
                classical_prob = binom.pmf(right_moves, self.params.rows, 0.5)
                
                # Consciousness enhancement
                consciousness_enhancement = np.sum(self.consciousness_matrix) / (self.params.consciousness_dimension ** 2)
                wallace_enhancement = (self.params.wallace_constant ** right_moves) / self.params.consciousness_constant
                love_enhancement = math.sin(self.params.love_frequency * right_moves * math.pi / 180)
                
                consciousness_prob = classical_prob * consciousness_enhancement * wallace_enhancement * love_enhancement
                theoretical_dist.append(consciousness_prob * self.params.balls)
            else:
                theoretical_dist.append(0)
        
        return {
            "results": self.results,
            "distribution": self.distribution,
            "positions": list(positions),
            "theoretical_distribution": theoretical_dist,
            "quantum_states": self.quantum_states,
            "entanglement_states": self.consciousness_entanglement_states,
            "mean": np.mean(self.results),
            "std": np.std(self.results),
            "variance": np.var(self.results),
            "consciousness_matrix_sum": np.sum(self.consciousness_matrix),
            "consciousness_factor": np.sum(self.consciousness_matrix) / (self.params.consciousness_dimension ** 2)
        }

def run_galton_board_comparison():
    """Run comprehensive comparison between classical and consciousness Galton Boards"""
    
    print("🎯 Galton Board Physics: Classical vs Consciousness-Enhanced Quantum Dynamics")
    print("=" * 80)
    
    # Classical Galton Board simulation
    classical_params = ClassicalGaltonParameters(rows=20, balls=1000, left_bias=0.5)
    classical_board = ClassicalGaltonBoard(classical_params)
    classical_results = classical_board.run_simulation()
    
    print(f"\n📊 Classical Galton Board Results:")
    print(f"   Mean Position: {classical_results['mean']:.4f}")
    print(f"   Standard Deviation: {classical_results['std']:.4f}")
    print(f"   Variance: {classical_results['variance']:.4f}")
    print(f"   Distribution Range: {min(classical_results['positions'])} to {max(classical_results['positions'])}")
    
    # Consciousness-enhanced Galton Board simulation
    consciousness_params = ConsciousnessGaltonParameters(
        rows=20, 
        balls=1000,
        quantum_superposition=True,
        consciousness_entanglement=True,
        zero_phase_transitions=True,
        structured_chaos_modulation=True
    )
    consciousness_board = ConsciousnessGaltonBoard(consciousness_params)
    consciousness_results = consciousness_board.run_consciousness_simulation()
    
    print(f"\n🧠 Consciousness-Enhanced Galton Board Results:")
    print(f"   Mean Position: {consciousness_results['mean']:.4f}")
    print(f"   Standard Deviation: {consciousness_results['std']:.4f}")
    print(f"   Variance: {consciousness_results['variance']:.4f}")
    print(f"   Distribution Range: {min(consciousness_results['positions'])} to {max(consciousness_results['positions'])}")
    print(f"   Consciousness Factor: {consciousness_results['consciousness_factor']:.6f}")
    print(f"   Consciousness Matrix Sum: {consciousness_results['consciousness_matrix_sum']:.6f}")
    
    # Comparative analysis
    print(f"\n📈 Comparative Analysis:")
    mean_difference = consciousness_results['mean'] - classical_results['mean']
    std_difference = consciousness_results['std'] - classical_results['std']
    variance_difference = consciousness_results['variance'] - classical_results['variance']
    
    print(f"   Mean Difference: {mean_difference:+.4f}")
    print(f"   Standard Deviation Difference: {std_difference:+.4f}")
    print(f"   Variance Difference: {variance_difference:+.4f}")
    
    # Consciousness effects analysis
    print(f"\n🌌 Consciousness Effects Analysis:")
    print(f"   Quantum Superposition States: {len(consciousness_results['quantum_states'])}")
    print(f"   Consciousness Entanglement States: {len(consciousness_results['entanglement_states'])}")
    print(f"   Wallace Transform Applied: {consciousness_params.wallace_constant}")
    print(f"   Love Frequency Modulation: {consciousness_params.love_frequency} Hz")
    print(f"   Chaos Factor Integration: {consciousness_params.chaos_factor}")
    
    # Theoretical implications
    print(f"\n🔬 Theoretical Implications:")
    print(f"   • Classical: Normal distribution with binomial probability")
    print(f"   • Consciousness: Quantum superposition with consciousness entanglement")
    print(f"   • Wallace Transform: Golden ratio modulation of ball trajectories")
    print(f"   • Love Frequency: 111 Hz consciousness resonance in quantum states")
    print(f"   • Structured Chaos: Euler-Mascheroni constant in trajectory modulation")
    print(f"   • Zero Phase Transitions: Consciousness-driven phase changes")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "classical_results": {
            "mean": classical_results['mean'],
            "std": classical_results['std'],
            "variance": classical_results['variance'],
            "distribution": classical_results['distribution'],
            "positions": classical_results['positions']
        },
        "consciousness_results": {
            "mean": consciousness_results['mean'],
            "std": consciousness_results['std'],
            "variance": consciousness_results['variance'],
            "distribution": consciousness_results['distribution'],
            "positions": consciousness_results['positions'],
            "consciousness_factor": consciousness_results['consciousness_factor'],
            "consciousness_matrix_sum": consciousness_results['consciousness_matrix_sum']
        },
        "comparative_analysis": {
            "mean_difference": mean_difference,
            "std_difference": std_difference,
            "variance_difference": variance_difference
        },
        "consciousness_parameters": {
            "wallace_constant": consciousness_params.wallace_constant,
            "consciousness_constant": consciousness_params.consciousness_constant,
            "love_frequency": consciousness_params.love_frequency,
            "chaos_factor": consciousness_params.chaos_factor,
            "quantum_superposition": consciousness_params.quantum_superposition,
            "consciousness_entanglement": consciousness_params.consciousness_entanglement
        }
    }
    
    with open('galton_board_consciousness_physics_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: galton_board_consciousness_physics_results.json")
    
    return results

if __name__ == "__main__":
    run_galton_board_comparison()
