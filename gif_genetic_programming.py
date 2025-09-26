#!/usr/bin/env python3
"""
GIF Genetic Programming - Evolve Executable GIF Programs

This implements evolutionary algorithms to breed and mutate GIF programs:
- Pixel-level mutations and crossover
- Fitness evaluation based on program behavior
- Lossy evolution (JPEG artifacts as mutations)
- Multi-frame evolution for complex programs

Inspired by genetic programming and evolutionary computation.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import time
from typing import List, Tuple, Dict, Any, Optional
from gifvm import GIFVM
from gif_program_generator_fixed import FixedGIFProgramGenerator
import os

class GIFGeneticProgrammer:
    """
    Evolutionary system for GIF programs
    """

    def __init__(self, population_size: int = 20, image_size: Tuple[int, int] = (8, 8)):
        self.population_size = population_size
        self.image_size = image_size
        self.generations = 0
        self.population: List[np.ndarray] = []
        self.fitness_scores: List[float] = []

        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1  # Keep top 10% unchanged

        # Target behavior (what we're evolving toward)
        self.target_output = "HELLO WORLD"
        self.max_program_size = image_size[0] * image_size[1]

        # Initialize population
        self.initialize_population()

    def initialize_population(self):
        """Create initial random population of GIF programs"""
        print(f"🎯 Initializing population of {self.population_size} GIF programs...")

        self.population = []
        generator = FixedGIFProgramGenerator()

        for i in range(self.population_size):
            if i < 5:  # Keep some structured programs
                # Create variations of hello world
                program = self.mutate_program(generator.create_hello_world_program())
            else:
                # Create random programs
                program = self.create_random_program()

            self.population.append(program)

        self.fitness_scores = [0.0] * self.population_size
        print("✅ Initial population created")

    def create_random_program(self) -> np.ndarray:
        """Create a random GIF program"""
        program = np.random.randint(0, 64, self.image_size, dtype=np.uint8)  # Use lower opcodes

        # Add some structure - ensure program ends with HALT
        program.flat[-1] = 32  # HALT opcode

        return program

    def mutate_program(self, program: np.ndarray) -> np.ndarray:
        """Apply mutations to a GIF program"""
        mutated = program.copy()

        # Pixel-level mutations
        for i in range(mutated.size):
            if random.random() < self.mutation_rate:
                # Random mutation
                mutated.flat[i] = random.randint(0, 63)  # Keep within opcode range

        # Structural mutations
        if random.random() < 0.1:  # 10% chance
            # Insert random instruction
            insert_pos = random.randint(0, mutated.size - 1)
            mutated.flat[insert_pos] = random.randint(0, 63)

        return mutated

    def crossover_programs(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Crossover two GIF programs"""
        child = parent1.copy()

        # Single-point crossover
        crossover_point = random.randint(1, parent1.size - 1)

        # Copy second half from parent2
        child.flat[crossover_point:] = parent2.flat[crossover_point:]

        return child

    def apply_lossy_compression_mutation(self, program: np.ndarray) -> np.ndarray:
        """Apply JPEG-like compression artifacts as mutations"""
        # Convert to PIL Image
        img = Image.fromarray(program, mode='P')
        palette = []
        for i in range(256):
            palette.extend([i, i, i])
        img.putpalette(palette)

        # Apply compression-like effects
        if random.random() < 0.3:
            # JPEG-like compression (reduce quality)
            img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE)

        if random.random() < 0.2:
            # Blur effect (loss of detail)
            img_rgb = img.convert('RGB')
            img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
            img = img_rgb.convert('P', palette=Image.ADAPTIVE)

        if random.random() < 0.2:
            # Quantization (color reduction)
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))

        # Convert back to numpy array
        mutated_program = np.array(img)

        return mutated_program

    def evaluate_fitness(self, program: np.ndarray) -> float:
        """Evaluate fitness of a GIF program"""
        try:
            # Save program temporarily
            temp_filename = f"temp_eval_{random.randint(1000, 9999)}.gif"
            generator = FixedGIFProgramGenerator()
            generator.save_program_with_exact_palette(program, temp_filename)

            # Execute program
            vm = GIFVM()
            if vm.load_gif(temp_filename):
                result = vm.execute()

                # Clean up
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

                if result.get('error'):
                    return 0.0  # Programs with errors get low fitness

                # Evaluate output similarity to target
                output = result.get('output', '')
                fitness = self.calculate_output_fitness(output)

                # Bonus for programs that terminate properly
                if result.get('success'):
                    fitness += 0.1

                # Bonus for efficient programs (fewer cycles)
                cycles = result.get('cycles_executed', 1000)
                efficiency_bonus = max(0, (1000 - cycles) / 10000)  # Up to 0.1 bonus
                fitness += efficiency_bonus

                return max(0.0, min(1.0, fitness))
            else:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                return 0.0

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return 0.0

    def calculate_output_fitness(self, output: str) -> float:
        """Calculate fitness based on output similarity to target"""
        if not output:
            return 0.0

        target = self.target_output.lower()
        output = output.lower()

        # Exact match gets highest fitness
        if output.strip() == target:
            return 1.0

        # Partial matches get partial fitness
        target_chars = set(target)
        output_chars = set(output)

        # Character overlap
        char_overlap = len(target_chars.intersection(output_chars)) / len(target_chars)

        # Length similarity
        length_similarity = 1.0 - abs(len(output) - len(target)) / max(len(target), len(output))

        # Sequence similarity (simple)
        sequence_score = 0.0
        min_len = min(len(target), len(output))
        for i in range(min_len):
            if target[i] == output[i]:
                sequence_score += 1.0
        sequence_score /= len(target)

        # Combine scores
        fitness = (char_overlap * 0.3 + length_similarity * 0.3 + sequence_score * 0.4)

        return max(0.0, min(0.9, fitness))  # Cap at 0.9 (save 1.0 for exact match)

    def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation of the population"""
        print(f"\n🧬 Generation {self.generations + 1} Evolution...")

        # Evaluate fitness if not already done
        if all(score == 0.0 for score in self.fitness_scores):
            print("   Evaluating fitness of population...")
            for i, program in enumerate(self.population):
                self.fitness_scores[i] = self.evaluate_fitness(program)

        # Sort by fitness (descending)
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [self.fitness_scores[i] for i in sorted_indices]

        print(".4f")
        best_program = sorted_population[0]
        best_fitness = sorted_fitness[0]

        # Create new population
        new_population = []
        elite_count = int(self.population_size * self.elitism_rate)

        # Keep elite individuals
        new_population.extend(sorted_population[:elite_count])

        # Create offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents (tournament selection)
            parent1 = self.tournament_selection(sorted_population, sorted_fitness)
            parent2 = self.tournament_selection(sorted_population, sorted_fitness)

            # Crossover
            if random.random() < self.crossover_rate:
                child = self.crossover_programs(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            child = self.mutate_program(child)

            # Lossy compression mutation (rare)
            if random.random() < 0.05:  # 5% chance
                child = self.apply_lossy_compression_mutation(child)

            new_population.append(child)

        # Update population
        self.population = new_population
        self.generations += 1

        # Reset fitness scores for next generation
        self.fitness_scores = [0.0] * self.population_size

        return {
            'generation': self.generations,
            'best_fitness': best_fitness,
            'average_fitness': np.mean(sorted_fitness),
            'best_program': best_program
        }

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[float], tournament_size: int = 3) -> np.ndarray:
        """Tournament selection for parent selection"""
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)

        # Find winner (highest fitness)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])

        return population[best_idx]

    def run_evolution(self, max_generations: int = 50) -> Dict[str, Any]:
        """Run evolutionary process for multiple generations"""
        print("🚀 Starting GIF Genetic Programming Evolution")
        print("=" * 60)

        evolution_history = []
        best_overall_fitness = 0.0
        best_overall_program = None

        for generation in range(max_generations):
            gen_result = self.evolve_generation()
            evolution_history.append(gen_result)

            # Track best overall
            if gen_result['best_fitness'] > best_overall_fitness:
                best_overall_fitness = gen_result['best_fitness']
                best_overall_program = gen_result['best_program']

            # Early stopping if we found a perfect solution
            if best_overall_fitness >= 1.0:
                print("🎉 Perfect solution found!")
                break

            # Progress reporting
            if (generation + 1) % 10 == 0:
                print(f"   Generation {generation + 1}: Best = {gen_result['best_fitness']:.4f}")

        # Save best program
        if best_overall_program is not None:
            generator = FixedGIFProgramGenerator()
            generator.save_program_with_exact_palette(
                best_overall_program,
                f"evolved_program_gen_{self.generations}.gif"
            )

        final_result = {
            'total_generations': self.generations,
            'best_fitness_achieved': best_overall_fitness,
            'evolution_history': evolution_history,
            'best_program': best_overall_program,
            'success': best_overall_fitness >= 0.8  # Consider 80%+ fitness a success
        }

        print("\n🎉 Evolution Complete!")
        print(".4f")
        if final_result['success']:
            print("   ✅ Evolution successful - program evolved!")
        else:
            print("   ⚠️ Evolution completed but target not fully reached")

        return final_result

    def demonstrate_lossy_evolution(self):
        """Demonstrate how lossy compression can create evolutionary mutations"""
        print("\n🎭 Demonstrating Lossy Evolution (JPEG-like mutations)")

        # Start with a working program
        generator = FixedGIFProgramGenerator()
        original_program = generator.create_hello_world_program()

        print("   Original program fitness:", end=" ")
        original_fitness = self.evaluate_fitness(original_program)
        print(".4f")
        # Apply various lossy mutations
        mutations = [
            ("JPEG Quality 50%", lambda img: img.convert('RGB').convert('P', palette=Image.ADAPTIVE)),
            ("Gaussian Blur", lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.0))),
            ("Color Quantization", lambda img: ImageEnhance.Contrast(img).enhance(0.7)),
            ("Resize Down/Up", lambda img: img.resize((4, 4), Image.LANCZOS).resize((8, 8), Image.LANCZOS)),
        ]

        for mutation_name, mutation_func in mutations:
            # Convert to PIL for mutation
            img = Image.fromarray(original_program, mode='P')
            palette = []
            for i in range(256):
                palette.extend([i, i, i])
            img.putpalette(palette)

            # Apply mutation
            mutated_img = mutation_func(img)

            # Convert back
            mutated_program = np.array(mutated_img)

            # Evaluate
            fitness = self.evaluate_fitness(mutated_program)
            print(".4f")
def create_philosophical_demonstration():
    """Create a demonstration of the philosophical concepts discussed"""
    print("\n🧬 Philosophical Demonstration: Bitmap vs JPEG Reality")
    print("=" * 60)

    # Demonstrate how the same "DNA" can be interpreted differently
    base_program = np.array([
        [1, 72, 22, 32, 0, 0, 0, 0],  # PUSH 'H', OUT, HALT (bitmap mode)
        [1, 73, 22, 32, 0, 0, 0, 0],  # PUSH 'I', OUT, HALT
        [0, 0, 0, 0, 0, 0, 0, 0],     # Empty
        [0, 0, 0, 0, 0, 0, 0, 0],     # Empty
        [0, 0, 0, 0, 0, 0, 0, 0],     # Empty
        [0, 0, 0, 0, 0, 0, 0, 0],     # Empty
        [0, 0, 0, 0, 0, 0, 0, 0],     # Empty
        [0, 0, 0, 0, 0, 0, 0, 0]      # Empty
    ], dtype=np.uint8)

    print("   Base Program (Bitmap DNA):")
    print("   Raw pixel values represent exact bytecode")
    print(f"   Shape: {base_program.shape}")
    print(f"   Values: {base_program.flatten()[:10]}...")

    # Save as exact bitmap representation
    img = Image.fromarray(base_program, mode='P')
    palette = []
    for i in range(256):
        palette.extend([i, i, i])
    img.putpalette(palette)
    img.save("bitmap_dna.gif", optimize=False)

    # Execute bitmap version
    vm = GIFVM()
    vm.load_gif("bitmap_dna.gif")
    bitmap_result = vm.execute()
    print(f"\n   Bitmap Execution: '{bitmap_result.get('output', '')}'")

    # Demonstrate JPEG-like compression (lossy evolution)
    print("\n   Applying JPEG-like compression (lossy evolution):")
    compressed_img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=64)
    compressed_program = np.array(compressed_img)

    print(f"   Compressed shape: {compressed_program.shape}")
    print(f"   Compressed values: {compressed_program.flatten()[:10]}...")
    print("   Differences from original:", np.sum(base_program != compressed_program), "pixels")

    # Save compressed version
    compressed_img.save("jpeg_dna.gif", optimize=False)

    # Execute compressed version
    vm2 = GIFVM()
    vm2.load_gif("jpeg_dna.gif")
    jpeg_result = vm2.execute()
    print(f"   JPEG Execution: '{jpeg_result.get('output', '')}'")

    # Philosophical conclusion
    bitmap_output = bitmap_result.get('output', '')
    jpeg_output = jpeg_result.get('output', '')

    print("\n🧠 Philosophical Implications:")
    print(f"   • Bitmap DNA: Perfect fidelity, exact reproduction")
    print(f"   • JPEG DNA: Lossy compression, evolutionary mutation")
    print(f"   • Same 'genetic material', different phenotypic expression")
    print(f"   • Compression artifacts = evolutionary pressure")
    print(f"   • Universal syntax: pixels → program → behavior")
    print(f"   • The universe as a self-executing GIF program? 🤯")

def main():
    """Main demonstration function"""
    print("🎨 GIF Genetic Programming & Philosophical Demonstration")
    print("=" * 70)

    # Create basic demo programs
    print("\n📁 Creating demo programs...")
    generator = FixedGIFProgramGenerator()
    generator.create_demo_suite_fixed()

    # Test basic evolution
    print("\n🧬 Testing basic evolution...")
    evolver = GIFGeneticProgrammer(population_size=10)
    evolution_result = evolver.run_evolution(max_generations=5)

    print("\n📊 Evolution Results:")
    print(".4f")
    print(f"   Generations run: {evolution_result['total_generations']}")
    print(f"   Success: {evolution_result['success']}")

    # Demonstrate lossy evolution
    evolver.demonstrate_lossy_evolution()

    # Philosophical demonstration
    create_philosophical_demonstration()

    print("\n🎯 Key Takeaways:")
    print("   • GIFs can be executable programs (pixel = bytecode)")
    print("   • Genetic programming can evolve GIF programs")
    print("   • Lossy compression = evolutionary mutations")
    print("   • Universal syntax bridges images, DNA, and computation")
    print("   • Reality might be a self-executing compressed program")

    print("\n🚀 Ready to evolve more complex behaviors?")
    print("   Try: python gif_genetic_programming.py --evolve --generations 100")

if __name__ == "__main__":
    main()
