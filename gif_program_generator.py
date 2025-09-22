#!/usr/bin/env python3
"""
GIF Program Generator - Create executable GIF programs

This tool generates GIF images that can be executed by the GIF-VM.
Each pixel represents a bytecode instruction or data value.
"""

import numpy as np
from PIL import Image
import argparse
import math
from typing import List, Dict, Any

class GIFProgramGenerator:
    """
    Generate GIF programs from high-level specifications
    """

    def __init__(self):
        # Opcode mapping (reverse of VM's OPCODES)
        self.opcodes = {
            'NOP': 0, 'PUSH_IMM': 1, 'POP': 2, 'DUP': 3, 'SWAP': 4,
            'ADD': 5, 'SUB': 6, 'MUL': 7, 'DIV': 8, 'MOD': 9, 'NEG': 10,
            'EQ': 11, 'NE': 12, 'LT': 13, 'GT': 14, 'LE': 15, 'GE': 16,
            'JZ': 17, 'JNZ': 18, 'JMP': 19, 'CALL': 20, 'RET': 21,
            'OUT': 22, 'OUTNUM': 23, 'IN': 24, 'INNUM': 25,
            'LOAD': 26, 'STORE': 27, 'ALLOC': 28, 'FREE': 29,
            'RAND': 30, 'TIME': 31, 'HALT': 32,
            'SIN': 33, 'COS': 34, 'LOG': 35, 'EXP': 36, 'SQRT': 37, 'POW': 38, 'ABS': 39,
            'MUTATE': 40, 'CROSSOVER': 41, 'SELECT': 42, 'FITNESS': 43
        }

    def generate_hello_world(self) -> np.ndarray:
        """
        Generate Hello World program

        Program: PUSH 'H', OUT, PUSH 'I', OUT, PUSH '\\n', OUT, HALT
        """
        bytecode = [
            self.opcodes['PUSH_IMM'], ord('H'),
            self.opcodes['OUT'],
            self.opcodes['PUSH_IMM'], ord('I'),
            self.opcodes['OUT'],
            self.opcodes['PUSH_IMM'], ord('\n'),
            self.opcodes['OUT'],
            self.opcodes['HALT']
        ]

        return self._create_program_image(bytecode, width=8, height=8)

    def generate_fibonacci(self, n: int = 10) -> np.ndarray:
        """
        Generate Fibonacci sequence program

        Computes and prints first n Fibonacci numbers
        """
        bytecode = []

        # Initialize with 0, 1
        bytecode.extend([self.opcodes['PUSH_IMM'], 0])  # a = 0
        bytecode.extend([self.opcodes['PUSH_IMM'], 1])  # b = 1

        # Print first two numbers
        bytecode.extend([self.opcodes['DUP'], self.opcodes['OUTNUM'],
                        self.opcodes['PUSH_IMM'], ord(' '), self.opcodes['OUT']])
        bytecode.extend([self.opcodes['DUP'], self.opcodes['OUTNUM'],
                        self.opcodes['PUSH_IMM'], ord(' '), self.opcodes['OUT']])

        # Loop for remaining numbers
        for i in range(2, n):
            # Add a + b, print result
            bytecode.extend([
                self.opcodes['DUP'],           # DUP a
                self.opcodes['SWAP'],          # SWAP (get b on top)
                self.opcodes['ADD'],           # ADD (a + b)
                self.opcodes['DUP'],           # DUP result for next iteration
                self.opcodes['SWAP'],          # SWAP (result, old_b -> old_b, result)
                self.opcodes['OUTNUM'],        # Print current number
                self.opcodes['PUSH_IMM'], ord(' '),
                self.opcodes['OUT']            # Print space
            ])

        bytecode.append(self.opcodes['HALT'])

        return self._create_program_image(bytecode, width=16, height=16)

    def generate_artistic_program(self) -> np.ndarray:
        """
        Generate a program that creates ASCII art

        This demonstrates more complex control flow
        """
        # Program: Print a simple ASCII art pattern
        art = [
            "  *  ",
            " *** ",
            "*****",
            " *** ",
            "  *  "
        ]

        bytecode = []

        for line in art:
            for char in line:
                bytecode.extend([
                    self.opcodes['PUSH_IMM'], ord(char),
                    self.opcodes['OUT']
                ])
            # Newline
            bytecode.extend([
                self.opcodes['PUSH_IMM'], ord('\n'),
                self.opcodes['OUT']
            ])

        bytecode.append(self.opcodes['HALT'])

        return self._create_program_image(bytecode, width=16, height=16)

    def generate_math_demo(self) -> np.ndarray:
        """
        Generate a program that demonstrates mathematical operations

        Computes: (5 + 3) * 2 - 1 = 15
        """
        bytecode = [
            # Push 5 and 3, add them -> 8
            self.opcodes['PUSH_IMM'], 5,
            self.opcodes['PUSH_IMM'], 3,
            self.opcodes['ADD'],

            # Multiply by 2 -> 16
            self.opcodes['PUSH_IMM'], 2,
            self.opcodes['MUL'],

            # Subtract 1 -> 15
            self.opcodes['PUSH_IMM'], 1,
            self.opcodes['SUB'],

            # Print result
            self.opcodes['OUTNUM'],
            self.opcodes['PUSH_IMM'], ord('\n'),
            self.opcodes['OUT'],
            self.opcodes['HALT']
        ]

        return self._create_program_image(bytecode, width=8, height=8)

    def generate_loop_demo(self) -> np.ndarray:
        """
        Generate a program that demonstrates looping

        Prints numbers 1 through 5 with loop
        """
        bytecode = [
            # Initialize counter
            self.opcodes['PUSH_IMM'], 1,      # Counter starts at 1

            # Loop start
            self.opcodes['DUP'],              # DUP counter
            self.opcodes['OUTNUM'],           # Print number
            self.opcodes['PUSH_IMM'], ord(' '),
            self.opcodes['OUT'],              # Print space

            # Increment counter
            self.opcodes['PUSH_IMM'], 1,
            self.opcodes['ADD'],

            # Check if counter <= 5
            self.opcodes['DUP'],              # DUP counter
            self.opcodes['PUSH_IMM'], 6,      # Compare with 6
            self.opcodes['LT'],               # counter < 6?

            # If true, jump back to loop start
            self.opcodes['JNZ'], -12,         # Jump back if not zero (true)

            # End
            self.opcodes['HALT']
        ]

        return self._create_program_image(bytecode, width=12, height=12)

    def generate_evolutionary_seed(self) -> np.ndarray:
        """
        Generate a seed program for evolutionary development

        This creates a simple template that can be evolved
        """
        # Create a template with basic structure
        bytecode = [
            # Random operations that can be evolved
            self.opcodes['RAND'], self.opcodes['OUTNUM'],
            self.opcodes['PUSH_IMM'], ord(' '), self.opcodes['OUT'],
            self.opcodes['RAND'], self.opcodes['OUTNUM'],
            self.opcodes['PUSH_IMM'], ord('\n'), self.opcodes['OUT'],
            self.opcodes['HALT']
        ]

        # Add some padding for evolution
        while len(bytecode) < 64:
            bytecode.append(self.opcodes['NOP'])

        return self._create_program_image(bytecode, width=8, height=8)

    def _create_program_image(self, bytecode: List[int], width: int, height: int) -> np.ndarray:
        """
        Create a numpy array representing the GIF program

        Args:
            bytecode: List of bytecode values
            width: Image width
            height: Image height

        Returns:
            numpy array representing the program
        """
        # Create image array
        program = np.zeros((height, width), dtype=np.uint8)

        # Place bytecode in image (row-major order)
        for i, byte in enumerate(bytecode):
            if i >= width * height:
                break  # Don't exceed image size

            y = i // width
            x = i % width
            program[y, x] = byte % 256  # Ensure valid palette index

        return program

    def save_program(self, program: np.ndarray, filename: str, description: str = ""):
        """
        Save program as GIF file

        Args:
            program: Program array
            filename: Output filename
            description: Program description for metadata
        """
        # Create PIL Image
        img = Image.fromarray(program, mode='P')

        # Create palette (grayscale for simplicity)
        palette = []
        for i in range(256):
            palette.extend([i, i, i])  # RGB values

        img.putpalette(palette)

        # Add metadata
        metadata = {
            'Description': description or 'GIF-VM Program',
            'Software': 'GIF-VM Generator',
            'Type': 'Executable GIF Program'
        }

        # Save with metadata
        img.save(filename, **metadata)

        print(f"💾 Saved program: {filename}")
        print(f"   Size: {program.shape}")
        print(f"   Description: {description}")

    def create_demo_suite(self):
        """Create a suite of demo programs"""
        print("🎨 Creating GIF-VM Demo Suite...")

        # Hello World
        hello = self.generate_hello_world()
        self.save_program(hello, "hello_program.gif",
                         "Hello World program - prints 'HI\\n'")

        # Fibonacci
        fib = self.generate_fibonacci(10)
        self.save_program(fib, "fibonacci_program.gif",
                         "Fibonacci sequence generator - prints first 10 numbers")

        # ASCII Art
        art = self.generate_artistic_program()
        self.save_program(art, "ascii_art_program.gif",
                         "ASCII art generator - prints star pattern")

        # Math Demo
        math_demo = self.generate_math_demo()
        self.save_program(math_demo, "math_demo_program.gif",
                         "Math operations demo - computes (5+3)*2-1 = 15")

        # Loop Demo
        loop_demo = self.generate_loop_demo()
        self.save_program(loop_demo, "loop_demo_program.gif",
                         "Loop demonstration - prints numbers 1-5")

        # Evolutionary Seed
        seed = self.generate_evolutionary_seed()
        self.save_program(seed, "evolutionary_seed.gif",
                         "Evolutionary seed - template for genetic programming")

        print("\n✅ Demo suite created successfully!")
        print("📁 Generated files:")
        print("   • hello_program.gif - Hello World")
        print("   • fibonacci_program.gif - Fibonacci sequence")
        print("   • ascii_art_program.gif - ASCII art")
        print("   • math_demo_program.gif - Math operations")
        print("   • loop_demo_program.gif - Loop demonstration")
        print("   • evolutionary_seed.gif - Evolutionary template")

        print("\n🚀 Run any program with:")
        print("   python gifvm.py <program.gif>")

def create_multi_frame_demo():
    """Create a multi-frame GIF program demonstration"""
    print("🎬 Creating Multi-Frame GIF Program Demo...")

    # Create a program with multiple frames
    # Frame 0: Setup and initialization
    # Frame 1: Main computation
    # Frame 2: Output generation

    generator = GIFProgramGenerator()

    # Frame 0: Setup
    setup_frame = generator._create_program_image([
        generator.opcodes['PUSH_IMM'], 10,  # Setup counter
        generator.opcodes['PUSH_IMM'], 0,   # Initialize sum
    ], 8, 8)

    # Frame 1: Computation (calculate sum 1 to 10)
    compute_frame = generator._create_program_image([
        generator.opcodes['DUP'],           # DUP counter
        generator.opcodes['ADD'],           # Add to sum
        generator.opcodes['SWAP'],          # Swap sum and counter
        generator.opcodes['PUSH_IMM'], 1,   # Push 1
        generator.opcodes['SUB'],           # Decrement counter
        generator.opcodes['SWAP'],          # Swap back
        generator.opcodes['DUP'],           # DUP counter
        generator.opcodes['JZ'], 0,         # Jump to next frame if zero
    ], 8, 8)

    # Frame 2: Output
    output_frame = generator._create_program_image([
        generator.opcodes['OUTNUM'],        # Print sum
        generator.opcodes['PUSH_IMM'], ord('\n'),
        generator.opcodes['OUT'],
        generator.opcodes['HALT']
    ], 8, 8)

    # Create multi-frame GIF
    frames = [setup_frame, compute_frame, output_frame]

    # Create PIL images for each frame
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame, mode='P')
        # Create palette
        palette = []
        for i in range(256):
            palette.extend([i, i, i])
        img.putpalette(palette)
        pil_frames.append(img)

    # Save as animated GIF
    pil_frames[0].save(
        "multi_frame_demo.gif",
        save_all=True,
        append_images=pil_frames[1:],
        duration=[500, 500, 500],  # Frame delays in milliseconds
        loop=1
    )

    print("✅ Multi-frame demo created: multi_frame_demo.gif")
    print("   Frame 0: Setup (500ms)")
    print("   Frame 1: Computation (500ms)")
    print("   Frame 2: Output (500ms)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GIF Program Generator')
    parser.add_argument('--demo', action='store_true',
                       help='Create demo suite')
    parser.add_argument('--multi-frame', action='store_true',
                       help='Create multi-frame demo')
    parser.add_argument('--hello', action='store_true',
                       help='Create hello world program')
    parser.add_argument('--fib', type=int, default=10,
                       help='Create fibonacci program (default: 10 numbers)')
    parser.add_argument('--output', default='program.gif',
                       help='Output filename')

    args = parser.parse_args()

    generator = GIFProgramGenerator()

    if args.demo:
        generator.create_demo_suite()
    elif args.multi_frame:
        create_multi_frame_demo()
    elif args.hello:
        hello = generator.generate_hello_world()
        generator.save_program(hello, args.output, "Hello World program")
    elif args.fib > 0:
        fib = generator.generate_fibonacci(args.fib)
        generator.save_program(fib, f"fibonacci_{args.fib}_program.gif",
                              f"Fibonacci sequence (first {args.fib} numbers)")
    else:
        print("GIF Program Generator")
        print("Usage examples:")
        print("  python gif_program_generator.py --demo              # Create demo suite")
        print("  python gif_program_generator.py --hello             # Create hello world")
        print("  python gif_program_generator.py --fib 15            # Create fibonacci program")
        print("  python gif_program_generator.py --multi-frame       # Create multi-frame demo")

if __name__ == "__main__":
    main()
