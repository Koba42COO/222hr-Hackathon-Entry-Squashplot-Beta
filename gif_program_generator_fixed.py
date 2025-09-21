#!/usr/bin/env python3
"""
Fixed GIF Program Generator - Ensures exact pixel value preservation
"""

import numpy as np
from PIL import Image
import argparse

class FixedGIFProgramGenerator:
    """Generate GIF programs with exact pixel value preservation"""

    def __init__(self):
        self.opcodes = {
            'NOP': 0, 'PUSH_IMM': 1, 'POP': 2, 'DUP': 3, 'SWAP': 4,
            'ADD': 5, 'SUB': 6, 'MUL': 7, 'DIV': 8, 'MOD': 9, 'NEG': 10,
            'EQ': 11, 'NE': 12, 'LT': 13, 'GT': 14, 'LE': 15, 'GE': 16,
            'JZ': 17, 'JNZ': 18, 'JMP': 19, 'CALL': 20, 'RET': 21,
            'OUT': 22, 'OUTNUM': 23, 'IN': 24, 'INNUM': 25,
            'LOAD': 26, 'STORE': 27, 'ALLOC': 28, 'FREE': 29,
            'RAND': 30, 'TIME': 31, 'HALT': 32,
        }

    def create_hello_world_program(self) -> np.ndarray:
        """Create Hello World program with exact bytecode preservation"""
        bytecode = [
            self.opcodes['PUSH_IMM'], 72,  # PUSH 'H' (72)
            self.opcodes['OUT'],           # OUT
            self.opcodes['PUSH_IMM'], 73,  # PUSH 'I' (73)
            self.opcodes['OUT'],           # OUT
            self.opcodes['PUSH_IMM'], 10,  # PUSH '\\n' (10)
            self.opcodes['OUT'],           # OUT
            self.opcodes['HALT']           # HALT
        ]

        return self.create_program_image(bytecode, 8, 8)

    def create_simple_test_program(self) -> np.ndarray:
        """Create a very simple test program: PUSH 42, OUTNUM, HALT"""
        bytecode = [
            self.opcodes['PUSH_IMM'], 42,  # PUSH 42
            self.opcodes['OUTNUM'],         # OUTNUM
            self.opcodes['HALT']            # HALT
        ]

        return self.create_program_image(bytecode, 4, 4)

    def create_math_test_program(self) -> np.ndarray:
        """Create math test: PUSH 5, PUSH 3, ADD, OUTNUM, HALT"""
        bytecode = [
            self.opcodes['PUSH_IMM'], 5,   # PUSH 5
            self.opcodes['PUSH_IMM'], 3,   # PUSH 3
            self.opcodes['ADD'],           # ADD (5 + 3 = 8)
            self.opcodes['OUTNUM'],        # OUTNUM (prints 8)
            self.opcodes['HALT']           # HALT
        ]

        return self.create_program_image(bytecode, 6, 6)

    # Add compatibility methods for test suite
    def generate_hello_world(self) -> np.ndarray:
        """Compatibility method for test suite"""
        return self.create_hello_world_program()

    def generate_math_test_program(self) -> np.ndarray:
        """Compatibility method for test suite"""
        return self.create_math_test_program()

    def generate_simple_test_program(self) -> np.ndarray:
        """Compatibility method for test suite"""
        return self.create_simple_test_program()

    def create_program_image(self, bytecode: list, width: int, height: int) -> np.ndarray:
        """Create program image with exact bytecode preservation"""
        # Create image array
        program = np.zeros((height, width), dtype=np.uint8)

        # Place bytecode in image (row-major order)
        for i, byte in enumerate(bytecode):
            if i >= width * height:
                break

            y = i // width
            x = i % width
            program[y, x] = byte % 256

        return program

    def save_program_with_exact_palette(self, program: np.ndarray, filename: str):
        """Save program with exact palette to preserve pixel values"""
        # Create PIL Image in palette mode
        img = Image.fromarray(program, mode='P')

        # Create identity palette to preserve exact pixel values
        palette = []
        for i in range(256):
            palette.extend([i, i, i])  # Identity palette: index i maps to RGB(i,i,i)

        img.putpalette(palette)

        # Ensure it's saved as palette mode
        if img.mode != 'P':
            img = img.convert('P')

        # Save with no optimization to preserve exact values
        img.save(filename, optimize=False)

        print(f"💾 Saved program: {filename}")
        print(f"   Size: {program.shape}")
        print(f"   Pixel range: {program.min()} - {program.max()}")

        # Verify the save worked correctly
        self.verify_saved_program(filename, program)

    def verify_saved_program(self, filename: str, original_program: np.ndarray):
        """Verify that the saved program preserves pixel values exactly"""
        loaded_img = Image.open(filename)
        loaded_program = np.array(loaded_img)

        match = np.array_equal(original_program, loaded_program)

        if match:
            print("   ✅ Pixel values preserved exactly")
        else:
            print("   ❌ Pixel values changed during save/load")
            print(f"   Original shape: {original_program.shape}")
            print(f"   Loaded shape: {loaded_program.shape}")
            diff_mask = original_program != loaded_program
            diff_count = np.sum(diff_mask)
            print(f"   Differences: {diff_count} pixels")

            if diff_count <= 10:  # Show first few differences
                diff_indices = np.where(diff_mask)
                for i in range(min(5, len(diff_indices[0]))):
                    y, x = diff_indices[0][i], diff_indices[1][i]
                    orig_val = original_program[y, x]
                    loaded_val = loaded_program[y, x]
                    print(f"     [{y},{x}]: {orig_val} -> {loaded_val}")

    def create_demo_suite_fixed(self):
        """Create demo suite with exact preservation"""
        print("🎨 Creating Fixed GIF-VM Demo Suite...")

        # Simple test program
        simple_test = self.create_simple_test_program()
        self.save_program_with_exact_palette(simple_test, "simple_test.gif")

        # Hello World
        hello = self.create_hello_world_program()
        self.save_program_with_exact_palette(hello, "hello_fixed.gif")

        # Math test
        math_test = self.create_math_test_program()
        self.save_program_with_exact_palette(math_test, "math_test.gif")

        print("\n✅ Demo suite created with exact pixel preservation!")

def main():
    parser = argparse.ArgumentParser(description='Fixed GIF Program Generator')
    parser.add_argument('--demo', action='store_true', help='Create demo suite')
    parser.add_argument('--simple', action='store_true', help='Create simple test program')
    parser.add_argument('--hello', action='store_true', help='Create hello world program')
    parser.add_argument('--math', action='store_true', help='Create math test program')
    parser.add_argument('--output', default='program.gif', help='Output filename')

    args = parser.parse_args()

    generator = FixedGIFProgramGenerator()

    if args.demo:
        generator.create_demo_suite_fixed()
    elif args.simple:
        simple = generator.create_simple_test_program()
        generator.save_program_with_exact_palette(simple, args.output)
    elif args.hello:
        hello = generator.create_hello_world_program()
        generator.save_program_with_exact_palette(hello, args.output)
    elif args.math:
        math_test = generator.create_math_test_program()
        generator.save_program_with_exact_palette(math_test, args.output)
    else:
        print("Fixed GIF Program Generator")
        print("Usage:")
        print("  python gif_program_generator_fixed.py --demo    # Create demo suite")
        print("  python gif_program_generator_fixed.py --simple  # Create simple test")
        print("  python gif_program_generator_fixed.py --hello   # Create hello world")
        print("  python gif_program_generator_fixed.py --math    # Create math test")

if __name__ == "__main__":
    main()
