#!/usr/bin/env python3
"""
GIF-VM: A Virtual Machine that Executes GIF Images as Programs

Each pixel in the GIF represents a bytecode instruction:
- Pixel value (0-255) = opcode or immediate value
- VM walks pixels left→right, top→bottom
- Supports stack-based execution with various opcodes

This creates a "visual programming language" where images are executable DNA.
"""

import sys
import struct
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image
import numpy as np

class GIFVM:
    """
    GIF Virtual Machine - Executes GIF images as programs
    """

    # Opcodes (0-63 reserved for instructions, 64-255 for immediates)
    OPCODES = {
        # Stack operations
        0: 'NOP',        # No operation
        1: 'PUSH_IMM',   # Push immediate value (next pixel)
        2: 'POP',        # Pop from stack
        3: 'DUP',        # Duplicate top of stack
        4: 'SWAP',       # Swap top two stack elements

        # Arithmetic operations
        5: 'ADD',        # Add top two elements
        6: 'SUB',        # Subtract top two elements
        7: 'MUL',        # Multiply top two elements
        8: 'DIV',        # Divide top two elements
        9: 'MOD',        # Modulo top two elements
        10: 'NEG',       # Negate top element

        # Comparison operations
        11: 'EQ',        # Equal comparison
        12: 'NE',        # Not equal comparison
        13: 'LT',        # Less than comparison
        14: 'GT',        # Greater than comparison
        15: 'LE',        # Less or equal comparison
        16: 'GE',        # Greater or equal comparison

        # Control flow
        17: 'JZ',        # Jump if zero (relative, next pixel = offset)
        18: 'JNZ',       # Jump if not zero (relative)
        19: 'JMP',       # Unconditional jump (relative)
        20: 'CALL',      # Call subroutine
        21: 'RET',       # Return from subroutine

        # I/O operations
        22: 'OUT',       # Output ASCII character
        23: 'OUTNUM',    # Output number
        24: 'IN',        # Input character
        25: 'INNUM',     # Input number

        # Memory operations
        26: 'LOAD',      # Load from memory
        27: 'STORE',     # Store to memory
        28: 'ALLOC',     # Allocate memory
        29: 'FREE',      # Free memory

        # Special operations
        30: 'RAND',      # Push random number
        31: 'TIME',      # Push current time
        32: 'HALT',      # Halt execution

        # Advanced operations (for evolution)
        33: 'SIN',       # Sine function
        34: 'COS',       # Cosine function
        35: 'LOG',       # Logarithm
        36: 'EXP',       # Exponential
        37: 'SQRT',      # Square root
        38: 'POW',       # Power function
        39: 'ABS',       # Absolute value

        # Evolutionary operations
        40: 'MUTATE',    # Mutate next instruction
        41: 'CROSSOVER', # Crossover with another frame
        42: 'SELECT',    # Selection operation
        43: 'FITNESS',   # Calculate fitness
    }

    def __init__(self):
        self.stack: List[int] = []
        self.memory: Dict[int, int] = {}
        self.pc: int = 0  # Program counter
        self.frames: List[np.ndarray] = []
        self.current_frame: int = 0
        self.call_stack: List[int] = []
        self.output: str = ""
        self.input_buffer: str = ""
        self.halted: bool = False
        self.max_cycles: int = 10000  # Prevent infinite loops
        self.cycle_count: int = 0

    def load_gif(self, filename: str) -> bool:
        """
        Load GIF file and prepare for execution

        Args:
            filename: Path to GIF file

        Returns:
            bool: True if loaded successfully
        """
        try:
            gif = Image.open(filename)

            # Load all frames
            self.frames = []
            try:
                while True:
                    # Convert to numpy array
                    frame = np.array(gif.convert('P'))  # P mode for palette
                    self.frames.append(frame)
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass  # End of frames

            if not self.frames:
                print("❌ No frames found in GIF")
                return False

            print(f"✅ Loaded GIF with {len(self.frames)} frames")
            for i, frame in enumerate(self.frames):
                print(f"   Frame {i}: {frame.shape} pixels")

            return True

        except Exception as e:
            print(f"❌ Failed to load GIF: {e}")
            return False

    def execute(self) -> Dict[str, Any]:
        """
        Execute the loaded GIF program

        Returns:
            Dict with execution results
        """
        if not self.frames:
            return {'error': 'No GIF loaded'}

        self.halted = False
        self.pc = 0
        self.current_frame = 0
        self.cycle_count = 0
        self.output = ""
        self.stack = []

        print("🚀 Starting GIF-VM execution...")

        while not self.halted and self.cycle_count < self.max_cycles:
            try:
                result = self._execute_step()
                if result.get('error'):
                    return result
                self.cycle_count += 1

            except Exception as e:
                return {
                    'error': f'Execution error at cycle {self.cycle_count}: {e}',
                    'cycles_executed': self.cycle_count,
                    'output': self.output,
                    'final_stack': self.stack.copy()
                }

        if self.cycle_count >= self.max_cycles:
            return {
                'error': 'Maximum cycles exceeded (possible infinite loop)',
                'cycles_executed': self.cycle_count,
                'output': self.output,
                'final_stack': self.stack.copy()
            }

        return {
            'success': True,
            'cycles_executed': self.cycle_count,
            'output': self.output,
            'final_stack': self.stack.copy(),
            'final_pc': self.pc,
            'memory_used': len(self.memory)
        }

    def _execute_step(self) -> Dict[str, Any]:
        """Execute a single step of the program"""
        try:
            # Get current frame
            frame = self.frames[self.current_frame]
            height, width = frame.shape

            # Calculate current position
            y = self.pc // width
            x = self.pc % width

            if y >= height:
                # End of current frame
                if self.current_frame + 1 < len(self.frames):
                    # Move to next frame
                    self.current_frame += 1
                    self.pc = 0
                    return {'continue': True}
                else:
                    # End of program
                    self.halted = True
                    return {'halted': True}

            # Get opcode
            opcode = frame[y, x]
            self.pc += 1

            # Execute opcode
            return self._execute_opcode(opcode)

        except Exception as e:
            return {'error': f'Step execution error: {e}'}

    def _execute_opcode(self, opcode: int) -> Dict[str, Any]:
        """Execute a single opcode"""
        if opcode in self.OPCODES:
            op_name = self.OPCODES[opcode]
        else:
            # Immediate value (64-255)
            if opcode >= 64:
                return self._push_immediate(opcode)
            else:
                return {'error': f'Unknown opcode: {opcode}'}

        # Execute operation
        if op_name == 'NOP':
            pass  # Do nothing

        elif op_name == 'PUSH_IMM':
            return self._push_immediate_from_next_pixel()

        elif op_name == 'POP':
            if self.stack:
                self.stack.pop()
            else:
                return {'error': 'Stack underflow on POP'}

        elif op_name == 'DUP':
            if self.stack:
                self.stack.append(self.stack[-1])
            else:
                return {'error': 'Stack underflow on DUP'}

        elif op_name == 'SWAP':
            if len(self.stack) >= 2:
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
            else:
                return {'error': 'Stack underflow on SWAP'}

        elif op_name == 'ADD':
            if len(self.stack) >= 2:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)
            else:
                return {'error': 'Stack underflow on ADD'}

        elif op_name == 'SUB':
            if len(self.stack) >= 2:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)
            else:
                return {'error': 'Stack underflow on SUB'}

        elif op_name == 'MUL':
            if len(self.stack) >= 2:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)
            else:
                return {'error': 'Stack underflow on MUL'}

        elif op_name == 'DIV':
            if len(self.stack) >= 2:
                b, a = self.stack.pop(), self.stack.pop()
                if b != 0:
                    self.stack.append(a // b)  # Integer division
                else:
                    return {'error': 'Division by zero'}
            else:
                return {'error': 'Stack underflow on DIV'}

        elif op_name == 'MOD':
            if len(self.stack) >= 2:
                b, a = self.stack.pop(), self.stack.pop()
                if b != 0:
                    self.stack.append(a % b)
                else:
                    return {'error': 'Modulo by zero'}
            else:
                return {'error': 'Stack underflow on MOD'}

        elif op_name == 'NEG':
            if self.stack:
                self.stack[-1] = -self.stack[-1]
            else:
                return {'error': 'Stack underflow on NEG'}

        elif op_name == 'JZ':
            return self._jump_if_zero()

        elif op_name == 'JNZ':
            return self._jump_if_not_zero()

        elif op_name == 'JMP':
            return self._jump_unconditional()

        elif op_name == 'OUT':
            if self.stack:
                char_code = self.stack.pop()
                self.output += chr(char_code % 256)
            else:
                return {'error': 'Stack underflow on OUT'}

        elif op_name == 'OUTNUM':
            if self.stack:
                num = self.stack.pop()
                self.output += str(num)
            else:
                return {'error': 'Stack underflow on OUTNUM'}

        elif op_name == 'HALT':
            self.halted = True
            return {'halted': True}

        elif op_name == 'EQ':
            if len(self.stack) >= 2:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1 if a == b else 0)
            else:
                return {'error': 'Stack underflow on EQ'}

        elif op_name == 'SIN':
            if self.stack:
                val = self.stack.pop()
                self.stack.append(int(np.sin(val) * 1000))  # Scale for integer
            else:
                return {'error': 'Stack underflow on SIN'}

        elif op_name == 'RAND':
            import random
            self.stack.append(random.randint(0, 255))

        elif op_name == 'TIME':
            import time
            self.stack.append(int(time.time()) % 256)

        # Add more opcodes as needed...

        return {'continue': True}

    def _push_immediate(self, value: int) -> Dict[str, Any]:
        """Push an immediate value onto the stack"""
        self.stack.append(value)
        return {'continue': True}

    def _push_immediate_from_next_pixel(self) -> Dict[str, Any]:
        """Push immediate value from next pixel"""
        try:
            frame = self.frames[self.current_frame]
            height, width = frame.shape

            y = self.pc // width
            x = self.pc % width

            if y < height and x < width:
                value = frame[y, x]
                self.stack.append(value)
                self.pc += 1
                return {'continue': True}
            else:
                return {'error': 'End of frame reached while reading immediate'}

        except Exception as e:
            return {'error': f'Error reading immediate: {e}'}

    def _jump_if_zero(self) -> Dict[str, Any]:
        """Jump if top of stack is zero"""
        if self.stack:
            condition = self.stack.pop()
            if condition == 0:
                return self._jump_unconditional()
        else:
            return {'error': 'Stack underflow on JZ'}
        return {'continue': True}

    def _jump_if_not_zero(self) -> Dict[str, Any]:
        """Jump if top of stack is not zero"""
        if self.stack:
            condition = self.stack.pop()
            if condition != 0:
                return self._jump_unconditional()
        else:
            return {'error': 'Stack underflow on JNZ'}
        return {'continue': True}

    def _jump_unconditional(self) -> Dict[str, Any]:
        """Unconditional jump"""
        try:
            frame = self.frames[self.current_frame]
            height, width = frame.shape

            y = self.pc // width
            x = self.pc % width

            if y < height and x < width:
                offset = frame[y, x]
                self.pc += 1

                # Apply relative jump
                new_pc = self.pc + offset - 128  # Center around 0
                if new_pc < 0:
                    new_pc = 0

                # Check bounds
                max_pc = height * width
                if new_pc >= max_pc:
                    # Wrap to next frame or halt
                    if self.current_frame + 1 < len(self.frames):
                        self.current_frame += 1
                        self.pc = 0
                    else:
                        self.halted = True
                        return {'halted': True}
                else:
                    self.pc = new_pc

                return {'continue': True}
            else:
                return {'error': 'End of frame reached while reading jump offset'}

        except Exception as e:
            return {'error': f'Jump error: {e}'}

class GIFProgramGenerator:
    """
    Generate GIF programs from bytecode
    """

    def __init__(self):
        self.opcode_map = {v: k for k, v in GIFVM.OPCODES.items()}

    def create_hello_world_program(self) -> np.ndarray:
        """
        Create a GIF program that prints "HI\\n"

        Program:
        PUSH 72 ('H')
        OUT
        PUSH 73 ('I')
        OUT
        PUSH 10 ('\\n')
        OUT
        HALT
        """
        # Define the bytecode sequence
        bytecode = [
            self.opcode_map['PUSH_IMM'], 72,   # PUSH 72
            self.opcode_map['OUT'],            # OUT
            self.opcode_map['PUSH_IMM'], 73,   # PUSH 73
            self.opcode_map['OUT'],            # OUT
            self.opcode_map['PUSH_IMM'], 10,   # PUSH 10
            self.opcode_map['OUT'],            # OUT
            self.opcode_map['HALT']            # HALT
        ]

        # Create a small 8x8 GIF image
        program = np.zeros((8, 8), dtype=np.uint8)

        # Place bytecode in the image
        for i, byte in enumerate(bytecode):
            if i < 64:  # Fit in 8x8 grid
                y = i // 8
                x = i % 8
                program[y, x] = byte

        return program

    def create_fibonacci_program(self) -> np.ndarray:
        """
        Create a GIF program that computes Fibonacci numbers

        Program computes first 10 Fibonacci numbers and prints them
        """
        bytecode = [
            # Initialize: PUSH 0, PUSH 1
            self.opcode_map['PUSH_IMM'], 0,    # PUSH 0
            self.opcode_map['PUSH_IMM'], 1,    # PUSH 1

            # Loop 10 times
            self.opcode_map['PUSH_IMM'], 10,   # Counter = 10
            self.opcode_map['DUP'],            # DUP counter
            self.opcode_map['OUTNUM'],         # Print current number
            self.opcode_map['PUSH_IMM'], 32,   # Space
            self.opcode_map['OUT'],            # Print space

            # Fibonacci calculation
            self.opcode_map['DUP'],            # DUP a
            self.opcode_map['SWAP'],           # SWAP (get b)
            self.opcode_map['ADD'],            # ADD (a + b)
            self.opcode_map['SWAP'],           # SWAP (new b, old b)

            # Decrement counter
            self.opcode_map['PUSH_IMM'], 1,    # PUSH 1
            self.opcode_map['SUB'],            # SUB (counter - 1)

            # Loop condition
            self.opcode_map['DUP'],            # DUP counter
            self.opcode_map['JZ'], 0,          # JZ to end if counter == 0

            self.opcode_map['HALT']            # HALT
        ]

        # Create larger image for Fibonacci program
        program = np.zeros((16, 16), dtype=np.uint8)

        for i, byte in enumerate(bytecode):
            if i < 256:  # Fit in 16x16 grid
                y = i // 16
                x = i % 16
                program[y, x] = byte

        return program

    def save_program_as_gif(self, program: np.ndarray, filename: str):
        """Save a program array as a GIF file"""
        # Create PIL Image from numpy array
        img = Image.fromarray(program, mode='P')

        # Create a simple palette (grayscale)
        palette = []
        for i in range(256):
            palette.extend([i, i, i])  # RGB values

        img.putpalette(palette)
        img.save(filename)

        print(f"💾 Saved program as {filename}")

def main():
    """Main function to run GIF-VM"""
    if len(sys.argv) != 2:
        print("Usage: python gifvm.py <program.gif>")
        print("\nExample programs:")
        print("  python gifvm.py hello_program.gif  # Prints 'HI\\n'")
        print("  python gifvm.py fib_program.gif     # Computes Fibonacci")
        return

    gif_file = sys.argv[1]

    # Create VM and load program
    vm = GIFVM()

    print(f"🎨 Loading GIF program: {gif_file}")
    if not vm.load_gif(gif_file):
        return

    # Execute program
    print("🚀 Executing GIF program...")
    result = vm.execute()

    # Display results
    print("\n📊 Execution Results:")
    print("=" * 50)

    if result.get('error'):
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Execution successful!")
        print(f"   Cycles executed: {result['cycles_executed']}")
        print(f"   Memory used: {result.get('memory_used', 0)} cells")
        print(f"   Final stack: {result['final_stack']}")

    if result.get('output'):
        print(f"\n📝 Program Output:\n{result['output']}")

    print("\n🎉 GIF-VM execution complete!")

if __name__ == "__main__":
    main()
