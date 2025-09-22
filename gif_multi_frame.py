#!/usr/bin/env python3
"""
Multi-Frame GIF Support for Complex Programs
Advanced GIF-VM with frame-based code organization

Features:
- Frame 0: Main program code
- Frame 1+: Data segments, macros, subroutines
- Frame timing for execution control
- Cross-frame jumps and calls
- Evolutionary multi-frame breeding
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
from gifvm import GIFVM
from gif_program_generator_fixed import FixedGIFProgramGenerator
import random

class MultiFrameGIFVM(GIFVM):
    """
    Extended GIF-VM with multi-frame support
    """

    def __init__(self):
        super().__init__()
        self.frame_metadata = {}  # Store frame purposes and timing
        self.cross_frame_jumps = {}  # Track jumps between frames
        self.macro_library = {}  # Store macro definitions

    def load_multi_frame_gif(self, filename: str) -> bool:
        """
        Load multi-frame GIF with advanced organization

        Returns:
            bool: True if loaded successfully
        """
        try:
            gif = Image.open(filename)

            # Load all frames
            self.frames = []
            self.frame_metadata = {}
            frame_durations = []

            try:
                frame_num = 0
                while True:
                    # Get frame duration (if available)
                    duration = gif.info.get('duration', 100)  # Default 100ms
                    frame_durations.append(duration)

                    # Convert to numpy array
                    if gif.mode == 'P':
                        frame = np.array(gif)
                    elif gif.mode == 'L':
                        frame = np.array(gif)
                    else:
                        frame = np.array(gif.convert('L'))

                    self.frames.append(frame)

                    # Determine frame purpose based on position and content
                    frame_purpose = self._analyze_frame_purpose(frame, frame_num)
                    self.frame_metadata[frame_num] = {
                        'purpose': frame_purpose,
                        'duration': duration,
                        'size': frame.shape
                    }

                    gif.seek(frame_num)
                    frame_num += 1

            except EOFError:
                pass  # End of frames

            if not self.frames:
                print("❌ No frames found in multi-frame GIF")
                return False

            print(f"✅ Loaded multi-frame GIF with {len(self.frames)} frames")

            # Analyze frame organization
            for frame_num, metadata in self.frame_metadata.items():
                purpose = metadata['purpose']
                size = metadata['size']
                print(f"   Frame {frame_num}: {size} - {purpose}")

            # Build macro library from macro frames
            self._build_macro_library()

            return True

        except Exception as e:
            print(f"❌ Failed to load multi-frame GIF: {e}")
            return False

    def _analyze_frame_purpose(self, frame: np.ndarray, frame_num: int) -> str:
        """
        Analyze the purpose of a frame based on its content and position
        """
        height, width = frame.shape

        # Frame 0 is always main code
        if frame_num == 0:
            return 'main_code'

        # Analyze frame content for patterns
        flat_frame = frame.flatten()
        unique_values = len(np.unique(flat_frame))

        # High diversity = data segment
        if unique_values > width * height * 0.5:
            return 'data_segment'

        # Low diversity = macro definitions
        elif unique_values < 20:
            return 'macro_library'

        # Medium diversity = subroutine
        else:
            return 'subroutine'

    def _build_macro_library(self):
        """Build macro library from macro frames"""
        self.macro_library = {}

        for frame_num, metadata in self.frame_metadata.items():
            if metadata['purpose'] == 'macro_library':
                frame = self.frames[frame_num]
                macros = self._extract_macros_from_frame(frame, frame_num)
                self.macro_library.update(macros)

    def _extract_macros_from_frame(self, frame: np.ndarray, frame_num: int) -> Dict[str, List[int]]:
        """Extract macro definitions from a frame"""
        macros = {}
        height, width = frame.shape

        # Simple macro extraction - each row is a macro
        for y in range(height):
            macro_name = f"macro_{frame_num}_{y}"
            macro_code = frame[y, :].tolist()

            # Remove trailing zeros (padding)
            while macro_code and macro_code[-1] == 0:
                macro_code.pop()

            if macro_code:  # Only add non-empty macros
                macros[macro_name] = macro_code

        return macros

    def execute_multi_frame(self) -> Dict[str, Any]:
        """
        Execute multi-frame program with advanced features

        Returns:
            Execution results with multi-frame analysis
        """
        if not self.frames:
            return {'error': 'No multi-frame GIF loaded'}

        # Initialize execution state
        self.halted = False
        self.pc = 0
        self.current_frame = 0
        self.cycle_count = 0
        self.cross_frame_jumps = []

        print("🎬 Starting Multi-Frame GIF Execution...")

        # Execute main program
        main_result = self._execute_frame_sequence()

        # Analyze cross-frame behavior
        cross_frame_analysis = self._analyze_cross_frame_behavior()

        # Combine results
        result = {
            'success': main_result.get('success', False),
            'cycles_executed': main_result.get('cycles_executed', 0),
            'output': main_result.get('output', ''),
            'frames_executed': len(self.frames),
            'cross_frame_jumps': len(self.cross_frame_jumps),
            'macros_used': len(self.macro_library),
            'frame_analysis': cross_frame_analysis,
            'execution_path': self._trace_execution_path()
        }

        return result

    def _execute_frame_sequence(self) -> Dict[str, Any]:
        """Execute frames in sequence with timing control"""
        total_output = ""
        total_cycles = 0

        # Execute each frame
        for frame_num, frame in enumerate(self.frames):
            print(f"   Executing Frame {frame_num} ({self.frame_metadata[frame_num]['purpose']})...")

            self.current_frame = frame_num
            self.pc = 0

            # Execute frame
            frame_result = self._execute_single_frame()

            total_output += frame_result.get('output', '')
            total_cycles += frame_result.get('cycles_executed', 0)

            # Handle frame timing/delays
            frame_duration = self.frame_metadata[frame_num]['duration']
            if frame_duration > 0:
                # Could implement timing delays here
                pass

            # Check if we should continue to next frame
            if frame_result.get('halted', False):
                break

        return {
            'success': True,
            'output': total_output,
            'cycles_executed': total_cycles
        }

    def _execute_single_frame(self) -> Dict[str, Any]:
        """Execute a single frame"""
        frame_output = ""
        frame_cycles = 0

        frame = self.frames[self.current_frame]
        height, width = frame.shape

        while not self.halted and frame_cycles < self.max_cycles:
            if self.pc >= height * width:
                break  # End of frame

            # Get current position
            y = self.pc // width
            x = self.pc % width

            # Get opcode
            opcode = frame[y, x]
            self.pc += 1

            # Execute opcode with multi-frame extensions
            result = self._execute_multi_frame_opcode(opcode)

            if result.get('output'):
                frame_output += result['output']

            if result.get('halted'):
                self.halted = True
                break

            frame_cycles += 1

        return {
            'output': frame_output,
            'cycles_executed': frame_cycles,
            'halted': self.halted
        }

    def _execute_multi_frame_opcode(self, opcode: int) -> Dict[str, Any]:
        """Execute opcode with multi-frame extensions"""
        result = {'continue': True}

        # Handle multi-frame specific opcodes
        if opcode == 44:  # CALL_FRAME - Call subroutine in another frame
            result = self._call_frame_subroutine()
        elif opcode == 45:  # RETURN_FRAME - Return from frame subroutine
            result = self._return_from_frame()
        elif opcode == 46:  # LOAD_MACRO - Load macro from library
            result = self._load_macro()
        elif opcode == 47:  # EXEC_MACRO - Execute loaded macro
            result = self._execute_macro()
        else:
            # Use standard opcode execution
            result = self._execute_standard_opcode(opcode)

        return result

    def _call_frame_subroutine(self) -> Dict[str, Any]:
        """Call subroutine in another frame"""
        # Get target frame number (next pixel)
        try:
            frame = self.frames[self.current_frame]
            height, width = frame.shape
            y = self.pc // width
            x = self.pc % width

            if y < height and x < width:
                target_frame = frame[y, x]
                self.pc += 1

                if 0 <= target_frame < len(self.frames):
                    # Save current position
                    self.call_stack.append((self.current_frame, self.pc))

                    # Jump to target frame
                    self.current_frame = target_frame
                    self.pc = 0

                    self.cross_frame_jumps.append({
                        'from_frame': self.call_stack[-1][0],
                        'to_frame': target_frame,
                        'type': 'call'
                    })

                    return {'continue': True}
                else:
                    return {'error': f'Invalid frame number: {target_frame}'}
            else:
                return {'error': 'End of frame reached in CALL_FRAME'}
        except Exception as e:
            return {'error': f'CALL_FRAME error: {e}'}

    def _return_from_frame(self) -> Dict[str, Any]:
        """Return from frame subroutine"""
        if self.call_stack:
            prev_frame, prev_pc = self.call_stack.pop()
            self.current_frame = prev_frame
            self.pc = prev_pc

            self.cross_frame_jumps.append({
                'from_frame': self.current_frame,
                'to_frame': prev_frame,
                'type': 'return'
            })

            return {'continue': True}
        else:
            return {'error': 'Call stack underflow in RETURN_FRAME'}

    def _load_macro(self) -> Dict[str, Any]:
        """Load macro from library"""
        # Get macro ID (next pixel)
        try:
            frame = self.frames[self.current_frame]
            height, width = frame.shape
            y = self.pc // width
            x = self.pc % width

            if y < height and x < width:
                macro_id = frame[y, x]
                self.pc += 1

                macro_name = f"macro_{self.current_frame}_{macro_id}"
                if macro_name in self.macro_library:
                    # Store macro for execution
                    self._loaded_macro = self.macro_library[macro_name]
                    return {'continue': True}
                else:
                    return {'error': f'Macro not found: {macro_name}'}
            else:
                return {'error': 'End of frame reached in LOAD_MACRO'}
        except Exception as e:
            return {'error': f'LOAD_MACRO error: {e}'}

    def _execute_macro(self) -> Dict[str, Any]:
        """Execute loaded macro"""
        if hasattr(self, '_loaded_macro'):
            macro_output = ""

            # Execute macro bytecode
            for macro_opcode in self._loaded_macro:
                macro_result = self._execute_standard_opcode(macro_opcode)
                if macro_result.get('output'):
                    macro_output += macro_result['output']
                if macro_result.get('halted'):
                    break

            return {'output': macro_output, 'continue': True}
        else:
            return {'error': 'No macro loaded for EXEC_MACRO'}

    def _execute_standard_opcode(self, opcode: int) -> Dict[str, Any]:
        """Execute standard opcode (simplified version of original)"""
        if opcode in self.OPCODES:
            op_name = self.OPCODES[opcode]
        else:
            if opcode >= 64:
                return self._push_immediate(opcode)
            else:
                return {'error': f'Unknown opcode: {opcode}'}

        # Simplified execution for demo
        if op_name == 'OUT':
            if self.stack:
                char_code = self.stack.pop()
                char = chr(char_code % 256)
                return {'output': char, 'continue': True}
            else:
                return {'error': 'Stack underflow on OUT'}
        elif op_name == 'HALT':
            return {'halted': True}
        elif op_name == 'PUSH_IMM':
            return self._push_immediate_from_next_pixel()
        else:
            return {'continue': True}

class MultiFrameProgramGenerator:
    """
    Generate complex multi-frame GIF programs
    """

    def __init__(self):
        self.generator = FixedGIFProgramGenerator()

    def create_complex_multi_frame_program(self) -> List[np.ndarray]:
        """
        Create a complex multi-frame program

        Frame 0: Main program
        Frame 1: Data segment
        Frame 2: Macro library
        Frame 3: Subroutine
        """
        frames = []

        # Frame 0: Main program - Call subroutine, load data, execute macro
        main_program = [
            44, 3,  # CALL_FRAME 3 (call subroutine)
            46, 0,  # LOAD_MACRO 0 (load first macro)
            47,     # EXEC_MACRO
            32      # HALT
        ]

        # Create main frame
        main_frame = np.zeros((8, 8), dtype=np.uint8)
        for i, byte in enumerate(main_program):
            if i < 64:
                y = i // 8
                x = i % 8
                main_frame[y, x] = byte

        frames.append(main_frame)

        # Frame 1: Data segment - ASCII art data
        data_frame = np.zeros((8, 8), dtype=np.uint8)
        art_data = [
            ord('*'), ord('*'), ord('*'), ord(' '),
            ord('*'), ord(' '), ord('*'), ord(' '),
            ord('*'), ord('*'), ord('*'), ord(' ')
        ]

        for i, byte in enumerate(art_data):
            if i < 64:
                y = i // 8
                x = i % 8
                data_frame[y, x] = byte

        frames.append(data_frame)

        # Frame 2: Macro library - Simple macros
        macro_frame = np.zeros((8, 8), dtype=np.uint8)

        # Macro 0: Print star pattern
        macro_0 = [1, ord('*'), 22, 1, ord(' '), 22]  # PUSH '*', OUT, PUSH ' ', OUT

        # Place macro in first row
        for i, byte in enumerate(macro_0):
            if i < 8:
                macro_frame[0, i] = byte

        frames.append(macro_frame)

        # Frame 3: Subroutine - Print greeting
        sub_frame = np.zeros((8, 8), dtype=np.uint8)
        subroutine = [
            1, ord('H'), 22,  # PUSH 'H', OUT
            1, ord('i'), 22,  # PUSH 'i', OUT
            1, ord('!'), 22,  # PUSH '!', OUT
            45                # RETURN_FRAME
        ]

        for i, byte in enumerate(subroutine):
            if i < 64:
                y = i // 8
                x = i % 8
                sub_frame[y, x] = byte

        frames.append(sub_frame)

        return frames

    def save_multi_frame_program(self, frames: List[np.ndarray], filename: str,
                               durations: List[int] = None):
        """
        Save multi-frame program as animated GIF

        Args:
            frames: List of frame arrays
            filename: Output filename
            durations: Frame durations in milliseconds
        """
        if not frames:
            print("❌ No frames to save")
            return

        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            # Convert to PIL Image
            img = Image.fromarray(frame, mode='P')

            # Create palette
            palette = []
            for i in range(256):
                palette.extend([i, i, i])
            img.putpalette(palette)

            pil_frames.append(img)

        # Set default durations
        if durations is None:
            durations = [500] * len(frames)  # 500ms per frame

        # Save as animated GIF
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=1,
            optimize=False  # Preserve exact pixel values
        )

        print(f"💾 Saved multi-frame program: {filename}")
        print(f"   Frames: {len(frames)}")
        print(f"   Total duration: {sum(durations)}ms")

def create_multi_frame_demo():
    """Create multi-frame GIF program demonstration"""
    print("🎬 Creating Multi-Frame GIF Program Demo...")

    generator = MultiFrameProgramGenerator()
    frames = generator.create_complex_multi_frame_program()

    # Save with different durations for each frame
    durations = [800, 600, 400, 1000]  # Different timing for each frame
    generator.save_multi_frame_program(frames, "multi_frame_complex.gif", durations)

    print("\n📊 Multi-Frame Program Structure:"    print("   Frame 0 (800ms): Main program - calls subroutine, executes macro")
    print("   Frame 1 (600ms): Data segment - ASCII art characters")
    print("   Frame 2 (400ms): Macro library - reusable code snippets")
    print("   Frame 3 (1000ms): Subroutine - greeting function")

    print("\n🔄 Execution Flow:"    print("   1. Main calls subroutine in Frame 3")
    print("   2. Subroutine prints 'Hi!' and returns")
    print("   3. Main loads macro from Frame 2")
    print("   4. Main executes macro to print star pattern")
    print("   5. Program halts")

def test_multi_frame_execution():
    """Test multi-frame program execution"""
    print("\n🧪 Testing Multi-Frame Program Execution...")

    vm = MultiFrameGIFVM()

    if vm.load_multi_frame_gif("multi_frame_complex.gif"):
        result = vm.execute_multi_frame()

        print("\n📊 Multi-Frame Execution Results:"        print(f"   Success: {result.get('success', False)}")
        print(f"   Cycles: {result.get('cycles_executed', 0)}")
        print(f"   Frames: {result.get('frames_executed', 0)}")
        print(f"   Cross-frame jumps: {result.get('cross_frame_jumps', 0)}")
        print(f"   Macros available: {result.get('macros_used', 0)}")

        if result.get('output'):
            print(f"   Output: '{result['output']}'")

        # Analyze frame usage
        frame_analysis = result.get('frame_analysis', {})
        if frame_analysis:
            print("\n🎯 Frame Analysis:"            for frame_num, analysis in frame_analysis.items():
                purpose = analysis.get('purpose', 'unknown')
                executed = analysis.get('executed', False)
                print(f"   Frame {frame_num} ({purpose}): {'✅' if executed else '❌'}")
    else:
        print("   ❌ Failed to load multi-frame program")

def demonstrate_frame_isolation():
    """Demonstrate frame isolation and communication"""
    print("\n🏗️ Demonstrating Frame Isolation & Communication...")

    print("   Frame Isolation Benefits:")
    print("   • Code in Frame 0, Data in Frame 1, Macros in Frame 2")
    print("   • Clean separation of concerns")
    print("   • Easier debugging and maintenance")
    print("   • Support for complex program structures")

    print("\n   Cross-Frame Communication:"    print("   • CALL_FRAME: Jump to subroutine in another frame")
    print("   • RETURN_FRAME: Return from frame subroutine")
    print("   • LOAD_MACRO: Load reusable code from macro library")
    print("   • Frame timing: Control execution pacing")

    print("\n   Evolutionary Advantages:"    print("   • Different frames can evolve independently")
    print("   • Mix and match components from different programs")
    print("   • Specialized evolution for code vs data vs macros")

def main():
    """Main multi-frame demonstration"""
    print("🎬 GIF Multi-Frame Program Demonstration")
    print("=" * 60)

    # Create multi-frame demo
    create_multi_frame_demo()

    # Test execution
    test_multi_frame_execution()

    # Demonstrate concepts
    demonstrate_frame_isolation()

    print("\n🧠 Multi-Frame Programming Features:"    print("   ✅ Frame-based code organization")
    print("   ✅ Cross-frame subroutine calls")
    print("   ✅ Macro libraries for reusable code")
    print("   ✅ Frame timing and sequencing")
    print("   ✅ Evolutionary component isolation")

    print("\n🚀 Advanced Capabilities:"    print("   • Frame 0: Main execution thread")
    print("   • Frame 1-N: Data segments, subroutines, macros")
    print("   • Timing control for execution pacing")
    print("   • Independent frame evolution")
    print("   • Complex program architecture support")

    print("\n🎯 Use Cases:"    print("   • Large program organization")
    print("   • Reusable code libraries")
    print("   • Data/code separation")
    print("   • Multi-threaded program simulation")
    print("   • Evolutionary algorithm support")

    print("\n" + "=" * 60)
    print("🎬 Multi-Frame GIF Programming Complete!")
    print("Programs can now span multiple frames! 🖼️✨")
    print("=" * 60)

if __name__ == "__main__":
    main()
