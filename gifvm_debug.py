#!/usr/bin/env python3
"""
GIF-VM Debug Version - Simplified for testing
"""

import sys
import numpy as np
from PIL import Image

def debug_gif_pixels(filename):
    """Debug function to examine GIF pixel values"""
    print(f"🔍 Debugging GIF: {filename}")

    # Open GIF
    gif = Image.open(filename)
    print(f"   Mode: {gif.mode}")
    print(f"   Size: {gif.size}")
    print(f"   Format: {gif.format}")

    # Convert to numpy array
    if gif.mode == 'P':
        # Palette mode - get raw palette indices
        pixels = np.array(gif)
    elif gif.mode == 'L':
        # Luminance mode - already grayscale values
        pixels = np.array(gif)
    else:
        # Convert to grayscale
        gray_gif = gif.convert('L')
        pixels = np.array(gray_gif)

    print(f"   Final mode: {gif.mode}")
    print(f"   Pixel shape: {pixels.shape}")
    print(f"   Pixel dtype: {pixels.dtype}")
    print(f"   Pixel range: {pixels.min()} - {pixels.max()}")

    # Print first few pixels
    print("   First 20 pixels:")
    flat_pixels = pixels.flatten()[:20]
    for i, pixel in enumerate(flat_pixels):
        print("2d")

    # Check if we have our expected bytecode (for hello program)
    expected_start = [1, 72, 22, 1, 73, 22, 1, 10, 22, 32]  # PUSH_IMM, 'H', OUT, etc.
    actual_start = flat_pixels[:10]
    print(f"\n   Expected start: {expected_start}")
    print(f"   Actual start:   {actual_start}")
    print(f"   Match: {np.array_equal(actual_start, expected_start)}")

    return pixels

def simple_vm_test(pixels):
    """Simple VM test with manual bytecode"""
    print("\n🧠 Running Simple VM Test...")

    # Manual bytecode for "HI\n"
    manual_bytecode = [1, 72, 22, 1, 73, 22, 1, 10, 22, 32]  # PUSH_IMM, 'H', OUT, etc.

    stack = []
    output = ""
    pc = 0

    print(f"   Manual bytecode: {manual_bytecode}")

    while pc < len(manual_bytecode):
        opcode = manual_bytecode[pc]
        pc += 1

        print(f"   PC={pc-1}: opcode={opcode}")

        if opcode == 1:  # PUSH_IMM
            if pc < len(manual_bytecode):
                value = manual_bytecode[pc]
                pc += 1
                stack.append(value)
                print(f"     PUSH_IMM {value}, stack={stack}")
        elif opcode == 22:  # OUT
            if stack:
                char_code = stack.pop()
                char = chr(char_code)
                output += char
                print(f"     OUT '{char}' ({char_code}), output='{output}'")
        elif opcode == 32:  # HALT
            print(f"     HALT, final output='{output}'")
            break
        else:
            print(f"     Unknown opcode {opcode}")

    return output

def test_pixel_reading():
    """Test different ways of reading pixels"""
    print("\n🔬 Testing Pixel Reading Methods...")

    # Create a simple test GIF programmatically
    test_pixels = np.array([
        [1, 72, 22, 0],
        [1, 73, 22, 0],
        [1, 10, 22, 0],
        [32, 0, 0, 0]
    ], dtype=np.uint8)

    print(f"   Test pixels shape: {test_pixels.shape}")
    print(f"   Test pixels:\n{test_pixels}")

    # Save as GIF
    img = Image.fromarray(test_pixels, mode='P')
    palette = []
    for i in range(256):
        palette.extend([i, i, i])
    img.putpalette(palette)
    img.save("debug_test.gif")

    print("   Saved debug_test.gif")

    # Read it back
    loaded = Image.open("debug_test.gif")
    loaded_pixels = np.array(loaded)

    print(f"   Loaded pixels:\n{loaded_pixels}")
    print(f"   Match: {np.array_equal(test_pixels, loaded_pixels)}")

    # Test VM on loaded pixels
    result = simple_vm_test(loaded_pixels.flatten())
    print(f"   VM result: '{result}'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python gifvm_debug.py <gif_file>")
        print("Or: python gifvm_debug.py --test-pixel-reading")
        return

    if sys.argv[1] == "--test-pixel-reading":
        test_pixel_reading()
        return

    filename = sys.argv[1]

    # Debug the GIF
    pixels = debug_gif_pixels(filename)

    # Test simple VM
    flat_pixels = pixels.flatten()
    result = simple_vm_test(flat_pixels)
    print(f"\n🎯 Final result: '{result}'")

if __name__ == "__main__":
    main()
