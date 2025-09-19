#!/usr/bin/env python3
"""
Test script for GPT Project Exporter
Tests the basic functionality without connecting to ChatGPT
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gpt_project_exporter import GPTProjectExporter

async def test_exporter():
    """Test the exporter functionality"""
    print("🧪 Testing GPT Project Exporter...")
    
    # Create test exporter
    exporter = GPTProjectExporter(
        destination="./test_export",
        project_filter="test",
        headful=False
    )
    
    # Test filename sanitization
    test_filenames = [
        "Normal File Name",
        "File with <invalid> characters",
        "File with spaces and dots...",
        "Very long filename that should be truncated to a reasonable length and not cause any issues with the filesystem",
        "File with /path/separators",
        "File with \"quotes\" and 'apostrophes'"
    ]
    
    print("\n📝 Testing filename sanitization:")
    for filename in test_filenames:
        sanitized = exporter.sanitize_filename(filename)
        print(f"  '{filename}' -> '{sanitized}'")
    
    # Test timestamp formatting
    test_timestamps = [
        "2025-01-15T10:30:00Z",
        "2025-01-15T10:30:00.123Z",
        "invalid-timestamp"
    ]
    
    print("\n⏰ Testing timestamp formatting:")
    for timestamp in test_timestamps:
        formatted = exporter.format_timestamp(timestamp)
        print(f"  '{timestamp}' -> '{formatted}'")
    
    # Test markdown content creation
    test_conversation = {
        'id': 'test-123',
        'title': 'Test Conversation',
        'create_time': '2025-01-15T10:30:00Z',
        'update_time': '2025-01-15T11:45:00Z',
        'current_node': {'parent_id': 'Test Project'}
    }
    
    test_messages = [
        {
            'message': {
                'author': {'role': 'user'},
                'content': {'parts': ['Hello, this is a test message']}
            }
        },
        {
            'message': {
                'author': {'role': 'assistant'},
                'content': {'parts': ['This is a test response with some code:']}
            }
        },
        {
            'message': {
                'author': {'role': 'assistant'},
                'content': {
                    'parts': [
                        {
                            'type': 'code',
                            'language': 'python',
                            'text': 'print("Hello, World!")'
                        }
                    ]
                }
            }
        }
    ]
    
    print("\n📄 Testing markdown content creation:")
    content = exporter.create_markdown_content(test_conversation, test_messages)
    print("Generated content preview:")
    print("-" * 40)
    print(content[:500] + "..." if len(content) > 500 else content)
    print("-" * 40)
    
    # Test destination creation
    print(f"\n📁 Testing destination creation: {exporter.destination}")
    exporter.destination.mkdir(parents=True, exist_ok=True)
    print(f"✅ Destination created: {exporter.destination.exists()}")
    
    # Clean up
    if exporter.destination.exists():
        import shutil
        shutil.rmtree(exporter.destination)
        print("🧹 Test directory cleaned up")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_exporter())
