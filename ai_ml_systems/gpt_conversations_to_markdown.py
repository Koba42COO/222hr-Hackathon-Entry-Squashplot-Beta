#!/usr/bin/env python3
"""
GPT Conversations to Markdown Converter
Converts GPT conversation JSON files to readable Markdown format

Usage:
    python3 gpt_conversations_to_markdown.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

def convert_gpt_conversations_to_markdown(src_dir: str = "~/dev/gpt/conversations", 
                                        dst_dir: str = "~/dev/gpt/markdown") -> None:
    """
    Convert GPT conversation JSON files to Markdown format
    
    Args:
        src_dir: Source directory containing JSON conversation files
        dst_dir: Destination directory for Markdown files
    """
    
    # Expand user paths
    src_path = Path(os.path.expanduser(src_dir))
    dst_path = Path(os.path.expanduser(dst_dir))
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Scanning for JSON files in: {src_path}")
    print(f"📁 Output directory: {dst_path}")
    
    # Check if source directory exists
    if not src_path.exists():
        print(f"❌ Source directory does not exist: {src_path}")
        print("📝 Creating sample conversation file...")
        create_sample_conversation(src_path)
    
    # Find all JSON files
    json_files = list(src_path.glob("*.json"))
    
    if not json_files:
        print("⚠️ No JSON files found in source directory")
        return
    
    print(f"📄 Found {len(json_files)} JSON file(s)")
    
    converted_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            print(f"🔄 Converting: {json_file.name}")
            
            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract conversation text
            conversation_parts = []
            
            if 'mapping' in data:
                # Extract messages from mapping
                for msg_id, msg_data in data['mapping'].items():
                    if 'content' in msg_data and 'parts' in msg_data['content']:
                        for part in msg_data['content']['parts']:
                            if isinstance(part, str) and part.strip():
                                conversation_parts.append(part.strip())
            elif 'messages' in data:
                # Alternative format with messages array
                for msg in data['messages']:
                    if 'content' in msg and isinstance(msg['content'], str):
                        conversation_parts.append(msg['content'].strip())
            else:
                # Try to find any text content
                conversation_parts = extract_text_from_json(data)
            
            if not conversation_parts:
                print(f"⚠️ No conversation content found in: {json_file.name}")
                continue
            
            # Join conversation parts
            conversation_text = "\n\n".join(conversation_parts)
            
            # Create markdown file
            markdown_file = dst_path / f"{json_file.stem}.md"
            
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(conversation_text)
            
            print(f"✅ Converted: {json_file.name} → {markdown_file.name}")
            converted_count += 1
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in {json_file.name}: {e}")
            error_count += 1
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            error_count += 1
    
    # Summary
    print(f"\n📊 Conversion Summary:")
    print(f"  ✅ Successfully converted: {converted_count} file(s)")
    print(f"  ❌ Errors: {error_count} file(s)")
    print(f"  📁 Output directory: {dst_path}")

def extract_text_from_json(data: Any) -> List[str]:
    """
    Recursively extract text content from JSON structure
    
    Args:
        data: JSON data structure
        
    Returns:
        List of text strings found in the data
    """
    text_parts = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ['content', 'text', 'message', 'parts']:
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            text_parts.append(item)
                        else:
                            text_parts.extend(extract_text_from_json(item))
                else:
                    text_parts.extend(extract_text_from_json(value))
            else:
                text_parts.extend(extract_text_from_json(value))
    elif isinstance(data, list):
        for item in data:
            text_parts.extend(extract_text_from_json(item))
    
    return text_parts

def create_sample_conversation(src_path: Path) -> None:
    """
    Create a sample conversation file for testing
    
    Args:
        src_path: Path to create the sample file
    """
    src_path.mkdir(parents=True, exist_ok=True)
    
    sample_data = {
        "mapping": {
            "msg_1": {
                "content": {
                    "parts": ["Hello! How can I help you today?"]
                }
            },
            "msg_2": {
                "content": {
                    "parts": ["I need help with a Python script."]
                }
            },
            "msg_3": {
                "content": {
                    "parts": ["I'd be happy to help! What kind of Python script are you working on?"]
                }
            },
            "msg_4": {
                "content": {
                    "parts": ["I want to convert JSON files to Markdown format."]
                }
            },
            "msg_5": {
                "content": {
                    "parts": ["Perfect! I can help you create a script to convert GPT conversation JSON files to Markdown."]
                }
            }
        }
    }
    
    sample_file = src_path / "sample_conversation.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"📝 Created sample file: {sample_file}")

def main():
    """Main function"""
    print("🔄 GPT Conversations to Markdown Converter")
    print("=" * 50)
    
    # Get command line arguments
    if len(sys.argv) > 1:
        src_dir = sys.argv[1]
    else:
        src_dir = "~/dev/gpt/conversations"
    
    if len(sys.argv) > 2:
        dst_dir = sys.argv[2]
    else:
        dst_dir = "~/dev/gpt/markdown"
    
    try:
        convert_gpt_conversations_to_markdown(src_dir, dst_dir)
        print("\n🎉 Conversion completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Conversion interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
