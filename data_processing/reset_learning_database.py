#!/usr/bin/env python3
"""
Reset and Clean Möbius Loop Learning Database
Fixes corrupted JSON files and prepares for new cybersecurity curriculum
"""

import json
import os
from pathlib import Path
from datetime import datetime

def reset_learning_database():
    """Reset and clean all learning database files."""

    print("🔧 Resetting Möbius Loop Learning Database...")
    print("=" * 60)

    research_dir = Path("research_data")

    # Backup existing files (if they exist)
    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = research_dir / f"backup_{backup_timestamp}"
    backup_dir.mkdir(exist_ok=True)

    files_to_backup = [
        "moebius_learning_objectives.json",
        "moebius_scraping_log.json",
        "moebius_learning_history.json"
    ]

    for filename in files_to_backup:
        filepath = research_dir / filename
        if filepath.exists():
            backup_path = backup_dir / filename
            try:
                with open(filepath, 'rb') as src, open(backup_path, 'wb') as dst:
                    dst.write(src.read())
                print(f"📁 Backed up: {filename}")
            except Exception as e:
                print(f"⚠️  Failed to backup {filename}: {e}")

    # Initialize clean learning objectives database
    print("\n📚 Initializing clean learning objectives database...")

    # First, load the existing objectives to preserve the new cybersecurity subjects
    learning_objectives = {}
    objectives_file = research_dir / "moebius_learning_objectives.json"

    if objectives_file.exists():
        try:
            with open(objectives_file, 'r', encoding='utf-8') as f:
                learning_objectives = json.load(f)
            print("✅ Loaded existing learning objectives")
        except Exception as e:
            print(f"⚠️  Could not load existing objectives: {e}")
            learning_objectives = {}

    # Ensure all subjects have proper structure
    cleaned_objectives = {}
    for subject_name, subject_data in learning_objectives.items():
        # Clean and standardize the subject data
        cleaned_subject = {
            "status": subject_data.get("status", "pending"),
            "completion_percentage": subject_data.get("completion_percentage", 0),
            "prerequisites": subject_data.get("prerequisites", []),
            "category": subject_data.get("category", "general"),
            "difficulty": subject_data.get("difficulty", "intermediate"),
            "estimated_hours": subject_data.get("estimated_hours", 100),
            "description": subject_data.get("description", f"Study of {subject_name}"),
            "sources": subject_data.get("sources", [f"{subject_name}_research", f"{subject_name}_academic"]),
            "last_attempt": subject_data.get("last_attempt"),
            "wallace_completion_score": subject_data.get("wallace_completion_score", 0),
            "learning_efficiency": subject_data.get("learning_efficiency", 0),
            "universal_math_enhancement": subject_data.get("universal_math_enhancement", 1.618033988749895)
        }
        cleaned_objectives[subject_name] = cleaned_subject

    # Save clean objectives
    with open(objectives_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_objectives, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(cleaned_objectives)} clean learning objectives")

    # Initialize clean scraping log
    print("\n🔍 Initializing clean scraping log...")
    scraping_log = {
        "total_sources_scraped": 0,
        "sources_by_status": {
            "pending": [],
            "in_progress": [],
            "completed": [],
            "failed": []
        },
        "scraping_history": []
    }

    scraping_file = research_dir / "moebius_scraping_log.json"
    with open(scraping_file, 'w', encoding='utf-8') as f:
        json.dump(scraping_log, f, indent=2, ensure_ascii=False)

    # Initialize clean learning history
    print("\n📊 Initializing clean learning history...")
    learning_history = {
        "total_iterations": 0,
        "successful_learnings": 0,
        "failed_learnings": 0,
        "average_completion_time": 0,
        "most_valuable_subjects": [],
        "learning_efficiency_trend": [],
        "records": []
    }

    history_file = research_dir / "moebius_learning_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(learning_history, f, indent=2, ensure_ascii=False)

    # Show curriculum summary
    categories = {}
    difficulties = {}

    for subject_name, subject_data in cleaned_objectives.items():
        category = subject_data.get('category', 'general')
        difficulty = subject_data.get('difficulty', 'intermediate')

        categories[category] = categories.get(category, 0) + 1
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

    print("\n🎉 DATABASE RESET COMPLETE!")
    print("=" * 60)
    print(f"📁 Backup created: {backup_dir}")
    print(f"📚 Total subjects: {len(cleaned_objectives)}")
    print(f"🔐 Cybersecurity subjects: {categories.get('cybersecurity', 0)}")
    print(f"💻 Programming subjects: {categories.get('programming', 0)}")
    print(f"🧠 Computer Science subjects: {categories.get('computer_science', 0)}")
    print("\n📊 CURRICULUM SUMMARY:")
    print(f"   Categories: {categories}")
    print(f"   Difficulty Levels: {difficulties}")

    print("\n🚀 READY FOR NEW LEARNING CYCLES!")
    print("The Möbius Loop Trainer is now reset and ready to begin")
    print("learning all the new cybersecurity and programming subjects!")

if __name__ == "__main__":
    reset_learning_database()
