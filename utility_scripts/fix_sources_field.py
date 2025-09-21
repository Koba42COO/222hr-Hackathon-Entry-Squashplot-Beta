#!/usr/bin/env python3
"""
Fix Sources Field in Möbius Learning Objectives
Adds missing sources field to all subjects
"""

import json
from pathlib import Path

def fix_sources_field():
    """Add sources field to all subjects that don't have it."""
    learning_db = Path("research_data/moebius_learning_objectives.json")

    if not learning_db.exists():
        print("❌ Learning objectives file not found!")
        return

    try:
        with open(learning_db, 'r') as f:
            objectives = json.load(f)

        fixed_count = 0
        for subject, data in objectives.items():
            if "sources" not in data:
                # Add default sources based on subject name
                objectives[subject]["sources"] = [
                    f"{subject}_research",
                    f"{subject}_academic",
                    f"{subject}_wikipedia"
                ]
                fixed_count += 1
                print(f"🔧 Added sources to {subject}")

        if fixed_count > 0:
            with open(learning_db, 'w') as f:
                json.dump(objectives, f, indent=2)
            print(f"✅ Successfully added sources to {fixed_count} subjects")
        else:
            print("ℹ️  All subjects already have sources field")

    except Exception as e:
        print(f"❌ Error fixing sources: {e}")

if __name__ == "__main__":
    fix_sources_field()
