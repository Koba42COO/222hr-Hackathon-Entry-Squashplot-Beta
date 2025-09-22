#!/usr/bin/env python3
"""
Reset Failed Subjects in Möbius Learning Tracker
This script fixes subjects that are stuck in "failed" status
"""

import json
from pathlib import Path

def reset_failed_subjects():
    """Reset all failed subjects back to pending status."""
    learning_db = Path("research_data/moebius_learning_objectives.json")

    if not learning_db.exists():
        print("❌ Learning objectives file not found!")
        return

    try:
        with open(learning_db, 'r') as f:
            objectives = json.load(f)

        reset_count = 0
        for subject, data in objectives.items():
            if data.get("status") == "failed":
                # Reset to pending and clear completion percentage if it was 0
                objectives[subject]["status"] = "pending"
                if data.get("completion_percentage", 0) == 0:
                    objectives[subject]["completion_percentage"] = 0
                reset_count += 1
                print(f"🔄 Reset {subject} from 'failed' to 'pending'")

        if reset_count > 0:
            with open(learning_db, 'w') as f:
                json.dump(objectives, f, indent=2)
            print(f"✅ Successfully reset {reset_count} failed subjects")
        else:
            print("ℹ️  No failed subjects found to reset")

    except Exception as e:
        print(f"❌ Error resetting subjects: {e}")

if __name__ == "__main__":
    reset_failed_subjects()
