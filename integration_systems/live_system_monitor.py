#!/usr/bin/env python3
"""
Live System Monitor for Revolutionary Continuous Learning System
Shows real-time activity from the enhanced academic learning system
"""

import time
import os
import json
from datetime import datetime
from pathlib import Path

def print_header():
    print("\n" + "="*80)
    print("🌌 REVOLUTIONARY CONTINUOUS LEARNING SYSTEM - LIVE MONITOR")
    print("="*80)
    print("🎓 Enhanced with Premium Academic Sources")
    print("📚 Learning from: arXiv, MIT OCW, Stanford, Harvard, Nature, Science")
    print("="*80)

def check_system_status():
    """Check if main system components are running"""
    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - System Status:")

    # Check for running processes
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout

        components = {
            'orchestrator': False,
            'knowledge_manager': False,
            'scraper_system': False,
            'backend': False,
            'frontend': False
        }

        for component in components:
            if component in processes:
                components[component] = True
                print(f"✅ {component}: RUNNING")
            else:
                print(f"❌ {component}: NOT FOUND")

        return components

    except Exception as e:
        print(f"❌ Error checking processes: {e}")
        return {}

def check_learning_cycles():
    """Check for new learning cycle reports"""
    reports_dir = Path("learning_cycle_reports")
    if not reports_dir.exists():
        print("📊 No learning cycle reports directory found")
        return 0

    reports = list(reports_dir.glob("cycle_report_*.json"))
    print(f"📊 Total learning cycles completed: {len(reports)}")

    if reports:
        # Show latest report
        latest_report = max(reports, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_report, 'r') as f:
                data = json.load(f)

            print(f"🔄 Latest cycle: {data.get('cycle_id', 'Unknown')}")
            print(f"🎯 Objectives: {len(data.get('objectives', []))}")
            print(f"🤖 Systems involved: {len(data.get('systems_involved', []))}")

        except Exception as e:
            print(f"❌ Error reading latest report: {e}")

    return len(reports)

def check_scraping_activity():
    """Check for scraping activity indicators"""
    print("\n🌐 Academic Scraping Activity:")

    # Check for scraped content database
    research_db = Path("research_data/scraped_content.db")
    if research_db.exists():
        size = research_db.stat().st_size
        print(".1f")
    else:
        print("📊 No scraped content database found yet")

    # Check for knowledge fragments
    knowledge_db = Path("research_data/knowledge_fragments.db")
    if knowledge_db.exists():
        size = knowledge_db.stat().st_size
        print(".1f")
    else:
        print("🧠 No knowledge fragments database found yet")

def check_ml_training():
    """Check for ML training activity"""
    print("\n🧠 ML F2 Training Status:")

    # Look for training logs or reports
    training_logs = list(Path(".").glob("*training*.log"))
    if training_logs:
        latest_training = max(training_logs, key=lambda x: x.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest_training.stat().st_mtime)
        print(f"🚀 Latest training log: {latest_training.name}")
        print(f"⏰ Last modified: {mtime.strftime('%H:%M:%S')}")
    else:
        print("📝 No training logs found yet")

def check_academic_sources():
    """Show configured academic sources"""
    print("\n🎓 Configured Academic Sources:")

    sources = [
        "arXiv (Quantum, AI, Physics, Math)",
        "MIT OpenCourseWare (EECS, Physics, Math)",
        "Stanford University (AI, CS Research)",
        "Harvard University (Physics, Sciences)",
        "Nature Journal (Breakthrough Research)",
        "Science Magazine (Scientific Reviews)",
        "Phys.org (Physics News)",
        "Coursera (ML Courses)",
        "edX (AI Education)",
        "Google AI Research",
        "OpenAI Research",
        "DeepMind"
    ]

    for i, source in enumerate(sources, 1):
        print("2d")

def main():
    """Main monitoring loop"""
    print_header()

    cycle_count = 0

    while True:
        try:
            # Check system status
            components = check_system_status()

            # Check learning cycles
            current_cycles = check_learning_cycles()

            # Show activity if cycles increased
            if current_cycles > cycle_count:
                print("\n🎉 NEW LEARNING CYCLE DETECTED!")
                print("🔍 System is actively learning from academic sources...")
                cycle_count = current_cycles

            # Check other activities
            check_scraping_activity()
            check_ml_training()

            # Show academic sources reminder
            if cycle_count == 0:
                check_academic_sources()

            print("\n" + "-"*50)
            print("🔄 Monitoring... (Press Ctrl+C to stop)")

            # Wait before next check
            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
