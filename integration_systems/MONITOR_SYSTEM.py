#!/usr/bin/env python3
"""
🌌 SYSTEM MONITOR DASHBOARD
Real-time monitoring of the Revolutionary Continuous Learning System
"""

import time
import psutil
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

def get_system_status():
    """Get comprehensive system status."""
    print("🌌 REVOLUTIONARY CONTINUOUS LEARNING SYSTEM - LIVE MONITOR")
    print("="*80)

    # Check if main process is running
    main_process = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'REVOLUTIONARY_CONTINUOUS_LEARNING_SYSTEM.py' in ' '.join(proc.info['cmdline'] or []):
                main_process = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if main_process:
        print(f"✅ Main System: RUNNING (PID: {main_process.pid})")
    else:
        print("❌ Main System: NOT RUNNING")
        return

    # Check subsystem processes
    subsystems = {
        'orchestrator': 'CONTINUOUS_AGENTIC_LEARNING_ORCHESTRATOR.py',
        'knowledge_manager': 'CONTINUOUS_KNOWLEDGE_BASE_MANAGER.py',
        'scraper_system': 'UNIFIED_CONTINUOUS_SCRAPER_SYSTEM.py',
        'backend': 'uvicorn main:app',
        'frontend': 'next dev'
    }

    print("\n🤖 SUBSYSTEMS:")
    print("-" * 50)

    for name, cmd_pattern in subsystems.items():
        found = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if cmd_pattern in cmdline and proc.pid != os.getpid():
                    cpu = proc.info['cpu_percent']
                    mem = proc.info['memory_percent']
                    print(f"✅ {name}: RUNNING (PID: {proc.info['pid']}, CPU: {cpu:.1f}%, MEM: {mem:.1f}%)")
                    found = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not found:
            print(f"❌ {name}: NOT RUNNING")

    # System Resources
    print("\n💻 SYSTEM RESOURCES:")
    print("-" * 50)
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")
    print(f"Disk: {disk.percent:.1f}% ({disk.used/1024/1024/1024:.1f}GB/{disk.total/1024/1024/1024:.1f}GB)")

    # Knowledge Base Stats
    print("\n🧠 KNOWLEDGE BASE:")
    print("-" * 50)

    knowledge_db = Path("research_data/continuous_knowledge_base.db")
    if knowledge_db.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(knowledge_db))
            cursor = conn.cursor()

            # Count fragments
            cursor.execute("SELECT COUNT(*) FROM knowledge_fragments")
            fragment_count = cursor.fetchone()[0]

            # Count integrated fragments
            cursor.execute("SELECT COUNT(*) FROM knowledge_fragments WHERE integration_status = 'integrated'")
            integrated_count = cursor.fetchone()[0]

            # Count high-quality fragments
            cursor.execute("SELECT COUNT(*) FROM knowledge_fragments WHERE quality_score > 0.8")
            high_quality_count = cursor.fetchone()[0]

            print(f"Total Knowledge Fragments: {fragment_count}")
            print(f"Integrated Fragments: {integrated_count}")
            print(f"High-Quality Fragments: {high_quality_count}")

            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    else:
        print("Knowledge base not yet initialized")

    # Recent Activity
    print("\n📊 RECENT ACTIVITY:")
    print("-" * 50)

    # Check log files for recent activity
    log_files = [
        'revolutionary_learning_system.log',
        'continuous_orchestrator.log',
        'continuous_knowledge_manager.log',
        'unified_scraper_system.log'
    ]

    for log_file in log_files:
        if Path(log_file).exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-3:]  # Last 3 lines
                    if lines:
                        log_name = log_file.replace('.log', '').replace('_', ' ').title()
                        print(f"\n{log_name}:")
                        for line in lines:
                            # Extract just the message part
                            if ' - ' in line:
                                message = line.split(' - ', 3)[-1].strip()
                                print(f"  • {message}")
            except Exception as e:
                print(f"Error reading {log_file}: {e}")

    print("\n" + "="*80)

def main():
    """Main monitoring loop."""
    try:
        while True:
            # Clear screen (Unix/Linux/Mac)
            print("\033c", end="")

            get_system_status()

            print("\n🔄 Refreshing in 10 seconds... (Ctrl+C to exit)")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    main()
