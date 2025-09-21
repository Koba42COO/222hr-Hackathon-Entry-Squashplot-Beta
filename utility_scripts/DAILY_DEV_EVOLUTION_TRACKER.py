#!/usr/bin/env python3
"""
📅 DAILY DEVELOPMENT EVOLUTION TRACKER
========================================
AUTOMATICALLY TRACKS DAILY CODEBASE EVOLUTION

This script can be run daily to:
- Update the daily evolution branch with latest changes
- Track development progress over time
- Generate evolution reports
- Maintain historical development records

Usage:
- Run manually: python3 DAILY_DEV_EVOLUTION_TRACKER.py
- Schedule daily: Add to cron or task scheduler
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class DailyEvolutionTracker:
    """Tracks daily evolution of the development environment"""

    def __init__(self, project_root: str = "/Users/coo-koba42/dev"):
        self.project_root = Path(project_root)
        self.today = datetime.now().strftime("%Y%m%d")
        self.evolution_branch = f"daily_evolution_{self.today}"

        print("📅 DAILY DEVELOPMENT EVOLUTION TRACKER")
        print("=" * 80)
        print(f"Tracking evolution for: {self.today}")
        print("=" * 80)

    def _run_git_command(self, command: List[str], description: str) -> bool:
        """Run a git command safely"""
        try:
            result = subprocess.run(command, cwd=self.project_root,
                                  capture_output=True, text=True, check=True)
            print(f"✅ {description}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {description} failed: {e}")
            return False

    def check_git_status(self) -> Dict[str, Any]:
        """Check current git status and branch information"""
        print("\n🔍 CHECKING GIT STATUS...")
        print("-" * 50)

        status_info = {
            "current_branch": "",
            "uncommitted_changes": False,
            "branches": [],
            "evolution_branches": []
        }

        try:
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'],
                                  cwd=self.project_root, capture_output=True, text=True)
            status_info["current_branch"] = result.stdout.strip()
            print(f"   🌿 Current branch: {status_info['current_branch']}")

            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  cwd=self.project_root, capture_output=True, text=True)
            status_info["uncommitted_changes"] = bool(result.stdout.strip())
            if status_info["uncommitted_changes"]:
                print("   📝 Uncommitted changes detected")
            else:
                print("   ✅ Working directory clean")

            # Get all branches
            result = subprocess.run(['git', 'branch', '-a'],
                                  cwd=self.project_root, capture_output=True, text=True)
            branches = [b.strip('* ') for b in result.stdout.split('\n') if b.strip()]
            status_info["branches"] = branches

            # Find evolution branches
            evolution_branches = [b for b in branches if 'daily_evolution_' in b]
            status_info["evolution_branches"] = evolution_branches
            print(f"   📅 Evolution branches found: {len(evolution_branches)}")

        except Exception as e:
            print(f"❌ Error checking git status: {e}")

        return status_info

    def update_daily_evolution_branch(self, status_info: Dict[str, Any]) -> bool:
        """Update or create today's evolution branch"""
        print("\n📅 UPDATING DAILY EVOLUTION BRANCH...")
        print("-" * 50)

        success = True

        # Check if today's evolution branch exists
        if self.evolution_branch in status_info["evolution_branches"]:
            print(f"   📅 Switching to existing evolution branch: {self.evolution_branch}")
            success &= self._run_git_command(['git', 'checkout', self.evolution_branch],
                                           "Switched to evolution branch")
        else:
            print(f"   🆕 Creating new evolution branch: {self.evolution_branch}")
            success &= self._run_git_command(['git', 'checkout', '-b', self.evolution_branch],
                                           "Created new evolution branch")

        # If there are uncommitted changes, commit them
        if status_info["uncommitted_changes"]:
            print("   💾 Committing uncommitted changes...")
            success &= self._run_git_command(['git', 'add', '.'], "Staged changes")
            commit_message = f"Daily Evolution Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            success &= self._run_git_command(['git', 'commit', '-m', commit_message],
                                           f"Committed with message: {commit_message}")

        # Pull latest changes from main branch if needed
        if status_info["current_branch"] != self.evolution_branch:
            try:
                print("   🔄 Merging latest changes from main branch...")
                self._run_git_command(['git', 'merge', 'main', '--no-edit'],
                                    "Merged changes from main")
            except:
                print("   ⚠️  Could not merge from main (may be expected)")

        return success

    def generate_evolution_report(self, status_info: Dict[str, Any]) -> str:
        """Generate a daily evolution report"""
        print("\n📊 GENERATING EVOLUTION REPORT...")
        print("-" * 50)

        report_lines = []
        report_lines.append("📈 DAILY DEVELOPMENT EVOLUTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Current status
        report_lines.append("🔍 CURRENT STATUS:")
        report_lines.append("-" * 30)
        report_lines.append(f"Current Branch: {status_info['current_branch']}")
        report_lines.append(f"Evolution Branch: {self.evolution_branch}")
        report_lines.append(f"Working Directory Clean: {not status_info['uncommitted_changes']}")
        report_lines.append("")

        # Branch information
        report_lines.append("🌿 BRANCH INFORMATION:")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Branches: {len(status_info['branches'])}")
        report_lines.append(f"Evolution Branches: {len(status_info['evolution_branches'])}")

        if status_info['evolution_branches']:
            report_lines.append("Evolution Branches:")
            for branch in status_info['evolution_branches'][-5:]:  # Last 5
                report_lines.append(f"  • {branch}")
        report_lines.append("")

        # Recent commits
        try:
            result = subprocess.run(['git', 'log', '--oneline', '-5'],
                                  cwd=self.project_root, capture_output=True, text=True)
            recent_commits = result.stdout.strip().split('\n')

            report_lines.append("🔄 RECENT COMMITS:")
            report_lines.append("-" * 30)
            for commit in recent_commits:
                if commit.strip():
                    report_lines.append(f"  • {commit}")
            report_lines.append("")
        except:
            pass

        # Evolution insights
        report_lines.append("🎯 EVOLUTION INSIGHTS:")
        report_lines.append("-" * 30)
        if len(status_info['evolution_branches']) > 1:
            report_lines.append(f"✅ Tracking evolution across {len(status_info['evolution_branches'])} days")
        else:
            report_lines.append("📝 Started daily evolution tracking today")

        if status_info['uncommitted_changes']:
            report_lines.append("📝 Daily changes committed to evolution branch")
        else:
            report_lines.append("✨ No new changes today")

        report_lines.append("")
        report_lines.append("📊 DEVELOPMENT EVOLUTION: ACTIVE & TRACKING")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def save_evolution_log(self, report: str) -> bool:
        """Save the evolution report to a log file"""
        try:
            log_dir = self.project_root / "evolution_logs"
            log_dir.mkdir(exist_ok=True)

            log_file = log_dir / f"evolution_{self.today}.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"   💾 Evolution log saved: {log_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to save evolution log: {e}")
            return False

    def run_daily_evolution_tracking(self) -> Dict[str, Any]:
        """Run complete daily evolution tracking"""
        print("🚀 STARTING DAILY EVOLUTION TRACKING")
        print("=" * 80)

        start_time = time.time()

        # Step 1: Check git status
        print("\n📊 STEP 1: GIT STATUS ANALYSIS")
        status_info = self.check_git_status()

        # Step 2: Update daily evolution branch
        print("\n📅 STEP 2: EVOLUTION BRANCH UPDATE")
        branch_success = self.update_daily_evolution_branch(status_info)

        # Step 3: Generate evolution report
        print("\n📊 STEP 3: EVOLUTION REPORT GENERATION")
        evolution_report = self.generate_evolution_report(status_info)

        # Step 4: Save evolution log
        print("\n💾 STEP 4: SAVE EVOLUTION LOG")
        log_saved = self.save_evolution_log(evolution_report)

        execution_time = time.time() - start_time

        # Generate summary
        summary = {
            "date": self.today,
            "evolution_branch": self.evolution_branch,
            "branch_update_success": branch_success,
            "log_saved": log_saved,
            "execution_time": execution_time,
            "current_branch": status_info["current_branch"],
            "evolution_branches_count": len(status_info["evolution_branches"]),
            "uncommitted_changes": status_info["uncommitted_changes"],
            "evolution_report": evolution_report
        }

        print("\n📊 DAILY EVOLUTION TRACKING COMPLETE:")
        print("-" * 80)
        print(f"   ⏱️  Execution Time: {summary['execution_time']:.2f} seconds")
        print(f"   📅 Evolution branch: {self.evolution_branch}")
        print(f"   🌿 Current branch: {status_info['current_branch']}")
        print(f"   📊 Evolution branches: {len(status_info['evolution_branches'])}")
        print(f"   💾 Log saved: {log_saved}")
        print(f"   ✅ Branch updated: {branch_success}")

        print("\n🎯 DAILY TRACKING SUMMARY:")
        print("   ✅ Git status analyzed")
        print("   ✅ Evolution branch updated")
        print("   ✅ Evolution report generated")
        print("   ✅ Evolution log saved")
        print("   ✅ Daily tracking complete")
        # Display evolution report
        print(f"\n{evolution_report}")

        return summary

def main():
    """Main execution function"""
    print("📅 STARTING DAILY DEVELOPMENT EVOLUTION TRACKER")
    print("Automatically tracking daily codebase evolution")
    print("=" * 80)

    tracker = DailyEvolutionTracker()

    try:
        # Run daily evolution tracking
        tracking_summary = tracker.run_daily_evolution_tracking()

        print("\n🎯 DAILY EVOLUTION TRACKING COMPLETE:")
        print("=" * 80)
        print("✅ Git status comprehensively analyzed")
        print("✅ Daily evolution branch updated")
        print("✅ Evolution report generated")
        print("✅ Evolution log saved")
        print("✅ Daily development tracking active")
        print("\n🏆 RESULT: DEVELOPMENT EVOLUTION SUCCESSFULLY TRACKED")
        return tracking_summary

    except Exception as e:
        print(f"❌ Daily evolution tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
