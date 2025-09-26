#!/usr/bin/env python3
"""
CODE QUALITY CHECK SCRIPT
Runs comprehensive code quality checks on the entire codebase

This script performs:
- Code formatting with black
- Import sorting with isort
- Linting with flake8
- Type checking with mypy
- Test coverage analysis
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

class CodeQualityChecker:
    """Comprehensive code quality checker"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path.cwd())
        self.results = {}

    def run_command(self, command: List[str], cwd: Path = None) -> Tuple[bool, str, str]:
        """Run a command and return success status with output"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except FileNotFoundError:
            return False, "", f"Command not found: {command[0]}"

    def check_black(self) -> Dict[str, Any]:
        """Check code formatting with black"""
        print("🔍 Checking code formatting with black...")
        success, stdout, stderr = self.run_command(["black", "--check", "--diff", "."])

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": "black --check --diff ."
        }

    def check_isort(self) -> Dict[str, Any]:
        """Check import sorting with isort"""
        print("🔍 Checking import sorting with isort...")
        success, stdout, stderr = self.run_command(["isort", "--check-only", "--diff", "."])

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": "isort --check-only --diff ."
        }

    def check_flake8(self) -> Dict[str, Any]:
        """Check code linting with flake8"""
        print("🔍 Checking code linting with flake8...")
        success, stdout, stderr = self.run_command(["flake8", "."])

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": "flake8 ."
        }

    def check_mypy(self) -> Dict[str, Any]:
        """Check type hints with mypy"""
        print("🔍 Checking type hints with mypy...")
        success, stdout, stderr = self.run_command(["mypy", "."])

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": "mypy ."
        }

    def run_tests_with_coverage(self) -> Dict[str, Any]:
        """Run tests with coverage analysis"""
        print("🧪 Running tests with coverage...")
        success, stdout, stderr = self.run_command([
            "python", "-m", "pytest",
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-fail-under=80"
        ])

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "command": "pytest --cov=. --cov-report=term-missing --cov-fail-under=80"
        }

    def fix_formatting(self) -> Dict[str, Any]:
        """Auto-fix formatting issues"""
        print("🔧 Auto-fixing code formatting...")

        # Fix with black
        success_black, _, _ = self.run_command(["black", "."])

        # Fix imports with isort
        success_isort, _, _ = self.run_command(["isort", "."])

        return {
            "success": success_black and success_isort,
            "black_success": success_black,
            "isort_success": success_isort
        }

    def run_all_checks(self, fix: bool = False) -> Dict[str, Any]:
        """Run all code quality checks"""

        print("🚀 Starting comprehensive code quality check...")
        print("=" * 60)

        # Run all checks
        self.results = {
            "black": self.check_black(),
            "isort": self.check_isort(),
            "flake8": self.check_flake8(),
            "mypy": self.check_mypy(),
            "tests": self.run_tests_with_coverage()
        }

        # Auto-fix if requested
        if fix:
            self.results["autofix"] = self.fix_formatting()

        return self.results

    def print_report(self) -> None:
        """Print comprehensive report"""

        print("\n" + "=" * 60)
        print("CODE QUALITY REPORT")
        print("=" * 60)

        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result.get("success", False))

        print(f"\n📊 SUMMARY: {passed_checks}/{total_checks} checks passed")

        for check_name, result in self.results.items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            print(f"\n{check_name.upper()}: {status}")

            if not result.get("success", False):
                if result.get("stdout"):
                    print("  STDOUT:")
                    for line in result["stdout"].split('\n')[:10]:  # First 10 lines
                        if line.strip():
                            print(f"    {line}")
                if result.get("stderr"):
                    print("  STDERR:")
                    for line in result["stderr"].split('\n')[:10]:  # First 10 lines
                        if line.strip():
                            print(f"    {line}")

        print("\n" + "=" * 60)

        if passed_checks == total_checks:
            print("🎉 ALL CODE QUALITY CHECKS PASSED!")
        else:
            print("⚠️  SOME CHECKS FAILED - REVIEW OUTPUT ABOVE")
            print("\n💡 QUICK FIXES:")
            print("  Run: python scripts/code_quality_check.py --fix")
            print("  Or manually:")
            print("    black .")
            print("    isort .")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Code Quality Checker")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting issues")
    parser.add_argument("--project-root", help="Project root directory")

    args = parser.parse_args()

    checker = CodeQualityChecker(args.project_root)
    results = checker.run_all_checks(fix=args.fix)
    checker.print_report()

    # Exit with appropriate code
    all_passed = all(result.get("success", False) for result in results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
