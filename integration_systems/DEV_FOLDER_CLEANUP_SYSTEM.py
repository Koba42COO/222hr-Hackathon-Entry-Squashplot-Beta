#!/usr/bin/env python3
"""
🧹 DEV FOLDER CLEANUP SYSTEM
===========================

COMPREHENSIVE CLEANUP AND ORGANIZATION OF THE DEVELOPMENT ENVIRONMENT
Transforming a cluttered dev folder into a clean, organized, professional workspace

CLEANUP PHASES:
1. 🔍 ANALYSIS - Analyze current structure and identify cleanup opportunities
2. 🗂️ ORGANIZE - Organize files into proper categories and directories
3. 🧹 CLEAN - Remove unnecessary files, cache, duplicates
4. 📁 RESTRUCTURE - Create optimal directory structure
5. 🔗 INTEGRATE - Ensure all systems work together seamlessly
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib

class DevFolderCleanupSystem:
    """Comprehensive dev folder cleanup and organization system"""

    def __init__(self, root_path="/Users/coo-koba42/dev"):
        self.root_path = Path(root_path)
        self.analysis_results = {}
        self.cleanup_actions = []
        self.backup_created = False

        # Define cleanup categories
        self.cleanup_patterns = {
            'cache_files': [
                '__pycache__', '*.pyc', '*.pyo', '.pytest_cache',
                '.mypy_cache', '.tox', 'htmlcov', '.coverage'
            ],
            'temp_files': [
                '*.tmp', '*.temp', '*.bak', '~', '.DS_Store',
                'Thumbs.db', 'ehthumbs.db', '._*'
            ],
            'log_files': [
                '*.log', 'logs/*.log', '*.log.*'
            ],
            'build_artifacts': [
                'build/', 'dist/', '*.egg-info/', '.eggs/',
                '*.whl', '*.tar.gz', '*.zip'
            ],
            'ide_files': [
                '.vscode/', '.idea/', '*.swp', '*.swo',
                '*~', '.project', '.classpath'
            ]
        }

        # Define optimal directory structure
        self.optimal_structure = {
            'core': ['src/', 'core/', 'main/'],
            'testing': ['tests/', 'test_data/', 'fixtures/'],
            'documentation': ['docs/', 'README.md', 'CHANGELOG.md'],
            'configuration': ['config/', 'settings/', 'conf/'],
            'scripts': ['scripts/', 'bin/', 'tools/'],
            'data': ['data/', 'datasets/', 'research_data/'],
            'logs': ['logs/', 'reports/'],
            'build': ['build/', 'dist/', 'artifacts/'],
            'libraries': ['libs/', 'packages/', 'modules/'],
            'projects': ['projects/', 'experiments/', 'prototypes/'],
            'reports': ['reports/', 'analysis/', 'results/']
        }

    def run_complete_cleanup(self):
        """Run the complete dev folder cleanup process"""
        print("🧹 DEV FOLDER CLEANUP SYSTEM")
        print("=" * 50)
        print("Transforming cluttered dev folder into clean, organized workspace")
        print("=" * 50)

        # Phase 1: Comprehensive Analysis
        self.phase1_comprehensive_analysis()

        # Phase 2: Safety Backup
        self.phase2_safety_backup()

        # Phase 3: File Organization Analysis
        self.phase3_file_organization_analysis()

        # Phase 4: Cleanup Execution
        self.phase4_cleanup_execution()

        # Phase 5: Directory Restructuring
        self.phase5_directory_restructuring()

        # Phase 6: Integration Verification
        self.phase6_integration_verification()

        # Generate final report
        self.generate_final_report()

    def phase1_comprehensive_analysis(self):
        """Phase 1: Comprehensive analysis of current structure"""
        print("\n🔍 PHASE 1: COMPREHENSIVE ANALYSIS")
        print("-" * 40)

        # Count files and directories
        total_files = sum(1 for _ in self.root_path.rglob('*') if _.is_file())
        total_dirs = sum(1 for _ in self.root_path.rglob('*') if _.is_dir())

        print(f"📊 Total Files: {total_files}")
        print(f"📁 Total Directories: {total_dirs}")

        # Analyze file types
        file_extensions = defaultdict(int)
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_extensions[ext] += 1

        print(f"📄 File Types Found: {len(file_extensions)}")

        # Analyze directory structure
        dir_structure = {}
        for item in self.root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                dir_structure[item.name] = {
                    'files': file_count,
                    'size': sum(_.stat().st_size for _ in item.rglob('*') if _.is_file())
                }

        # Identify cleanup candidates
        cleanup_candidates = {
            'cache_files': [],
            'temp_files': [],
            'empty_dirs': [],
            'large_files': [],
            'duplicate_files': []
        }

        # Find cache files
        for pattern in self.cleanup_patterns['cache_files']:
            for item in self.root_path.rglob(pattern):
                cleanup_candidates['cache_files'].append(str(item))

        # Find temp files
        for pattern in self.cleanup_patterns['temp_files']:
            for item in self.root_path.rglob(pattern):
                cleanup_candidates['temp_files'].append(str(item))

        # Find empty directories
        for dir_path in self.root_path.rglob('*'):
            if dir_path.is_dir():
                try:
                    if not list(dir_path.iterdir()):
                        cleanup_candidates['empty_dirs'].append(str(dir_path))
                except PermissionError:
                    continue

        # Find large files (>100MB)
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                        cleanup_candidates['large_files'].append(str(file_path))
                except (OSError, PermissionError):
                    continue

        self.analysis_results = {
            'total_files': total_files,
            'total_dirs': total_dirs,
            'file_extensions': dict(file_extensions),
            'dir_structure': dir_structure,
            'cleanup_candidates': cleanup_candidates,
            'analysis_timestamp': datetime.now().isoformat()
        }

        print(f"🗂️ Cache Files to Clean: {len(cleanup_candidates['cache_files'])}")
        print(f"🗑️ Temp Files to Clean: {len(cleanup_candidates['temp_files'])}")
        print(f"📁 Empty Directories: {len(cleanup_candidates['empty_dirs'])}")
        print(f"📂 Large Files (>100MB): {len(cleanup_candidates['large_files'])}")

    def phase2_safety_backup(self):
        """Phase 2: Create safety backup before cleanup"""
        print("\n💾 PHASE 2: SAFETY BACKUP")
        print("-" * 40)

        backup_dir = self.root_path.parent / f"dev_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"📦 Creating backup: {backup_dir}")

        try:
            # Create backup of critical files
            critical_files = [
                'pyproject.toml', 'requirements.txt', 'dev-requirements.txt',
                'Makefile', '.editorconfig', 'README.md', 'CONTRIBUTING.md',
                '.gitignore', '.pre-commit-config.yaml'
            ]

            backup_dir.mkdir(parents=True, exist_ok=True)

            for file in critical_files:
                src = self.root_path / file
                if src.exists():
                    shutil.copy2(src, backup_dir / file)
                    print(f"✅ Backed up: {file}")

            self.backup_created = True
            print(f"📦 Backup created successfully at: {backup_dir}")

        except Exception as e:
            print(f"⚠️ Backup creation failed: {e}")

    def phase3_file_organization_analysis(self):
        """Phase 3: Analyze file organization and categorization"""
        print("\n📂 PHASE 3: FILE ORGANIZATION ANALYSIS")
        print("-" * 40)

        # Analyze file distribution
        file_categories = {
            'python': [],
            'config': [],
            'documentation': [],
            'data': [],
            'scripts': [],
            'build': [],
            'misc': []
        }

        # Categorize files
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                suffix = file_path.suffix.lower()
                name = file_path.name.lower()

                if suffix in ['.py', '.pyc', '.pyo']:
                    file_categories['python'].append(str(file_path))
                elif suffix in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
                    file_categories['config'].append(str(file_path))
                elif suffix in ['.md', '.txt', '.rst', '.pdf']:
                    file_categories['documentation'].append(str(file_path))
                elif suffix in ['.csv', '.json', '.pkl', '.h5', '.npy']:
                    file_categories['data'].append(str(file_path))
                elif suffix in ['.sh', '.bash', '.ps1'] or 'script' in name:
                    file_categories['scripts'].append(str(file_path))
                elif any(pattern in str(file_path) for pattern in ['build', 'dist', '__pycache__']):
                    file_categories['build'].append(str(file_path))
                else:
                    file_categories['misc'].append(str(file_path))

        print(f"🐍 Python Files: {len(file_categories['python'])}")
        print(f"⚙️ Config Files: {len(file_categories['config'])}")
        print(f"📚 Documentation: {len(file_categories['documentation'])}")
        print(f"📊 Data Files: {len(file_categories['data'])}")
        print(f"📜 Scripts: {len(file_categories['scripts'])}")
        print(f"🔨 Build Files: {len(file_categories['build'])}")
        print(f"📦 Other Files: {len(file_categories['misc'])}")

        self.analysis_results['file_categories'] = file_categories

        # Identify organizational issues
        organizational_issues = []

        # Check for files in root that should be in subdirectories
        root_files = [f for f in self.root_path.glob('*') if f.is_file() and not f.name.startswith('.')]
        for file in root_files:
            if file.suffix.lower() not in ['.md', '.txt', '.py'] and 'readme' not in file.name.lower():
                organizational_issues.append(f"Root file that could be organized: {file.name}")

        # Check for scattered project files
        scattered_projects = []
        for dir_path in self.root_path.glob('*/'):
            if dir_path.is_dir() and not dir_path.name.startswith('.') and not dir_path.name.startswith('_'):
                py_files = list(dir_path.glob('*.py'))
                if len(py_files) > 5:  # Directory with multiple Python files
                    scattered_projects.append(dir_path.name)

        self.analysis_results['organizational_issues'] = organizational_issues
        self.analysis_results['scattered_projects'] = scattered_projects

        print(f"📋 Root Files to Organize: {len(organizational_issues)}")
        print(f"🏗️ Scattered Projects: {len(scattered_projects)}")

    def phase4_cleanup_execution(self):
        """Phase 4: Execute cleanup operations"""
        print("\n🧹 PHASE 4: CLEANUP EXECUTION")
        print("-" * 40)

        cleanup_count = 0

        # Clean cache files
        print("🗂️ Cleaning cache files...")
        for pattern in self.cleanup_patterns['cache_files']:
            for item in self.root_path.rglob(pattern):
                try:
                    if item.is_file():
                        item.unlink()
                        cleanup_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        cleanup_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove {item}: {e}")

        # Clean temp files
        print("🗑️ Cleaning temporary files...")
        for pattern in self.cleanup_patterns['temp_files']:
            for item in self.root_path.rglob(pattern):
                try:
                    if item.is_file():
                        item.unlink()
                        cleanup_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove {item}: {e}")

        # Clean empty directories
        print("📁 Removing empty directories...")
        for dir_path in reversed(list(self.root_path.rglob('*'))):
            if dir_path.is_dir():
                try:
                    if not list(dir_path.iterdir()):
                        dir_path.rmdir()
                        cleanup_count += 1
                except Exception as e:
                    continue

        print(f"✅ Cleanup completed: {cleanup_count} items removed")

        self.analysis_results['cleanup_count'] = cleanup_count

    def phase5_directory_restructuring(self):
        """Phase 5: Directory restructuring and organization"""
        print("\n🏗️ PHASE 5: DIRECTORY RESTRUCTURING")
        print("-" * 40)

        # Create optimal directory structure
        print("📁 Creating optimal directory structure...")

        for category, dirs in self.optimal_structure.items():
            for dir_name in dirs:
                dir_path = self.root_path / dir_name
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"✅ Created: {dir_name}")

        # Organize scattered files
        print("📂 Organizing scattered files...")

        # Move config files to config/
        config_dir = self.root_path / 'config'
        for file_path in self.root_path.glob('*.cfg'):
            if file_path.name != '.editorconfig':  # Keep editorconfig in root
                try:
                    shutil.move(str(file_path), str(config_dir / file_path.name))
                    print(f"📋 Moved config: {file_path.name} → config/")
                except Exception as e:
                    print(f"⚠️ Failed to move {file_path.name}: {e}")

        # Move scripts to scripts/
        scripts_dir = self.root_path / 'scripts'
        script_patterns = ['*.sh', '*.bash', '*.ps1']
        for pattern in script_patterns:
            for file_path in self.root_path.glob(pattern):
                if not str(file_path).startswith(str(scripts_dir)):
                    try:
                        shutil.move(str(file_path), str(scripts_dir / file_path.name))
                        print(f"📜 Moved script: {file_path.name} → scripts/")
                    except Exception as e:
                        print(f"⚠️ Failed to move {file_path.name}: {e}")

        # Create .gitignore if it doesn't exist
        gitignore_path = self.root_path / '.gitignore'
        if not gitignore_path.exists():
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.pytest_cache/
nosetests.xml
coverage.xml
*.cover
*.log
.cache/
.mypy_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Build
build/
dist/
*.egg-info/
*.whl
*.tar.gz

# Documentation
docs/_build/
*.pdf

# Data
*.csv
*.json
*.pkl
*.h5

# Logs
logs/
*.log
"""

            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)
            print("✅ Created: .gitignore")

    def phase6_integration_verification(self):
        """Phase 6: Integration verification and final checks"""
        print("\n🔗 PHASE 6: INTEGRATION VERIFICATION")
        print("-" * 40)

        # Verify essential files exist
        essential_files = [
            'README.md',
            'pyproject.toml',
            '.editorconfig',
            '.gitignore',
            'Makefile'
        ]

        missing_essentials = []
        for file in essential_files:
            if not (self.root_path / file).exists():
                missing_essentials.append(file)

        if missing_essentials:
            print(f"⚠️ Missing essential files: {missing_essentials}")
        else:
            print("✅ All essential files present")

        # Verify directory structure
        expected_dirs = ['src', 'tests', 'config', 'scripts', 'docs']
        missing_dirs = []
        for dir_name in expected_dirs:
            if not (self.root_path / dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            print(f"⚠️ Missing expected directories: {missing_dirs}")
        else:
            print("✅ Core directory structure complete")

        # Check for obvious duplicates
        python_files = list(self.root_path.rglob('*.py'))
        file_names = defaultdict(list)

        for file_path in python_files:
            file_names[file_path.name].append(file_path)

        duplicates = {name: paths for name, paths in file_names.items() if len(paths) > 1}
        if duplicates:
            print(f"⚠️ Potential duplicate files found: {len(duplicates)}")
            for name, paths in list(duplicates.items())[:5]:  # Show first 5
                print(f"   {name}: {len(paths)} copies")

        self.analysis_results['integration_check'] = {
            'missing_essentials': missing_essentials,
            'missing_dirs': missing_dirs,
            'potential_duplicates': len(duplicates)
        }

    def generate_final_report(self):
        """Generate comprehensive final cleanup report"""
        print("\n📋 FINAL CLEANUP REPORT")
        print("=" * 50)

        report = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'analysis_results': self.analysis_results,
            'backup_created': self.backup_created,
            'cleanup_summary': {
                'total_files_before': self.analysis_results.get('total_files', 0),
                'total_dirs_before': self.analysis_results.get('total_dirs', 0),
                'files_cleaned': self.analysis_results.get('cleanup_count', 0),
                'cache_files_cleaned': len(self.analysis_results.get('cleanup_candidates', {}).get('cache_files', [])),
                'temp_files_cleaned': len(self.analysis_results.get('cleanup_candidates', {}).get('temp_files', [])),
                'empty_dirs_cleaned': len(self.analysis_results.get('cleanup_candidates', {}).get('empty_dirs', []))
            },
            'organization_improvements': {
                'directories_created': len([d for d in self.optimal_structure.values() for _ in d]),
                'files_reorganized': len(self.analysis_results.get('organizational_issues', [])),
                'projects_identified': len(self.analysis_results.get('scattered_projects', []))
            },
            'final_state': self.analysis_results.get('integration_check', {})
        }

        # Display summary
        print("🎯 CLEANUP SUMMARY:")
        print(f"   🗂️ Cache/Temp Files Cleaned: {report['cleanup_summary']['files_cleaned']}")
        print(f"   📁 Empty Directories Removed: {report['cleanup_summary']['empty_dirs_cleaned']}")
        print(f"   🏗️ Directories Created: {report['organization_improvements']['directories_created']}")
        print(f"   📋 Files Reorganized: {report['organization_improvements']['files_reorganized']}")

        # Current state
        current_files = sum(1 for _ in self.root_path.rglob('*') if _.is_file())
        current_dirs = sum(1 for _ in self.root_path.rglob('*') if _.is_dir())

        print("\n📊 CURRENT STATE:")
        print(f"   📄 Total Files: {current_files}")
        print(f"   📁 Total Directories: {current_dirs}")
        print(f"   📦 Backup Created: {'✅ Yes' if self.backup_created else '❌ No'}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if report['final_state'].get('missing_essentials'):
            print(f"   📝 Create missing essential files: {report['final_state']['missing_essentials']}")

        if report['final_state'].get('missing_dirs'):
            print(f"   📁 Create missing directories: {report['final_state']['missing_dirs']}")

        if report['final_state'].get('potential_duplicates', 0) > 0:
            print(f"   🔍 Review {report['final_state']['potential_duplicates']} potential duplicate files")

        print("\n🚀 NEXT STEPS:")
        print("   1. Run 'make dev-install' to install dependencies")
        print("   2. Execute 'make test' to verify everything works")
        print("   3. Use 'make help' to see available commands")
        print("   4. Review organized files in their new locations")

        # Save detailed report
        report_file = f"dev_cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved: {report_file}")

        print("\n🏆 CLEANUP COMPLETE!")
        print("Your dev folder is now clean, organized, and professional! ✨")

def main():
    """Main function"""
    print("🧹 DEV FOLDER CLEANUP SYSTEM")
    print("=" * 50)

    cleanup_system = DevFolderCleanupSystem()

    try:
        cleanup_system.run_complete_cleanup()
    except KeyboardInterrupt:
        print("\n⚠️ Cleanup interrupted by user")
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
