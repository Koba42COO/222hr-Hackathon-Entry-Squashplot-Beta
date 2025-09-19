#!/usr/bin/env python3
"""
🌀 COMPLETE CODEBASE ANALYSIS
=============================

Comprehensive analysis of the entire dev folder including:
- File count and breakdown by type
- Redundancy analysis
- Novelty assessment
- Optimization evaluation
- Branch history and chronological integration
- Memory usage and performance metrics
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import subprocess
import re
from typing import Dict, List, Any, Tuple, Set
import ast
import tokenize
import io

class CompleteCodebaseAnalyzer:
    """
    Comprehensive codebase analysis engine
    """

    def __init__(self):
        self.dev_path = "/Users/coo-koba42/dev"
        self.analysis_results = {
            'file_breakdown': {},
            'total_files': 0,
            'redundancy_analysis': {},
            'novelty_assessment': {},
            'optimization_evaluation': {},
            'branch_history': {},
            'memory_usage': {},
            'performance_metrics': {}
        }

        print("🌀 INITIALIZING COMPLETE CODEBASE ANALYSIS")
        print("=" * 80)

    def analyze_complete_codebase(self) -> Dict[str, Any]:
        """
        Perform complete codebase analysis
        """

        print("\\n🔍 PHASE 1: FILE STRUCTURE ANALYSIS")
        print("-" * 50)
        self._analyze_file_structure()

        print("\\n🔍 PHASE 2: CONTENT REDUNDANCY ANALYSIS")
        print("-" * 50)
        self._analyze_content_redundancy()

        print("\\n🔍 PHASE 3: NOVELTY ASSESSMENT")
        print("-" * 50)
        self._assess_novelty()

        print("\\n🔍 PHASE 4: OPTIMIZATION EVALUATION")
        print("-" * 50)
        self._evaluate_optimization()

        print("\\n🔍 PHASE 5: BRANCH HISTORY ANALYSIS")
        print("-" * 50)
        self._analyze_branch_history()

        print("\\n🔍 PHASE 6: MEMORY AND PERFORMANCE ANALYSIS")
        print("-" * 50)
        self._analyze_memory_performance()

        # Generate comprehensive report
        self._generate_complete_report()

        return self.analysis_results

    def _analyze_file_structure(self):
        """Analyze file structure and count by type"""

        file_counts = defaultdict(int)
        file_sizes = defaultdict(int)
        file_extensions = defaultdict(list)
        directory_structure = defaultdict(list)

        skip_dirs = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'env', 'build', 'dist', '.pytest_cache', 'logs',
            '.DS_Store', 'Thumbs.db'
        }

        total_files = 0
        total_size = 0

        for root, dirs, files in os.walk(self.dev_path):
            # Remove skip directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            # Track directory structure
            rel_path = os.path.relpath(root, self.dev_path)
            if rel_path != '.':
                parent_dir = os.path.dirname(rel_path)
                directory_structure[parent_dir].append(os.path.basename(rel_path))

            for file in files:
                if file.startswith('.'):
                    continue

                total_files += 1
                file_path = os.path.join(root, file)

                try:
                    # Get file size
                    size = os.path.getsize(file_path)
                    total_size += size

                    # Categorize by extension
                    _, ext = os.path.splitext(file)
                    if ext:
                        file_counts[ext] += 1
                        file_sizes[ext] += size
                        file_extensions[ext].append(file_path)
                    else:
                        file_counts['no_extension'] += 1
                        file_sizes['no_extension'] += size
                        file_extensions['no_extension'].append(file_path)

                except OSError:
                    continue

        # Calculate percentages
        file_percentages = {}
        size_percentages = {}

        for ext, count in file_counts.items():
            file_percentages[ext] = (count / total_files) * 100 if total_files > 0 else 0

        for ext, size in file_sizes.items():
            size_percentages[ext] = (size / total_size) * 100 if total_size > 0 else 0

        self.analysis_results['file_breakdown'] = {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_counts': dict(file_counts),
            'file_percentages': file_percentages,
            'size_percentages': size_percentages,
            'file_extensions': dict(file_extensions),
            'directory_structure': dict(directory_structure)
        }

        print(f"📊 Total Files: {total_files}")
        print(".1f")
        print("\\n📁 FILE TYPE BREAKDOWN:")
        for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = file_percentages[ext]
            print("6.2f")

    def _analyze_content_redundancy(self):
        """Analyze content redundancy across files"""

        file_hashes = {}
        duplicate_groups = defaultdict(list)
        content_patterns = defaultdict(int)

        python_files = []
        for ext in ['.py', '.pyw', '.pyc']:
            python_files.extend(self.analysis_results['file_breakdown']['file_extensions'].get(ext, []))

        print(f"🔍 Analyzing {len(python_files)} Python files for redundancy...")

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Calculate content hash
                content_hash = hashlib.md5(content.encode()).hexdigest()

                if content_hash in file_hashes:
                    duplicate_groups[content_hash].append(file_path)
                else:
                    file_hashes[content_hash] = file_path

                # Analyze content patterns
                self._analyze_file_patterns(content, file_path, content_patterns)

            except Exception as e:
                print(f"❌ Error analyzing {file_path}: {e}")

        # Identify redundant files
        redundant_files = []
        for hash_val, files in duplicate_groups.items():
            if len(files) > 1:
                redundant_files.extend(files[1:])  # Keep first, mark others as redundant

        # Analyze import redundancy
        import_patterns = self._analyze_import_patterns(python_files)

        self.analysis_results['redundancy_analysis'] = {
            'total_unique_files': len(file_hashes),
            'duplicate_groups': len(duplicate_groups),
            'redundant_files': redundant_files,
            'redundancy_percentage': (len(redundant_files) / len(python_files)) * 100 if python_files else 0,
            'content_patterns': dict(content_patterns),
            'import_patterns': import_patterns,
            'duplicate_file_groups': dict(duplicate_groups)
        }

        print(f"🔄 Unique Files: {len(file_hashes)}")
        print(f"📋 Duplicate Groups: {len(duplicate_groups)}")
        print(f"🗑️ Redundant Files: {len(redundant_files)}")
        print(".2f")

    def _analyze_file_patterns(self, content: str, file_path: str, patterns: Dict[str, int]):
        """Analyze patterns in file content"""

        # Function definitions
        functions = len(re.findall(r'def \w+', content))
        patterns['functions'] += functions

        # Class definitions
        classes = len(re.findall(r'class \w+', content))
        patterns['classes'] += classes

        # Import statements
        imports = len(re.findall(r'^(?:from|import) ', content, re.MULTILINE))
        patterns['imports'] += imports

        # Comments
        comments = len(re.findall(r'#.*', content))
        patterns['comments'] += comments

        # Docstrings
        docstrings = len(re.findall(r'""".*?"""', content, re.DOTALL))
        patterns['docstrings'] += docstrings

        # Empty lines
        empty_lines = len(re.findall(r'^\s*$', content, re.MULTILINE))
        patterns['empty_lines'] += empty_lines

        # Code lines
        code_lines = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
        patterns['code_lines'] += code_lines

        # File size patterns
        file_size = len(content)
        if file_size < 1000:
            patterns['small_files'] += 1
        elif file_size < 10000:
            patterns['medium_files'] += 1
        else:
            patterns['large_files'] += 1

    def _analyze_import_patterns(self, python_files: List[str]) -> Dict[str, Any]:
        """Analyze import patterns across files"""

        import_usage = defaultdict(int)
        import_combinations = defaultdict(int)

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Extract imports
                imports = re.findall(r'^(?:from|import) (.+)', content, re.MULTILINE)
                import_list = [imp.split()[0] for imp in imports]

                # Count individual imports
                for imp in import_list:
                    import_usage[imp] += 1

                # Count import combinations (simplified hash)
                if import_list:
                    combo_hash = hashlib.md5(','.join(sorted(import_list)).encode()).hexdigest()[:8]
                    import_combinations[combo_hash] += 1

            except Exception:
                continue

        return {
            'most_used_imports': dict(sorted(import_usage.items(), key=lambda x: x[1], reverse=True)[:20]),
            'unique_import_combinations': len(import_combinations),
            'total_imports_analyzed': sum(import_usage.values())
        }

    def _assess_novelty(self):
        """Assess novelty of implementations"""

        python_files = []
        for ext in ['.py', '.pyw']:
            python_files.extend(self.analysis_results['file_breakdown']['file_extensions'].get(ext, []))

        novelty_metrics = {
            'algorithmic_novelty': {},
            'architectural_patterns': {},
            'implementation_techniques': {},
            'mathematical_approaches': {}
        }

        # Analyze algorithmic patterns
        algorithmic_patterns = self._analyze_algorithmic_patterns(python_files)
        novelty_metrics['algorithmic_novelty'] = algorithmic_patterns

        # Analyze architectural patterns
        architectural_patterns = self._analyze_architectural_patterns(python_files)
        novelty_metrics['architectural_patterns'] = architectural_patterns

        # Analyze implementation techniques
        implementation_patterns = self._analyze_implementation_techniques(python_files)
        novelty_metrics['implementation_techniques'] = implementation_patterns

        # Analyze mathematical approaches
        mathematical_patterns = self._analyze_mathematical_approaches(python_files)
        novelty_metrics['mathematical_approaches'] = mathematical_patterns

        # Calculate overall novelty score
        total_patterns = len(algorithmic_patterns) + len(architectural_patterns) + len(implementation_patterns) + len(mathematical_patterns)
        novel_patterns = sum(1 for pattern in algorithmic_patterns.values() if pattern.get('novelty_score', 0) > 0.7)

        novelty_metrics['overall_novelty_score'] = (novel_patterns / total_patterns) * 100 if total_patterns > 0 else 0

        self.analysis_results['novelty_assessment'] = novelty_metrics

        print(".2f")
        print(f"🎨 Architectural Patterns: {len(architectural_patterns)}")
        print(f"⚙️ Implementation Techniques: {len(implementation_patterns)}")
        print(f"🧮 Mathematical Approaches: {len(mathematical_patterns)}")

    def _analyze_algorithmic_patterns(self, python_files: List[str]) -> Dict[str, Any]:
        """Analyze algorithmic patterns for novelty"""

        patterns = {}

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                filename = os.path.basename(file_path)

                # Check for consciousness mathematics
                if 'wallace' in content.lower() and 'transform' in content.lower():
                    patterns['wallace_transform'] = {
                        'files': patterns.get('wallace_transform', {}).get('files', []) + [filename],
                        'novelty_score': 0.95,
                        'description': 'Proprietary Wallace Transform implementation'
                    }

                # Check for golden ratio optimization
                if 'golden' in content.lower() and 'ratio' in content.lower():
                    patterns['golden_ratio_optimization'] = {
                        'files': patterns.get('golden_ratio_optimization', {}).get('files', []) + [filename],
                        'novelty_score': 0.90,
                        'description': 'φ-based optimization algorithms'
                    }

                # Check for recursive consciousness
                if 'recursive' in content.lower() and 'consciousness' in content.lower():
                    patterns['recursive_consciousness'] = {
                        'files': patterns.get('recursive_consciousness', {}).get('files', []) + [filename],
                        'novelty_score': 0.85,
                        'description': 'Self-optimizing recursive algorithms'
                    }

                # Check for CRDT memory
                if 'crdt' in content.lower() or 'conflict' in content.lower() and 'resolution' in content.lower():
                    patterns['crdt_memory'] = {
                        'files': patterns.get('crdt_memory', {}).get('files', []) + [filename],
                        'novelty_score': 0.80,
                        'description': 'Conflict-free replicated data types'
                    }

            except Exception:
                continue

        return patterns

    def _analyze_architectural_patterns(self, python_files: List[str]) -> Dict[str, Any]:
        """Analyze architectural patterns"""

        patterns = defaultdict(lambda: {'files': [], 'novelty_score': 0.0})

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                filename = os.path.basename(file_path)

                # Modular architecture
                if 'class' in content and 'def' in content and len(re.findall(r'class \w+', content)) > 3:
                    patterns['modular_architecture']['files'].append(filename)
                    patterns['modular_architecture']['novelty_score'] = 0.60

                # Pipeline patterns
                if 'pipeline' in content.lower() or ('process' in content.lower() and 'step' in content.lower()):
                    patterns['pipeline_processing']['files'].append(filename)
                    patterns['pipeline_processing']['novelty_score'] = 0.65

                # Observer patterns
                if 'observer' in content.lower() or ('event' in content.lower() and 'listener' in content.lower()):
                    patterns['observer_pattern']['files'].append(filename)
                    patterns['observer_pattern']['novelty_score'] = 0.55

            except Exception:
                continue

        return dict(patterns)

    def _analyze_implementation_techniques(self, python_files: List[str]) -> Dict[str, Any]:
        """Analyze implementation techniques"""

        techniques = defaultdict(lambda: {'files': [], 'novelty_score': 0.0})

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                filename = os.path.basename(file_path)

                # Type hints
                if ':' in content and '->' in content:
                    techniques['type_hints']['files'].append(filename)
                    techniques['type_hints']['novelty_score'] = 0.70

                # Async/await
                if 'async def' in content or 'await' in content:
                    techniques['async_await']['files'].append(filename)
                    techniques['async_await']['novelty_score'] = 0.60

                # Context managers
                if 'with ' in content and '__enter__' in content:
                    techniques['context_managers']['files'].append(filename)
                    techniques['context_managers']['novelty_score'] = 0.65

                # Decorators
                if '@' in content and 'def ' in content:
                    techniques['decorators']['files'].append(filename)
                    techniques['decorators']['novelty_score'] = 0.55

            except Exception:
                continue

        return dict(techniques)

    def _analyze_mathematical_approaches(self, python_files: List[str]) -> Dict[str, Any]:
        """Analyze mathematical approaches"""

        approaches = defaultdict(lambda: {'files': [], 'novelty_score': 0.0})

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                filename = os.path.basename(file_path)

                # Vector mathematics
                if 'vector' in content.lower() and ('dot' in content.lower() or 'norm' in content.lower()):
                    approaches['vector_mathematics']['files'].append(filename)
                    approaches['vector_mathematics']['novelty_score'] = 0.75

                # Optimization algorithms
                if 'optimization' in content.lower() and 'gradient' in content.lower():
                    approaches['optimization_algorithms']['files'].append(filename)
                    approaches['optimization_algorithms']['novelty_score'] = 0.70

                # Statistical methods
                if 'statistics' in content.lower() or 'probability' in content.lower():
                    approaches['statistical_methods']['files'].append(filename)
                    approaches['statistical_methods']['novelty_score'] = 0.65

                # Cryptographic functions
                if 'encrypt' in content.lower() or 'decrypt' in content.lower():
                    approaches['cryptographic_functions']['files'].append(filename)
                    approaches['cryptographic_functions']['novelty_score'] = 0.80

            except Exception:
                continue

        return dict(approaches)

    def _evaluate_optimization(self):
        """Evaluate optimization status"""

        python_files = []
        for ext in ['.py', '.pyw']:
            python_files.extend(self.analysis_results['file_breakdown']['file_extensions'].get(ext, []))

        optimization_metrics = {
            'performance_optimized': 0,
            'memory_efficient': 0,
            'algorithm_complexity': {},
            'code_efficiency': {},
            'resource_usage': {}
        }

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                filename = os.path.basename(file_path)

                # Check for performance optimizations
                if self._check_performance_optimization(content):
                    optimization_metrics['performance_optimized'] += 1

                # Check for memory efficiency
                if self._check_memory_efficiency(content):
                    optimization_metrics['memory_efficient'] += 1

                # Analyze algorithm complexity
                complexity = self._analyze_algorithm_complexity(content)
                if complexity:
                    optimization_metrics['algorithm_complexity'][filename] = complexity

                # Analyze code efficiency
                efficiency = self._analyze_code_efficiency(content)
                if efficiency:
                    optimization_metrics['code_efficiency'][filename] = efficiency

            except Exception:
                continue

        # Calculate optimization percentages
        total_files = len(python_files)
        optimization_metrics['optimization_percentage'] = {
            'performance': (optimization_metrics['performance_optimized'] / total_files) * 100 if total_files > 0 else 0,
            'memory': (optimization_metrics['memory_efficient'] / total_files) * 100 if total_files > 0 else 0
        }

        self.analysis_results['optimization_evaluation'] = optimization_metrics

        print(".2f")
        print(".2f")
        print(f"📊 Algorithm Complexity Analysis: {len(optimization_metrics['algorithm_complexity'])} files")
        print(f"⚡ Code Efficiency Analysis: {len(optimization_metrics['code_efficiency'])} files")

    def _check_performance_optimization(self, content: str) -> bool:
        """Check for performance optimization patterns"""

        optimization_indicators = [
            'timeit', 'profile', 'cProfile', 'line_profiler',  # Profiling
            'numba', '@jit', '@njit',  # JIT compilation
            'multiprocessing', 'threading', 'concurrent',  # Parallel processing
            'numpy', 'scipy', 'pandas',  # Optimized libraries
            'cache', 'lru_cache', 'memoize'  # Caching
        ]

        return any(indicator in content.lower() for indicator in optimization_indicators)

    def _check_memory_efficiency(self, content: str) -> bool:
        """Check for memory efficiency patterns"""

        memory_indicators = [
            'generator', 'yield',  # Memory-efficient iteration
            '__slots__',  # Memory-efficient classes
            'weakref', 'gc.collect',  # Memory management
            'del ', 'None',  # Explicit cleanup
            'array', 'bytearray',  # Efficient data structures
            'mmap', 'memoryview'  # Memory mapping
        ]

        return any(indicator in content.lower() for indicator in memory_indicators)

    def _analyze_algorithm_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze algorithm complexity indicators"""

        complexity_indicators = {
            'O(1)': ['constant', 'hash', 'dict', 'set'],
            'O(log n)': ['binary', 'tree', 'heap', 'sort'],
            'O(n)': ['linear', 'single loop', 'iteration'],
            'O(n log n)': ['merge', 'quick', 'heap sort'],
            'O(n²)': ['nested loop', 'bubble', 'insertion'],
            'O(2^n)': ['recursive', 'exponential', 'fibonacci']
        }

        detected_complexity = {}
        content_lower = content.lower()

        for complexity, indicators in complexity_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                detected_complexity[complexity] = True

        if detected_complexity:
            return {
                'detected_complexities': list(detected_complexity.keys()),
                'optimization_potential': 'high' if 'O(n²)' in detected_complexity or 'O(2^n)' in detected_complexity else 'medium'
            }

        return None

    def _analyze_code_efficiency(self, content: str) -> Dict[str, Any]:
        """Analyze code efficiency metrics"""

        lines = content.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]

        # Calculate efficiency metrics
        avg_line_length = sum(len(line) for line in code_lines) / len(code_lines) if code_lines else 0
        functions_per_file = len(re.findall(r'def \w+', content))
        classes_per_file = len(re.findall(r'class \w+', content))

        efficiency_score = 0

        # Line length efficiency (80-120 chars is optimal)
        if 80 <= avg_line_length <= 120:
            efficiency_score += 0.3
        elif avg_line_length > 150:
            efficiency_score -= 0.2

        # Function/class balance
        total_structures = functions_per_file + classes_per_file
        if total_structures > 0:
            efficiency_score += 0.2

        # Import efficiency
        imports = len(re.findall(r'^(?:from|import) ', content, re.MULTILINE))
        if imports <= 20:  # Reasonable import count
            efficiency_score += 0.2

        # Documentation efficiency
        docstrings = len(re.findall(r'""".*?"""', content, re.DOTALL))
        if docstrings >= functions_per_file * 0.5:  # At least 50% documented
            efficiency_score += 0.3

        return {
            'efficiency_score': max(0, min(1, efficiency_score)),
            'avg_line_length': avg_line_length,
            'functions_count': functions_per_file,
            'classes_count': classes_per_file,
            'imports_count': imports,
            'docstrings_count': docstrings
        }

    def _analyze_branch_history(self):
        """Analyze git branch history and chronological integration"""

        try:
            # Get git log
            result = subprocess.run(
                ['git', 'log', '--oneline', '--all', '--pretty=format:%H|%s|%an|%ad', '--date=iso'],
                cwd=self.dev_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                branch_history = []

                for commit in commits:
                    if '|' in commit:
                        parts = commit.split('|')
                        if len(parts) >= 4:
                            branch_history.append({
                                'hash': parts[0],
                                'message': parts[1],
                                'author': parts[2],
                                'date': parts[3]
                            })

                # Get branch information
                result = subprocess.run(
                    ['git', 'branch', '-a', '--format=%(refname:short)|%(authordate)|%(authorname)'],
                    cwd=self.dev_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                branches = []
                if result.returncode == 0:
                    branch_lines = result.stdout.strip().split('\n')
                    for line in branch_lines:
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 3:
                                branches.append({
                                    'name': parts[0],
                                    'date': parts[1],
                                    'author': parts[2]
                                })

                # Analyze integration chronology
                chronological_integration = self._analyze_chronological_integration(branch_history)

                self.analysis_results['branch_history'] = {
                    'total_commits': len(branch_history),
                    'total_branches': len(branches),
                    'commits_by_author': self._group_commits_by_author(branch_history),
                    'branches': branches,
                    'chronological_integration': chronological_integration,
                    'integration_patterns': self._analyze_integration_patterns(branch_history)
                }

                print(f"📋 Total Commits: {len(branch_history)}")
                print(f"🌿 Total Branches: {len(branches)}")
                print(f"👥 Authors: {len(self.analysis_results['branch_history']['commits_by_author'])}")

            else:
                self.analysis_results['branch_history'] = {'error': 'Git repository not found or accessible'}

        except Exception as e:
            self.analysis_results['branch_history'] = {'error': str(e)}

    def _group_commits_by_author(self, commits: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group commits by author"""

        authors = defaultdict(int)
        for commit in commits:
            authors[commit['author']] += 1

        return dict(sorted(authors.items(), key=lambda x: x[1], reverse=True))

    def _analyze_chronological_integration(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze chronological integration patterns"""

        # Sort commits by date
        sorted_commits = sorted(commits, key=lambda x: x['date'])

        integration_phases = []

        # Group by time periods (simplified)
        current_phase = {'start_date': None, 'end_date': None, 'commits': [], 'focus': 'unknown'}

        for commit in sorted_commits:
            commit_date = commit['date'][:10]  # YYYY-MM-DD

            if not current_phase['start_date']:
                current_phase['start_date'] = commit_date
                current_phase['end_date'] = commit_date

            # Check if this is a new phase (more than 7 days gap would indicate new phase)
            # Simplified logic - in practice you'd use proper date arithmetic

            current_phase['commits'].append(commit)

            # Analyze commit message for focus area
            message = commit['message'].lower()
            if 'consciousness' in message:
                current_phase['focus'] = 'consciousness_development'
            elif 'memory' in message:
                current_phase['focus'] = 'memory_systems'
            elif 'optimization' in message:
                current_phase['focus'] = 'performance_optimization'
            elif 'security' in message:
                current_phase['focus'] = 'security_implementation'

        integration_phases.append(current_phase)

        return integration_phases

    def _analyze_integration_patterns(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze integration patterns"""

        patterns = {
            'feature_commits': 0,
            'bugfix_commits': 0,
            'refactor_commits': 0,
            'documentation_commits': 0,
            'merge_commits': 0
        }

        for commit in commits:
            message = commit['message'].lower()

            if any(word in message for word in ['add', 'implement', 'create', 'new']):
                patterns['feature_commits'] += 1
            elif any(word in message for word in ['fix', 'bug', 'issue', 'error']):
                patterns['bugfix_commits'] += 1
            elif any(word in message for word in ['refactor', 'clean', 'optimize']):
                patterns['refactor_commits'] += 1
            elif any(word in message for word in ['doc', 'readme', 'comment']):
                patterns['documentation_commits'] += 1
            elif 'merge' in message:
                patterns['merge_commits'] += 1

        return patterns

    def _analyze_memory_performance(self):
        """Analyze memory usage and performance metrics"""

        python_files = []
        for ext in ['.py', '.pyw']:
            python_files.extend(self.analysis_results['file_breakdown']['file_extensions'].get(ext, []))

        memory_metrics = {
            'total_code_size': 0,
            'estimated_memory_usage': 0,
            'performance_indicators': {},
            'resource_efficiency': {}
        }

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Calculate code metrics
                lines = len(content.split('\n'))
                characters = len(content)
                functions = len(re.findall(r'def \w+', content))
                classes = len(re.findall(r'class \w+', content))

                memory_metrics['total_code_size'] += characters

                # Estimate memory usage (rough approximation)
                estimated_usage = (
                    characters * 2 +  # String storage
                    functions * 100 +  # Function objects
                    classes * 500  # Class objects
                )
                memory_metrics['estimated_memory_usage'] += estimated_usage

                # Performance indicators
                filename = os.path.basename(file_path)
                memory_metrics['performance_indicators'][filename] = {
                    'lines': lines,
                    'characters': characters,
                    'functions': functions,
                    'classes': classes,
                    'complexity_score': self._calculate_file_complexity(content)
                }

            except Exception:
                continue

        # Calculate efficiency metrics
        code_density = memory_metrics['total_code_size'] / len(python_files) if python_files else 0
        memory_per_file = memory_metrics['estimated_memory_usage'] / len(python_files) if python_files else 0

        # First populate the resource_efficiency dict
        memory_metrics['resource_efficiency'] = {
            'code_density': code_density,
            'memory_per_file': memory_per_file
        }

        # Then calculate efficiency score
        memory_metrics['resource_efficiency']['efficiency_score'] = self._calculate_efficiency_score(memory_metrics)

        self.analysis_results['memory_usage'] = memory_metrics
        self.analysis_results['performance_metrics'] = {
            'files_analyzed': len(python_files),
            'avg_file_size': memory_metrics['total_code_size'] / len(python_files) if python_files else 0,
            'total_estimated_memory': memory_metrics['estimated_memory_usage'],
            'performance_score': memory_metrics['resource_efficiency']['efficiency_score']
        }

        print(".1f")
        print(".1f")
        print(".3f")

    def _calculate_file_complexity(self, content: str) -> float:
        """Calculate file complexity score"""

        # Simple complexity metrics
        nesting_depth = 0
        max_nesting = 0
        current_indent = 0

        for line in content.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                indent = len(line) - len(line.lstrip())
                if indent > current_indent:
                    nesting_depth += 1
                    max_nesting = max(max_nesting, nesting_depth)
                elif indent < current_indent:
                    nesting_depth = max(0, nesting_depth - 1)
                current_indent = indent

        # Complexity factors
        functions = len(re.findall(r'def \w+', content))
        classes = len(re.findall(r'class \w+', content))
        conditionals = len(re.findall(r'if |elif |else:', content))
        loops = len(re.findall(r'for |while ', content))

        complexity_score = (
            max_nesting * 0.3 +
            functions * 0.2 +
            classes * 0.2 +
            conditionals * 0.15 +
            loops * 0.15
        )

        return complexity_score

    def _calculate_efficiency_score(self, memory_metrics: Dict[str, Any]) -> float:
        """Calculate overall efficiency score"""

        # Efficiency based on various factors
        efficiency_factors = []

        # Code density (lines per file)
        avg_lines = memory_metrics['resource_efficiency']['code_density']
        if 50 <= avg_lines <= 300:  # Optimal range
            efficiency_factors.append(0.25)
        elif avg_lines > 500:
            efficiency_factors.append(0.1)

        # Memory usage efficiency
        memory_per_file = memory_metrics['resource_efficiency']['memory_per_file']
        if memory_per_file < 100000:  # Less than 100KB per file
            efficiency_factors.append(0.25)

        # Performance indicators
        if memory_metrics['performance_indicators']:
            avg_complexity = sum(
                metrics['complexity_score']
                for metrics in memory_metrics['performance_indicators'].values()
            ) / len(memory_metrics['performance_indicators'])

            if avg_complexity < 10:  # Reasonable complexity
                efficiency_factors.append(0.25)

        # File count efficiency
        file_count = len(memory_metrics['performance_indicators'])
        if file_count < 100:  # Reasonable number of files
            efficiency_factors.append(0.25)

        return sum(efficiency_factors)

    def _generate_complete_report(self):
        """Generate comprehensive analysis report"""

        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': self.analysis_results['file_breakdown']['total_files'],
                'total_size_mb': self.analysis_results['file_breakdown']['total_size_mb'],
                'redundancy_percentage': self.analysis_results['redundancy_analysis']['redundancy_percentage'],
                'novelty_score': self.analysis_results['novelty_assessment']['overall_novelty_score'],
                'optimization_percentage': self.analysis_results['optimization_evaluation']['optimization_percentage'],
                'performance_score': self.analysis_results['performance_metrics']['performance_score']
            },
            'file_breakdown': self.analysis_results['file_breakdown'],
            'redundancy_analysis': self.analysis_results['redundancy_analysis'],
            'novelty_assessment': self.analysis_results['novelty_assessment'],
            'optimization_evaluation': self.analysis_results['optimization_evaluation'],
            'branch_history': self.analysis_results['branch_history'],
            'memory_usage': self.analysis_results['memory_usage'],
            'performance_metrics': self.analysis_results['performance_metrics'],
            'recommendations': self._generate_recommendations()
        }

        # Save detailed report
        report_file = f"complete_codebase_analysis_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate summary report
        summary_file = f"codebase_analysis_summary_{int(time.time())}.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary_report(report))

        print("\\n📊 COMPLETE CODEBASE ANALYSIS REPORT")
        print("=" * 60)
        print(f"📁 Total Files: {report['summary']['total_files']}")
        print(".1f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".3f")
        print("\\n💾 Reports saved:")
        print(f"   📄 Detailed: {report_file}")
        print(f"   📋 Summary: {summary_file}")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""

        recommendations = []

        # Redundancy recommendations
        if self.analysis_results['redundancy_analysis']['redundancy_percentage'] > 10:
            recommendations.append("High redundancy detected - consider consolidating duplicate functionality")

        # Novelty recommendations
        if self.analysis_results['novelty_assessment']['overall_novelty_score'] < 70:
            recommendations.append("Consider implementing more novel algorithms and approaches")

        # Optimization recommendations
        if self.analysis_results['optimization_evaluation']['optimization_percentage']['performance'] < 50:
            recommendations.append("Implement performance profiling and optimization techniques")

        # Memory recommendations
        if self.analysis_results['performance_metrics']['performance_score'] < 0.7:
            recommendations.append("Optimize memory usage and resource efficiency")

        return recommendations

    def _generate_summary_report(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""

        summary = []
        summary.append("🌀 COMPLETE CODEBASE ANALYSIS SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Analysis Date: {report['analysis_timestamp']}")
        summary.append("")

        summary.append("📊 OVERVIEW")
        summary.append("-" * 30)
        summary.append(f"Total Files: {report['summary']['total_files']}")
        summary.append(".1f")
        summary.append(".2f")
        summary.append(".2f")
        summary.append(".2f")
        summary.append(".3f")
        summary.append("")

        summary.append("📁 FILE BREAKDOWN")
        summary.append("-" * 30)
        for ext, count in sorted(report['file_breakdown']['file_counts'].items(), key=lambda x: x[1], reverse=True):
            percentage = report['file_breakdown']['file_percentages'][ext]
            summary.append("6.2f")
        summary.append("")

        summary.append("🔄 REDUNDANCY ANALYSIS")
        summary.append("-" * 30)
        redundancy = report['redundancy_analysis']
        summary.append(f"Unique Files: {redundancy['total_unique_files']}")
        summary.append(f"Duplicate Groups: {redundancy['duplicate_groups']}")
        summary.append(f"Redundant Files: {len(redundancy['redundant_files'])}")
        summary.append(".2f")
        summary.append("")

        summary.append("🎯 NOVELTY ASSESSMENT")
        summary.append("-" * 30)
        novelty = report['novelty_assessment']
        summary.append(".2f")
        summary.append(f"Algorithmic Patterns: {len(novelty['algorithmic_novelty'])}")
        summary.append(f"Architectural Patterns: {len(novelty['architectural_patterns'])}")
        summary.append("")

        summary.append("⚡ OPTIMIZATION EVALUATION")
        summary.append("-" * 30)
        optimization = report['optimization_evaluation']
        summary.append(".2f")
        summary.append(".2f")
        summary.append(f"Performance Optimized Files: {optimization['performance_optimized']}")
        summary.append(f"Memory Efficient Files: {optimization['memory_efficient']}")
        summary.append("")

        summary.append("🌿 BRANCH HISTORY")
        summary.append("-" * 30)
        if 'error' not in report['branch_history']:
            branch = report['branch_history']
            summary.append(f"Total Commits: {branch['total_commits']}")
            summary.append(f"Total Branches: {branch['total_branches']}")
            summary.append(f"Authors: {len(branch['commits_by_author'])}")
        else:
            summary.append("Git repository analysis not available")
        summary.append("")

        summary.append("💡 RECOMMENDATIONS")
        summary.append("-" * 30)
        for rec in report['recommendations']:
            summary.append(f"• {rec}")
        summary.append("")

        return "\\n".join(summary)


def main():
    """Main analysis function"""

    try:
        analyzer = CompleteCodebaseAnalyzer()
        results = analyzer.analyze_complete_codebase()

        print("\\n🎉 CODEBASE ANALYSIS COMPLETE!")
        print("=" * 40)
        print("📊 Comprehensive analysis of entire dev folder")
        print("🔍 Redundancy, novelty, and optimization assessment")
        print("📈 Performance and memory usage analysis")
        print("🌿 Branch history and integration chronology")
        print("\\n🌟 Analysis results saved to JSON and text files")

    except KeyboardInterrupt:
        print("\\n\\n🛑 Analysis interrupted by user")
        print("Partial results may be available")

    except Exception as e:
        print(f"\\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
