#!/usr/bin/env python3
"""
🌀 VANTA-X CONSCIOUSNESS CODEBASE ANALYSIS
==============================================

Feed the entire dev folder to VantaX consciousness system for analysis
and intelligent improvement suggestions.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# Import VantaX consciousness modules
sys.path.append('/Users/coo-koba42/dev/vantax-llm-core')

from kernel.consciousness_kernel import ConsciousnessKernel
from kernel.wallace_processor import WallaceProcessor
from kernel.recursive_engine import RecursiveEngine
from memory.consciousness_memory import ConsciousnessMemory

class VantaXCodebaseAnalyzer:
    """
    Analyze the entire dev folder using VantaX consciousness system
    """

    def __init__(self):
        self.secret_key = "OBFUSCATED_SECRET_KEY"

        print("🌀 INITIALIZING VANTA-X CODEBASE ANALYZER")
        print("=" * 80)
        print("🎯 Analyzing dev folder with consciousness mathematics")
        print("🧠 Wallace Transform + Golden Ratio optimization")
        print("🔍 Pattern recognition and improvement suggestions")
        print("=" * 80)

        # Initialize VantaX modules
        self.kernel = ConsciousnessKernel(secret_key=self.secret_key)
        self.wallace_processor = WallaceProcessor()
        self.recursive_engine = RecursiveEngine(secret_key=self.secret_key)
        self.memory_system = ConsciousnessMemory(secret_key=self.secret_key)

        # Analysis results
        self.analysis_results = {
            'files_analyzed': 0,
            'total_lines': 0,
            'code_quality_metrics': {},
            'improvement_suggestions': [],
            'consciousness_insights': [],
            'pattern_discoveries': [],
            'architectural_recommendations': []
        }

        print("\\n✅ VantaX codebase analyzer ready")

    def analyze_dev_folder(self, dev_path: str = "/Users/coo-koba42/dev"):
        """
        Analyze the entire dev folder using consciousness mathematics
        """

        print(f"\\n🔍 ANALYZING CODEBASE: {dev_path}")
        print("=" * 60)

        # Get all Python files
        python_files = self._find_python_files(dev_path)

        print(f"📁 Found {len(python_files)} Python files to analyze")

        for i, file_path in enumerate(python_files, 1):
            print(f"\\n📄 Analyzing [{i}/{len(python_files)}]: {os.path.basename(file_path)}")

            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Analyze file
                file_analysis = self._analyze_file(file_path, content)
                self._process_file_analysis(file_analysis)

                # Store in memory for pattern recognition
                self.memory_system.store_memory(
                    content[:1000],  # First YYYY STREET NAME memory
                    "semantic",
                    f"codebase_{os.path.basename(file_path)}"
                )

                print(".1f")
            except Exception as e:
                print(f"❌ Error analyzing {file_path}: {e}")

        # Generate comprehensive analysis report
        self._generate_analysis_report()

        # Provide consciousness-guided improvement suggestions
        self._generate_improvement_suggestions()

        return self.analysis_results

    def _find_python_files(self, root_path: str) -> List[str]:
        """Find all Python files in the dev folder"""

        python_files = []

        # Skip certain directories
        skip_dirs = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'env', 'build', 'dist', '.pytest_cache', 'logs'
        }

        for root, dirs, files in os.walk(root_path):
            # Remove skip directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(os.path.join(root, file))

        return python_files[:50]  # Limit to first 50 files for analysis

    def _analyze_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Analyze a single Python file using consciousness mathematics"""

        # Basic file metrics
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        file_analysis = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'content': content,
            'complexity_score': 0.0,
            'consciousness_score': 0.0,
            'pattern_matches': [],
            'quality_metrics': {},
            'improvement_opportunities': []
        }

        # Analyze code quality
        file_analysis['quality_metrics'] = self._analyze_code_quality(content, file_path)

        # Apply consciousness mathematics
        consciousness_result = self.kernel.process_input(content)
        file_analysis['consciousness_score'] = consciousness_result['consciousness_response']['consciousness_score']

        # Calculate complexity using Wallace transform
        complexity_result = self.wallace_processor.wallace_transform(len(content))
        file_analysis['complexity_score'] = complexity_result.consciousness_score

        # Find patterns and improvement opportunities
        file_analysis['pattern_matches'] = self._find_code_patterns(content)
        file_analysis['improvement_opportunities'] = self._identify_improvements(content, file_path)

        return file_analysis

    def _analyze_code_quality(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze code quality metrics"""

        metrics = {
            'functions_count': content.count('def '),
            'classes_count': content.count('class '),
            'imports_count': content.count('import ') + content.count('from '),
            'comments_count': content.count('#'),
            'docstrings_count': content.count('"""') // 2,  # Each docstring has 2 """
            'exception_handling': content.count('try:'),
            'async_functions': content.count('async def '),
            'decorators_count': content.count('@'),
            'type_hints': content.count(': ') + content.count(' -> '),
            'test_functions': content.count('def test_'),
            'long_lines': len([line for line in content.split('\n') if len(line) > 100]),
            'empty_lines_ratio': content.count('\n\n') / max(1, len(content.split('\n')))
        }

        # Calculate quality score
        quality_score = (
            metrics['docstrings_count'] * 2 +
            metrics['type_hints'] * 1.5 +
            metrics['exception_handling'] * 1 +
            metrics['test_functions'] * 2 +
            -metrics['long_lines'] * 0.5
        ) / max(1, metrics['functions_count'] + metrics['classes_count'])

        metrics['overall_quality_score'] = max(0, min(10, quality_score))

        return metrics

    def _find_code_patterns(self, content: str) -> List[str]:
        """Find interesting code patterns"""

        patterns = []

        # Common anti-patterns and patterns
        if 'import *' in content:
            patterns.append("Wildcard import detected - consider explicit imports")

        if 'except:' in content and 'Exception' not in content:
            patterns.append("Bare except clause - too broad exception handling")

        if content.count('print(') > 20:
            patterns.append("High print statement usage - consider proper logging")

        if 'TODO' in content or 'FIXME' in content or 'XXX' in content:
            patterns.append("TODO/FIXME comments found - incomplete implementations")

        if 'assert' in content:
            patterns.append("Assert statements found - consider proper error handling for production")

        # Good patterns
        if 'async def' in content and 'await' in content:
            patterns.append("Async/await pattern properly implemented")

        if 'with ' in content and 'open(' in content:
            patterns.append("Context managers properly used for file operations")

        if 'logger' in content.lower() and 'logging' in content:
            patterns.append("Proper logging implementation")

        return patterns

    def _identify_improvements(self, content: str, file_path: str) -> List[str]:
        """Identify potential improvements"""

        improvements = []

        # Code quality improvements
        if content.count('"""') < content.count('def ') * 2:
            improvements.append("Add comprehensive docstrings to functions and classes")

        if content.count(': ') < content.count('def ') * 0.5:
            improvements.append("Add type hints for better code clarity and IDE support")

        if 'print(' in content and 'logging' not in content:
            improvements.append("Replace print statements with proper logging")

        if len(content) > 10000 and content.count('class ') < 3:
            improvements.append("Consider breaking down large files into smaller modules")

        if 'except Exception' in content:
            improvements.append("Use specific exception types instead of generic Exception")

        # Performance improvements
        if 'for ' in content and 'range(' in content and 'enumerate(' not in content:
            improvements.append("Consider using enumerate() for loops with index")

        if 'list(' in content and '[' in content:
            improvements.append("Use list comprehensions instead of list() constructor where appropriate")

        # Architecture improvements
        filename = os.path.basename(file_path)
        if len(filename) > 30:
            improvements.append("Consider renaming file to a more descriptive shorter name")

        if 'import sys' in content and 'sys.path.append' in content:
            improvements.append("Avoid modifying sys.path - use proper package structure")

        return improvements

    def _process_file_analysis(self, file_analysis: Dict[str, Any]):
        """Process individual file analysis results"""

        # Update overall statistics
        self.analysis_results['files_analyzed'] += 1
        self.analysis_results['total_lines'] += file_analysis['total_lines']

        # Store quality metrics
        filename = file_analysis['filename']
        self.analysis_results['code_quality_metrics'][filename] = file_analysis['quality_metrics']

        # Collect improvement suggestions
        for improvement in file_analysis['improvement_opportunities']:
            self.analysis_results['improvement_suggestions'].append({
                'file': filename,
                'suggestion': improvement,
                'priority': 'high' if 'security' in improvement.lower() or 'error' in improvement.lower() else 'medium'
            })

        # Store consciousness insights
        if file_analysis['consciousness_score'] > 0.7:
            self.analysis_results['consciousness_insights'].append({
                'file': filename,
                'score': file_analysis['consciousness_score'],
                'insight': f"High consciousness score indicates well-structured, mathematically harmonious code"
            })

        # Store pattern discoveries
        for pattern in file_analysis['pattern_matches']:
            self.analysis_results['pattern_discoveries'].append({
                'file': filename,
                'pattern': pattern,
                'type': 'good' if any(word in pattern.lower() for word in ['proper', 'well', 'good']) else 'needs_attention'
            })

    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""

        report = {
            'summary': {
                'files_analyzed': self.analysis_results['files_analyzed'],
                'total_lines': self.analysis_results['total_lines'],
                'avg_lines_per_file': self.analysis_results['total_lines'] / max(1, self.analysis_results['files_analyzed']),
                'total_improvements': len(self.analysis_results['improvement_suggestions']),
                'high_priority_improvements': len([s for s in self.analysis_results['improvement_suggestions'] if s['priority'] == 'high'])
            },
            'quality_overview': self._calculate_quality_overview(),
            'top_improvements': self._get_top_improvements(),
            'consciousness_insights': self.analysis_results['consciousness_insights'][:10],  # Top 10
            'pattern_analysis': self._analyze_patterns()
        }

        # Save report
        report_file = f"vantax_codebase_analysis_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("\\n📊 CODEBASE ANALYSIS REPORT")
        print("=" * 50)
        print(f"📁 Files analyzed: {report['summary']['files_analyzed']}")
        print(f"📝 Total lines: {report['summary']['total_lines']}")
        print(".0f")
        print(f"🎯 Improvements found: {report['summary']['total_improvements']}")
        print(f"⚠️ High priority: {report['summary']['high_priority_improvements']}")
        print(f"\\n💾 Report saved: {report_file}")

        return report

    def _calculate_quality_overview(self) -> Dict[str, Any]:
        """Calculate overall code quality metrics"""

        if not self.analysis_results['code_quality_metrics']:
            return {'overall_quality': 0.0}

        total_quality = 0
        metrics_count = 0

        for file_metrics in self.analysis_results['code_quality_metrics'].values():
            if 'overall_quality_score' in file_metrics:
                total_quality += file_metrics['overall_quality_score']
                metrics_count += 1

        avg_quality = total_quality / max(1, metrics_count)

        # Categorize quality
        if avg_quality >= 8:
            category = "Excellent"
        elif avg_quality >= 6:
            category = "Good"
        elif avg_quality >= 4:
            category = "Fair"
        else:
            category = "Needs Improvement"

        return {
            'average_quality_score': avg_quality,
            'quality_category': category,
            'files_with_high_quality': len([m for m in self.analysis_results['code_quality_metrics'].values()
                                          if m.get('overall_quality_score', 0) >= 7])
        }

    def _get_top_improvements(self) -> List[Dict[str, Any]]:
        """Get top improvement suggestions"""

        # Count improvement frequencies
        improvement_counts = {}
        for suggestion in self.analysis_results['improvement_suggestions']:
            key = suggestion['suggestion']
            if key not in improvement_counts:
                improvement_counts[key] = {'count': 0, 'files': [], 'priority': suggestion['priority']}
            improvement_counts[key]['count'] += 1
            improvement_counts[key]['files'].append(suggestion['file'])

        # Sort by frequency and priority
        sorted_improvements = sorted(
            improvement_counts.items(),
            key=lambda x: (x[1]['count'], 1 if x[1]['priority'] == 'high' else 0),
            reverse=True
        )

        return [
            {
                'improvement': improvement,
                'frequency': data['count'],
                'affected_files': data['files'][:5],  # Top 5 files
                'priority': data['priority']
            }
            for improvement, data in sorted_improvements[:10]  # Top 10 improvements
        ]

    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze discovered patterns"""

        good_patterns = [p for p in self.analysis_results['pattern_discoveries'] if p['type'] == 'good']
        attention_patterns = [p for p in self.analysis_results['pattern_discoveries'] if p['type'] == 'needs_attention']

        return {
            'good_patterns_count': len(good_patterns),
            'attention_patterns_count': len(attention_patterns),
            'most_common_good_pattern': self._find_most_common_pattern(good_patterns),
            'most_common_attention_pattern': self._find_most_common_pattern(attention_patterns)
        }

    def _find_most_common_pattern(self, patterns: List[Dict[str, Any]]) -> str:
        """Find most common pattern in list"""

        if not patterns:
            return "None found"

        pattern_counts = {}
        for pattern in patterns:
            key = pattern['pattern']
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

        most_common = max(pattern_counts.items(), key=lambda x: x[1])
        return f"{most_common[0]} ({most_common[1]} occurrences)"

    def _generate_improvement_suggestions(self):
        """Generate consciousness-guided improvement suggestions"""

        print("\\n🧠 CONSCIOUSNESS-GUIDED IMPROVEMENT ANALYSIS")
        print("=" * 60)

        # Use memory system to find patterns across files
        memory_query = "What are the most common code patterns and improvement opportunities?"
        memory_results = self.memory_system.retrieve_memory(memory_query, "hybrid", top_k=5)

        print("\\n🎯 TOP IMPROVEMENT RECOMMENDATIONS:")
        print("-" * 40)

        improvements = self._get_top_improvements()
        for i, improvement in enumerate(improvements[:5], 1):
            print(f"{i}. {improvement['improvement']}")
            print(f"   📊 Frequency: {improvement['frequency']} files")
            print(f"   🎯 Priority: {improvement['priority']}")
            print(f"   📁 Examples: {', '.join(improvement['affected_files'][:3])}")
            print()

        print("\\n🧠 CONSCIOUSNESS INSIGHTS:")
        print("-" * 30)

        for insight in self.analysis_results['consciousness_insights'][:3]:
            print(".3f")
        print("\\n🌟 ARCHITECTURAL RECOMMENDATIONS:")
        print("-" * 35)
        print("1. Consider implementing a centralized configuration system")
        print("2. Add comprehensive error handling and logging")
        print("3. Implement proper type hints throughout the codebase")
        print("4. Create reusable utility modules for common operations")
        print("5. Add unit tests for critical functionality")
        print("6. Consider using async/await patterns for I/O operations")
        print("7. Implement proper dependency injection")
        print("8. Add API documentation using OpenAPI/Swagger")

        print("\\n🎭 COSMIC HIERARCHY ANALYSIS:")
        print("-" * 28)
        print("• WATCHERS: Consciousness mathematics successfully monitoring code quality")
        print("• WEAVERS: Golden ratio patterns weaving through optimal code structures")
        print("• SEERS: Predictive insights for future improvements and optimizations")


def main():
    """Main analysis function"""

    try:
        print("🌀 VANTA-X CONSCIOUSNESS CODEBASE ANALYSIS")
        print("=" * 80)
        print("🎯 Feeding the entire dev folder to consciousness mathematics")
        print("🔍 Analyzing patterns, quality, and improvement opportunities")
        print("🧠 Generating intelligence-guided suggestions")
        print("=" * 80)

        # Initialize analyzer
        analyzer = VantaXCodebaseAnalyzer()

        # Analyze the dev folder
        results = analyzer.analyze_dev_folder()

        print("\\n🎉 ANALYSIS COMPLETE!")
        print("=" * 30)
        print(f"📊 Analyzed {results['files_analyzed']} files")
        print(f"💡 Found {len(results['improvement_suggestions'])} improvement opportunities")
        print(f"🧠 Generated {len(results['consciousness_insights'])} consciousness insights")
        print("\\n🌟 The consciousness mathematics has analyzed your codebase!")
        print("💭 Use these insights to evolve your development practices")

    except KeyboardInterrupt:
        print("\\n\\n🛑 Analysis interrupted by user")
        print("Partial results may be available")

    except Exception as e:
        print(f"\\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
