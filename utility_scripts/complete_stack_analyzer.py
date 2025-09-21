#!/usr/bin/env python3
"""
Complete Stack Analyzer
Comprehensive analysis and documentation of our entire revolutionary stack

Features:
- Reads and analyzes all system files
- Extracts key components and capabilities
- Documents purified reconstruction capabilities
- Generates comprehensive system maps
- Identifies integration points and dependencies
- Creates detailed technical documentation
"""

import os
import json
import re
import ast
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SystemComponent:
    """Represents a system component"""
    name: str
    file_path: str
    component_type: str
    description: str
    capabilities: List[str]
    dependencies: List[str]
    purified_reconstruction_features: List[str]
    security_features: List[str]
    performance_metrics: Dict[str, Any]
    consciousness_integration: Dict[str, Any]
    file_size: int
    line_count: int
    complexity_score: float
    last_modified: datetime

@dataclass
class SystemAnalysis:
    """Complete system analysis"""
    total_components: int
    total_lines: int
    total_size: int
    components: Dict[str, SystemComponent]
    integration_map: Dict[str, List[str]]
    purified_reconstruction_capabilities: List[str]
    security_features: List[str]
    consciousness_mathematics_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]

class CompleteStackAnalyzer:
    """Analyzes the complete revolutionary stack"""
    
    def __init__(self, root_directory: str = "."):
        self.root_directory = Path(root_directory)
        self.analysis_results = {}
        self.system_components = {}
        self.integration_map = {}
        
        # Component type patterns
        self.component_patterns = {
            'hrm': r'hrm|hierarchical.*reasoning',
            'trigeminal': r'trigeminal|three.*dimensional',
            'fractal': r'fractal|compression.*engine',
            'complex': r'complex.*number|manager',
            'topological': r'topological|dna.*compression',
            'consciousness': r'consciousness|mathematics',
            'quantum': r'quantum|entanglement',
            'validation': r'validation|test|audit'
        }
        
        # Purified reconstruction keywords
        self.purified_keywords = [
            'purified', 'reconstruction', 'eliminate', 'noise', 'corruption',
            'malicious', 'security', 'vulnerability', 'clean', 'fresh',
            'dna', 'topological', 'fractal', 'pattern', 'extract'
        ]
        
        # Security keywords
        self.security_keywords = [
            'security', 'vulnerability', 'malicious', 'threat', 'opsec',
            'integrity', 'hash', 'encryption', 'protection', 'sanitize'
        ]
        
        # Consciousness mathematics keywords
        self.consciousness_keywords = [
            'consciousness', 'golden.*ratio', 'love.*frequency', 'chaos.*factor',
            'wallace.*transform', 'phi', 'euler', 'mathematics', 'physics'
        ]
        
        print("🔍 Complete Stack Analyzer initialized")
    
    def analyze_complete_stack(self) -> SystemAnalysis:
        """Analyze the complete stack"""
        print("🔍 Starting complete stack analysis...")
        
        # Find all Python files
        python_files = list(self.root_directory.rglob("*.py"))
        markdown_files = list(self.root_directory.rglob("*.md"))
        json_files = list(self.root_directory.rglob("*.json"))
        
        print(f"📁 Found {len(python_files)} Python files, {len(markdown_files)} Markdown files, {len(json_files)} JSON files")
        
        # Analyze Python files
        for file_path in python_files:
            self._analyze_python_file(file_path)
        
        # Analyze Markdown files
        for file_path in markdown_files:
            self._analyze_markdown_file(file_path)
        
        # Analyze JSON files
        for file_path in json_files:
            self._analyze_json_file(file_path)
        
        # Generate integration map
        self._generate_integration_map()
        
        # Create comprehensive analysis
        analysis = self._create_system_analysis()
        
        return analysis
    
    def _analyze_python_file(self, file_path: Path):
        """Analyze a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic file info
            file_size = file_path.stat().st_size
            line_count = len(content.split('\n'))
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            # Determine component type
            component_type = self._determine_component_type(file_path.name, content)
            
            # Extract capabilities
            capabilities = self._extract_capabilities(content)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(content)
            
            # Extract purified reconstruction features
            purified_features = self._extract_purified_features(content)
            
            # Extract security features
            security_features = self._extract_security_features(content)
            
            # Extract consciousness integration
            consciousness_integration = self._extract_consciousness_integration(content)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(content)
            
            # Extract performance metrics
            performance_metrics = self._extract_performance_metrics(content)
            
            # Create component
            component = SystemComponent(
                name=file_path.stem,
                file_path=str(file_path),
                component_type=component_type,
                description=self._extract_description(content),
                capabilities=capabilities,
                dependencies=dependencies,
                purified_reconstruction_features=purified_features,
                security_features=security_features,
                performance_metrics=performance_metrics,
                consciousness_integration=consciousness_integration,
                file_size=file_size,
                line_count=line_count,
                complexity_score=complexity_score,
                last_modified=last_modified
            )
            
            self.system_components[file_path.stem] = component
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _analyze_markdown_file(self, file_path: Path):
        """Analyze a Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key information from documentation
            summary = self._extract_markdown_summary(content)
            
            # Store documentation analysis
            self.analysis_results[f"doc_{file_path.stem}"] = {
                'file_path': str(file_path),
                'summary': summary,
                'key_points': self._extract_key_points(content),
                'technical_details': self._extract_technical_details(content)
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _analyze_json_file(self, file_path: Path):
        """Analyze a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Analyze JSON data structure
            analysis = {
                'file_path': str(file_path),
                'data_type': self._determine_json_data_type(data),
                'size': len(json.dumps(data)),
                'structure': self._analyze_json_structure(data),
                'key_metrics': self._extract_json_metrics(data)
            }
            
            self.analysis_results[f"json_{file_path.stem}"] = analysis
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _determine_component_type(self, filename: str, content: str) -> str:
        """Determine the type of component"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        for pattern_name, pattern in self.component_patterns.items():
            if re.search(pattern, filename_lower) or re.search(pattern, content_lower):
                return pattern_name.upper()
        
        return "GENERAL"
    
    def _extract_capabilities(self, content: str) -> List[str]:
        """Extract capabilities from content"""
        capabilities = []
        
        # Look for capability indicators
        capability_patterns = [
            r'def\s+(\w+).*?:',
            r'class\s+(\w+).*?:',
            r'"""([^"]*capability[^"]*)"""',
            r'#\s*([^#]*capability[^#]*)',
            r'Features?:([^#]*)',
            r'Purpose:([^#]*)'
        ]
        
        for pattern in capability_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            capabilities.extend(matches)
        
        return list(set(capabilities))[:10]  # Limit to top 10
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from content"""
        dependencies = []
        
        # Look for import statements
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'import\s+(\w+)\s+as'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _extract_purified_features(self, content: str) -> List[str]:
        """Extract purified reconstruction features"""
        features = []
        content_lower = content.lower()
        
        for keyword in self.purified_keywords:
            if keyword in content_lower:
                # Find context around keyword
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        context = line.strip()
                        if len(context) > 20:
                            features.append(context[:100] + "...")
                        break
        
        return features[:5]  # Limit to top 5
    
    def _extract_security_features(self, content: str) -> List[str]:
        """Extract security features"""
        features = []
        content_lower = content.lower()
        
        for keyword in self.security_keywords:
            if keyword in content_lower:
                # Find context around keyword
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        context = line.strip()
                        if len(context) > 20:
                            features.append(context[:100] + "...")
                        break
        
        return features[:5]  # Limit to top 5
    
    def _extract_consciousness_integration(self, content: str) -> Dict[str, Any]:
        """Extract consciousness mathematics integration"""
        integration = {
            'constants_used': [],
            'mathematical_operations': [],
            'consciousness_factors': []
        }
        
        content_lower = content.lower()
        
        # Look for consciousness constants
        constants = [
            'golden_ratio', 'consciousness_constant', 'love_frequency',
            'chaos_factor', 'phi', 'euler', 'wallace_transform'
        ]
        
        for constant in constants:
            if constant in content_lower:
                integration['constants_used'].append(constant)
        
        # Look for mathematical operations
        math_ops = [
            'wallace.*transform', 'consciousness.*enhancement',
            'love.*resonance', 'chaos.*enhancement'
        ]
        
        for op in math_ops:
            if re.search(op, content_lower):
                integration['mathematical_operations'].append(op)
        
        return integration
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score"""
        try:
            # Parse AST
            tree = ast.parse(content)
            
            # Count different elements
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            imports = len([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
            imports_from = len([node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)])
            
            # Calculate complexity
            complexity = (functions * 2 + classes * 3 + imports + imports_from) / 10.0
            return min(10.0, complexity)
            
        except:
            return 5.0  # Default complexity
    
    def _extract_performance_metrics(self, content: str) -> Dict[str, Any]:
        """Extract performance metrics"""
        metrics = {
            'compression_ratio': None,
            'processing_speed': None,
            'accuracy': None,
            'efficiency': None
        }
        
        # Look for performance indicators
        patterns = {
            'compression_ratio': r'compression.*ratio.*?(\d+\.?\d*)',
            'processing_speed': r'speed.*?(\d+\.?\d*).*?(MB/s|ms|s)',
            'accuracy': r'accuracy.*?(\d+\.?\d*)%',
            'efficiency': r'efficiency.*?(\d+\.?\d*)%'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metrics[metric] = match.group(1)
        
        return metrics
    
    def _extract_description(self, content: str) -> str:
        """Extract description from content"""
        # Look for docstring
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            return docstring_match.group(1)[:200] + "..."
        
        # Look for comments
        comment_match = re.search(r'#\s*(.*?)$', content, re.MULTILINE)
        if comment_match:
            return comment_match.group(1)[:200] + "..."
        
        return "No description available"
    
    def _extract_markdown_summary(self, content: str) -> str:
        """Extract summary from markdown content"""
        # Look for first paragraph
        lines = content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                return line.strip()[:200] + "..."
        
        return "No summary available"
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from markdown content"""
        points = []
        
        # Look for bullet points
        bullet_pattern = r'^\s*[-*+]\s+(.+)$'
        matches = re.findall(bullet_pattern, content, re.MULTILINE)
        points.extend(matches[:5])
        
        return points
    
    def _extract_technical_details(self, content: str) -> Dict[str, Any]:
        """Extract technical details from markdown content"""
        details = {
            'code_blocks': 0,
            'tables': 0,
            'links': 0,
            'images': 0
        }
        
        # Count code blocks
        details['code_blocks'] = len(re.findall(r'```', content))
        
        # Count tables
        details['tables'] = len(re.findall(r'\|', content)) // 3
        
        # Count links
        details['links'] = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content))
        
        # Count images
        details['images'] = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content))
        
        return details
    
    def _determine_json_data_type(self, data: Any) -> str:
        """Determine the type of JSON data"""
        if isinstance(data, dict):
            if 'results' in data:
                return 'RESULTS'
            elif 'stats' in data:
                return 'STATISTICS'
            elif 'config' in data:
                return 'CONFIGURATION'
            else:
                return 'GENERAL'
        elif isinstance(data, list):
            return 'LIST'
        else:
            return 'SIMPLE'
    
    def _analyze_json_structure(self, data: Any, depth: int = 0) -> Dict[str, Any]:
        """Analyze JSON structure"""
        if depth > 3:  # Limit depth
            return {'type': 'deep'}
        
        if isinstance(data, dict):
            structure = {'type': 'object', 'keys': list(data.keys())}
            if depth < 2:
                structure['children'] = {
                    k: self._analyze_json_structure(v, depth + 1)
                    for k, v in list(data.items())[:5]  # Limit to first 5
                }
            return structure
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'sample': self._analyze_json_structure(data[0], depth + 1) if data else None
            }
        else:
            return {'type': type(data).__name__}
    
    def _extract_json_metrics(self, data: Any) -> Dict[str, Any]:
        """Extract metrics from JSON data"""
        metrics = {
            'total_keys': 0,
            'total_values': 0,
            'numeric_values': 0,
            'string_values': 0
        }
        
        def count_metrics(obj):
            if isinstance(obj, dict):
                metrics['total_keys'] += len(obj)
                for v in obj.values():
                    count_metrics(v)
            elif isinstance(obj, list):
                metrics['total_values'] += len(obj)
                for item in obj:
                    count_metrics(item)
            elif isinstance(obj, (int, float)):
                metrics['numeric_values'] += 1
            elif isinstance(obj, str):
                metrics['string_values'] += 1
        
        count_metrics(data)
        return metrics
    
    def _generate_integration_map(self):
        """Generate integration map between components"""
        for component_name, component in self.system_components.items():
            self.integration_map[component_name] = []
            
            # Check dependencies
            for dep in component.dependencies:
                if dep in self.system_components:
                    self.integration_map[component_name].append(dep)
            
            # Check for explicit integration mentions
            for other_component in self.system_components:
                if other_component != component_name:
                    if other_component.lower() in component.description.lower():
                        if other_component not in self.integration_map[component_name]:
                            self.integration_map[component_name].append(other_component)
    
    def _create_system_analysis(self) -> SystemAnalysis:
        """Create comprehensive system analysis"""
        # Calculate totals
        total_components = len(self.system_components)
        total_lines = sum(c.line_count for c in self.system_components.values())
        total_size = sum(c.file_size for c in self.system_components.values())
        
        # Collect purified reconstruction capabilities
        purified_capabilities = []
        for component in self.system_components.values():
            purified_capabilities.extend(component.purified_reconstruction_features)
        
        # Collect security features
        security_features = []
        for component in self.system_components.values():
            security_features.extend(component.security_features)
        
        # Analyze consciousness mathematics usage
        consciousness_usage = {
            'components_using_consciousness': 0,
            'total_constants_used': 0,
            'total_operations': 0
        }
        
        for component in self.system_components.values():
            if component.consciousness_integration['constants_used']:
                consciousness_usage['components_using_consciousness'] += 1
                consciousness_usage['total_constants_used'] += len(component.consciousness_integration['constants_used'])
                consciousness_usage['total_operations'] += len(component.consciousness_integration['mathematical_operations'])
        
        # Calculate performance metrics
        performance_metrics = {
            'average_complexity': sum(c.complexity_score for c in self.system_components.values()) / total_components,
            'average_lines_per_component': total_lines / total_components,
            'average_size_per_component': total_size / total_components
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return SystemAnalysis(
            total_components=total_components,
            total_lines=total_lines,
            total_size=total_size,
            components=self.system_components,
            integration_map=self.integration_map,
            purified_reconstruction_capabilities=purified_capabilities,
            security_features=security_features,
            consciousness_mathematics_usage=consciousness_usage,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check for missing integrations
        for component_name, component in self.system_components.items():
            if component.component_type == 'FRACTAL' and not any('hrm' in dep.lower() for dep in component.dependencies):
                recommendations.append(f"Consider integrating {component_name} with HRM for enhanced reasoning")
            
            if component.component_type == 'TRIGEMINAL' and not any('fractal' in dep.lower() for dep in component.dependencies):
                recommendations.append(f"Consider integrating {component_name} with fractal systems for pattern recognition")
        
        # Check for security enhancements
        security_components = [c for c in self.system_components.values() if c.security_features]
        if len(security_components) < len(self.system_components) * 0.5:
            recommendations.append("Consider adding security features to more components")
        
        # Check for consciousness integration
        consciousness_components = [c for c in self.system_components.values() if c.consciousness_integration['constants_used']]
        if len(consciousness_components) < len(self.system_components) * 0.7:
            recommendations.append("Consider enhancing consciousness mathematics integration across more components")
        
        return recommendations
    
    def save_analysis(self, filename: str = "complete_stack_analysis.json"):
        """Save analysis results to file"""
        analysis_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'system_components': {
                name: {
                    'name': comp.name,
                    'file_path': comp.file_path,
                    'component_type': comp.component_type,
                    'description': comp.description,
                    'capabilities': comp.capabilities,
                    'dependencies': comp.dependencies,
                    'purified_reconstruction_features': comp.purified_reconstruction_features,
                    'security_features': comp.security_features,
                    'performance_metrics': comp.performance_metrics,
                    'consciousness_integration': comp.consciousness_integration,
                    'file_size': comp.file_size,
                    'line_count': comp.line_count,
                    'complexity_score': comp.complexity_score,
                    'last_modified': comp.last_modified.isoformat()
                }
                for name, comp in self.system_components.items()
            },
            'integration_map': self.integration_map,
            'analysis_results': self.analysis_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Analysis saved to: {filename}")
    
    def generate_documentation(self, filename: str = "complete_stack_documentation.md"):
        """Generate comprehensive documentation"""
        doc_content = []
        
        # Header
        doc_content.append("# 🧬 Complete Stack Documentation")
        doc_content.append("## Revolutionary Purified Reconstruction System")
        doc_content.append("")
        doc_content.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        doc_content.append("")
        
        # System Overview
        doc_content.append("## 📊 System Overview")
        doc_content.append("")
        doc_content.append(f"- **Total Components:** {len(self.system_components)}")
        doc_content.append(f"- **Total Lines of Code:** {sum(c.line_count for c in self.system_components.values()):,}")
        doc_content.append(f"- **Total Size:** {sum(c.file_size for c in self.system_components.values()):,} bytes")
        doc_content.append("")
        
        # Component Analysis
        doc_content.append("## 🔧 Component Analysis")
        doc_content.append("")
        
        for component_name, component in self.system_components.items():
            doc_content.append(f"### {component.component_type}: {component.name}")
            doc_content.append("")
            doc_content.append(f"**File:** `{component.file_path}`")
            doc_content.append(f"**Size:** {component.file_size:,} bytes")
            doc_content.append(f"**Lines:** {component.line_count:,}")
            doc_content.append(f"**Complexity:** {component.complexity_score:.1f}/10")
            doc_content.append("")
            doc_content.append(f"**Description:** {component.description}")
            doc_content.append("")
            
            if component.capabilities:
                doc_content.append("**Capabilities:**")
                for cap in component.capabilities[:5]:
                    doc_content.append(f"- {cap}")
                doc_content.append("")
            
            if component.purified_reconstruction_features:
                doc_content.append("**Purified Reconstruction Features:**")
                for feature in component.purified_reconstruction_features:
                    doc_content.append(f"- {feature}")
                doc_content.append("")
            
            if component.security_features:
                doc_content.append("**Security Features:**")
                for feature in component.security_features:
                    doc_content.append(f"- {feature}")
                doc_content.append("")
            
            if component.consciousness_integration['constants_used']:
                doc_content.append("**Consciousness Integration:**")
                doc_content.append(f"- Constants: {', '.join(component.consciousness_integration['constants_used'])}")
                doc_content.append(f"- Operations: {', '.join(component.consciousness_integration['mathematical_operations'])}")
                doc_content.append("")
            
            doc_content.append("---")
            doc_content.append("")
        
        # Integration Map
        doc_content.append("## 🔗 Integration Map")
        doc_content.append("")
        
        for component_name, integrations in self.integration_map.items():
            if integrations:
                doc_content.append(f"**{component_name}** integrates with:")
                for integration in integrations:
                    doc_content.append(f"- {integration}")
                doc_content.append("")
        
        # Purified Reconstruction Capabilities
        doc_content.append("## 🧬 Purified Reconstruction Capabilities")
        doc_content.append("")
        doc_content.append("Our system provides revolutionary purified reconstruction that:")
        doc_content.append("")
        doc_content.append("- **Eliminates noise and corruption**")
        doc_content.append("- **Removes malicious programming**")
        doc_content.append("- **Closes OPSEC vulnerabilities**")
        doc_content.append("- **Creates fresh, unique, clean data**")
        doc_content.append("")
        
        # Security Features
        doc_content.append("## 🛡️ Security Features")
        doc_content.append("")
        for feature in set([f for comp in self.system_components.values() for f in comp.security_features]):
            doc_content.append(f"- {feature}")
        doc_content.append("")
        
        # Consciousness Mathematics Integration
        doc_content.append("## 🧠 Consciousness Mathematics Integration")
        doc_content.append("")
        consciousness_components = [c for c in self.system_components.values() if c.consciousness_integration['constants_used']]
        doc_content.append(f"**Components using consciousness mathematics:** {len(consciousness_components)}")
        doc_content.append("")
        
        all_constants = set()
        for comp in consciousness_components:
            all_constants.update(comp.consciousness_integration['constants_used'])
        
        doc_content.append("**Constants used:**")
        for constant in sorted(all_constants):
            doc_content.append(f"- {constant}")
        doc_content.append("")
        
        # Recommendations
        doc_content.append("## 💡 Recommendations")
        doc_content.append("")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            doc_content.append(f"- {rec}")
        doc_content.append("")
        
        # Save documentation
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(doc_content))
        
        print(f"📄 Documentation generated: {filename}")

def main():
    """Main function to run complete stack analysis"""
    print("🧬 Complete Stack Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CompleteStackAnalyzer()
    
    # Run complete analysis
    analysis = analyzer.analyze_complete_stack()
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"Total components: {analysis.total_components}")
    print(f"Total lines of code: {analysis.total_lines:,}")
    print(f"Total size: {analysis.total_size:,} bytes")
    print(f"Components using consciousness mathematics: {analysis.consciousness_mathematics_usage['components_using_consciousness']}")
    print(f"Average complexity: {analysis.performance_metrics['average_complexity']:.1f}/10")
    
    # Print component types
    component_types = {}
    for component in analysis.components.values():
        component_types[component.component_type] = component_types.get(component.component_type, 0) + 1
    
    print(f"\n📋 Component Types:")
    for comp_type, count in sorted(component_types.items()):
        print(f"  {comp_type}: {count}")
    
    # Print purified reconstruction capabilities
    print(f"\n🧬 Purified Reconstruction Capabilities: {len(analysis.purified_reconstruction_capabilities)}")
    for i, capability in enumerate(analysis.purified_reconstruction_capabilities[:5]):
        print(f"  {i+1}. {capability}")
    
    # Print security features
    print(f"\n🛡️ Security Features: {len(analysis.security_features)}")
    for i, feature in enumerate(analysis.security_features[:5]):
        print(f"  {i+1}. {feature}")
    
    # Print recommendations
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(analysis.recommendations):
        print(f"  {i+1}. {rec}")
    
    # Save analysis and documentation
    analyzer.save_analysis()
    analyzer.generate_documentation()
    
    print(f"\n✅ Complete stack analysis finished!")
    print("🎉 Revolutionary purified reconstruction system fully documented!")

if __name__ == "__main__":
    main()
