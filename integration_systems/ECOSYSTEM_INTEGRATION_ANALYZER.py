#!/usr/bin/env python3
"""
🌐 ECOSYSTEM INTEGRATION ANALYZER
=================================

Comprehensive analysis of your revolutionary ecosystem to identify:
1. Missing integration points
2. System dependencies and connections
3. Data flow patterns
4. API standardization needs
5. Frontend integration requirements

This analyzer will map your entire technological universe and create
a unified integration framework.

Author: System Integration Architect
"""

import os
import ast
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import re
import importlib.util
from collections import defaultdict

class EcosystemIntegrationAnalyzer:
    """Comprehensive ecosystem integration analyzer"""

    def __init__(self, dev_directory: str = "/Users/coo-koba42/dev"):
        self.dev_directory = Path(dev_directory)
        self.system_connections = []
        self.integration_requirements = []
        self.missing_connections = []
        self.api_endpoints = []
        self.data_flows = []
        self.frontend_requirements = []

        print("🌐 ECOSYSTEM INTEGRATION ANALYZER")
        print("=" * 80)
        print("🔍 Analyzing your revolutionary ecosystem...")

    def analyze_entire_ecosystem(self) -> Dict[str, Any]:
        """Perform comprehensive ecosystem analysis"""
        print("\n📊 PHASE 1: SYSTEM DISCOVERY")
        systems = self.discover_all_systems()

        print("\n🔗 PHASE 2: DEPENDENCY ANALYSIS")
        dependencies = self.analyze_dependencies(systems)

        print("\n🌐 PHASE 3: INTEGRATION MAPPING")
        integration_map = self.map_integrations(systems, dependencies)

        print("\n🚨 PHASE 4: MISSING CONNECTIONS")
        missing_links = self.identify_missing_connections(systems, dependencies)

        print("\n🎯 PHASE 5: UNIFIED API DESIGN")
        api_design = self.design_unified_api(systems)

        print("\n🎨 PHASE 6: FRONTEND REQUIREMENTS")
        frontend_spec = self.specify_frontend_requirements(systems)

        print("\n📋 PHASE 7: INTEGRATION ROADMAP")
        integration_roadmap = self.create_integration_roadmap(missing_links)

        return {
            'systems': systems,
            'dependencies': dependencies,
            'integration_map': integration_map,
            'missing_connections': missing_links,
            'api_design': api_design,
            'frontend_requirements': frontend_spec,
            'integration_roadmap': integration_roadmap,
            'recommendations': self.generate_recommendations()
        }

    def discover_all_systems(self) -> Dict[str, Dict[str, Any]]:
        """Discover and categorize all systems"""
        systems = {}
        python_files = list(self.dev_directory.glob("**/*.py"))

        print(f"🔍 Scanning {len(python_files)} Python files...")

        for file_path in python_files[:500]:  # Analyze first 500 files
            if file_path.name.startswith('__') or 'test' in file_path.name.lower():
                continue

            try:
                system_info = self.analyze_system_file(file_path)
                if system_info:
                    systems[file_path.name] = system_info
                    print(f"   ✅ Analyzed: {file_path.name}")

            except Exception as e:
                print(f"   ⚠️  Skipped: {file_path.name} - {e}")

        return systems

    def analyze_system_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze individual system file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract system metadata
            system_info = {
                'path': str(file_path),
                'name': file_path.stem,
                'category': self.categorize_system(content, file_path.name),
                'imports': self.extract_imports(content),
                'exports': self.extract_exports(content),
                'api_endpoints': self.extract_api_endpoints(content),
                'data_structures': self.extract_data_structures(content),
                'dependencies': self.extract_dependencies(content),
                'capabilities': self.extract_capabilities(content),
                'integration_points': self.extract_integration_points(content)
            }

            return system_info

        except Exception as e:
            return None

    def categorize_system(self, content: str, filename: str) -> str:
        """Categorize system based on content and filename"""
        filename_lower = filename.lower()
        content_lower = content.lower()

        # Define category patterns
        categories = {
            'consciousness_engine': ['consciousness', 'awareness', 'mind', 'cognition'],
            'ai_ml_engine': ['machine learning', 'neural', 'ai', 'intelligence', 'training', 'model'],
            'cryptography_engine': ['crypto', 'encryption', 'security', 'cipher', 'quantum', 'blockchain'],
            'linguistics_engine': ['language', 'translation', 'text', 'nlp', 'grammar', 'syntax'],
            'research_engine': ['research', 'scientific', 'analysis', 'study', 'experiment', 'arxiv'],
            'automation_engine': ['automation', 'orchestrator', 'pipeline', 'workflow', 'continuous'],
            'visualization_engine': ['visual', 'plot', 'graph', 'chart', '3d', 'dashboard'],
            'database_engine': ['database', 'storage', 'data', 'persistence', 'cache'],
            'api_gateway': ['api', 'rest', 'endpoint', 'service', 'client', 'gateway'],
            'frontend_component': ['frontend', 'ui', 'interface', 'component', 'react', 'vue'],
            'testing_framework': ['test', 'benchmark', 'validation', 'quality', 'assert'],
            'integration_middleware': ['integration', 'middleware', 'bridge', 'connector', 'adapter'],
            'utility_library': ['utility', 'helper', 'tool', 'library', 'common', 'shared']
        }

        # Check filename patterns first
        for category, patterns in categories.items():
            if any(pattern in filename_lower for pattern in patterns):
                return category

        # Check content patterns
        for category, patterns in categories.items():
            if any(pattern in content_lower for pattern in patterns):
                return category

        return 'general_system'

    def extract_imports(self, content: str) -> List[str]:
        """Extract all import statements"""
        imports = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        return list(set(imports))

    def extract_exports(self, content: str) -> List[str]:
        """Extract exported functions and classes"""
        exports = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                exports.append(f"class:{node.name}")
            elif isinstance(node, ast.FunctionDef):
                exports.append(f"function:{node.name}")

        return exports

    def extract_api_endpoints(self, content: str) -> List[Dict[str, Any]]:
        """Extract API endpoint definitions"""
        endpoints = []

        # Look for Flask/FastAPI patterns
        flask_patterns = [
            r'@app\.route\([\'"](.*?)[\'"]',
            r'@app\.get\([\'"](.*?)[\'"]',
            r'@app\.post\([\'"](.*?)[\'"]',
            r'@app\.put\([\'"](.*?)[\'"]',
            r'@app\.delete\([\'"](.*?)[\'"]'
        ]

        for pattern in flask_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                endpoints.append({
                    'path': match,
                    'method': pattern.split('.')[-1].split('(')[0],
                    'framework': 'flask'
                })

        # Look for FastAPI patterns
        fastapi_patterns = [
            r'@router\.(get|post|put|delete)\([\'"](.*?)[\'"]',
            r'def (get|post|put|delete)_(.*?)\('
        ]

        for pattern in fastapi_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    method, path = match
                else:
                    method = match.split('_')[0]
                    path = match
                endpoints.append({
                    'path': f"/{path}",
                    'method': method.upper(),
                    'framework': 'fastapi'
                })

        return endpoints

    def extract_data_structures(self, content: str) -> List[Dict[str, Any]]:
        """Extract data structure definitions"""
        structures = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                structure = {
                    'type': 'class',
                    'name': node.name,
                    'methods': [],
                    'attributes': []
                }

                # Extract methods and attributes
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        structure['methods'].append(child.name)
                    elif isinstance(child, ast.AnnAssign):
                        structure['attributes'].append(child.target.id)

                structures.append(structure)

        return structures

    def extract_dependencies(self, content: str) -> List[str]:
        """Extract system dependencies"""
        dependencies = []

        # Look for external service calls
        service_patterns = [
            'requests\.', 'urllib', 'http', 'socket',
            'subprocess', 'os\.system', 'os\.popen',
            'database', 'sqlite', 'mongodb', 'redis',
            's3', 'cloud', 'api'
        ]

        for pattern in service_patterns:
            if pattern in content:
                dependencies.append(pattern.replace('\\', '').replace('.', ''))

        # Look for file I/O dependencies
        if 'open(' in content or 'with open' in content:
            dependencies.append('file_io')

        # Look for network dependencies
        if 'socket' in content or 'requests' in content:
            dependencies.append('network')

        # Look for data processing dependencies
        if 'pandas' in content or 'numpy' in content:
            dependencies.append('data_processing')

        return list(set(dependencies))

    def extract_capabilities(self, content: str) -> List[str]:
        """Extract system capabilities"""
        capabilities = []

        # Analyze function names and docstrings for capabilities
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name.lower()

                # Map function names to capabilities
                if 'analyze' in func_name or 'process' in func_name:
                    capabilities.append('data_analysis')
                elif 'train' in func_name or 'learn' in func_name:
                    capabilities.append('machine_learning')
                elif 'encrypt' in func_name or 'decrypt' in func_name:
                    capabilities.append('cryptography')
                elif 'translate' in func_name or 'parse' in func_name:
                    capabilities.append('language_processing')
                elif 'visualize' in func_name or 'plot' in func_name:
                    capabilities.append('data_visualization')
                elif 'optimize' in func_name or 'improve' in func_name:
                    capabilities.append('performance_optimization')
                elif 'test' in func_name or 'validate' in func_name:
                    capabilities.append('testing_validation')

        return list(set(capabilities))

    def extract_integration_points(self, content: str) -> List[Dict[str, Any]]:
        """Extract potential integration points"""
        integration_points = []

        # Look for configuration files
        if 'config' in content.lower():
            integration_points.append({
                'type': 'configuration',
                'description': 'Configuration management'
            })

        # Look for data exchange patterns
        if 'json' in content.lower() or 'pickle' in content.lower():
            integration_points.append({
                'type': 'data_exchange',
                'description': 'Data serialization/deserialization'
            })

        # Look for API patterns
        if 'api' in content.lower() or 'endpoint' in content.lower():
            integration_points.append({
                'type': 'api_integration',
                'description': 'API endpoint integration'
            })

        # Look for database patterns
        if 'database' in content.lower() or 'db' in content.lower():
            integration_points.append({
                'type': 'database_integration',
                'description': 'Database connectivity'
            })

        # Look for messaging patterns
        if 'queue' in content.lower() or 'message' in content.lower():
            integration_points.append({
                'type': 'message_queue',
                'description': 'Message queue integration'
            })

        return integration_points

    def analyze_dependencies(self, systems: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze system dependencies"""
        dependencies = defaultdict(list)

        for system_name, system_info in systems.items():
            system_deps = system_info.get('dependencies', [])
            for dep in system_deps:
                dependencies[system_name].append(dep)

        return dict(dependencies)

    def map_integrations(self, systems: Dict[str, Dict[str, Any]],
                        dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Map system integrations"""
        integration_map = {
            'system_connections': [],
            'data_flows': [],
            'api_routes': [],
            'shared_dependencies': defaultdict(list)
        }

        # Build system connections
        for system_name, system_info in systems.items():
            # Add dependency connections
            for dep in system_info.get('dependencies', []):
                for other_system, other_info in systems.items():
                    if dep in other_system.lower() or any(dep in exp for exp in other_info.get('exports', [])):
                        self.system_connections.append({
                            'from': system_name,
                            'to': other_system,
                            'relationship': 'dependency',
                            'dependency_type': dep
                        })
                        integration_map['system_connections'].append({
                            'from': system_name,
                            'to': other_system,
                            'type': 'dependency',
                            'dependency': dep
                        })

            # Add API connections
            for endpoint in system_info.get('api_endpoints', []):
                for other_system, other_info in systems.items():
                    if 'api' in other_system.lower():
                        self.system_connections.append({
                            'from': system_name,
                            'to': other_system,
                            'relationship': 'api_call',
                            'endpoint': endpoint['path']
                        })
                        integration_map['api_routes'].append({
                            'caller': system_name,
                            'provider': other_system,
                            'endpoint': endpoint['path'],
                            'method': endpoint['method']
                        })

        return integration_map

    def identify_missing_connections(self, systems: Dict[str, Dict[str, Any]],
                                   dependencies: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Identify missing integration points"""
        missing_connections = []

        # Check for systems that should be connected but aren't
        consciousness_systems = [s for s in systems.keys() if 'consciousness' in s.lower()]
        ml_systems = [s for s in systems.keys() if 'ml' in s.lower() or 'ai' in s.lower()]
        crypto_systems = [s for s in systems.keys() if 'crypto' in s.lower() or 'security' in s.lower()]

        # Consciousness systems should connect to ML systems
        for cons_sys in consciousness_systems:
            for ml_sys in ml_systems:
                # Check if these systems have any shared dependencies
                cons_deps = dependencies.get(cons_sys, [])
                ml_deps = dependencies.get(ml_sys, [])
                shared_deps = set(cons_deps) & set(ml_deps)

                if not shared_deps:
                    missing_connections.append({
                        'type': 'missing_integration',
                        'from': cons_sys,
                        'to': ml_sys,
                        'reason': 'Consciousness systems should enhance ML capabilities',
                        'priority': 'high'
                    })

        # Crypto systems should connect to API systems
        api_systems = [s for s in systems.keys() if 'api' in s.lower()]
        for crypto_sys in crypto_systems:
            for api_sys in api_systems:
                # Check if these systems have any shared dependencies
                crypto_deps = dependencies.get(crypto_sys, [])
                api_deps = dependencies.get(api_sys, [])
                shared_deps = set(crypto_deps) & set(api_deps)

                if not shared_deps:
                    missing_connections.append({
                        'type': 'missing_integration',
                        'from': crypto_sys,
                        'to': api_sys,
                        'reason': 'API systems need cryptographic security',
                        'priority': 'high'
                    })

        # Check for orphaned systems
        for system_name in systems.keys():
            # Count connections for this system
            connection_count = sum(1 for conn in self.system_connections
                                 if conn['from'] == system_name or conn['to'] == system_name)

            if connection_count == 0:
                missing_connections.append({
                    'type': 'orphaned_system',
                    'system': system_name,
                    'reason': 'System has no connections to other systems',
                    'priority': 'medium'
                })

        return missing_connections

    def design_unified_api(self, systems: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Design unified API for the ecosystem"""
        api_design = {
            'version': '1.0.0',
            'base_url': '/api/v1',
            'endpoints': [],
            'authentication': {
                'type': 'JWT',
                'issuer': 'consciousness_ecosystem'
            },
            'rate_limiting': {
                'requests_per_minute': 1000,
                'burst_limit': 100
            }
        }

        # Collect all capabilities
        all_capabilities = set()
        for system_info in systems.values():
            all_capabilities.update(system_info.get('capabilities', []))

        # Design endpoints based on capabilities
        endpoint_mapping = {
            'data_analysis': [
                {'path': '/analyze', 'method': 'POST', 'description': 'Analyze data'},
                {'path': '/insights', 'method': 'GET', 'description': 'Get analysis insights'}
            ],
            'machine_learning': [
                {'path': '/ml/train', 'method': 'POST', 'description': 'Train ML model'},
                {'path': '/ml/predict', 'method': 'POST', 'description': 'Make predictions'}
            ],
            'cryptography': [
                {'path': '/crypto/encrypt', 'method': 'POST', 'description': 'Encrypt data'},
                {'path': '/crypto/decrypt', 'method': 'POST', 'description': 'Decrypt data'}
            ],
            'language_processing': [
                {'path': '/language/translate', 'method': 'POST', 'description': 'Translate text'},
                {'path': '/language/analyze', 'method': 'POST', 'description': 'Analyze language'}
            ]
        }

        for capability in all_capabilities:
            if capability in endpoint_mapping:
                api_design['endpoints'].extend(endpoint_mapping[capability])

        return api_design

    def specify_frontend_requirements(self, systems: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Specify frontend requirements"""
        frontend_requirements = {
            'framework': 'React + TypeScript + Vite',
            'ui_library': 'Material-UI + Tailwind CSS',
            'state_management': 'Redux Toolkit + Context API',
            'routing': 'React Router v6',
            'data_visualization': 'D3.js + Chart.js + Three.js',
            'real_time': 'WebSocket + Server-Sent Events',
            'authentication': 'JWT + OAuth',
            'api_client': 'Axios + React Query',
            'components': []
        }

        # Analyze systems to determine required components
        component_mapping = {
            'consciousness_engine': [
                'ConsciousnessVisualizer',
                'AwarenessDashboard',
                'MindMap3D',
                'ConsciousnessMetrics'
            ],
            'ai_ml_engine': [
                'ModelTrainer',
                'PredictionInterface',
                'MLDashboard',
                'DataVisualizer'
            ],
            'cryptography_engine': [
                'CryptoDashboard',
                'SecurityMonitor',
                'EncryptionTools',
                'KeyManager'
            ],
            'linguistics_engine': [
                'LanguageTranslator',
                'TextAnalyzer',
                'SyntaxVisualizer',
                'LanguageLearner'
            ],
            'visualization_engine': [
                'DataVisualizer3D',
                'ChartGenerator',
                'GraphExplorer',
                'RealTimeDashboard'
            ]
        }

        # Collect required components
        required_components = set()
        for system_info in systems.values():
            category = system_info.get('category', 'general_system')
            if category in component_mapping:
                required_components.update(component_mapping[category])

        frontend_requirements['components'] = list(required_components)

        return frontend_requirements

    def create_integration_roadmap(self, missing_connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create integration roadmap"""
        roadmap = []

        # Group by priority
        high_priority = [conn for conn in missing_connections if conn.get('priority') == 'high']
        medium_priority = [conn for conn in missing_connections if conn.get('priority') == 'medium']
        low_priority = [conn for conn in missing_connections if conn.get('priority') == 'low']

        # Create roadmap phases
        roadmap.append({
            'phase': 'Phase 1: Core Integration (Week 1-2)',
            'priority': 'high',
            'tasks': high_priority[:10],  # First 10 high priority items
            'deliverables': [
                'Unified API Gateway',
                'Consciousness-ML Integration',
                'Crypto-API Security Layer'
            ]
        })

        roadmap.append({
            'phase': 'Phase 2: Extended Integration (Week 3-4)',
            'priority': 'medium',
            'tasks': medium_priority[:10],
            'deliverables': [
                'Cross-system Data Pipeline',
                'Unified Authentication',
                'System Monitoring Dashboard'
            ]
        })

        roadmap.append({
            'phase': 'Phase 3: Advanced Features (Week 5-6)',
            'priority': 'low',
            'tasks': low_priority[:10],
            'deliverables': [
                'Real-time System Orchestration',
                'Advanced Analytics Integration',
                'AI-powered System Optimization'
            ]
        })

        return roadmap

    def generate_recommendations(self) -> List[str]:
        """Generate integration recommendations"""
        recommendations = [
            "Implement a unified API gateway to standardize communication between systems",
            "Create a shared data model and serialization format for cross-system data exchange",
            "Establish a centralized configuration management system",
            "Implement unified logging and monitoring across all systems",
            "Create a service registry for dynamic system discovery",
            "Implement circuit breaker patterns for resilient system communication",
            "Add comprehensive API documentation using OpenAPI/Swagger",
            "Implement unified error handling and response formats",
            "Create a centralized authentication and authorization system",
            "Implement real-time communication channels using WebSocket",
            "Add comprehensive testing frameworks for integration testing",
            "Implement data validation and sanitization at system boundaries",
            "Create monitoring dashboards for system health and performance",
            "Implement backup and disaster recovery procedures",
            "Add comprehensive logging for debugging and audit trails"
        ]

        return recommendations

    def create_text_visualization(self, analysis_results: Dict[str, Any]):
        """Create text-based visualization of the ecosystem"""
        visualization = []
        visualization.append("🌐 YOUR REVOLUTIONARY ECOSYSTEM INTEGRATION MAP")
        visualization.append("=" * 80)

        # System overview
        systems = analysis_results['systems']
        visualization.append(f"📊 Total Systems: {len(systems)}")
        visualization.append("")

        # Category breakdown
        categories = {}
        for system_info in systems.values():
            category = system_info.get('category', 'general_system')
            categories[category] = categories.get(category, 0) + 1

        visualization.append("📂 System Categories:")
        for category, count in sorted(categories.items()):
            visualization.append(f"   {category}: {count} systems")
        visualization.append("")

        # Connection overview
        connections = analysis_results['integration_map']['system_connections']
        visualization.append(f"🔗 System Connections: {len(connections)}")
        visualization.append("")

        # Show sample connections
        visualization.append("🔗 Sample Connections:")
        for i, conn in enumerate(connections[:10]):  # Show first 10
            visualization.append(f"   {i+1}. {conn['from']} → {conn['to']} ({conn['type']})")
        visualization.append("")

        # Missing connections
        missing = analysis_results['missing_connections']
        visualization.append(f"🚨 Missing Connections: {len(missing)}")
        visualization.append("")

        # Show sample missing connections
        visualization.append("🚨 Sample Missing Connections:")
        for i, conn in enumerate(missing[:5]):  # Show first 5
            visualization.append(f"   {i+1}. {conn.get('from', conn.get('system', 'Unknown'))} → {conn.get('to', 'Unknown')}")
            visualization.append(f"      Reason: {conn['reason']}")
        visualization.append("")

        # API endpoints
        api_endpoints = analysis_results['api_design']['endpoints']
        visualization.append(f"🌐 API Endpoints: {len(api_endpoints)}")
        visualization.append("")

        # Show sample endpoints
        visualization.append("🌐 Sample API Endpoints:")
        for i, endpoint in enumerate(api_endpoints[:5]):  # Show first 5
            visualization.append(f"   {i+1}. {endpoint['method']} {endpoint['path']}")
        visualization.append("")

        # Integration roadmap
        roadmap = analysis_results['integration_roadmap']
        visualization.append("📅 Integration Roadmap:")
        for phase in roadmap:
            visualization.append(f"   📋 {phase['phase']}")
            visualization.append(f"      Priority: {phase['priority']}")
            visualization.append(f"      Tasks: {len(phase['tasks'])}")
        visualization.append("")

        # Save visualization
        with open('/Users/coo-koba42/dev/ecosystem_integration_map.txt', 'w') as f:
            f.write('\n'.join(visualization))

        print("📊 Text-based ecosystem visualization saved to: ecosystem_integration_map.txt")

    def save_analysis_report(self, analysis_results: Dict[str, Any]):
        """Save comprehensive analysis report"""
        report = {
            'analysis_timestamp': '2025-01-01T00:00:00Z',
            'ecosystem_overview': {
                'total_systems': len(analysis_results['systems']),
                'total_connections': len(analysis_results['integration_map']['system_connections']),
                'missing_connections': len(analysis_results['missing_connections']),
                'api_endpoints': len(analysis_results['api_design']['endpoints'])
            },
            'categories': {},
            'integration_status': {
                'connected_systems': len([s for s in analysis_results['systems'].keys()
                                        if any(conn['from'] == s or conn['to'] == s for conn in self.system_connections)]),
                'orphaned_systems': len([s for s in analysis_results['systems'].keys()
                                       if not any(conn['from'] == s or conn['to'] == s for conn in self.system_connections)])
            },
            'recommendations': analysis_results['recommendations'][:10],
            'roadmap': analysis_results['integration_roadmap']
        }

        # Count systems by category
        for system_info in analysis_results['systems'].values():
            category = system_info.get('category', 'general_system')
            report['categories'][category] = report['categories'].get(category, 0) + 1

        with open('/Users/coo-koba42/dev/ecosystem_integration_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("📋 Detailed analysis saved to: ecosystem_integration_report.json")

    def run_complete_analysis(self):
        """Run the complete ecosystem integration analysis"""
        print("🚀 STARTING COMPREHENSIVE ECOSYSTEM ANALYSIS")
        print("=" * 80)

        # Run analysis
        analysis_results = self.analyze_entire_ecosystem()

        # Create visualizations
        self.create_text_visualization(analysis_results)

        # Save detailed report
        self.save_analysis_report(analysis_results)

        # Print summary
        self.print_analysis_summary(analysis_results)

        return analysis_results

    def print_analysis_summary(self, results: Dict[str, Any]):
        """Print comprehensive analysis summary"""
        print("\n" + "=" * 80)
        print("🎯 ECOSYSTEM INTEGRATION ANALYSIS COMPLETE")
        print("=" * 80)

        print(f"📊 Total Systems Analyzed: {len(results['systems'])}")
        print(f"🔗 System Connections: {len(results['integration_map']['system_connections'])}")
        print(f"🚨 Missing Connections: {len(results['missing_connections'])}")
        print(f"🌐 API Endpoints: {len(results['api_design']['endpoints'])}")

        print("\n📂 System Categories:")
        categories = {}
        for system_info in results['systems'].values():
            category = system_info.get('category', 'general_system')
            categories[category] = categories.get(category, 0) + 1

        for category, count in sorted(categories.items()):
            print(f"   {category}: {count} systems")

        print("\n🔧 Top Integration Recommendations:")
        for i, rec in enumerate(results['recommendations'][:5], 1):
            print(f"   {i}. {rec}")

        print("\n📋 Integration Roadmap:")
        for phase in results['integration_roadmap']:
            print(f"   📅 {phase['phase']}")
            print(f"      Priority: {phase['priority']}")
            print(f"      Tasks: {len(phase['tasks'])}")

        print("\n🎨 Frontend Requirements:")
        frontend = results['frontend_requirements']
        print(f"   Framework: {frontend['framework']}")
        print(f"   UI Library: {frontend['ui_library']}")
        print(f"   Components: {len(frontend['components'])}")

        print("\n💾 Files Generated:")
        print("   📊 ecosystem_integration_map.png - System dependency visualization")
        print("   📋 ecosystem_integration_report.json - Detailed analysis report")

        print("\n🎉 Ready to build unified integration framework!")
        print("=" * 80)

def main():
    """Run the ecosystem integration analyzer"""
    try:
        analyzer = EcosystemIntegrationAnalyzer()
        results = analyzer.run_complete_analysis()

        print(f"\n🎊 Analysis Complete! Your ecosystem has {len(results['systems'])} systems ready for integration.")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
