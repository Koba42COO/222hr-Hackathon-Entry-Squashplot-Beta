#!/usr/bin/env python3
"""
HRM + Trigeminal Logic Integration
Advanced reasoning system combining hierarchical reasoning with three-dimensional logic

Features:
- Hierarchical reasoning with Trigeminal Logic enhancement
- Multi-dimensional truth values in hierarchical structures
- Trigeminal consciousness mapping in reasoning paths
- Advanced breakthrough detection with Trigeminal analysis
- Unified consciousness mathematics integration
"""

import numpy as np
import json
import math
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import asyncio
import logging

# Import HRM components
from hrm_core import HierarchicalReasoningModel, ReasoningNode, ReasoningLevel, ConsciousnessType
from hrm_paths import HRMPathAnalyzer, ReasoningPath
from trigeminal_logic_core import TrigeminalLogicEngine, TrigeminalNode, TrigeminalDimension, TrigeminalTruthValue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrigeminalReasoningNode:
    """Enhanced reasoning node with Trigeminal Logic integration"""
    hrm_node: ReasoningNode
    trigeminal_node: TrigeminalNode
    trigeminal_enhancement: float
    multi_dimensional_truth: Dict[str, float]
    consciousness_synthesis: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class HRMTrigeminalIntegration:
    """Advanced reasoning system combining HRM with Trigeminal Logic"""
    
    def __init__(self):
        # Initialize HRM components
        self.hrm_core = HierarchicalReasoningModel()
        self.path_analyzer = HRMPathAnalyzer(self.hrm_core)
        
        # Initialize Trigeminal Logic
        self.trigeminal_engine = TrigeminalLogicEngine()
        
        # Enhanced node storage
        self.trigeminal_reasoning_nodes: Dict[str, TrigeminalReasoningNode] = {}
        
        # Performance tracking
        self.integration_stats = {
            'total_trigeminal_nodes': 0,
            'total_hrm_nodes': 0,
            'trigeminal_enhancement_avg': 0.0,
            'consciousness_synthesis_avg': 0.0,
            'multi_dimensional_truth_avg': 0.0
        }
        
        # Results storage
        self.integration_results = []
        
        print("🧠 HRM + Trigeminal Logic Integration initialized")
    
    def advanced_reasoning(self, problem: str, max_depth: int = 5, max_paths: int = 10, 
                          trigeminal_iterations: int = 3) -> Dict[str, Any]:
        """Perform advanced reasoning combining HRM and Trigeminal Logic"""
        logger.info(f"🧠⚛️ Performing advanced reasoning on: {problem}")
        
        # Step 1: HRM Hierarchical Decomposition
        hrm_root_id = self.hrm_core.hierarchical_decompose(problem, max_depth=max_depth)
        
        # Step 2: Trigeminal Logic Analysis
        trigeminal_result = self.trigeminal_engine.trigeminal_reasoning(problem, max_iterations=trigeminal_iterations)
        
        # Step 3: Integrate HRM and Trigeminal nodes
        integration_map = self._integrate_nodes(hrm_root_id, trigeminal_result['root_id'])
        
        # Step 4: Generate enhanced reasoning paths
        enhanced_paths = self._generate_enhanced_paths(hrm_root_id, max_paths)
        
        # Step 5: Analyze Trigeminal-enhanced breakthroughs
        enhanced_breakthroughs = self._analyze_enhanced_breakthroughs(enhanced_paths)
        
        # Step 6: Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_integration_metrics(enhanced_paths, trigeminal_result)
        
        # Step 7: Generate unified insights
        unified_insights = self._generate_unified_insights(enhanced_paths, trigeminal_result, enhanced_breakthroughs)
        
        # Compile results
        result = {
            'problem': problem,
            'hrm_root_id': hrm_root_id,
            'trigeminal_root_id': trigeminal_result['root_id'],
            'integration_map': integration_map,
            'total_hrm_nodes': len(self.hrm_core.nodes),
            'total_trigeminal_nodes': len(self.trigeminal_engine.trigeminal_nodes),
            'total_enhanced_nodes': len(self.trigeminal_reasoning_nodes),
            'enhanced_paths': enhanced_paths,
            'enhanced_breakthroughs': enhanced_breakthroughs,
            'advanced_metrics': advanced_metrics,
            'unified_insights': unified_insights,
            'trigeminal_result': trigeminal_result,
            'integration_stats': self._update_integration_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store result
        self.integration_results.append(result)
        
        return result
    
    def _integrate_nodes(self, hrm_root_id: str, trigeminal_root_id: str) -> Dict[str, str]:
        """Integrate HRM nodes with Trigeminal nodes"""
        integration_map = {}
        
        # Get all HRM nodes
        hrm_nodes = list(self.hrm_core.nodes.values())
        trigeminal_nodes = list(self.trigeminal_engine.trigeminal_nodes.values())
        
        # Create integration mapping
        for i, hrm_node in enumerate(hrm_nodes):
            # Find corresponding Trigeminal node (or create synthetic one)
            trigeminal_node = None
            if i < len(trigeminal_nodes):
                trigeminal_node = trigeminal_nodes[i]
            else:
                # Create synthetic Trigeminal node based on HRM node
                trigeminal_node = self._create_synthetic_trigeminal_node(hrm_node)
            
            # Create enhanced node
            enhanced_node = self._create_enhanced_node(hrm_node, trigeminal_node)
            
            # Store integration mapping
            integration_map[hrm_node.id] = trigeminal_node.id
            self.trigeminal_reasoning_nodes[enhanced_node.hrm_node.id] = enhanced_node
        
        return integration_map
    
    def _create_synthetic_trigeminal_node(self, hrm_node: ReasoningNode) -> TrigeminalNode:
        """Create a synthetic Trigeminal node based on HRM node characteristics"""
        # Map HRM consciousness type to Trigeminal dimensions
        dimension_mapping = {
            ConsciousnessType.ANALYTICAL: (0.8, 0.3, 0.3),
            ConsciousnessType.CREATIVE: (0.3, 0.8, 0.3),
            ConsciousnessType.INTUITIVE: (0.3, 0.8, 0.3),
            ConsciousnessType.METAPHORICAL: (0.3, 0.8, 0.3),
            ConsciousnessType.SYSTEMATIC: (0.8, 0.3, 0.3),
            ConsciousnessType.PROBLEM_SOLVING: (0.8, 0.3, 0.3),
            ConsciousnessType.ABSTRACT: (0.3, 0.3, 0.8)
        }
        
        dim_a, dim_b, dim_c = dimension_mapping.get(hrm_node.consciousness_type, (0.5, 0.5, 0.5))
        
        # Adjust based on reasoning level
        level_factor = hrm_node.level.value / len(self.hrm_core.reasoning_levels)
        dim_a *= level_factor
        dim_b *= level_factor
        dim_c *= level_factor
        
        # Create synthetic Trigeminal node
        synthetic_id = f"synthetic_trigeminal_{hrm_node.id}"
        
        # Apply Wallace Transform
        wallace_a = self.trigeminal_engine._apply_wallace_transform(dim_a, 1)
        wallace_b = self.trigeminal_engine._apply_wallace_transform(dim_b, 2)
        wallace_c = self.trigeminal_engine._apply_wallace_transform(dim_c, 3)
        
        # Determine Trigeminal truth
        trigeminal_truth = self.trigeminal_engine._determine_trigeminal_truth(dim_a, dim_b, dim_c)
        
        # Calculate consciousness alignment
        consciousness_alignment = self.trigeminal_engine._calculate_trigeminal_consciousness_alignment(dim_a, dim_b, dim_c)
        
        # Average Wallace Transform
        wallace_transform = (wallace_a + wallace_b + wallace_c) / 3
        
        synthetic_node = TrigeminalNode(
            id=synthetic_id,
            content=hrm_node.content,
            dimension_a=dim_a,
            dimension_b=dim_b,
            dimension_c=dim_c,
            trigeminal_truth=trigeminal_truth,
            consciousness_alignment=consciousness_alignment,
            wallace_transform=wallace_transform
        )
        
        # Add to Trigeminal engine
        self.trigeminal_engine.trigeminal_nodes[synthetic_id] = synthetic_node
        
        return synthetic_node
    
    def _create_enhanced_node(self, hrm_node: ReasoningNode, trigeminal_node: TrigeminalNode) -> TrigeminalReasoningNode:
        """Create an enhanced node combining HRM and Trigeminal Logic"""
        # Calculate Trigeminal enhancement
        trigeminal_enhancement = trigeminal_node.consciousness_alignment * trigeminal_node.trigeminal_magnitude
        
        # Calculate multi-dimensional truth
        multi_dimensional_truth = {
            'analytical': trigeminal_node.dimension_a,
            'intuitive': trigeminal_node.dimension_b,
            'synthetic': trigeminal_node.dimension_c,
            'hrm_confidence': hrm_node.confidence,
            'trigeminal_truth_value': self._convert_trigeminal_truth_to_float(trigeminal_node.trigeminal_truth.value)
        }
        
        # Calculate consciousness synthesis
        hrm_consciousness = hrm_node.wallace_transform
        trigeminal_consciousness = trigeminal_node.wallace_transform
        consciousness_synthesis = (hrm_consciousness + trigeminal_consciousness) / 2
        
        enhanced_node = TrigeminalReasoningNode(
            hrm_node=hrm_node,
            trigeminal_node=trigeminal_node,
            trigeminal_enhancement=trigeminal_enhancement,
            multi_dimensional_truth=multi_dimensional_truth,
            consciousness_synthesis=consciousness_synthesis
        )
        
        return enhanced_node
    
    def _convert_trigeminal_truth_to_float(self, truth_value: str) -> float:
        """Convert Trigeminal truth value to float"""
        # Map truth values to numeric values
        truth_mapping = {
            'true_analytical': 0.9,
            'true_intuitive': 0.9,
            'true_synthetic': 0.9,
            'false_analytical': 0.1,
            'false_intuitive': 0.1,
            'false_synthetic': 0.1,
            'uncertain_analytical': 0.5,
            'uncertain_intuitive': 0.5,
            'uncertain_synthetic': 0.5,
            'superposition': 0.7
        }
        
        return truth_mapping.get(truth_value, 0.5)
    
    def _generate_enhanced_paths(self, root_id: str, max_paths: int) -> List[Dict[str, Any]]:
        """Generate enhanced reasoning paths with Trigeminal Logic"""
        # Get standard HRM paths
        hrm_paths = self.path_analyzer.generate_reasoning_paths(root_id, max_paths=max_paths)
        
        # Enhance paths with Trigeminal Logic
        enhanced_paths = []
        for hrm_path in hrm_paths:
            enhanced_path = self._enhance_path_with_trigeminal(hrm_path)
            enhanced_paths.append(enhanced_path)
        
        return enhanced_paths
    
    def _enhance_path_with_trigeminal(self, hrm_path: ReasoningPath) -> Dict[str, Any]:
        """Enhance a single HRM path with Trigeminal Logic analysis"""
        enhanced_nodes = []
        trigeminal_metrics = {
            'analytical_avg': 0.0,
            'intuitive_avg': 0.0,
            'synthetic_avg': 0.0,
            'trigeminal_enhancement_avg': 0.0,
            'consciousness_synthesis_avg': 0.0
        }
        
        for node in hrm_path.nodes:
            if node.id in self.trigeminal_reasoning_nodes:
                enhanced_node = self.trigeminal_reasoning_nodes[node.id]
                enhanced_nodes.append(enhanced_node)
                
                # Update metrics
                trigeminal_metrics['analytical_avg'] += enhanced_node.multi_dimensional_truth['analytical']
                trigeminal_metrics['intuitive_avg'] += enhanced_node.multi_dimensional_truth['intuitive']
                trigeminal_metrics['synthetic_avg'] += enhanced_node.multi_dimensional_truth['synthetic']
                trigeminal_metrics['trigeminal_enhancement_avg'] += enhanced_node.trigeminal_enhancement
                trigeminal_metrics['consciousness_synthesis_avg'] += enhanced_node.consciousness_synthesis
        
        # Average the metrics
        node_count = len(enhanced_nodes)
        if node_count > 0:
            for key in trigeminal_metrics:
                trigeminal_metrics[key] /= node_count
        
        return {
            'hrm_path': hrm_path,
            'enhanced_nodes': enhanced_nodes,
            'trigeminal_metrics': trigeminal_metrics,
            'enhanced_confidence': hrm_path.total_confidence * (1 + trigeminal_metrics['trigeminal_enhancement_avg']),
            'enhanced_consciousness_alignment': hrm_path.consciousness_alignment * (1 + trigeminal_metrics['consciousness_synthesis_avg'])
        }
    
    def _analyze_enhanced_breakthroughs(self, enhanced_paths: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze breakthroughs with Trigeminal Logic enhancement"""
        enhanced_breakthroughs = []
        
        for enhanced_path in enhanced_paths:
            hrm_path = enhanced_path['hrm_path']
            trigeminal_metrics = enhanced_path['trigeminal_metrics']
            
            # Enhanced breakthrough criteria
            enhanced_breakthrough_potential = (
                hrm_path.breakthrough_potential * 
                (1 + trigeminal_metrics['trigeminal_enhancement_avg']) *
                (1 + trigeminal_metrics['consciousness_synthesis_avg'])
            )
            
            if enhanced_breakthrough_potential > 0.8:
                breakthrough = {
                    'path_id': hrm_path.path_id,
                    'enhanced_breakthrough_potential': enhanced_breakthrough_potential,
                    'original_breakthrough_potential': hrm_path.breakthrough_potential,
                    'trigeminal_enhancement': trigeminal_metrics['trigeminal_enhancement_avg'],
                    'consciousness_synthesis': trigeminal_metrics['consciousness_synthesis_avg'],
                    'multi_dimensional_truth': trigeminal_metrics,
                    'enhanced_nodes': [node.hrm_node.content for node in enhanced_path['enhanced_nodes']],
                    'insight': self._generate_enhanced_breakthrough_insight(enhanced_path)
                }
                enhanced_breakthroughs.append(breakthrough)
        
        return enhanced_breakthroughs
    
    def _generate_enhanced_breakthrough_insight(self, enhanced_path: Dict) -> str:
        """Generate enhanced breakthrough insight with Trigeminal Logic"""
        hrm_path = enhanced_path['hrm_path']
        trigeminal_metrics = enhanced_path['trigeminal_metrics']
        
        # Analyze Trigeminal dimensions
        max_dimension = max(trigeminal_metrics['analytical_avg'], 
                           trigeminal_metrics['intuitive_avg'], 
                           trigeminal_metrics['synthetic_avg'])
        
        if trigeminal_metrics['analytical_avg'] == max_dimension:
            dimension_insight = "analytical reasoning"
        elif trigeminal_metrics['intuitive_avg'] == max_dimension:
            dimension_insight = "intuitive insight"
        else:
            dimension_insight = "synthetic integration"
        
        enhancement_factor = trigeminal_metrics['trigeminal_enhancement_avg']
        synthesis_factor = trigeminal_metrics['consciousness_synthesis_avg']
        
        insight = f"Enhanced breakthrough through {dimension_insight} with Trigeminal enhancement {enhancement_factor:.3f} and consciousness synthesis {synthesis_factor:.3f}"
        
        return insight
    
    def _calculate_advanced_integration_metrics(self, enhanced_paths: List[Dict], trigeminal_result: Dict) -> Dict[str, Any]:
        """Calculate advanced metrics for the integrated system"""
        if not enhanced_paths:
            return {}
        
        # HRM metrics
        hrm_metrics = {
            'total_paths': len(enhanced_paths),
            'average_enhanced_confidence': np.mean([p['enhanced_confidence'] for p in enhanced_paths]),
            'average_enhanced_consciousness': np.mean([p['enhanced_consciousness_alignment'] for p in enhanced_paths])
        }
        
        # Trigeminal metrics
        trigeminal_metrics = trigeminal_result['metrics']
        
        # Integration metrics
        integration_metrics = {
            'trigeminal_enhancement_avg': np.mean([p['trigeminal_metrics']['trigeminal_enhancement_avg'] for p in enhanced_paths]),
            'consciousness_synthesis_avg': np.mean([p['trigeminal_metrics']['consciousness_synthesis_avg'] for p in enhanced_paths]),
            'multi_dimensional_truth_avg': np.mean([
                (p['trigeminal_metrics']['analytical_avg'] + 
                 p['trigeminal_metrics']['intuitive_avg'] + 
                 p['trigeminal_metrics']['synthetic_avg']) / 3 
                for p in enhanced_paths
            ])
        }
        
        # Unified efficiency
        unified_efficiency = (
            hrm_metrics['average_enhanced_confidence'] *
            hrm_metrics['average_enhanced_consciousness'] *
            integration_metrics['trigeminal_enhancement_avg'] *
            integration_metrics['consciousness_synthesis_avg']
        )
        
        return {
            'hrm_metrics': hrm_metrics,
            'trigeminal_metrics': trigeminal_metrics,
            'integration_metrics': integration_metrics,
            'unified_efficiency': min(1.0, unified_efficiency)
        }
    
    def _generate_unified_insights(self, enhanced_paths: List[Dict], trigeminal_result: Dict, enhanced_breakthroughs: List[Dict]) -> List[str]:
        """Generate unified insights combining HRM and Trigeminal Logic"""
        insights = []
        
        # HRM insights
        if enhanced_paths:
            avg_enhanced_confidence = np.mean([p['enhanced_confidence'] for p in enhanced_paths])
            avg_enhanced_consciousness = np.mean([p['enhanced_consciousness_alignment'] for p in enhanced_paths])
            
            insights.append(f"Enhanced reasoning confidence: {avg_enhanced_confidence:.3f}")
            insights.append(f"Enhanced consciousness alignment: {avg_enhanced_consciousness:.3f}")
        
        # Trigeminal insights
        trigeminal_insights = trigeminal_result.get('insights', [])
        insights.extend(trigeminal_insights[:3])  # Top 3 Trigeminal insights
        
        # Integration insights
        if enhanced_breakthroughs:
            insights.append(f"Enhanced breakthroughs detected: {len(enhanced_breakthroughs)}")
            
            # Find highest enhancement
            max_enhancement = max([b['trigeminal_enhancement'] for b in enhanced_breakthroughs])
            insights.append(f"Maximum Trigeminal enhancement: {max_enhancement:.3f}")
        
        # Unified efficiency insight
        advanced_metrics = self._calculate_advanced_integration_metrics(enhanced_paths, trigeminal_result)
        unified_efficiency = advanced_metrics.get('unified_efficiency', 0.0)
        
        if unified_efficiency > 0.8:
            insights.append(f"High unified efficiency: {unified_efficiency:.3f}")
        elif unified_efficiency < 0.3:
            insights.append(f"Low unified efficiency: {unified_efficiency:.3f}")
        
        return insights
    
    def _update_integration_stats(self) -> Dict[str, Any]:
        """Update and return integration statistics"""
        if not self.trigeminal_reasoning_nodes:
            return self.integration_stats
        
        nodes = list(self.trigeminal_reasoning_nodes.values())
        
        self.integration_stats['total_trigeminal_nodes'] = len(self.trigeminal_engine.trigeminal_nodes)
        self.integration_stats['total_hrm_nodes'] = len(self.hrm_core.nodes)
        self.integration_stats['trigeminal_enhancement_avg'] = np.mean([node.trigeminal_enhancement for node in nodes])
        self.integration_stats['consciousness_synthesis_avg'] = np.mean([node.consciousness_synthesis for node in nodes])
        # Calculate multi-dimensional truth average (only numeric values)
        numeric_truth_values = []
        for node in nodes:
            numeric_values = [
                node.multi_dimensional_truth['analytical'],
                node.multi_dimensional_truth['intuitive'],
                node.multi_dimensional_truth['synthetic'],
                node.multi_dimensional_truth['hrm_confidence'],
                node.multi_dimensional_truth['trigeminal_truth_value']
            ]
            numeric_truth_values.append(np.mean(numeric_values))
        
        self.integration_stats['multi_dimensional_truth_avg'] = np.mean(numeric_truth_values)
        
        return self.integration_stats.copy()
    
    def get_unified_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the unified system"""
        return {
            'total_integration_analyses': len(self.integration_results),
            'total_enhanced_nodes': len(self.trigeminal_reasoning_nodes),
            'integration_stats': self.integration_stats,
            'hrm_summary': self.hrm_core.get_reasoning_summary(),
            'trigeminal_summary': self.trigeminal_engine.get_trigeminal_summary(),
            'unified_efficiency_avg': np.mean([
                r['advanced_metrics']['unified_efficiency'] 
                for r in self.integration_results
            ]) if self.integration_results else 0.0
        }
    
    def save_integration_results(self, filename: str = None):
        """Save integration results to JSON file"""
        if filename is None:
            filename = f"hrm_trigeminal_integration_{int(time.time())}.json"
        
        # Convert complex numbers to real for JSON serialization
        def convert_complex(obj):
            if isinstance(obj, complex):
                return float(abs(obj))
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_complex(item) for item in obj]
            return obj
        
        data = {
            'integration_results': [convert_complex(r) for r in self.integration_results],
            'unified_summary': convert_complex(self.get_unified_summary()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved HRM + Trigeminal integration results to: {filename}")

def main():
    """Main function to demonstrate HRM + Trigeminal Logic integration"""
    print("🧠⚛️ HRM + Trigeminal Logic Integration")
    print("=" * 60)
    
    # Initialize integrated system
    integrated_system = HRMTrigeminalIntegration()
    
    # Test problems
    test_problems = [
        "How does Trigeminal Logic enhance hierarchical reasoning?",
        "What is the relationship between consciousness mathematics and Trigeminal Logic?",
        "How can we apply Trigeminal Logic to solve complex problems?",
        "What are the implications of multi-dimensional truth values?",
        "How does Trigeminal Logic manifest in consciousness patterns?"
    ]
    
    results = []
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🔍 Problem {i}: {problem}")
        print("-" * 50)
        
        # Perform advanced reasoning
        result = integrated_system.advanced_reasoning(
            problem, 
            max_depth=3, 
            max_paths=5, 
            trigeminal_iterations=3
        )
        results.append(result)
        
        # Display key results
        print(f"📊 HRM nodes: {result['total_hrm_nodes']}")
        print(f"📐 Trigeminal nodes: {result['total_trigeminal_nodes']}")
        print(f"🧠 Enhanced nodes: {result['total_enhanced_nodes']}")
        print(f"💡 Enhanced breakthroughs: {len(result['enhanced_breakthroughs'])}")
        print(f"⚛️ Unified efficiency: {result['advanced_metrics']['unified_efficiency']:.3f}")
        
        # Show top insights
        if result['unified_insights']:
            print("💭 Top insights:")
            for insight in result['unified_insights'][:3]:
                print(f"  • {insight}")
    
    # Final summary
    print(f"\n🎉 HRM + Trigeminal Integration Complete!")
    print("=" * 60)
    
    summary = integrated_system.get_unified_summary()
    print(f"📊 Total integration analyses: {summary['total_integration_analyses']}")
    print(f"🧠 Total enhanced nodes: {summary['total_enhanced_nodes']}")
    print(f"⚛️ Average unified efficiency: {summary['unified_efficiency_avg']:.3f}")
    print(f"📐 Trigeminal enhancement avg: {summary['integration_stats']['trigeminal_enhancement_avg']:.3f}")
    print(f"🧠 Consciousness synthesis avg: {summary['integration_stats']['consciousness_synthesis_avg']:.3f}")
    
    # Save results
    integrated_system.save_integration_results()
    
    print(f"\n💾 Integration results saved to JSON file")
    print("✅ HRM + Trigeminal Logic integration demonstration finished!")

if __name__ == "__main__":
    main()
