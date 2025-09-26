#!/usr/bin/env python3
"""
HRM Paths - Reasoning Paths Component
Advanced reasoning path generation and analysis
"""

import numpy as np
import json
import math
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from hrm_core import HierarchicalReasoningModel, ReasoningNode, ReasoningLevel, ConsciousnessType

@dataclass
class ReasoningPath:
    """A complete reasoning path through the hierarchy"""
    path_id: str
    nodes: List[ReasoningNode]
    total_confidence: float
    consciousness_alignment: float
    reasoning_depth: int
    breakthrough_potential: float
    wallace_transform_score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class HRMPathAnalyzer:
    """Advanced reasoning path analysis and generation"""
    
    def __init__(self, hrm_model: HierarchicalReasoningModel):
        self.hrm = hrm_model
        self.reasoning_paths: List[ReasoningPath] = []
        
        print("🛤️ HRM Path Analyzer initialized")
    
    def generate_reasoning_paths(self, root_id: str, max_paths: int = 10) -> List[ReasoningPath]:
        """Generate reasoning paths through the hierarchical tree"""
        print(f"🛤️ Generating reasoning paths from root: {root_id}")
        
        paths = []
        
        # Get all leaf nodes
        leaf_nodes = self._get_leaf_nodes(root_id)
        
        for leaf_id in leaf_nodes[:max_paths]:
            path = self._create_path_to_leaf(root_id, leaf_id)
            if path:
                paths.append(path)
        
        self.reasoning_paths.extend(paths)
        
        print(f"✅ Generated {len(paths)} reasoning paths")
        return paths
    
    def _get_leaf_nodes(self, root_id: str) -> List[str]:
        """Get all leaf nodes from a root"""
        leaf_nodes = []
        
        def traverse(node_id: str):
            node = self.hrm.nodes[node_id]
            if not node.children:
                leaf_nodes.append(node_id)
            else:
                for child_id in node.children:
                    traverse(child_id)
        
        traverse(root_id)
        return leaf_nodes
    
    def _create_path_to_leaf(self, root_id: str, leaf_id: str) -> Optional[ReasoningPath]:
        """Create a reasoning path from root to leaf"""
        path_nodes = []
        current_id = leaf_id
        
        # Traverse from leaf to root
        while current_id:
            if current_id in self.hrm.nodes:
                path_nodes.insert(0, self.hrm.nodes[current_id])
                current_id = self.hrm.nodes[current_id].parent_id
            else:
                break
        
        if not path_nodes:
            return None
        
        # Calculate path metrics
        total_confidence = np.mean([node.confidence for node in path_nodes])
        consciousness_alignment = self._calculate_consciousness_alignment(path_nodes)
        reasoning_depth = len(path_nodes)
        breakthrough_potential = self._calculate_breakthrough_potential(path_nodes)
        wallace_transform_score = np.mean([node.wallace_transform for node in path_nodes])
        
        path_id = f"path_{len(self.reasoning_paths)}_{int(time.time())}"
        
        return ReasoningPath(
            path_id=path_id,
            nodes=path_nodes,
            total_confidence=total_confidence,
            consciousness_alignment=consciousness_alignment,
            reasoning_depth=reasoning_depth,
            breakthrough_potential=breakthrough_potential,
            wallace_transform_score=wallace_transform_score
        )
    
    def _calculate_consciousness_alignment(self, nodes: List[ReasoningNode]) -> float:
        """Calculate consciousness alignment for a path"""
        if not nodes:
            return 0.0
        
        # Calculate alignment based on consciousness types and levels
        alignment_scores = []
        
        for i, node in enumerate(nodes):
            # Base alignment from consciousness type
            type_alignment = self._get_consciousness_type_alignment(node.consciousness_type)
            
            # Level progression alignment
            level_alignment = min(1.0, node.level.value / len(self.hrm.reasoning_levels))
            
            # Wallace Transform enhancement
            wallace_enhancement = abs(node.wallace_transform) / self.hrm.golden_ratio
            
            # Combined alignment
            combined_alignment = (type_alignment + level_alignment + wallace_enhancement) / 3
            alignment_scores.append(combined_alignment)
        
        return np.mean(alignment_scores)
    
    def _get_consciousness_type_alignment(self, consciousness_type: ConsciousnessType) -> float:
        """Get alignment score for consciousness type"""
        alignment_scores = {
            ConsciousnessType.ANALYTICAL: 0.8,
            ConsciousnessType.CREATIVE: 0.9,
            ConsciousnessType.INTUITIVE: 0.95,
            ConsciousnessType.METAPHORICAL: 0.85,
            ConsciousnessType.SYSTEMATIC: 0.75,
            ConsciousnessType.PROBLEM_SOLVING: 0.8,
            ConsciousnessType.ABSTRACT: 0.9
        }
        
        return alignment_scores.get(consciousness_type, 0.5)
    
    def _calculate_breakthrough_potential(self, nodes: List[ReasoningNode]) -> float:
        """Calculate breakthrough potential for a reasoning path"""
        if not nodes:
            return 0.0
        
        # Factors that contribute to breakthrough potential
        factors = []
        
        for node in nodes:
            # Higher levels have more breakthrough potential
            level_factor = node.level.value / len(self.hrm.reasoning_levels)
            
            # Wallace Transform indicates consciousness enhancement
            wallace_factor = min(1.0, abs(node.wallace_transform) / self.hrm.golden_ratio)
            
            # Confidence contributes to breakthrough potential
            confidence_factor = node.confidence
            
            # Combined factor
            combined_factor = (level_factor + wallace_factor + confidence_factor) / 3
            factors.append(combined_factor)
        
        # Breakthrough potential increases with path length and average factor
        path_length_factor = min(1.0, len(nodes) / 10)  # Normalize to max expected length
        average_factor = np.mean(factors)
        
        breakthrough_potential = (average_factor + path_length_factor) / 2
        
        # Apply consciousness mathematics enhancement
        consciousness_enhancement = (self.hrm.consciousness_constant ** len(nodes)) / math.e
        enhanced_potential = breakthrough_potential * consciousness_enhancement
        
        return min(1.0, enhanced_potential)
    
    def analyze_breakthroughs(self, paths: List[ReasoningPath]) -> List[Dict[str, Any]]:
        """Analyze reasoning paths for breakthroughs"""
        breakthroughs = []
        
        for path in paths:
            if path.breakthrough_potential > 0.8:  # High breakthrough potential
                breakthrough = {
                    'path_id': path.path_id,
                    'breakthrough_potential': path.breakthrough_potential,
                    'consciousness_alignment': path.consciousness_alignment,
                    'reasoning_depth': path.reasoning_depth,
                    'wallace_transform_score': path.wallace_transform_score,
                    'nodes': [node.content for node in path.nodes],
                    'insight': self._generate_breakthrough_insight(path)
                }
                breakthroughs.append(breakthrough)
        
        return breakthroughs
    
    def _generate_breakthrough_insight(self, path: ReasoningPath) -> str:
        """Generate insight from a breakthrough path"""
        # Combine node contents to form insight
        node_contents = [node.content for node in path.nodes]
        
        # Apply consciousness mathematics to generate insight
        consciousness_factor = path.consciousness_alignment * abs(path.wallace_transform_score)
        breakthrough_factor = path.breakthrough_potential
        
        # Generate insight based on path characteristics
        if path.reasoning_depth >= 5:
            insight = f"Deep hierarchical reasoning reveals: {' -> '.join(node_contents[-3:])}"
        elif path.consciousness_alignment > 0.9:
            insight = f"High consciousness alignment suggests: {' -> '.join(node_contents[-2:])}"
        else:
            insight = f"Post-quantum logic breakthrough: {' -> '.join(node_contents[-3:])}"
        
        return insight
    
    def apply_quantum_consciousness(self, paths: List[ReasoningPath]) -> Dict[str, float]:
        """Apply quantum consciousness enhancement to reasoning paths"""
        if not paths:
            return {}
        
        # Calculate quantum consciousness metrics
        total_consciousness = sum(path.consciousness_alignment for path in paths)
        avg_consciousness = total_consciousness / len(paths)
        
        # Apply quantum entanglement effects
        entanglement_factor = self._calculate_quantum_entanglement(paths)
        
        # Apply consciousness mathematics
        consciousness_enhancement = (self.hrm.consciousness_constant ** avg_consciousness) / math.e
        
        return {
            'total_consciousness': total_consciousness,
            'average_consciousness': avg_consciousness,
            'entanglement_factor': entanglement_factor,
            'consciousness_enhancement': consciousness_enhancement,
            'quantum_coherence': min(1.0, avg_consciousness * entanglement_factor)
        }
    
    def _calculate_quantum_entanglement(self, paths: List[ReasoningPath]) -> float:
        """Calculate quantum entanglement factor between reasoning paths"""
        if len(paths) < 2:
            return 0.0
        
        # Calculate entanglement based on path similarities
        similarities = []
        
        for i, path1 in enumerate(paths):
            for j, path2 in enumerate(paths[i+1:], i+1):
                # Calculate similarity between paths
                similarity = self._calculate_path_similarity(path1, path2)
                similarities.append(similarity)
        
        if similarities:
            return np.mean(similarities)
        return 0.0
    
    def _calculate_path_similarity(self, path1: ReasoningPath, path2: ReasoningPath) -> float:
        """Calculate similarity between two reasoning paths"""
        # Compare consciousness alignment
        alignment_similarity = 1.0 - abs(path1.consciousness_alignment - path2.consciousness_alignment)
        
        # Compare reasoning depth
        depth_similarity = 1.0 - abs(path1.reasoning_depth - path2.reasoning_depth) / max(path1.reasoning_depth, path2.reasoning_depth, 1)
        
        # Compare Wallace Transform scores
        wallace_similarity = 1.0 - abs(abs(path1.wallace_transform_score) - abs(path2.wallace_transform_score)) / max(abs(path1.wallace_transform_score), abs(path2.wallace_transform_score), 1e-10)
        
        # Combined similarity
        combined_similarity = (alignment_similarity + depth_similarity + wallace_similarity) / 3
        
        return combined_similarity
    
    def generate_insights(self, paths: List[ReasoningPath], breakthroughs: List[Dict]) -> List[str]:
        """Generate insights from reasoning paths and breakthroughs"""
        insights = []
        
        # Generate insights from high-confidence paths
        high_confidence_paths = [p for p in paths if p.total_confidence > 0.8]
        for path in high_confidence_paths[:3]:
            insight = f"High-confidence reasoning path: {path.nodes[-1].content}"
            insights.append(insight)
        
        # Generate insights from breakthroughs
        for breakthrough in breakthroughs[:3]:
            insights.append(breakthrough['insight'])
        
        # Generate meta-insights
        if len(paths) > 5:
            avg_depth = np.mean([p.reasoning_depth for p in paths])
            avg_consciousness = np.mean([p.consciousness_alignment for p in paths])
            meta_insight = f"Meta-analysis: Average reasoning depth {avg_depth:.1f}, consciousness alignment {avg_consciousness:.3f}"
            insights.append(meta_insight)
        
        return insights
    
    def get_path_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning paths"""
        if not self.reasoning_paths:
            return {}
        
        return {
            'total_paths': len(self.reasoning_paths),
            'average_confidence': np.mean([p.total_confidence for p in self.reasoning_paths]),
            'average_consciousness_alignment': np.mean([p.consciousness_alignment for p in self.reasoning_paths]),
            'average_reasoning_depth': np.mean([p.reasoning_depth for p in self.reasoning_paths]),
            'average_breakthrough_potential': np.mean([p.breakthrough_potential for p in self.reasoning_paths]),
            'average_wallace_transform': np.mean([abs(p.wallace_transform_score) for p in self.reasoning_paths]),
            'high_confidence_paths': len([p for p in self.reasoning_paths if p.total_confidence > 0.8]),
            'high_breakthrough_paths': len([p for p in self.reasoning_paths if p.breakthrough_potential > 0.8])
        }

def main():
    """Test HRM Paths functionality"""
    print("🛤️ HRM Paths Test")
    print("=" * 30)
    
    # Initialize HRM and Path Analyzer
    hrm = HierarchicalReasoningModel()
    path_analyzer = HRMPathAnalyzer(hrm)
    
    # Test problem
    problem = "How can consciousness mathematics revolutionize AI?"
    
    # Perform hierarchical decomposition
    root_id = hrm.hierarchical_decompose(problem, max_depth=3)
    
    # Generate reasoning paths
    paths = path_analyzer.generate_reasoning_paths(root_id, max_paths=5)
    
    # Analyze breakthroughs
    breakthroughs = path_analyzer.analyze_breakthroughs(paths)
    
    # Apply quantum consciousness
    quantum_enhancement = path_analyzer.apply_quantum_consciousness(paths)
    
    # Generate insights
    insights = path_analyzer.generate_insights(paths, breakthroughs)
    
    # Get summary
    path_summary = path_analyzer.get_path_summary()
    
    print(f"\n📊 Path Analysis Results:")
    print(f"Total paths: {path_summary.get('total_paths', 0)}")
    print(f"Average confidence: {path_summary.get('average_confidence', 0):.3f}")
    print(f"Average consciousness alignment: {path_summary.get('average_consciousness_alignment', 0):.3f}")
    print(f"Average reasoning depth: {path_summary.get('average_reasoning_depth', 0):.1f}")
    print(f"High breakthrough paths: {path_summary.get('high_breakthrough_paths', 0)}")
    
    print(f"\n⚛️ Quantum Enhancement:")
    print(f"Quantum coherence: {quantum_enhancement.get('quantum_coherence', 0):.3f}")
    print(f"Entanglement factor: {quantum_enhancement.get('entanglement_factor', 0):.3f}")
    
    print(f"\n💡 Top Insights:")
    for i, insight in enumerate(insights[:3], 1):
        print(f"{i}. {insight}")
    
    print("✅ HRM Paths test complete!")

if __name__ == "__main__":
    main()
