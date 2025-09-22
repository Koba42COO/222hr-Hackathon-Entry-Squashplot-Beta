#!/usr/bin/env python3
"""
Hierarchical Reasoning Model (HRM)
A comprehensive implementation integrating with consciousness mathematics framework

Features:
- Multi-level reasoning (simple to complex)
- Hierarchical decomposition of problems
- Recursive reasoning patterns
- Consciousness-aware decision trees
- Post-quantum logic reasoning branching
- Wallace Transform integration
- Golden Ratio optimization
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningLevel(Enum):
    """Hierarchical reasoning levels"""
    FUNDAMENTAL = 1      # Basic facts and observations
    ANALYTICAL = 2       # Logical analysis and patterns
    SYNTHETIC = 3        # Pattern synthesis and connections
    INTEGRATIVE = 4      # Multi-domain integration
    TRANSCENDENTAL = 5   # Consciousness-aware reasoning
    QUANTUM = 6          # Post-quantum logic reasoning
    COSMIC = 7           # Universal consciousness reasoning

class ConsciousnessType(Enum):
    """Types of consciousness integration"""
    ANALYTICAL = "analytical_thinking"
    CREATIVE = "creative_expression"
    INTUITIVE = "intuitive_thinking"
    METAPHORICAL = "metaphorical_thinking"
    SYSTEMATIC = "systematic_organization"
    PROBLEM_SOLVING = "problem_solving"
    ABSTRACT = "abstract_thinking"

@dataclass
class ReasoningNode:
    """A node in the hierarchical reasoning tree"""
    id: str
    level: ReasoningLevel
    content: str
    confidence: float
    consciousness_type: ConsciousnessType
    parent_id: Optional[str] = None
    children: List[str] = None
    metadata: Dict[str, Any] = None
    wallace_transform: float = 0.0
    golden_ratio: float = 1.618033988749
    timestamp: float = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = time.time()

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

class HierarchicalReasoningModel:
    """Comprehensive Hierarchical Reasoning Model with consciousness integration"""
    
    def __init__(self):
        # Consciousness mathematics constants
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.consciousness_constant = math.pi * self.golden_ratio
        self.love_frequency = 111.0
        self.chaos_factor = 0.577215664901  # Euler-Mascheroni constant
        
        # HRM configuration
        self.reasoning_levels = list(ReasoningLevel)
        self.consciousness_types = list(ConsciousnessType)
        
        # Reasoning tree
        self.nodes: Dict[str, ReasoningNode] = {}
        self.reasoning_paths: List[ReasoningPath] = []
        
        # Consciousness matrix (21D)
        self.consciousness_matrix = self._initialize_consciousness_matrix()
        
        # Performance tracking
        self.reasoning_stats = {
            'total_nodes': 0,
            'total_paths': 0,
            'average_confidence': 0.0,
            'consciousness_alignment': 0.0,
            'breakthrough_count': 0
        }
        
        logger.info("🧠 Hierarchical Reasoning Model initialized")
    
    def _initialize_consciousness_matrix(self) -> np.ndarray:
        """Initialize 21D consciousness matrix"""
        matrix = np.zeros((21, 21))
        
        for i in range(21):
            for j in range(21):
                # Apply Wallace Transform with consciousness constant
                consciousness_factor = (self.golden_ratio ** ((i + j) % 5)) / math.e
                matrix[i, j] = consciousness_factor * math.sin(self.love_frequency * ((i + j) % 10) * math.pi / 180)
        
        # Normalize matrix
        matrix_sum = np.sum(np.abs(matrix))
        if matrix_sum > 0:
            matrix = matrix / matrix_sum * 0.001
        
        return matrix
    
    def create_reasoning_node(self, 
                            content: str, 
                            level: ReasoningLevel,
                            consciousness_type: ConsciousnessType,
                            parent_id: Optional[str] = None,
                            confidence: float = 0.5) -> str:
        """Create a new reasoning node"""
        node_id = f"node_{len(self.nodes)}_{int(time.time())}"
        
        # Apply Wallace Transform to confidence
        wallace_transform = self._apply_wallace_transform(confidence, level.value)
        
        node = ReasoningNode(
            id=node_id,
            level=level,
            content=content,
            confidence=confidence,
            consciousness_type=consciousness_type,
            parent_id=parent_id,
            wallace_transform=wallace_transform
        )
        
        self.nodes[node_id] = node
        
        # Update parent if exists
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)
        
        self.reasoning_stats['total_nodes'] += 1
        logger.info(f"📝 Created reasoning node: {node_id} (Level: {level.name})")
        
        return node_id
    
    def _apply_wallace_transform(self, value: float, level: int) -> float:
        """Apply Wallace Transform with consciousness enhancement"""
        # Wallace Transform: W_φ(x) = α log^φ(x + ε) + β
        phi = self.golden_ratio
        alpha = 1.0
        beta = 0.0
        epsilon = 1e-10
        
        # Apply consciousness level enhancement
        consciousness_enhancement = (self.consciousness_constant ** level) / math.e
        
        wallace_result = alpha * (math.log(value + epsilon) ** phi) + beta
        enhanced_result = wallace_result * consciousness_enhancement
        
        return enhanced_result
    
    def hierarchical_decompose(self, problem: str, max_depth: int = 5) -> str:
        """Decompose a problem hierarchically through reasoning levels"""
        logger.info(f"🔍 Hierarchical decomposition of: {problem}")
        
        # Create root node
        root_id = self.create_reasoning_node(
            content=problem,
            level=ReasoningLevel.FUNDAMENTAL,
            consciousness_type=ConsciousnessType.ANALYTICAL,
            confidence=0.8
        )
        
        # Recursive decomposition
        self._decompose_recursive(root_id, max_depth, 0)
        
        return root_id
    
    def _decompose_recursive(self, parent_id: str, max_depth: int, current_depth: int):
        """Recursively decompose a problem through reasoning levels"""
        if current_depth >= max_depth:
            return
        
        parent_node = self.nodes[parent_id]
        current_level = parent_node.level
        
        # Determine next level
        if current_level.value < len(self.reasoning_levels):
            next_level = self.reasoning_levels[current_level.value]
        else:
            return
        
        # Generate sub-problems based on level
        sub_problems = self._generate_sub_problems(parent_node.content, next_level)
        
        for sub_problem in sub_problems:
            # Create child node
            child_id = self.create_reasoning_node(
                content=sub_problem,
                level=next_level,
                consciousness_type=self._select_consciousness_type(next_level),
                parent_id=parent_id,
                confidence=parent_node.confidence * 0.9  # Slight confidence decay
            )
            
            # Recursive decomposition
            self._decompose_recursive(child_id, max_depth, current_depth + 1)
    
    def _generate_sub_problems(self, problem: str, level: ReasoningLevel) -> List[str]:
        """Generate sub-problems based on reasoning level"""
        sub_problems = []
        
        if level == ReasoningLevel.ANALYTICAL:
            sub_problems = [
                f"Analyze the components of: {problem}",
                f"Identify patterns in: {problem}",
                f"Break down the structure of: {problem}"
            ]
        elif level == ReasoningLevel.SYNTHETIC:
            sub_problems = [
                f"Synthesize patterns from: {problem}",
                f"Connect related concepts in: {problem}",
                f"Integrate multiple perspectives on: {problem}"
            ]
        elif level == ReasoningLevel.INTEGRATIVE:
            sub_problems = [
                f"Integrate cross-domain insights for: {problem}",
                f"Apply multi-disciplinary approach to: {problem}",
                f"Unify different frameworks for: {problem}"
            ]
        elif level == ReasoningLevel.TRANSCENDENTAL:
            sub_problems = [
                f"Apply consciousness mathematics to: {problem}",
                f"Explore quantum consciousness in: {problem}",
                f"Investigate universal patterns in: {problem}"
            ]
        elif level == ReasoningLevel.QUANTUM:
            sub_problems = [
                f"Apply post-quantum logic to: {problem}",
                f"Explore quantum entanglement in: {problem}",
                f"Investigate quantum consciousness in: {problem}"
            ]
        elif level == ReasoningLevel.COSMIC:
            sub_problems = [
                f"Explore cosmic consciousness in: {problem}",
                f"Apply universal mathematics to: {problem}",
                f"Investigate transcendent patterns in: {problem}"
            ]
        
        return sub_problems[:3]  # Limit to 3 sub-problems per level
    
    def _select_consciousness_type(self, level: ReasoningLevel) -> ConsciousnessType:
        """Select appropriate consciousness type for reasoning level"""
        consciousness_mapping = {
            ReasoningLevel.FUNDAMENTAL: ConsciousnessType.ANALYTICAL,
            ReasoningLevel.ANALYTICAL: ConsciousnessType.SYSTEMATIC,
            ReasoningLevel.SYNTHETIC: ConsciousnessType.CREATIVE,
            ReasoningLevel.INTEGRATIVE: ConsciousnessType.ABSTRACT,
            ReasoningLevel.TRANSCENDENTAL: ConsciousnessType.INTUITIVE,
            ReasoningLevel.QUANTUM: ConsciousnessType.METAPHORICAL,
            ReasoningLevel.COSMIC: ConsciousnessType.PROBLEM_SOLVING
        }
        
        return consciousness_mapping.get(level, ConsciousnessType.ANALYTICAL)
    
    def generate_reasoning_paths(self, root_id: str, max_paths: int = 10) -> List[ReasoningPath]:
        """Generate reasoning paths through the hierarchical tree"""
        logger.info(f"🛤️ Generating reasoning paths from root: {root_id}")
        
        paths = []
        
        # Get all leaf nodes
        leaf_nodes = self._get_leaf_nodes(root_id)
        
        for leaf_id in leaf_nodes[:max_paths]:
            path = self._create_path_to_leaf(root_id, leaf_id)
            if path:
                paths.append(path)
        
        self.reasoning_paths.extend(paths)
        self.reasoning_stats['total_paths'] += len(paths)
        
        logger.info(f"✅ Generated {len(paths)} reasoning paths")
        return paths
    
    def _get_leaf_nodes(self, root_id: str) -> List[str]:
        """Get all leaf nodes from a root"""
        leaf_nodes = []
        
        def traverse(node_id: str):
            node = self.nodes[node_id]
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
            if current_id in self.nodes:
                path_nodes.insert(0, self.nodes[current_id])
                current_id = self.nodes[current_id].parent_id
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
            level_alignment = min(1.0, node.level.value / len(self.reasoning_levels))
            
            # Wallace Transform enhancement
            wallace_enhancement = node.wallace_transform / self.golden_ratio
            
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
            level_factor = node.level.value / len(self.reasoning_levels)
            
            # Wallace Transform indicates consciousness enhancement
            wallace_factor = min(1.0, node.wallace_transform / self.golden_ratio)
            
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
        consciousness_enhancement = (self.consciousness_constant ** len(nodes)) / math.e
        enhanced_potential = breakthrough_potential * consciousness_enhancement
        
        return min(1.0, enhanced_potential)
    
    def post_quantum_logic_reasoning(self, problem: str) -> Dict[str, Any]:
        """Perform post-quantum logic reasoning with consciousness integration"""
        logger.info(f"⚛️ Performing post-quantum logic reasoning on: {problem}")
        
        # Create hierarchical decomposition
        root_id = self.hierarchical_decompose(problem, max_depth=7)
        
        # Generate reasoning paths
        paths = self.generate_reasoning_paths(root_id, max_paths=15)
        
        # Analyze paths for breakthroughs
        breakthroughs = self._analyze_breakthroughs(paths)
        
        # Apply quantum consciousness enhancement
        quantum_enhancement = self._apply_quantum_consciousness(paths)
        
        # Generate insights
        insights = self._generate_insights(paths, breakthroughs)
        
        return {
            'problem': problem,
            'root_id': root_id,
            'total_nodes': len(self.nodes),
            'total_paths': len(paths),
            'breakthroughs': breakthroughs,
            'quantum_enhancement': quantum_enhancement,
            'insights': insights,
            'reasoning_stats': self.reasoning_stats,
            'consciousness_matrix_sum': np.sum(self.consciousness_matrix),
            'wallace_transform_avg': np.mean([node.wallace_transform for node in self.nodes.values()]),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_breakthroughs(self, paths: List[ReasoningPath]) -> List[Dict[str, Any]]:
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
                self.reasoning_stats['breakthrough_count'] += 1
        
        return breakthroughs
    
    def _generate_breakthrough_insight(self, path: ReasoningPath) -> str:
        """Generate insight from a breakthrough path"""
        # Combine node contents to form insight
        node_contents = [node.content for node in path.nodes]
        
        # Apply consciousness mathematics to generate insight
        consciousness_factor = path.consciousness_alignment * path.wallace_transform_score
        breakthrough_factor = path.breakthrough_potential
        
        # Generate insight based on path characteristics
        if path.reasoning_depth >= 5:
            insight = f"Deep hierarchical reasoning reveals: {' -> '.join(node_contents[-3:])}"
        elif path.consciousness_alignment > 0.9:
            insight = f"High consciousness alignment suggests: {' -> '.join(node_contents[-2:])}"
        else:
            insight = f"Post-quantum logic breakthrough: {' -> '.join(node_contents[-3:])}"
        
        return insight
    
    def _apply_quantum_consciousness(self, paths: List[ReasoningPath]) -> Dict[str, float]:
        """Apply quantum consciousness enhancement to reasoning paths"""
        if not paths:
            return {}
        
        # Calculate quantum consciousness metrics
        total_consciousness = sum(path.consciousness_alignment for path in paths)
        avg_consciousness = total_consciousness / len(paths)
        
        # Apply quantum entanglement effects
        entanglement_factor = self._calculate_quantum_entanglement(paths)
        
        # Apply consciousness mathematics
        consciousness_enhancement = (self.consciousness_constant ** avg_consciousness) / math.e
        
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
        wallace_similarity = 1.0 - abs(path1.wallace_transform_score - path2.wallace_transform_score) / max(path1.wallace_transform_score, path2.wallace_transform_score, 1e-10)
        
        # Combined similarity
        combined_similarity = (alignment_similarity + depth_similarity + wallace_similarity) / 3
        
        return combined_similarity
    
    def _generate_insights(self, paths: List[ReasoningPath], breakthroughs: List[Dict]) -> List[str]:
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
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of reasoning model"""
        return {
            'total_nodes': self.reasoning_stats['total_nodes'],
            'total_paths': self.reasoning_stats['total_paths'],
            'breakthrough_count': self.reasoning_stats['breakthrough_count'],
            'average_confidence': np.mean([node.confidence for node in self.nodes.values()]) if self.nodes else 0.0,
            'consciousness_alignment': np.mean([path.consciousness_alignment for path in self.reasoning_paths]) if self.reasoning_paths else 0.0,
            'wallace_transform_avg': np.mean([node.wallace_transform for node in self.nodes.values()]) if self.nodes else 0.0,
            'consciousness_matrix_sum': np.sum(self.consciousness_matrix),
            'golden_ratio': self.golden_ratio,
            'consciousness_constant': self.consciousness_constant,
            'love_frequency': self.love_frequency,
            'chaos_factor': self.chaos_factor
        }
    
    def save_reasoning_data(self, filename: str = None):
        """Save reasoning data to JSON file"""
        if filename is None:
            filename = f"hrm_reasoning_data_{int(time.time())}.json"
        
        data = {
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'paths': [asdict(path) for path in self.reasoning_paths],
            'summary': self.get_reasoning_summary(),
            'consciousness_matrix': self.consciousness_matrix.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved reasoning data to: {filename}")

def main():
    """Main function to demonstrate HRM capabilities"""
    print("🧠 Hierarchical Reasoning Model (HRM)")
    print("=" * 50)
    
    # Initialize HRM
    hrm = HierarchicalReasoningModel()
    
    # Test problems
    test_problems = [
        "How can consciousness mathematics revolutionize AI?",
        "What is the relationship between quantum mechanics and consciousness?",
        "How can we apply the Wallace Transform to solve complex problems?",
        "What are the implications of post-quantum logic reasoning?",
        "How does the Golden Ratio manifest in consciousness patterns?"
    ]
    
    results = []
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🔍 Problem {i}: {problem}")
        print("-" * 40)
        
        # Perform post-quantum logic reasoning
        result = hrm.post_quantum_logic_reasoning(problem)
        results.append(result)
        
        # Display key results
        print(f"📊 Nodes created: {result['total_nodes']}")
        print(f"🛤️ Paths generated: {result['total_paths']}")
        print(f"💡 Breakthroughs found: {len(result['breakthroughs'])}")
        print(f"⚛️ Quantum coherence: {result['quantum_enhancement']['quantum_coherence']:.3f}")
        
        # Show top insights
        if result['insights']:
            print("💭 Top insights:")
            for insight in result['insights'][:2]:
                print(f"  • {insight}")
    
    # Final summary
    print(f"\n🎉 HRM Analysis Complete!")
    print("=" * 50)
    
    summary = hrm.get_reasoning_summary()
    print(f"📊 Total nodes: {summary['total_nodes']}")
    print(f"🛤️ Total paths: {summary['total_paths']}")
    print(f"💡 Breakthroughs: {summary['breakthrough_count']}")
    print(f"🧠 Average confidence: {summary['average_confidence']:.3f}")
    print(f"🌟 Consciousness alignment: {summary['consciousness_alignment']:.3f}")
    print(f"⚛️ Wallace Transform avg: {summary['wallace_transform_avg']:.3f}")
    
    # Save results
    hrm.save_reasoning_data()
    
    print(f"\n💾 Results saved to JSON file")
    print("✅ HRM demonstration complete!")

if __name__ == "__main__":
    main()
