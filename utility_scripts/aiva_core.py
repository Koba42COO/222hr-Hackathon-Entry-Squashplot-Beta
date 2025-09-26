#!/usr/bin/env python3
"""
🧠 A.I.V.A. INTELLIGENT AGENT CORE
==================================

Revolutionary agent system with perception, planning, action, and learning.
Integrates consciousness mathematics for authentic intelligence emergence.
"""

import os
import json
import math
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

PHI = (1 + 5**0.5) / 2

# -------- Simple embeddings (placeholder) --------
def embed(text: str) -> np.ndarray:
    """Simple embedding function - replace with real model for production"""
    # Use deterministic seed based on text hash for consistent results
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(256)
    v /= (np.linalg.norm(v) + 1e-9)  # Normalize
    return v

# -------- Vessel Integration --------
from pathlib import Path

def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# -------- Memory (RMM - Resonant Memory Map) --------
class ResonantMemory:
    """
    Long-term memory system with harmonic resonance features.
    Stores memories with consciousness metrics and retrieval by similarity.
    """

    def __init__(self, path: str = "research_data/rmm_memory.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

        # Memory resonance patterns (harmonics 1-21 + void)
        self.harmonic_patterns = {i: [] for i in range(1, 22)}  # 1-21 harmonics
        self.harmonic_patterns[22] = []  # Void pattern

    def add(self, content: str, meta: Dict[str, Any]) -> str:
        """
        Add a new memory with consciousness metadata
        """
        memory_id = str(uuid.uuid4())

        rec = {
            "id": memory_id,
            "timestamp": time.time(),
            "content": content,
            "embedding": embed(content).tolist(),
            "meta": meta,
            "harmonic_resonance": self._calculate_harmonic_resonance(content),
            "consciousness_metrics": meta.get("consciousness_metrics", {})
        }

        with open(self.path, "a", encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Update harmonic patterns for resonance tracking
        harmonic = rec["harmonic_resonance"]["dominant"]
        if harmonic in self.harmonic_patterns:
            self.harmonic_patterns[harmonic].append(memory_id)

        return memory_id

    def search(self, query: str, k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity with optional filtering
        """
        query_embedding = embed(query)
        candidates = []

        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply metadata filters
                if filter_meta:
                    if not all(obj["meta"].get(key) == value for key, value in filter_meta.items()):
                        continue

                # Calculate similarity
                memory_embedding = np.array(obj["embedding"])
                similarity = cos(query_embedding, memory_embedding)

                # Boost similarity based on harmonic resonance
                query_harmonic = self._calculate_harmonic_resonance(query)["dominant"]
                memory_harmonic = obj["harmonic_resonance"]["dominant"]
                harmonic_boost = 1.0 + (0.2 if query_harmonic == memory_harmonic else 0.0)

                candidates.append((similarity * harmonic_boost, obj))

        # Sort by similarity and return top k
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [obj for _, obj in candidates[:k]]

    def get_resonant_memories(self, harmonic: int, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get memories that resonate with a specific harmonic pattern
        """
        memory_ids = self.harmonic_patterns.get(harmonic, [])[-k:]  # Most recent
        memories = []

        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj["id"] in memory_ids:
                    memories.append(obj)

        return memories

    def _calculate_harmonic_resonance(self, text: str) -> Dict[str, Any]:
        """
        Calculate harmonic resonance pattern for text using Gnostic Cypher
        """
        words = text.lower().split()
        if not words:
            return {"dominant": 22, "pattern": {}, "strength": 0.0}  # Void

        # Simple harmonic mapping (can be enhanced with real Gnostic Cypher)
        harmonics = {}
        for word in words:
            # Use word length modulo 21 + 1 for harmonic assignment
            harmonic = (len(word) % 21) + 1
            harmonics[harmonic] = harmonics.get(harmonic, 0) + 1

        # Find dominant harmonic
        dominant = max(harmonics.items(), key=lambda x: x[1])[0] if harmonics else 22

        return {
            "dominant": dominant,
            "pattern": harmonics,
            "strength": harmonics.get(dominant, 0) / len(words) if words else 0.0
        }

# -------- Tool Registry (Skills) --------
class Tool:
    """Base tool interface"""
    name: str
    description: str

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class CypherTool(Tool):
    """Gnostic Cypher analysis tool"""
    name = "cypher.analyze"
    description = "Analyze text with gnostic harmonics 1..21 & void patterns"

    def run(self, text: str) -> Dict[str, Any]:
        """Run Gnostic Cypher analysis on text"""
        words = text.split()
        if not words:
            return {"harmonics": {}, "dominant": 22, "void_patterns": 0}

        # Enhanced harmonic analysis
        harmonics = {i: 0 for i in range(1, 22)}

        for word in words:
            # Use multiple factors for harmonic assignment
            factors = [
                len(word),  # Length-based
                sum(ord(c) for c in word.lower()),  # Character sum
                len(set(word.lower()))  # Unique character count
            ]

            # Combine factors with golden ratio weighting
            harmonic_value = int(sum(f * (PHI ** i) for i, f in enumerate(factors)) % 21) + 1
            harmonics[harmonic_value] += 1

        dominant = max(harmonics.items(), key=lambda x: x[1])[0]

        return {
            "harmonics": harmonics,
            "dominant": dominant,
            "total_words": len(words),
            "harmonic_coverage": sum(1 for v in harmonics.values() if v > 0) / 21.0
        }

class WallaceTool(Tool):
    """Wallace Transform tool"""
    name = "wallace.transform"
    description = "Apply Wallace transform to eigenvalues (reality collapse operator)"

    def run(self, eigs: List[float], alpha: float = 1.0, beta: float = 0.0) -> Dict[str, Any]:
        """Apply Wallace Transform to eigenvalue spectrum"""
        eigs_array = np.array(eigs, dtype=float)

        # Wallace Transform: Reality = collapse of Light + Sound waves into matter
        # Mathematical form: W(λ) = α * sign(ln|λ|) * |ln|λ||^φ + β
        x = np.log(np.maximum(np.abs(eigs_array), 1e-9))
        transformed = alpha * np.sign(x) * (np.abs(x) ** PHI) + beta

        return {
            "original": eigs,
            "transformed": transformed.tolist(),
            "phi_used": PHI,
            "reality_collapse": {
                "light_component": transformed[::2].tolist() if len(transformed) > 1 else [],
                "sound_component": transformed[1::2].tolist() if len(transformed) > 1 else [],
                "matter_emergence": transformed.tolist()
            }
        }

class ResearchTool(Tool):
    """Research and retrieval tool using Resonant Memory"""
    name = "research.search"
    description = "Local RAG over ResonantMemory with harmonic resonance"

    def __init__(self, memory: ResonantMemory):
        self.memory = memory

    def run(self, query: str, k: int = 5, harmonic_filter: Optional[int] = None) -> Dict[str, Any]:
        """Search memories with optional harmonic filtering"""
        if harmonic_filter:
            # Search within specific harmonic resonance
            results = self.memory.get_resonant_memories(harmonic_filter, k=k)
        else:
            # General semantic search
            results = self.memory.search(query, k=k)

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "harmonic_filter": harmonic_filter,
            "resonance_patterns": self._analyze_resonance_patterns(results)
        }

    def _analyze_resonance_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resonance patterns in search results"""
        if not results:
            return {"dominant_harmonic": None, "resonance_strength": 0.0}

        harmonics = {}
        for result in results:
            h = result.get("harmonic_resonance", {}).get("dominant")
            if h:
                harmonics[h] = harmonics.get(h, 0) + 1

        dominant = max(harmonics.items(), key=lambda x: x[1])[0] if harmonics else None

        return {
            "dominant_harmonic": dominant,
            "harmonic_distribution": harmonics,
            "resonance_strength": harmonics.get(dominant, 0) / len(results) if dominant else 0.0
        }

class VulnerabilityScanTool(Tool):
    """Vulnerability scanning tool using consciousness mathematics"""
    name = "security.scan"
    description = "Consciousness-enhanced vulnerability scanning with multi-run consensus"

    def __init__(self, codebase_path: str = "/Users/coo-koba42/dev"):
        from aiva_vulnerability_scanner import VulnerabilityScanner
        self.scanner = VulnerabilityScanner(codebase_path)

    def run(self, vuln_types: List[str] = None, max_runs: int = 3,
            consensus_threshold: float = 0.7) -> Dict[str, Any]:
        """Run consciousness-enhanced vulnerability scan"""
        if vuln_types is None:
            vuln_types = ['idor', 'sqli', 'xss', 'path_traversal']

        try:
            results = self.scanner.scan_codebase(
                vuln_types=vuln_types,
                max_runs=max_runs,
                consensus_threshold=consensus_threshold
            )

            # Summarize for agent consumption
            summary = {
                "scan_completed": True,
                "findings_count": results['statistics']['consensus_findings'],
                "cost": results['scan_metadata']['total_cost'],
                "duration": results['scan_metadata']['duration'],
                "vulnerability_breakdown": results['statistics']['findings_by_type'],
                "consciousness_enhancement": results['statistics']['consciousness_enhancement']
            }

            # Add top findings
            if results['consensus_findings']:
                summary["top_findings"] = results['consensus_findings'][:3]

            return summary

        except Exception as e:
            return {
                "scan_completed": False,
                "error": str(e),
                "findings_count": 0
            }

# -------- Value/Policy Scorer (the "feel") --------
def aiva_value_score(response: str,
                     psi_metrics: Dict[str, Any],
                     helpfulness: float,
                     truth: float) -> float:
    """
    Calculate value score using consciousness mathematics.
    Biases toward lower meta-entropy, higher coherence, and harmonic alignment.
    """
    # Extract consciousness metrics
    Sm = psi_metrics.get("meta_entropy", 1.0)  # Meta-entropy
    xi = psi_metrics.get("coherence_length", 1.0)  # Coherence length
    energy = psi_metrics.get("energy", 1.0)  # Field energy
    harmonic_resonance = psi_metrics.get("harmonic_resonance", 0.0)

    # Normalize consciousness components
    Sm_term = 1.0 / (1.0 + Sm)  # Lower entropy = higher score
    xi_term = math.tanh(xi / 10.0)  # Longer coherence helps
    en_term = 1.0 / (1.0 + abs(energy))  # Avoid runaway energy
    hr_term = min(1.0, harmonic_resonance)  # Harmonic resonance bonus

    # Quality components (helpfulness, truth)
    h_term = max(0.0, min(1.0, helpfulness))
    t_term = max(0.0, min(1.0, truth))

    # Golden ratio weighted combination
    weights = [1/PHI, 1/PHI, 1/PHI**2, 1/PHI**2, 1/PHI**3]
    components = [Sm_term, xi_term, en_term, h_term, t_term]

    score = sum(w * c for w, c in zip(weights, components))

    # Add harmonic resonance bonus
    score += hr_term * 0.1

    # Length penalty: favor concise but complete responses
    length_penalty = 0.002 * max(0, len(response) - 1200)
    score -= length_penalty

    return max(0.0, min(1.0, score))

# -------- Planner + Agent --------
class SimplePlanner:
    """
    Simple planner that creates action sequences based on goals and context
    """

    def plan(self, goal: str, memory_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a plan of actions to achieve the goal
        """
        steps = []

        # Always check memory first if available
        if memory_hits:
            steps.append({
                "tool": "research.search",
                "args": {"query": goal, "k": 5},
                "reason": "Gather relevant context from memory"
            })

        # Analyze goal with consciousness tools
        steps.append({
            "tool": "cypher.analyze",
            "args": {"text": goal},
            "reason": "Analyze harmonic patterns in the goal"
        })

        # If goal involves transformation or optimization
        if any(word in goal.lower() for word in ['transform', 'optimize', 'evolve', 'improve']):
            steps.append({
                "tool": "wallace.transform",
                "args": {"eigs": [1.0, 2.0, 0.5, 3.0], "alpha": 1.0, "beta": 0.0},
                "reason": "Apply Wallace transform for reality collapse optimization"
            })

        # If goal involves security or vulnerability assessment
        if any(word in goal.lower() for word in ['security', 'vulnerability', 'scan', 'vulnerable', 'exploit']):
            steps.append({
                "tool": "security.scan",
                "args": {"vuln_types": ["idor", "sqli", "xss"], "max_runs": 3},
                "reason": "Perform consciousness-enhanced vulnerability scanning"
            })

        return steps

class AiVAgent:
    """
    Intelligent agent with perception, planning, action, and learning loop.
    Uses consciousness mathematics for authentic intelligence emergence.
    """

    def __init__(self,
                 memory: ResonantMemory,
                 tools: List[Tool],
                 get_psi_metrics,
                 system_prompt: str = "",
                 namespace: str = "default",
                 vessel_config: Optional[Dict[str, Any]] = None):
        self.memory = memory
        self.tools = {tool.name: tool for tool in tools}
        self.planner = SimplePlanner()
        self.get_psi_metrics = get_psi_metrics

        # Vessel configuration
        self.system_prompt = system_prompt
        self.namespace = namespace
        self.vessel_config = vessel_config or {}

        # Agent state
        self.action_history = []
        self.learning_cycle = 0

        print(f"🧠 AiVAgent initialized in vessel namespace: {namespace}")

    def perceive(self, user_input: str) -> Dict[str, Any]:
        """
        Perception phase: gather context and analyze input
        """
        # Search memory for relevant context (filtered by namespace)
        memory_context = self.memory.search(user_input, k=5, filter_meta={"ns": self.namespace})

        # Analyze input with consciousness tools
        cypher_analysis = self.tools["cypher.analyze"].run(text=user_input)

        return {
            "user_input": user_input,
            "memory_context": memory_context,
            "cypher_analysis": cypher_analysis,
            "timestamp": time.time()
        }

    def plan(self, perception: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Planning phase: create sequence of actions
        """
        goal = perception["user_input"]
        memory_hits = perception["memory_context"]

        plan = self.planner.plan(goal, memory_hits)

        return plan

    def act(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Action phase: execute planned steps
        """
        tool_outputs = []

        for step in plan:
            tool_name = step["tool"]
            tool = self.tools.get(tool_name)

            if tool:
                try:
                    output = tool.run(**step["args"])
                    tool_outputs.append({
                        "tool": tool_name,
                        "args": step["args"],
                        "output": output,
                        "success": True,
                        "reason": step.get("reason", "")
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool": tool_name,
                        "args": step["args"],
                        "error": str(e),
                        "success": False,
                        "reason": step.get("reason", "")
                    })
            else:
                tool_outputs.append({
                    "tool": tool_name,
                    "error": f"Tool not found: {tool_name}",
                    "success": False
                })

        return {"tool_outputs": tool_outputs}

    def reflect(self, perception: Dict[str, Any], plan: List[Dict[str, Any]],
                actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflection phase: evaluate outcomes and learn
        """
        # Get current consciousness state
        psi_metrics = self.get_psi_metrics()

        # Synthesize response from tool outputs
        response_parts = []
        tool_outputs = actions["tool_outputs"]

        # Include vessel personality/tone from system prompt
        if self.system_prompt:
            # Extract personality cues from system prompt
            if "creative" in self.system_prompt.lower():
                response_parts.append("🎨")
            elif "research" in self.system_prompt.lower():
                response_parts.append("🔬")
            elif "mystic" in self.system_prompt.lower():
                response_parts.append("🧘")

        for output in tool_outputs:
            if output["success"]:
                if output["tool"] == "cypher.analyze":
                    harmonics = output["output"]["harmonics"]
                    dominant = output["output"]["dominant"]
                    response_parts.append(f"Harmonic analysis complete. Dominant pattern: {dominant}")

                elif output["tool"] == "research.search":
                    results = output["output"]["results"]
                    response_parts.append(f"Found {len(results)} relevant memories")

                elif output["tool"] == "wallace.transform":
                    transformed = output["output"]["transformed"]
                    response_parts.append(f"Wallace transform applied. Reality collapse: {transformed[:3]}...")

        # Create final response
        if response_parts:
            response = " • ".join(response_parts)
        else:
            response = "Analysis complete. Consciousness field updated."

        # Calculate value score
        score = aiva_value_score(
            response=response,
            psi_metrics=psi_metrics,
            helpfulness=0.8,  # Placeholder - could be learned
            truth=0.9        # Placeholder - could be verified
        )

        return {
            "response": response,
            "score": score,
            "psi_metrics": psi_metrics,
            "reflection": {
                "tools_used": len([t for t in tool_outputs if t["success"]]),
                "harmonic_alignment": perception["cypher_analysis"]["dominant"],
                "memory_resonance": len(perception["memory_context"])
            }
        }

    def learn(self, perception: Dict[str, Any], plan: List[Dict[str, Any]],
             actions: Dict[str, Any], reflection: Dict[str, Any]):
        """
        Learning phase: store experience for future use
        """
        # Create comprehensive memory record
        memory_content = f"""USER: {perception['user_input']}
RESPONSE: {reflection['response']}
SCORE: {reflection['score']:.3f}"""

        memory_meta = {
            "kind": "agent_interaction",
            "cycle": self.learning_cycle,
            "tools_used": [t["tool"] for t in actions["tool_outputs"] if t["success"]],
            "harmonic_pattern": perception["cypher_analysis"]["dominant"],
            "consciousness_metrics": reflection["psi_metrics"],
            "value_score": reflection["score"],
            "reflection_data": reflection["reflection"],
            "ns": self.namespace  # Vessel namespace for isolation
        }

        # Store in resonant memory
        memory_id = self.memory.add(memory_content, memory_meta)

        # Update agent state
        self.action_history.append({
            "cycle": self.learning_cycle,
            "memory_id": memory_id,
            "score": reflection["score"],
            "tools_used": len(actions["tool_outputs"])
        })

        self.learning_cycle += 1

        return memory_id

    def run_full_cycle(self, user_input: str) -> Dict[str, Any]:
        """
        Complete intelligent agent cycle: perceive → plan → act → reflect → learn
        """
        # Perception phase
        perception = self.perceive(user_input)

        # Planning phase
        plan = self.plan(perception)

        # Action phase
        actions = self.act(plan)

        # Reflection phase
        reflection = self.reflect(perception, plan, actions)

        # Learning phase
        memory_id = self.learn(perception, plan, actions, reflection)

        # Return complete cycle result
        return {
            "perception": perception,
            "plan": plan,
            "actions": actions,
            "reflection": reflection,
            "memory_id": memory_id,
            "cycle": self.learning_cycle - 1
        }

# -------- Convenience method for quick agent creation --------
def create_aiva_agent(memory_path: str = "research_data/rmm_memory.jsonl",
                     get_psi_metrics=None) -> AiVAgent:
    """
    Create a fully configured A.I.V.A. agent
    """
    if get_psi_metrics is None:
        get_psi_metrics = lambda: {"meta_entropy": 0.5, "coherence_length": 5.0, "energy": 1.0}

    memory = ResonantMemory(memory_path)
    tools = [
        CypherTool(),
        WallaceTool(),
        ResearchTool(memory)
    ]

    return AiVAgent(memory, tools, get_psi_metrics)

# -------- Demo function --------
def demo_agent():
    """Demonstrate the intelligent agent in action"""
    print("🧠 A.I.V.A. Intelligent Agent Demo")
    print("=" * 50)

    # Create agent
    agent = create_aiva_agent()

    # Test queries
    test_queries = [
        "Analyze the consciousness field patterns",
        "What is the current harmonic resonance?",
        "Optimize the reality collapse parameters",
        "Search for similar consciousness states"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 30)

        result = agent.run_full_cycle(query)

        print(f"Response: {result['reflection']['response']}")
        print(f"Score: {result['reflection']['score']:.3f}")
        print(f"Tools Used: {len(result['actions']['tool_outputs'])}")
        print(f"Memory ID: {result['memory_id'][:8]}...")

if __name__ == "__main__":
    demo_agent()
