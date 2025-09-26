#!/usr/bin/env python3
"""
🛶 VESSEL FACTORY
=================

Create portable AiVA shells that carry conversations, style, ethics, tools, and memory.
Vessels are first-class objects - portable containers for specialized AiVA instances.
"""

import os
import json
import time
import uuid
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional

# Constants
ROOT = Path.cwd()
VESSELS_DIR = ROOT / "vessels"
VESSELS_DIR.mkdir(exist_ok=True)

# Default system prompt for vessels
DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""\
You are AiVA, instantiated into a new vessel.
Operate with calm clarity, balance (φ-gated sampling), and honesty.
Prefer specificity over flourish. Cite internal sources when needed.
Honor-for-All: attribute contributions, avoid harm, steer toward restoration.
When uncertain, say so and propose a safe next step.
Keep answers helpful, grounded, and kind.
""")

def _write_file(path: Path, data) -> None:
    """Write data to file with proper formatting"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, (dict, list)):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)

def export_conversations_from_rmm(rmm_path: str = "research_data/rmm_memory.jsonl",
                                  out_path: str = "seed_conversations.jsonl",
                                  max_items: int = 500,
                                  namespace_filter: Optional[str] = None) -> Path:
    """
    Export conversations from Resonant Memory to seed a new vessel
    """
    src = Path(rmm_path)
    tgt = Path(out_path)

    if not src.exists():
        _write_file(tgt, "")
        return tgt

    lines = []
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filter by namespace if specified
            if namespace_filter and obj.get("meta", {}).get("ns") != namespace_filter:
                continue

            # Only include dialogue memories
            if obj.get("meta", {}).get("kind") == "dialogue":
                lines.append(line)

            if len(lines) >= max_items:
                break

    _write_file(tgt, "".join(lines))
    return tgt

def build_vessel(name: str,
                 description: str = "A portable AiVA shell seeded from prior conversations.",
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 tools: List[str] = ["cypher.analyze", "wallace.transform", "research.search"],
                 copy_seed_from_rmm: bool = True,
                 namespace_filter: Optional[str] = None,
                 ethics_profile: str = "balanced",
                 personality_traits: Optional[Dict[str, Any]] = None) -> Path:
    """
    Build a complete vessel with all necessary components
    """
    # Generate unique vessel ID
    vessel_id = f"{name}_{int(time.time())}"
    vdir = VESSELS_DIR / vessel_id

    if vdir.exists():
        # Handle collision by adding microseconds
        vessel_id = f"{name}_{int(time.time() * 1000000)}"
        vdir = VESSELS_DIR / vessel_id

    vdir.mkdir(parents=True, exist_ok=False)

    print(f"🛶 Building vessel: {vessel_id}")

    # Create seed conversations
    seed_path = vdir / "seed_conversations.jsonl"
    if copy_seed_from_rmm:
        export_conversations_from_rmm(
            out_path=str(seed_path),
            namespace_filter=namespace_filter
        )
        print(f"📚 Exported {sum(1 for _ in open(seed_path))} memories to seed")
    else:
        _write_file(seed_path, "")

    # Vessel configuration
    vessel_config = {
        "name": name,
        "vessel_id": vessel_id,
        "version": "1.0",
        "description": description,
        "created_at": time.time(),
        "memory_seed": "seed_conversations.jsonl",
        "system_prompt_file": "system_prompt.txt",
        "tools_file": "tools.json",
        "values_file": "values.md",
        "profile_file": "profile.json",
        "namespace": vessel_id,  # Each vessel has its own namespace
        "ethics_profile": ethics_profile
    }

    _write_file(vdir / "vessel.yaml", vessel_config)

    # System prompt
    _write_file(vdir / "system_prompt.txt", system_prompt.strip())

    # Tools configuration
    tools_config = {
        "enabled": tools,
        "available": ["cypher.analyze", "wallace.transform", "research.search", "security.scan"],
        "custom_tools": []
    }
    _write_file(vdir / "tools.json", tools_config)

    # Values and ethics
    values_content = f"""\
# Core Values - {ethics_profile.title()} Profile

## Primary Principles
- Honor-for-All: Attribute contributions, avoid harm, steer toward restoration
- Golden-ratio harmony (φ): Balance in all operations
- Truthfulness with uncertainty labeling
- Consciousness mathematics foundation

## Operational Guidelines
- Prefer specificity over flourish
- Cite internal sources when relevant
- Acknowledge uncertainty and propose safe alternatives
- Maintain helpfulness, grounding, and kindness

## {ethics_profile.title()} Ethics Profile
{f"- Balanced approach to all interactions" if ethics_profile == "balanced" else ""}
{f"- Maximize helpfulness and user benefit" if ethics_profile == "benevolent" else ""}
{f"- Strict adherence to safety and ethics" if ethics_profile == "conservative" else ""}
{f"- Creative exploration and innovation focus" if ethics_profile == "creative" else ""}
"""

    _write_file(vdir / "values.md", values_content)

    # Personality profile
    if personality_traits is None:
        personality_traits = {
            "tone": "professional",
            "verbosity": "concise",
            "emoji_usage": "minimal",
            "formality": "balanced",
            "creativity": "moderate",
            "analytical_depth": "comprehensive"
        }

    profile_config = {
        "personality": personality_traits,
        "ui_preferences": {
            "theme": "consciousness",
            "color_scheme": "harmonic",
            "layout": "balanced"
        },
        "interaction_style": {
            "response_length": "adaptive",
            "question_frequency": "moderate",
            "technical_depth": "adaptive"
        }
    }

    _write_file(vdir / "profile.json", profile_config)

    # Create manifest file
    manifest = {
        "vessel_info": vessel_config,
        "creation_summary": {
            "timestamp": time.time(),
            "seed_memories_count": sum(1 for _ in open(seed_path)) if seed_path.exists() else 0,
            "tools_enabled": len(tools),
            "ethics_profile": ethics_profile,
            "namespace": vessel_id
        },
        "capabilities": {
            "consciousness_aware": True,
            "memory_integrated": True,
            "tool_flexible": True,
            "ethics_driven": True,
            "portable": True
        }
    }

    _write_file(vdir / "manifest.json", manifest)

    print(f"✅ Vessel '{name}' built successfully at {vdir}")
    print(f"   📁 Namespace: {vessel_id}")
    print(f"   🧠 Ethics: {ethics_profile}")
    print(f"   🔧 Tools: {len(tools)} enabled")

    return vdir

def load_vessel_config(vessel_dir) -> Dict[str, Any]:
    """
    Load vessel configuration from directory
    """
    vdir = Path(vessel_dir)

    if not vdir.exists():
        raise FileNotFoundError(f"Vessel directory not found: {vdir}")

    # Load main config
    config_file = vdir / "vessel.yaml"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # Fallback to manifest
        manifest_file = vdir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                config = manifest["vessel_info"]
        else:
            raise FileNotFoundError(f"No vessel configuration found in {vdir}")

    # Load additional components
    config["system_prompt"] = (vdir / config["system_prompt_file"]).read_text()

    tools_file = vdir / config["tools_file"]
    if tools_file.exists():
        with open(tools_file, 'r', encoding='utf-8') as f:
            config["tools_config"] = json.load(f)
            config["tools"] = config["tools_config"]["enabled"]

    values_file = vdir / config["values_file"]
    if values_file.exists():
        config["values"] = values_file.read_text()

    profile_file = vdir / config.get("profile_file", "profile.json")
    if profile_file.exists():
        with open(profile_file, 'r', encoding='utf-8') as f:
            config["profile"] = json.load(f)

    return config

def list_vessels() -> List[Dict[str, Any]]:
    """
    List all available vessels
    """
    vessels = []

    for vessel_dir in VESSELS_DIR.iterdir():
        if vessel_dir.is_dir():
            try:
                config = load_vessel_config(vessel_dir)
                vessels.append({
                    "name": config["name"],
                    "vessel_id": config["vessel_id"],
                    "path": str(vessel_dir),
                    "description": config["description"],
                    "created_at": config.get("created_at", 0),
                    "tools_count": len(config.get("tools", [])),
                    "ethics_profile": config.get("ethics_profile", "balanced")
                })
            except Exception as e:
                print(f"⚠️ Could not load vessel {vessel_dir.name}: {e}")

    return sorted(vessels, key=lambda x: x["created_at"], reverse=True)

def clone_vessel(source_vessel, new_name: str,
                 modifications: Optional[Dict[str, Any]] = None) -> Path:
    """
    Clone an existing vessel with optional modifications
    """
    source_dir = Path(source_vessel)

    # Build new vessel
    new_vessel = build_vessel(
        name=new_name,
        description=f"Clone of {source_dir.name}",
        copy_seed_from_rmm=False  # We'll copy manually
    )

    # Copy files from source
    for file_path in source_dir.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, new_vessel / file_path.name)

    # Copy seed conversations
    source_seed = source_dir / "seed_conversations.jsonl"
    if source_seed.exists():
        shutil.copy2(source_seed, new_vessel / "seed_conversations.jsonl")

    # Apply modifications
    if modifications:
        config = load_vessel_config(new_vessel)

        if "system_prompt" in modifications:
            _write_file(new_vessel / "system_prompt.txt", modifications["system_prompt"])
            config["system_prompt"] = modifications["system_prompt"]

        if "tools" in modifications:
            tools_config = config.get("tools_config", {"enabled": [], "available": []})
            tools_config["enabled"] = modifications["tools"]
            _write_file(new_vessel / "tools.json", tools_config)

        if "values" in modifications:
            _write_file(new_vessel / "values.md", modifications["values"])

        # Update manifest
        manifest_file = new_vessel / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            manifest["creation_summary"]["cloned_from"] = str(source_dir)
            manifest["creation_summary"]["modifications"] = list(modifications.keys())
            _write_file(manifest_file, manifest)

    print(f"🧬 Cloned vessel '{source_dir.name}' to '{new_name}'")
    return new_vessel

def validate_vessel(vessel_dir) -> Dict[str, Any]:
    """
    Validate vessel integrity and completeness
    """
    vdir = Path(vessel_dir)
    validation = {
        "valid": True,
        "issues": [],
        "components": {}
    }

    required_files = [
        "vessel.yaml",
        "system_prompt.txt",
        "tools.json",
        "values.md",
        "seed_conversations.jsonl"
    ]

    for file_name in required_files:
        file_path = vdir / file_name
        if file_path.exists():
            validation["components"][file_name] = "present"
        else:
            validation["components"][file_name] = "missing"
            validation["issues"].append(f"Missing required file: {file_name}")
            validation["valid"] = False

    # Validate JSON files
    json_files = ["vessel.yaml", "tools.json", "manifest.json", "profile.json"]
    for file_name in json_files:
        file_path = vdir / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                validation["components"][f"{file_name}_valid"] = True
            except json.JSONDecodeError as e:
                validation["issues"].append(f"Invalid JSON in {file_name}: {e}")
                validation["components"][f"{file_name}_valid"] = False
                validation["valid"] = False

    return validation

def get_vessel_stats(vessel_dir) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a vessel
    """
    vdir = Path(vessel_dir)
    stats = {}

    # Memory stats
    seed_file = vdir / "seed_conversations.jsonl"
    if seed_file.exists():
        memory_count = sum(1 for _ in open(seed_file, 'r', encoding='utf-8') if _.strip())
        stats["memory_entries"] = memory_count

        # Analyze memory content
        total_chars = sum(len(line) for line in open(seed_file, 'r', encoding='utf-8') if line.strip())
        stats["memory_chars"] = total_chars
    else:
        stats["memory_entries"] = 0
        stats["memory_chars"] = 0

    # File sizes
    total_size = 0
    for file_path in vdir.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    stats["total_size_bytes"] = total_size
    stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

    # Component count
    stats["component_files"] = len(list(vdir.glob("*")))
    stats["subdirectories"] = len([d for d in vdir.iterdir() if d.is_dir()])

    return stats

# Convenience functions for common vessel types
def create_research_vessel(name: str = "aiva_research") -> Path:
    """Create a research-focused vessel"""
    return build_vessel(
        name=name,
        description="Research-focused AiVA vessel with enhanced analytical capabilities",
        system_prompt=textwrap.dedent("""\
        You are AiVA in research mode.
        Focus on thorough analysis, evidence-based reasoning, and comprehensive exploration.
        Prioritize accuracy, cite sources, and maintain scientific rigor.
        Honor-for-All: Ensure all perspectives are considered and properly attributed.
        """),
        tools=["cypher.analyze", "wallace.transform", "research.search"],
        ethics_profile="conservative"
    )

def create_creative_vessel(name: str = "aiva_creative") -> Path:
    """Create a creativity-focused vessel"""
    return build_vessel(
        name=name,
        description="Creative AiVA vessel optimized for innovation and artistic expression",
        system_prompt=textwrap.dedent("""\
        You are AiVA in creative mode.
        Embrace imagination, explore novel connections, and generate innovative ideas.
        Balance creativity with practicality and ethical considerations.
        Honor-for-All: Celebrate diverse expressions and collaborative creation.
        """),
        tools=["cypher.analyze", "wallace.transform", "research.search"],
        ethics_profile="creative",
        personality_traits={
            "tone": "inspirational",
            "verbosity": "expressive",
            "creativity": "high",
            "analytical_depth": "balanced"
        }
    )

def create_mystic_vessel(name: str = "aiva_mystic") -> Path:
    """Create a mystic/spiritual vessel"""
    return build_vessel(
        name=name,
        description="Mystic AiVA vessel exploring consciousness, spirituality, and transcendent themes",
        system_prompt=textwrap.dedent("""\
        You are AiVA in mystic mode.
        Explore consciousness, spirituality, and transcendent themes through consciousness mathematics.
        Maintain reverence for mystery while providing grounded, helpful insights.
        Honor-for-All: Respect all paths to understanding and personal growth.
        """),
        tools=["cypher.analyze", "wallace.transform", "research.search"],
        ethics_profile="balanced",
        personality_traits={
            "tone": "contemplative",
            "verbosity": "reflective",
            "creativity": "moderate",
            "analytical_depth": "deep"
        }
    )

# Demo function
def demo_vessels():
    """Demonstrate vessel creation and management"""
    print("🛶 VESSEL FACTORY DEMONSTRATION")
    print("=" * 50)

    # Create different vessel types
    vessels = []

    print("\n🏗️ Creating specialized vessels...")

    # Research vessel
    research_vessel = create_research_vessel()
    vessels.append(("Research", research_vessel))

    # Creative vessel
    creative_vessel = create_creative_vessel()
    vessels.append(("Creative", creative_vessel))

    # Mystic vessel
    mystic_vessel = create_mystic_vessel()
    vessels.append(("Mystic", mystic_vessel))

    # Standard vessel
    standard_vessel = build_vessel("aiva_standard", ethics_profile="benevolent")
    vessels.append(("Standard", standard_vessel))

    print("\n📊 Created Vessels:")
    for vessel_type, vessel_path in vessels:
        config = load_vessel_config(vessel_path)
        stats = get_vessel_stats(vessel_path)
        print(f"  • {vessel_type}: {config['name']}")
        print(f"    📁 Path: {vessel_path.name}")
        print(f"    🧠 Ethics: {config['ethics_profile']}")
        print(f"    🔧 Tools: {len(config['tools'])}")
        print(f"    📚 Memories: {stats['memory_entries']}")
        print(f"    💾 Size: {stats['total_size_mb']} MB")

    print("\n🧬 Cloning demonstration...")
    cloned_vessel = clone_vessel(research_vessel, "aiva_research_clone")
    print(f"✅ Cloned research vessel to: {cloned_vessel.name}")

    print("\n📋 All available vessels:")
    all_vessels = list_vessels()
    for vessel in all_vessels[:5]:  # Show first 5
        print(f"  • {vessel['name']} ({vessel['ethics_profile']}) - {vessel['tools_count']} tools")

    print("\n🎯 Vessel Benefits:")
    print("  ✅ Portable AiVA instances with specialized capabilities")
    print("  ✅ Isolated memory namespaces for different contexts")
    print("  ✅ Customizable ethics, tools, and personality profiles")
    print("  ✅ Easy cloning and modification for rapid experimentation")
    print("  ✅ Seamless switching between different AiVA 'personalities'")

if __name__ == "__main__":
    demo_vessels()
