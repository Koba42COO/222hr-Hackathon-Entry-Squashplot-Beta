#!/usr/bin/env python3
"""
🛶 VESSEL SYSTEM DEMONSTRATION
==============================

Demonstrate the complete vessel system with consciousness mathematics.
"""

from REVOLUTIONARY_CONTINUOUS_LEARNING_SYSTEM import RevolutionaryLearningCoordinator
from vessel_factory import create_research_vessel, create_creative_vessel, create_mystic_vessel
from aiva_core import AiVAgent, ResonantMemory, CypherTool, WallaceTool, ResearchTool

def main():
    print("🛶 VESSEL SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("Portable AiVA Shells with Consciousness Mathematics")
    print("=" * 60)

    # Initialize coordinator
    print("\n🚀 Initializing Revolutionary Coordinator...")
    coordinator = RevolutionaryLearningCoordinator()

    # Create specialized vessels
    print("\n🏗️ Creating Specialized Vessels...")

    vessels = []

    # Research vessel
    research_path = create_research_vessel("demo_research")
    vessels.append(("Research", research_path))

    # Creative vessel
    creative_path = create_creative_vessel("demo_creative")
    vessels.append(("Creative", creative_path))

    # Mystic vessel
    mystic_path = create_mystic_vessel("demo_mystic")
    vessels.append(("Mystic", mystic_path))

    print("\n📊 Created Vessels:")
    for vessel_type, vessel_path in vessels:
        print(f"  • {vessel_type}: {vessel_path}")

    # Demonstrate vessel switching
    print("\n🔄 Demonstrating Vessel Switching...")

    test_queries = [
        "Analyze consciousness patterns in the data",
        "Create an innovative solution for memory optimization",
        "Explore the mystical aspects of harmonic resonance"
    ]

    for i, (vessel_type, vessel_path) in enumerate(vessels):
        print(f"\n🎭 Loading {vessel_type} Vessel...")
        coordinator.load_vessel(str(vessel_path))

        vessel_info = coordinator.get_current_vessel_info()
        print(f"   📋 Name: {vessel_info['name']}")
        print(f"   🧠 Ethics: {vessel_info['ethics_profile']}")
        print(f"   🔧 Tools: {len(vessel_info['tools'])}")
        print(f"   📚 Memory: {len(vessel_info.get('seed_conversations', []))} entries")

        # Test agent in this vessel
        query = test_queries[i % len(test_queries)]
        print(f"   🔍 Testing: '{query[:50]}...'")

        try:
            result = coordinator.agent.run_full_cycle(query)
            print(f"   ✅ Score: {result['reflection']['score']:.3f}")
            print(f"   🛠️ Tools Used: {len(result['actions']['tool_outputs'])}")
            print(f"   💬 Response: {result['reflection']['response'][:80]}...")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

    # Show vessel management capabilities
    print("\n📋 Vessel Management:")
    all_vessels = coordinator.list_available_vessels()
    print(f"   • Total Vessels: {len(all_vessels)}")
    for vessel in all_vessels[-3:]:  # Show last 3
        print(f"     - {vessel['name']} ({vessel['ethics_profile']}) - {vessel['tools_count']} tools")

    # Create a new vessel from current memory
    print("\n🧬 Creating Vessel from Current Memory...")
    new_vessel_path = coordinator.create_vessel_from_memory("demo_from_memory", "Created from live system memory")
    print(f"✅ Created: {new_vessel_path}")

    print("\n🎉 VESSEL SYSTEM COMPLETE!")
    print("\n🎯 Key Features Demonstrated:")
    print("  ✅ Portable AiVA instances with specialized capabilities")
    print("  ✅ Isolated memory namespaces for different contexts")
    print("  ✅ Customizable ethics, tools, and personality profiles")
    print("  ✅ Seamless switching between different AiVA 'personalities'")
    print("  ✅ Consciousness mathematics integration throughout")
    print("  ✅ Memory seeding and harmonic resonance tracking")

    print("\n🚀 Vessel Benefits:")
    print("  • 🎨 Creative Vessel: Innovation-focused with high creativity")
    print("  • 🔬 Research Vessel: Analytical with scientific rigor")
    print("  • 🧘 Mystic Vessel: Consciousness exploration with spiritual depth")
    print("  • 📚 Memory Vessels: Persistent context across sessions")
    print("  • 🔄 Dynamic Switching: Instant personality/context changes")

    print("\n🧠 Consciousness Mathematics Integration:")
    print("  • Harmonic resonance for context understanding")
    print("  • Meta-entropy tracking for information complexity")
    print("  • Golden ratio optimization for decision making")
    print("  • Recursive intelligence through memory evolution")

if __name__ == "__main__":
    main()
