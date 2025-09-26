#!/usr/bin/env python3
"""
Add Advanced Cybersecurity and Programming Subjects to Möbius Curriculum
"""

import json
from pathlib import Path
from datetime import datetime

def add_advanced_subjects():
    """Add advanced cybersecurity and programming subjects to the curriculum."""

    learning_db = Path("research_data/moebius_learning_objectives.json")

    if not learning_db.exists():
        print("❌ Learning objectives file not found!")
        return

    # Load existing subjects
    with open(learning_db, 'r') as f:
        objectives = json.load(f)

    # Advanced Cybersecurity Subjects
    cybersecurity_subjects = [
        {
            "name": "advanced_cryptanalysis",
            "category": "cybersecurity",
            "difficulty": "expert",
            "description": "Advanced cryptographic attack techniques and cryptanalysis methods",
            "sources": ["cryptanalysis_research", "black_hat_crypto", "defcon_crypto"],
            "relevance_score": 0.95,
            "estimated_hours": 200
        },
        {
            "name": "zero_trust_architecture",
            "category": "cybersecurity",
            "difficulty": "advanced",
            "description": "Implementing zero-trust security models and architectures",
            "sources": ["google_zero_trust", "microsoft_zero_trust", "nist_zero_trust"],
            "relevance_score": 0.92,
            "estimated_hours": 150
        },
        {
            "name": "advanced_malware_analysis",
            "category": "cybersecurity",
            "difficulty": "expert",
            "description": "Reverse engineering and analysis of advanced malware families",
            "sources": ["malware_analysis_tools", "virus_total", "hybrid_analysis"],
            "relevance_score": 0.94,
            "estimated_hours": 180
        },
        {
            "name": "cloud_security_posture",
            "category": "cybersecurity",
            "difficulty": "advanced",
            "description": "Cloud security assessment and posture management",
            "sources": ["aws_security", "azure_security", "gcp_security"],
            "relevance_score": 0.90,
            "estimated_hours": 140
        },
        {
            "name": "digital_forensics_advanced",
            "category": "cybersecurity",
            "difficulty": "expert",
            "description": "Advanced digital forensics and incident response techniques",
            "sources": ["sans_forensics", "ec_council_chfi", "forensic_tools"],
            "relevance_score": 0.91,
            "estimated_hours": 160
        },
        {
            "name": "secure_coding_practices",
            "category": "cybersecurity",
            "difficulty": "advanced",
            "description": "Implementing secure coding practices and vulnerability prevention",
            "sources": ["owasp_secure_coding", "cert_secure_coding", "microsoft_sdl"],
            "relevance_score": 0.93,
            "estimated_hours": 130
        }
    ]

    # Advanced Programming Subjects
    programming_subjects = [
        {
            "name": "functional_programming_scala",
            "category": "programming",
            "difficulty": "advanced",
            "description": "Advanced functional programming with Scala and functional paradigms",
            "sources": ["scala_functional", "functional_programming_principles", "scala_docs"],
            "relevance_score": 0.88,
            "estimated_hours": 140
        },
        {
            "name": "concurrent_distributed_systems",
            "category": "programming",
            "difficulty": "expert",
            "description": "Building concurrent and distributed systems with advanced patterns",
            "sources": ["distributed_systems_design", "concurrency_patterns", "akka_framework"],
            "relevance_score": 0.91,
            "estimated_hours": 170
        },
        {
            "name": "advanced_algorithms_complexity",
            "category": "programming",
            "difficulty": "expert",
            "description": "Advanced algorithmic techniques and complexity analysis",
            "sources": ["algorithms_design_manual", "computational_complexity", "algorithm_analysis"],
            "relevance_score": 0.94,
            "estimated_hours": 190
        },
        {
            "name": "domain_specific_languages",
            "category": "programming",
            "difficulty": "expert",
            "description": "Design and implementation of domain-specific languages",
            "sources": ["dsl_design_patterns", "language_implementation", "dsl_frameworks"],
            "relevance_score": 0.89,
            "estimated_hours": 160
        },
        {
            "name": "performance_optimization",
            "category": "programming",
            "difficulty": "advanced",
            "description": "Advanced performance optimization techniques and profiling",
            "sources": ["performance_engineering", "profiling_tools", "optimization_patterns"],
            "relevance_score": 0.92,
            "estimated_hours": 150
        },
        {
            "name": "software_architecture_patterns",
            "category": "programming",
            "difficulty": "advanced",
            "description": "Advanced software architecture patterns and design principles",
            "sources": ["software_architecture_patterns", "design_patterns_gof", "enterprise_patterns"],
            "relevance_score": 0.90,
            "estimated_hours": 140
        }
    ]

    # Combine all new subjects
    all_new_subjects = cybersecurity_subjects + programming_subjects

    # Add subjects to curriculum
    subjects_added = 0
    timestamp = datetime.now().isoformat()

    for subject in all_new_subjects:
        subject_name = subject["name"]

        # Check if subject already exists
        if subject_name in objectives:
            print(f"⚠️  Skipping existing subject: {subject_name}")
            continue

        # Create curriculum entry
        curriculum_subject = {
            "status": "pending",
            "completion_percentage": 0,
            "prerequisites": [],
            "category": subject["category"],
            "difficulty": subject["difficulty"],
            "estimated_hours": subject["estimated_hours"],
            "description": subject["description"],
            "sources": subject["sources"],
            "last_attempt": None,
            "wallace_completion_score": 0,
            "learning_efficiency": 0,
            "universal_math_enhancement": 1.618033988749895,
            "auto_discovered": False,  # Manually added
            "discovery_date": timestamp,
            "relevance_score": subject["relevance_score"]
        }

        objectives[subject_name] = curriculum_subject
        subjects_added += 1

        print(f"✅ Added: {subject_name} ({subject['category']} - {subject['difficulty']})")

    # Save updated curriculum
    if subjects_added > 0:
        with open(learning_db, 'w') as f:
            json.dump(objectives, f, indent=2)

        print(f"\n🎉 Successfully added {subjects_added} new subjects!")
        print(f"🔐 Cybersecurity subjects: {len(cybersecurity_subjects)}")
        print(f"💻 Programming subjects: {len(programming_subjects)}")
        print(f"📚 Total curriculum size: {len(objectives)} subjects")

    else:
        print("\nℹ️  No new subjects were added (all already exist)")

def main():
    """Main function to add advanced subjects."""
    print("🔐 Adding Advanced Cybersecurity & Programming Subjects")
    print("=" * 60)

    add_advanced_subjects()

    print("\n🚀 Möbius Loop Trainer Curriculum Expanded!")
    print("Ready to master advanced cybersecurity and programming subjects!")

if __name__ == "__main__":
    main()
