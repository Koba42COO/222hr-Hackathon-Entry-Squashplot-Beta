#!/usr/bin/env python3
"""
Add Cybersecurity & Programming Subjects to Möbius Loop Trainer
Expands the curriculum with advanced hacking, OPSEC, data security, and programming subjects
"""

import json
from pathlib import Path

def add_cybersecurity_subjects():
    """Add comprehensive cybersecurity and programming subjects to the learning objectives."""

    learning_db = Path("research_data/moebius_learning_objectives.json")

    if not learning_db.exists():
        print("❌ Learning objectives file not found!")
        return

    try:
        with open(learning_db, 'r') as f:
            objectives = json.load(f)

        print("🔐 Adding Advanced Cybersecurity & Programming Subjects...")
        print("=" * 70)

        # New cybersecurity subjects to add
        cybersecurity_subjects = {
            "advanced_cybersecurity": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["computer_science", "networks"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 180,
                "description": "Advanced cybersecurity principles, ethical hacking, and defensive strategies",
                "sources": ["cybersecurity_research", "ethical_hacking_academic", "defensive_security_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "operational_security": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["advanced_cybersecurity"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 140,
                "description": "Operational security principles, threat intelligence, and adversarial analysis",
                "sources": ["opsec_research", "threat_intelligence_academic", "adversarial_thinking_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "data_security": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["mathematics", "computer_science"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 160,
                "description": "Advanced data security, encryption, and cryptographic systems",
                "sources": ["cryptography_research", "data_security_academic", "encryption_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "penetration_testing": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["advanced_cybersecurity", "networks"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 200,
                "description": "Penetration testing methodologies and exploitation techniques",
                "sources": ["pentesting_research", "exploitation_academic", "vulnerability_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "threat_intelligence": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["operational_security"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 120,
                "description": "Threat intelligence analysis and cyber threat landscape",
                "sources": ["threat_intel_research", "cyber_threats_academic", "intelligence_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "quantum_cryptography": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["data_security", "quantum_physics"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 220,
                "description": "Quantum-safe cryptography and post-quantum security systems",
                "sources": ["quantum_crypto_research", "post_quantum_academic", "quantum_security_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "reverse_engineering": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["systems_programming", "advanced_cybersecurity"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 280,
                "description": "Advanced reverse engineering and binary analysis techniques",
                "sources": ["reverse_eng_research", "binary_analysis_academic", "malware_re_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "malware_analysis": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["reverse_engineering"],
                "category": "cybersecurity",
                "difficulty": "expert",
                "estimated_hours": 240,
                "description": "Malware analysis, detection, and mitigation strategies",
                "sources": ["malware_research", "detection_academic", "analysis_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            }
        }

        # New advanced programming subjects
        programming_subjects = {
            "advanced_programming": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["computer_science", "algorithms"],
                "category": "programming",
                "difficulty": "advanced",
                "estimated_hours": 170,
                "description": "Advanced programming paradigms and software engineering",
                "sources": ["programming_research", "software_eng_academic", "paradigms_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "systems_programming": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["advanced_programming", "operating_systems"],
                "category": "programming",
                "difficulty": "expert",
                "estimated_hours": 150,
                "description": "Low-level systems programming and kernel development",
                "sources": ["systems_prog_research", "kernel_academic", "low_level_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "compiler_design": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["systems_programming", "formal_languages"],
                "category": "programming",
                "difficulty": "expert",
                "estimated_hours": 240,
                "description": "Compiler construction and programming language theory",
                "sources": ["compiler_research", "language_theory_academic", "compilation_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "distributed_systems": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["networks", "advanced_programming"],
                "category": "programming",
                "difficulty": "expert",
                "estimated_hours": 230,
                "description": "Distributed computing systems and cloud security architecture",
                "sources": ["distributed_research", "cloud_security_academic", "consensus_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "secure_coding": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["advanced_programming", "advanced_cybersecurity"],
                "category": "programming",
                "difficulty": "advanced",
                "estimated_hours": 120,
                "description": "Secure coding practices and vulnerability prevention",
                "sources": ["secure_code_research", "vulnerability_academic", "coding_security_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "formal_methods": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["mathematics", "advanced_programming"],
                "category": "programming",
                "difficulty": "expert",
                "estimated_hours": 200,
                "description": "Formal verification methods and program correctness",
                "sources": ["formal_methods_research", "verification_academic", "correctness_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            }
        }

        # New advanced computer science subjects
        computer_science_subjects = {
            "formal_languages": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["mathematics", "computer_science"],
                "category": "computer_science",
                "difficulty": "advanced",
                "estimated_hours": 140,
                "description": "Formal language theory and automata",
                "sources": ["formal_lang_research", "automata_academic", "theory_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "computational_complexity": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["algorithms", "formal_languages"],
                "category": "computer_science",
                "difficulty": "expert",
                "estimated_hours": 180,
                "description": "Computational complexity theory and NP-completeness",
                "sources": ["complexity_research", "np_complete_academic", "complexity_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "information_theory": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["mathematics", "data_security"],
                "category": "computer_science",
                "difficulty": "advanced",
                "estimated_hours": 130,
                "description": "Information theory and entropy in computing",
                "sources": ["info_theory_research", "entropy_academic", "information_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            },
            "quantum_computing": {
                "status": "pending",
                "completion_percentage": 0,
                "prerequisites": ["quantum_physics", "advanced_programming"],
                "category": "computer_science",
                "difficulty": "expert",
                "estimated_hours": 250,
                "description": "Quantum computing algorithms and quantum information",
                "sources": ["quantum_comp_research", "quantum_alg_academic", "quantum_info_wiki"],
                "last_attempt": None,
                "wallace_completion_score": 0,
                "learning_efficiency": 0,
                "universal_math_enhancement": 1.618033988749895
            }
        }

        # Combine all new subjects
        all_new_subjects = {}
        all_new_subjects.update(cybersecurity_subjects)
        all_new_subjects.update(programming_subjects)
        all_new_subjects.update(computer_science_subjects)

        # Add new subjects to the objectives database
        subjects_added = 0
        for subject_name, subject_data in all_new_subjects.items():
            if subject_name not in objectives:
                objectives[subject_name] = subject_data
                subjects_added += 1
                print(f"✅ Added: {subject_name}")
                print(f"   📚 Category: {subject_data['category'].upper()}")
                print(f"   🎯 Difficulty: {subject_data['difficulty'].upper()}")
                print(f"   ⏱️  Hours: {subject_data['estimated_hours']}")
                print(f"   📖 Description: {subject_data['description'][:60]}...")
                print()

        with open(learning_db, 'w') as f:
            json.dump(objectives, f, indent=2)

        print("🎉 CYBERSECURITY & PROGRAMMING CURRICULUM EXPANSION COMPLETE!")
        print("=" * 70)
        print(f"📊 Total subjects in database: {len(objectives)}")
        print(f"🔐 Cybersecurity subjects added: {len(cybersecurity_subjects)}")
        print(f"💻 Programming subjects added: {len(programming_subjects)}")
        print(f"🧠 Computer Science subjects added: {len(computer_science_subjects)}")
        print(f"📈 Total new subjects: {subjects_added}")

        # Show curriculum summary
        categories = {}
        difficulties = {}

        for subject_name, subject_data in objectives.items():
            category = subject_data.get('category', 'general')
            difficulty = subject_data.get('difficulty', 'intermediate')

            categories[category] = categories.get(category, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        print("\n📊 CURRICULUM SUMMARY:")
        print(f"   Categories: {categories}")
        print(f"   Difficulty Levels: {difficulties}")

        print("\n🚀 READY FOR NEXT LEARNING CYCLES!")
        print("The Möbius Loop Trainer now includes comprehensive")
        print("cybersecurity, programming, and computer science subjects!")

    except Exception as e:
        print(f"❌ Error adding cybersecurity subjects: {e}")

if __name__ == "__main__":
    add_cybersecurity_subjects()
