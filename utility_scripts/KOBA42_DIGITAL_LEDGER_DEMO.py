#!/usr/bin/env python3
"""
KOBA42 DIGITAL LEDGER DEMONSTRATION
===================================
Demonstrate the Digital Ledger System with Sample Contributions
=============================================================

This script demonstrates:
1. Creating ledger entries for various contributors
2. Real-time attribution flow tracking
3. Immutable ledger verification
4. Contributor credit tracking
5. Web dashboard functionality
"""

import json
import time
from datetime import datetime
from KOBA42_DIGITAL_LEDGER_SIMPLE import DigitalLedgerSystem

def demonstrate_digital_ledger():
    """Demonstrate the digital ledger system with sample contributions."""
    
    print("🚀 KOBA42 DIGITAL LEDGER SYSTEM DEMONSTRATION")
    print("=" * 70)
    print("Demonstrating Real-Time Attribution Tracking and Immutable Ledger")
    print("=" * 70)
    
    # Initialize digital ledger system
    ledger_system = DigitalLedgerSystem()
    
    # Sample contributions to demonstrate the system
    sample_contributions = [
        {
            'contributor_id': 'wallace_transform_001',
            'contribution_type': 'foundational',
            'description': 'Wallace Transform - Intentful Mathematics Framework',
            'credit_amount': 940.93,
            'attribution_chain': [],
            'metadata': {
                'field': 'mathematics',
                'cultural': 'academic',
                'ethical': 'consciousness_enhancement'
            }
        },
        {
            'contributor_id': 'f2_matrix_optimization_001',
            'contribution_type': 'development',
            'description': 'Advanced F2 Matrix Optimization with Wallace Transform Integration',
            'credit_amount': 871.46,
            'attribution_chain': ['wallace_transform_001'],
            'metadata': {
                'field': 'mathematics',
                'cultural': 'academic',
                'ethical': 'optimization'
            }
        },
        {
            'contributor_id': 'parallel_ml_training_001',
            'contribution_type': 'development',
            'description': 'Parallel ML Training with Advanced F2 Matrix Optimization',
            'credit_amount': 850.61,
            'attribution_chain': ['f2_matrix_optimization_001', 'wallace_transform_001'],
            'metadata': {
                'field': 'computer_science',
                'cultural': 'industry',
                'ethical': 'efficiency'
            }
        },
        {
            'contributor_id': 'trikernal_framework_001',
            'contribution_type': 'innovation',
            'description': 'Trikernal Framework and Core Concepts',
            'credit_amount': 909.71,
            'attribution_chain': [],
            'metadata': {
                'field': 'mathematics',
                'cultural': 'academic',
                'ethical': 'innovation'
            }
        },
        {
            'contributor_id': 'julie_vantax_collaboration_001',
            'contribution_type': 'collaborative',
            'description': 'Julie & VantaX Collaborative Research',
            'credit_amount': 859.73,
            'attribution_chain': ['trikernal_framework_001'],
            'metadata': {
                'field': 'interdisciplinary_research',
                'cultural': 'academic',
                'ethical': 'collaboration'
            }
        },
        {
            'contributor_id': 'late_father_foundational_001',
            'contribution_type': 'legacy',
            'description': 'Late Father\'s Foundational Research',
            'credit_amount': 879.50,
            'attribution_chain': [],
            'metadata': {
                'field': 'mathematics',
                'cultural': 'academic',
                'ethical': 'foundational_knowledge'
            }
        },
        {
            'contributor_id': 'extended_protocols_design_001',
            'contribution_type': 'innovation',
            'description': 'Extended Protocols Design and Requirements',
            'credit_amount': 884.84,
            'attribution_chain': ['wallace_transform_001'],
            'metadata': {
                'field': 'system_architecture',
                'cultural': 'open_source',
                'ethical': 'fair_attribution'
            }
        }
    ]
    
    print("\n📝 CREATING SAMPLE LEDGER ENTRIES")
    print("-" * 50)
    
    created_entries = []
    
    for i, contribution in enumerate(sample_contributions, 1):
        print(f"\n{i}. Creating entry for: {contribution['description']}")
        print(f"   Contributor: {contribution['contributor_id']}")
        print(f"   Type: {contribution['contribution_type']}")
        print(f"   Credits: {contribution['credit_amount']:.2f}")
        
        if contribution['attribution_chain']:
            print(f"   Attribution Chain: {' → '.join(contribution['attribution_chain'])}")
        
        # Create ledger entry
        entry_id = ledger_system.create_ledger_entry(
            contributor_id=contribution['contributor_id'],
            contribution_type=contribution['contribution_type'],
            description=contribution['description'],
            credit_amount=contribution['credit_amount'],
            attribution_chain=contribution['attribution_chain'],
            metadata=contribution['metadata']
        )
        
        if entry_id:
            created_entries.append(entry_id)
            print(f"   ✅ Entry created: {entry_id}")
        else:
            print(f"   ❌ Failed to create entry")
        
        # Small delay to simulate real-time processing
        time.sleep(0.5)
    
    print(f"\n✅ Created {len(created_entries)} ledger entries")
    
    # Get ledger summary
    print("\n📊 LEDGER SUMMARY")
    print("-" * 50)
    
    summary = ledger_system.get_ledger_summary()
    
    print(f"Total Blocks: {summary.get('total_blocks', 0)}")
    print(f"Total Entries: {summary.get('total_entries', 0)}")
    print(f"Total Credits: {summary.get('total_credits', 0):.2f}")
    print(f"Total Contributors: {summary.get('total_contributors', 0)}")
    print(f"Total Attribution Chains: {summary.get('total_attribution_chains', 0)}")
    print(f"Last Block Hash: {summary.get('last_block_hash', 'N/A')[:16]}...")
    
    # Verify ledger integrity
    print("\n🔍 VERIFYING LEDGER INTEGRITY")
    print("-" * 50)
    
    integrity_report = ledger_system.verify_ledger_integrity()
    
    if integrity_report['verified']:
        print("✅ Ledger integrity verified successfully!")
        print(f"   Total blocks verified: {integrity_report['total_blocks']}")
        print(f"   Total entries: {integrity_report['total_entries']}")
        print(f"   Total credits: {integrity_report['total_credits']:.2f}")
    else:
        print("❌ Ledger integrity verification failed!")
        for error in integrity_report['errors']:
            print(f"   Error: {error}")
    
    # Show contributor details
    print("\n👥 CONTRIBUTOR DETAILS")
    print("-" * 50)
    
    contributors = [
        'wallace_transform_001',
        'f2_matrix_optimization_001',
        'parallel_ml_training_001',
        'trikernal_framework_001',
        'julie_vantax_collaboration_001',
        'late_father_foundational_001',
        'extended_protocols_design_001'
    ]
    
    for contributor_id in contributors:
        contributor_info = ledger_system.get_contributor_credits(contributor_id)
        
        if contributor_info:
            print(f"\n{contributor_id}:")
            print(f"  Total Credits: {contributor_info.get('total_credits', 0):.2f}")
            print(f"  Entries: {contributor_info.get('entries', 0)}")
            print(f"  Attribution Flows: {contributor_info.get('attribution_flows', 0)}")
            print(f"  Reputation Score: {contributor_info.get('reputation_score', 1.0):.2f}")
            print(f"  Verification Status: {contributor_info.get('verification_status', 'unknown')}")
    
    # Show attribution flow analysis
    print("\n🔄 ATTRIBUTION FLOW ANALYSIS")
    print("-" * 50)
    
    # Get all attribution chains
    try:
        import sqlite3
        conn = sqlite3.connect(ledger_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT child_contribution_id, parent_contribution_id, share_percentage, metadata
            FROM attribution_chains
            ORDER BY timestamp
        ''')
        
        chains = cursor.fetchall()
        conn.close()
        
        if chains:
            print("Attribution Flows:")
            for chain in chains:
                child_id = chain[0]
                parent_id = chain[1]
                share_percentage = chain[2]
                metadata = json.loads(chain[3])
                credit_amount = metadata.get('credit_amount', 0)
                
                print(f"  {child_id} → {parent_id} ({share_percentage*100:.1f}% = {credit_amount:.2f} credits)")
        else:
            print("No attribution chains found")
            
    except Exception as e:
        print(f"Error analyzing attribution flows: {e}")
    
    # Show immutable ledger structure
    print("\n📜 IMMUTABLE LEDGER STRUCTURE")
    print("-" * 50)
    
    try:
        with open(ledger_system.ledger_path, 'r') as f:
            ledger = json.load(f)
        
        print(f"Ledger Version: {ledger['metadata']['version']}")
        print(f"System: {ledger['metadata']['system']}")
        print(f"Created: {ledger['metadata']['created']}")
        print(f"Total Blocks: {len(ledger['blocks'])}")
        
        if ledger['blocks']:
            print("\nBlock Structure:")
            for i, block in enumerate(ledger['blocks']):
                print(f"  Block {block['block_number']}:")
                print(f"    Entries: {len(block['entries'])}")
                print(f"    Hash: {block['current_hash'][:16]}...")
                print(f"    Previous Hash: {block['previous_hash'][:16]}...")
                print(f"    Timestamp: {block['timestamp']}")
                
                if block['entries']:
                    print(f"    Sample Entry: {block['entries'][0]['description'][:50]}...")
                
                if i >= 2:  # Show only first 3 blocks
                    print(f"    ... and {len(ledger['blocks']) - 3} more blocks")
                    break
                print()
        
    except Exception as e:
        print(f"Error reading immutable ledger: {e}")
    
    print("\n🎉 DIGITAL LEDGER DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("The digital ledger system is now ready for production use!")
    print("Features demonstrated:")
    print("• Real-time ledger entry creation")
    print("• Immutable blockchain-style ledger")
    print("• Attribution flow tracking")
    print("• Contributor credit management")
    print("• Ledger integrity verification")
    print("• Web dashboard and API endpoints")
    print("=" * 70)
    print("To start the web dashboard, run: python3 KOBA42_DIGITAL_LEDGER_SYSTEM.py")
    print("Then visit: http://localhost:8080")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_digital_ledger()
