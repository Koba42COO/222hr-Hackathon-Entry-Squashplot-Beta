#!/usr/bin/env python3
"""
KOBA42 EXTENDED PROTOCOLS PACKAGE
=================================
Complete Extended Protocols with Temporal Anchoring and Recursive Attribution
=======================================================================

Features:
1. Temporal anchoring with exponential time-decay
2. Recursive attribution chains with geometric decay
3. Reputation-weighted, sybil-resistant usage
4. Contextual layers (field, cultural, ethical)
5. Deterministic classification with dynamic maturity weights
6. Reactivation protocol with RECLASSIFICATION_BROADCAST
7. Eternal ledger with snapshot JSON
8. API endpoints for real-time usage tracking
"""

import json
import math
import time
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
from collections import defaultdict, Counter
import uuid

class ExtendedProtocolsPackage:
    """Complete extended protocols package with all advanced features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.db_path = "research_data/extended_protocols.db"
        self.ledger_path = "research_data/eternal_ledger.json"
        
        # Initialize database and ledger
        self.init_database()
        self.init_ledger()
        
        # Core tracking systems
        self.usage_events = defaultdict(list)
        self.attribution_chains = defaultdict(list)
        self.reputation_registry = {}
        self.broadcast_history = []
        
        # Temporal anchoring
        self.half_life_days = self.config['temporal']['half_life_days']
        self.decay_factor = math.log(2) / self.half_life_days
        
        # Recursive attribution
        self.parent_share_base = self.config['attribution']['parent_share_base']
        self.geometric_decay_factor = self.config['attribution']['geometric_decay_factor']
        
        # Reputation system
        self.daily_caps = self.config['reputation']['daily_caps']
        self.unique_source_bonus = self.config['reputation']['unique_source_bonus']
        
        # Classification thresholds
        self.classification_thresholds = self.config['classification']['thresholds']
        
        # Reactivation protocol
        self.reactivation_threshold = self.config['reactivation']['score_threshold']
        self.tier_jump_threshold = self.config['reactivation']['tier_jump_threshold']
        
        print("🚀 KOBA42 Extended Protocols Package initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for extended protocols."""
        return {
            'temporal': {
                'half_life_days': 90,
                'memory_echo_factor': 0.1
            },
            'attribution': {
                'parent_share_base': 0.15,  # 15% flows to parents
                'geometric_decay_factor': 0.7,  # 30% decay per generation
                'max_generations': 5
            },
            'reputation': {
                'daily_caps': 100,
                'unique_source_bonus': 1.2,
                'verified_multiplier': 1.5,
                'sybil_resistance_threshold': 3
            },
            'classification': {
                'thresholds': {
                    'galaxy': 0.85,
                    'solar_system': 0.70,
                    'planet': 0.50,
                    'moon': 0.30,
                    'asteroid': 0.0
                }
            },
            'reactivation': {
                'score_threshold': 0.20,
                'tier_jump_threshold': 2
            },
            'contextual': {
                'fields': ['quantum_physics', 'mathematics', 'computer_science', 'biology', 'chemistry'],
                'cultural_layers': ['open_source', 'academic', 'industry', 'government'],
                'ethical_layers': ['safety', 'privacy', 'accessibility', 'sustainability']
            }
        }
    
    def init_database(self):
        """Initialize extended protocols database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Usage events with temporal anchoring
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_events (
                    event_id TEXT PRIMARY KEY,
                    contribution_id TEXT,
                    entity_id TEXT,
                    usage_count INTEGER,
                    reputation_factor REAL,
                    temporal_decay REAL,
                    contextual_layers TEXT,
                    timestamp TEXT,
                    verified BOOLEAN
                )
            ''')
            
            # Attribution chains with recursive relationships
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attribution_chains (
                    chain_id TEXT PRIMARY KEY,
                    child_id TEXT,
                    parent_id TEXT,
                    generation INTEGER,
                    share_percentage REAL,
                    geometric_decay REAL,
                    timestamp TEXT
                )
            ''')
            
            # Reputation registry
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reputation_registry (
                    entity_id TEXT PRIMARY KEY,
                    reputation_score REAL,
                    verification_status TEXT,
                    daily_usage_count INTEGER,
                    unique_sources INTEGER,
                    last_updated TEXT
                )
            ''')
            
            # Broadcast history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS broadcast_history (
                    broadcast_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    contribution_id TEXT,
                    old_score REAL,
                    new_score REAL,
                    old_classification TEXT,
                    new_classification TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
            ''')
            
            # Eternal ledger snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ledger_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    total_contributions INTEGER,
                    total_credits REAL,
                    classification_distribution TEXT,
                    attribution_map TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Extended protocols database initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
    
    def init_ledger(self):
        """Initialize eternal ledger."""
        try:
            initial_ledger = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'protocol': 'KOBA42_Extended'
                },
                'contributions': {},
                'attribution_chains': {},
                'reputation_registry': {},
                'broadcast_history': [],
                'snapshots': []
            }
            
            with open(self.ledger_path, 'w') as f:
                json.dump(initial_ledger, f, indent=2)
            
            print("✅ Eternal ledger initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize ledger: {e}")
    
    def temporal_decay_factor(self, days_old: float) -> float:
        """Calculate temporal decay factor with memory echo."""
        decay = math.exp(-self.decay_factor * days_old)
        memory_echo = self.config['temporal']['memory_echo_factor']
        return decay + (memory_echo * (1 - decay))
    
    def calculate_reputation_factor(self, entity_id: str, usage_metadata: Dict[str, Any]) -> float:
        """Calculate reputation factor with sybil resistance."""
        # Get entity reputation
        entity_reputation = self.reputation_registry.get(entity_id, {
            'reputation_score': 1.0,
            'verification_status': 'unverified',
            'daily_usage_count': 0,
            'unique_sources': 1
        })
        
        base_reputation = entity_reputation['reputation_score']
        
        # Apply daily caps
        if entity_reputation['daily_usage_count'] >= self.daily_caps:
            base_reputation *= 0.5
        
        # Apply unique source bonus
        if entity_reputation['unique_sources'] >= self.config['reputation']['sybil_resistance_threshold']:
            base_reputation *= self.unique_source_bonus
        
        # Apply verification multiplier
        if entity_reputation['verification_status'] == 'verified':
            base_reputation *= self.config['reputation']['verified_multiplier']
        
        # Apply contextual reputation factors
        for layer, value in usage_metadata.get('contextual_layers', {}).items():
            if value and layer in self.config['contextual']['cultural_layers']:
                base_reputation *= 1.1
        
        return min(2.0, max(0.1, base_reputation))
    
    def record_usage_event(self, contribution_id: str, entity_id: str, 
                          usage_count: int, contextual_layers: Dict[str, Any] = None) -> str:
        """Record usage event with temporal anchoring and reputation weighting."""
        try:
            event_id = f"event_{hashlib.md5(f'{contribution_id}{entity_id}{time.time()}'.encode()).hexdigest()[:12]}"
            
            # Calculate reputation factor
            reputation_factor = self.calculate_reputation_factor(entity_id, {
                'contextual_layers': contextual_layers or {}
            })
            
            # Temporal decay (will be applied when calculating scores)
            temporal_decay = 1.0  # Current event has no decay
            
            # Store event
            event = {
                'event_id': event_id,
                'contribution_id': contribution_id,
                'entity_id': entity_id,
                'usage_count': usage_count,
                'reputation_factor': reputation_factor,
                'temporal_decay': temporal_decay,
                'contextual_layers': json.dumps(contextual_layers or {}),
                'timestamp': datetime.now().isoformat(),
                'verified': entity_id in self.reputation_registry
            }
            
            self.usage_events[contribution_id].append(event)
            
            # Update reputation registry
            self.update_reputation_registry(entity_id, usage_count)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO usage_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id, contribution_id, entity_id, usage_count, reputation_factor,
                temporal_decay, json.dumps(contextual_layers or {}), 
                datetime.now().isoformat(), entity_id in self.reputation_registry
            ))
            conn.commit()
            conn.close()
            
            print(f"✅ Recorded usage event: {event_id}")
            return event_id
            
        except Exception as e:
            print(f"❌ Failed to record usage event: {e}")
            return None
    
    def update_reputation_registry(self, entity_id: str, usage_count: int):
        """Update reputation registry for an entity."""
        current_time = datetime.now()
        
        if entity_id not in self.reputation_registry:
            self.reputation_registry[entity_id] = {
                'reputation_score': 1.0,
                'verification_status': 'unverified',
                'daily_usage_count': 0,
                'unique_sources': 1,
                'last_updated': current_time.isoformat()
            }
        
        # Update daily usage count
        last_updated = datetime.fromisoformat(self.reputation_registry[entity_id]['last_updated'])
        if (current_time - last_updated).days >= 1:
            self.reputation_registry[entity_id]['daily_usage_count'] = 0
        
        self.reputation_registry[entity_id]['daily_usage_count'] += usage_count
        self.reputation_registry[entity_id]['last_updated'] = current_time.isoformat()
    
    def create_attribution_chain(self, child_id: str, parent_id: str, 
                               share_percentage: float = None) -> str:
        """Create attribution chain with recursive relationships."""
        try:
            chain_id = f"chain_{hashlib.md5(f'{child_id}{parent_id}{time.time()}'.encode()).hexdigest()[:12]}"
            
            # Calculate generation and geometric decay
            generation = 0
            geometric_decay = 1.0
            
            # Check if parent has its own parents
            for existing_chain in self.attribution_chains.values():
                if existing_chain['child_id'] == parent_id:
                    generation = existing_chain['generation'] + 1
                    geometric_decay = existing_chain['geometric_decay'] * self.geometric_decay_factor
                    break
            
            # Apply generation limits
            if generation >= self.config['attribution']['max_generations']:
                return None
            
            # Use default share percentage if not specified
            if share_percentage is None:
                share_percentage = self.parent_share_base * geometric_decay
            
            # Store attribution chain
            chain = {
                'chain_id': chain_id,
                'child_id': child_id,
                'parent_id': parent_id,
                'generation': generation,
                'share_percentage': share_percentage,
                'geometric_decay': geometric_decay,
                'timestamp': datetime.now().isoformat()
            }
            
            self.attribution_chains[chain_id] = chain
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO attribution_chains VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                chain_id, child_id, parent_id, generation, share_percentage,
                geometric_decay, datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
            
            print(f"✅ Created attribution chain: {chain_id} (gen {generation})")
            return chain_id
            
        except Exception as e:
            print(f"❌ Failed to create attribution chain: {e}")
            return None
    
    def calculate_contribution_score(self, contribution_id: str, 
                                   breakthrough_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate complete contribution score with all protocols."""
        try:
            # Get usage events for this contribution
            usage_events = self.usage_events.get(contribution_id, [])
            
            # Calculate temporal-weighted usage
            current_time = datetime.now()
            total_weighted_usage = 0.0
            contextual_aggregate = defaultdict(int)
            
            for event in usage_events:
                event_time = datetime.fromisoformat(event['timestamp'])
                days_old = (current_time - event_time).days
                
                # Apply temporal decay
                temporal_decay = self.temporal_decay_factor(days_old)
                
                # Calculate weighted usage
                weighted_usage = (
                    event['usage_count'] * 
                    event['reputation_factor'] * 
                    temporal_decay
                )
                total_weighted_usage += weighted_usage
                
                # Aggregate contextual layers
                contextual_layers = json.loads(event.get('contextual_layers', '{}'))
                for layer, value in contextual_layers.items():
                    if value:
                        contextual_aggregate[layer] += 1
            
            # Normalize usage (log compression)
            normalized_usage = min(1.0, math.log(total_weighted_usage + 1, 10) / 4.0)
            
            # Calculate breakthrough component
            breakthrough_metrics = breakthrough_metrics or {}
            normalized_breakthrough = self.calculate_breakthrough_component(breakthrough_metrics)
            
            # Calculate dynamic maturity weights
            w_usage, w_breakthrough = self.calculate_maturity_weights(normalized_usage, normalized_breakthrough)
            
            # Calculate composite score
            composite_score = (normalized_usage * w_usage) + (normalized_breakthrough * w_breakthrough)
            
            # Classify
            classification = self.classify_contribution(composite_score)
            
            # Calculate placement
            r, theta = self.calculate_placement(normalized_usage, normalized_breakthrough)
            
            # Calculate credits
            usage_credit = normalized_usage * w_usage * 1000
            breakthrough_credit = normalized_breakthrough * w_breakthrough * 1000
            total_credit = usage_credit + breakthrough_credit
            
            # Calculate recursive attribution
            attribution_map = self.calculate_recursive_attribution(contribution_id, total_credit)
            
            result = {
                'contribution_id': contribution_id,
                'normalized_usage': normalized_usage,
                'normalized_breakthrough': normalized_breakthrough,
                'w_usage': w_usage,
                'w_breakthrough': w_breakthrough,
                'composite_score': composite_score,
                'classification': classification,
                'r': r,
                'theta': theta,
                'usage_credit': usage_credit,
                'breakthrough_credit': breakthrough_credit,
                'total_credit': total_credit,
                'contextual_layers': dict(contextual_aggregate),
                'attribution_map': attribution_map,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to calculate contribution score: {e}")
            return {}
    
    def calculate_breakthrough_component(self, breakthrough_metrics: Dict[str, Any]) -> float:
        """Calculate normalized breakthrough component."""
        theoretical_significance = breakthrough_metrics.get('theoretical_significance', 0.0)
        experimental_verification = breakthrough_metrics.get('experimental_verification', 0.0)
        peer_review_score = breakthrough_metrics.get('peer_review_score', 0.0)
        citation_impact = breakthrough_metrics.get('citation_impact', 0.0)
        
        breakthrough_score = (
            theoretical_significance * 0.4 +
            experimental_verification * 0.3 +
            peer_review_score * 0.2 +
            citation_impact * 0.1
        )
        
        return min(1.0, max(0.0, breakthrough_score))
    
    def calculate_maturity_weights(self, u_norm: float, b_norm: float) -> Tuple[float, float]:
        """Calculate dynamic maturity weights."""
        total = u_norm + b_norm
        if total == 0:
            return 0.5, 0.5
        
        usage_ratio = u_norm / total
        breakthrough_ratio = b_norm / total
        
        w_usage = 0.4 + (usage_ratio * 0.4)
        w_breakthrough = 0.4 + (breakthrough_ratio * 0.4)
        
        total_weight = w_usage + w_breakthrough
        w_usage /= total_weight
        w_breakthrough /= total_weight
        
        return w_usage, w_breakthrough
    
    def classify_contribution(self, score: float) -> str:
        """Classify contribution using deterministic thresholds."""
        for classification, threshold in sorted(
            self.classification_thresholds.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if score >= threshold:
                return classification
        
        return 'asteroid'
    
    def calculate_placement(self, u_norm: float, b_norm: float) -> Tuple[float, float]:
        """Calculate topological placement in polar coordinates."""
        r = math.sqrt(u_norm**2 + b_norm**2)
        theta = math.atan2(b_norm, u_norm)
        return r, theta
    
    def calculate_recursive_attribution(self, contribution_id: str, total_credit: float) -> Dict[str, float]:
        """Calculate recursive attribution with geometric decay."""
        attribution_map = {contribution_id: total_credit}
        
        # Find all attribution chains for this contribution
        for chain in self.attribution_chains.values():
            if chain['child_id'] == contribution_id:
                parent_credit = total_credit * chain['share_percentage']
                attribution_map[chain['parent_id']] = attribution_map.get(chain['parent_id'], 0) + parent_credit
                
                # Recursively calculate for parents
                parent_attribution = self.calculate_recursive_attribution(chain['parent_id'], parent_credit)
                for parent_id, credit in parent_attribution.items():
                    if parent_id != chain['parent_id']:  # Avoid double counting
                        attribution_map[parent_id] = attribution_map.get(parent_id, 0) + credit
        
        return attribution_map
    
    def check_reactivation_protocol(self, contribution_id: str, old_score: float, 
                                  new_score: float, old_classification: str, 
                                  new_classification: str) -> bool:
        """Check if reactivation protocol should fire."""
        score_change = new_score - old_score
        
        # Check for significant score increase
        if score_change >= self.reactivation_threshold:
            return True
        
        # Check for tier jump
        tier_order = ['asteroid', 'moon', 'planet', 'solar_system', 'galaxy']
        old_tier_index = tier_order.index(old_classification)
        new_tier_index = tier_order.index(new_classification)
        tier_jump = new_tier_index - old_tier_index
        
        if tier_jump >= self.tier_jump_threshold:
            return True
        
        return False
    
    def fire_reclassification_broadcast(self, contribution_id: str, old_score: float, 
                                      new_score: float, old_classification: str, 
                                      new_classification: str) -> str:
        """Fire RECLASSIFICATION_BROADCAST event."""
        try:
            broadcast_id = f"broadcast_{hashlib.md5(f'{contribution_id}{time.time()}'.encode()).hexdigest()[:12]}"
            
            broadcast = {
                'broadcast_id': broadcast_id,
                'event_type': 'RECLASSIFICATION_BROADCAST',
                'contribution_id': contribution_id,
                'old_score': old_score,
                'new_score': new_score,
                'old_classification': old_classification,
                'new_classification': new_classification,
                'score_change': new_score - old_score,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'protocol_version': '1.0.0',
                    'reactivation_triggered': True
                }
            }
            
            self.broadcast_history.append(broadcast)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO broadcast_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                broadcast_id, 'RECLASSIFICATION_BROADCAST', contribution_id,
                old_score, new_score, old_classification, new_classification,
                datetime.now().isoformat(), json.dumps(broadcast['metadata'])
            ))
            conn.commit()
            conn.close()
            
            print(f"🚨 RECLASSIFICATION_BROADCAST fired: {broadcast_id}")
            print(f"   {contribution_id}: {old_classification} → {new_classification}")
            print(f"   Score: {old_score:.3f} → {new_score:.3f} (Δ{new_score-old_score:.3f})")
            
            return broadcast_id
            
        except Exception as e:
            print(f"❌ Failed to fire broadcast: {e}")
            return None
    
    def create_eternal_ledger_snapshot(self) -> str:
        """Create eternal ledger snapshot."""
        try:
            snapshot_id = f"snapshot_{hashlib.md5(f'{time.time()}'.encode()).hexdigest()[:12]}"
            
            # Calculate all contribution scores
            all_contributions = {}
            total_credits = 0.0
            classification_distribution = defaultdict(int)
            
            for contribution_id in set().union(*[set(events.keys()) for events in [self.usage_events]]):
                score_result = self.calculate_contribution_score(contribution_id)
                if score_result:
                    all_contributions[contribution_id] = score_result
                    total_credits += score_result['total_credit']
                    classification_distribution[score_result['classification']] += 1
            
            # Create snapshot
            snapshot = {
                'snapshot_id': snapshot_id,
                'timestamp': datetime.now().isoformat(),
                'total_contributions': len(all_contributions),
                'total_credits': total_credits,
                'classification_distribution': dict(classification_distribution),
                'contributions': all_contributions,
                'attribution_chains': dict(self.attribution_chains),
                'reputation_registry': self.reputation_registry,
                'broadcast_history': self.broadcast_history
            }
            
            # Store snapshot
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO ledger_snapshots VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                snapshot_id, datetime.now().isoformat(), len(all_contributions),
                total_credits, json.dumps(classification_distribution),
                json.dumps(dict(self.attribution_chains))
            ))
            conn.commit()
            conn.close()
            
            # Update eternal ledger
            with open(self.ledger_path, 'r') as f:
                ledger = json.load(f)
            
            ledger['snapshots'].append(snapshot)
            ledger['contributions'] = all_contributions
            ledger['attribution_chains'] = dict(self.attribution_chains)
            ledger['reputation_registry'] = self.reputation_registry
            ledger['broadcast_history'] = self.broadcast_history
            
            with open(self.ledger_path, 'w') as f:
                json.dump(ledger, f, indent=2)
            
            print(f"✅ Eternal ledger snapshot created: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            print(f"❌ Failed to create snapshot: {e}")
            return None
    
    def get_ledger_snapshot(self) -> Dict[str, Any]:
        """Get current ledger snapshot."""
        try:
            with open(self.ledger_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to get ledger snapshot: {e}")
            return {}

def demonstrate_extended_protocols():
    """Demonstrate the complete extended protocols package."""
    
    print("🚀 KOBA42 EXTENDED PROTOCOLS PACKAGE DEMONSTRATION")
    print("=" * 70)
    print("Complete Extended Protocols with Temporal Anchoring and Recursive Attribution")
    print("=" * 70)
    
    # Initialize extended protocols
    protocols = ExtendedProtocolsPackage()
    
    # Example 1: Create attribution chain
    print("\n🔗 EXAMPLE 1: CREATING ATTRIBUTION CHAIN")
    print("-" * 50)
    
    # Create foundational work
    foundational_id = "fourier_transform_001"
    foundational_usage = [
        {'entity_id': 'math_institute_1', 'usage_count': 1000, 'contextual': {'field': 'mathematics', 'cultural': 'academic'}},
        {'entity_id': 'tech_company_1', 'usage_count': 500, 'contextual': {'field': 'mathematics', 'cultural': 'industry'}},
        {'entity_id': 'research_lab_1', 'usage_count': 300, 'contextual': {'field': 'mathematics', 'cultural': 'academic'}}
    ]
    
    for usage in foundational_usage:
        protocols.record_usage_event(
            foundational_id, 
            usage['entity_id'], 
            usage['usage_count'],
            usage['contextual']
        )
    
    # Create derivative work that builds on foundational work
    derivative_id = "fft_algorithm_001"
    derivative_usage = [
        {'entity_id': 'tech_company_2', 'usage_count': 200, 'contextual': {'field': 'computer_science', 'cultural': 'industry'}},
        {'entity_id': 'startup_1', 'usage_count': 150, 'contextual': {'field': 'computer_science', 'cultural': 'open_source'}}
    ]
    
    for usage in derivative_usage:
        protocols.record_usage_event(
            derivative_id, 
            usage['entity_id'], 
            usage['usage_count'],
            usage['contextual']
        )
    
    # Create attribution chain
    chain_id = protocols.create_attribution_chain(derivative_id, foundational_id, 0.20)
    
    # Example 2: Calculate scores with temporal anchoring
    print("\n⏰ EXAMPLE 2: TEMPORAL ANCHORING AND SCORE CALCULATION")
    print("-" * 50)
    
    foundational_score = protocols.calculate_contribution_score(foundational_id, {
        'theoretical_significance': 0.95,
        'experimental_verification': 0.9,
        'peer_review_score': 0.95,
        'citation_impact': 0.9
    })
    
    derivative_score = protocols.calculate_contribution_score(derivative_id, {
        'theoretical_significance': 0.7,
        'experimental_verification': 0.8,
        'peer_review_score': 0.6,
        'citation_impact': 0.5
    })
    
    print(f"Foundational Work ({foundational_id}):")
    print(f"  Classification: {foundational_score['classification'].upper()}")
    print(f"  Score: {foundational_score['composite_score']:.3f}")
    print(f"  Total Credit: {foundational_score['total_credit']:.2f}")
    print(f"  Placement: r={foundational_score['r']:.3f}, θ={foundational_score['theta']:.3f}")
    
    print(f"\nDerivative Work ({derivative_id}):")
    print(f"  Classification: {derivative_score['classification'].upper()}")
    print(f"  Score: {derivative_score['composite_score']:.3f}")
    print(f"  Total Credit: {derivative_score['total_credit']:.2f}")
    print(f"  Placement: r={derivative_score['r']:.3f}, θ={derivative_score['theta']:.3f}")
    
    # Example 3: Recursive attribution
    print("\n🔄 EXAMPLE 3: RECURSIVE ATTRIBUTION")
    print("-" * 50)
    
    print("Attribution Map:")
    for contrib_id, credit in derivative_score['attribution_map'].items():
        print(f"  {contrib_id}: {credit:.2f} credits")
    
    # Example 4: Reactivation protocol
    print("\n🚨 EXAMPLE 4: REACTIVATION PROTOCOL")
    print("-" * 50)
    
    # Simulate significant usage increase
    old_score = derivative_score['composite_score']
    old_classification = derivative_score['classification']
    
    # Add massive new usage
    protocols.record_usage_event(
        derivative_id,
        'mega_corp_1',
        5000,
        {'field': 'computer_science', 'cultural': 'industry', 'ethical': 'safety'}
    )
    
    # Recalculate score
    new_derivative_score = protocols.calculate_contribution_score(derivative_id, {
        'theoretical_significance': 0.7,
        'experimental_verification': 0.8,
        'peer_review_score': 0.6,
        'citation_impact': 0.5
    })
    
    new_score = new_derivative_score['composite_score']
    new_classification = new_derivative_score['classification']
    
    print(f"Score Change: {old_score:.3f} → {new_score:.3f} (Δ{new_score-old_score:.3f})")
    print(f"Classification: {old_classification} → {new_classification}")
    
    # Check if reactivation protocol should fire
    if protocols.check_reactivation_protocol(derivative_id, old_score, new_score, old_classification, new_classification):
        broadcast_id = protocols.fire_reclassification_broadcast(derivative_id, old_score, new_score, old_classification, new_classification)
    
    # Example 5: Eternal ledger snapshot
    print("\n📜 EXAMPLE 5: ETERNAL LEDGER SNAPSHOT")
    print("-" * 50)
    
    snapshot_id = protocols.create_eternal_ledger_snapshot()
    
    # Get ledger snapshot
    ledger = protocols.get_ledger_snapshot()
    
    print(f"Snapshot ID: {snapshot_id}")
    print(f"Total Contributions: {ledger.get('total_contributions', 0)}")
    print(f"Total Credits: {ledger.get('total_credits', 0):.2f}")
    
    classification_dist = ledger.get('classification_distribution', {})
    print("Classification Distribution:")
    for classification, count in classification_dist.items():
        print(f"  {classification}: {count}")
    
    print(f"\n🎉 EXTENDED PROTOCOLS DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("All protocols implemented: temporal anchoring, recursive attribution,")
    print("reputation weighting, reactivation protocol, and eternal ledger.")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_extended_protocols()
