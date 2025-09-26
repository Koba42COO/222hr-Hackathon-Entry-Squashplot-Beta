#!/usr/bin/env python3
"""
KOBA42 DIGITAL LEDGER SYSTEM (SIMPLIFIED)
=========================================
Simplified Digital Ledger System for Demonstration
=================================================

Features:
1. Real-time contribution tracking and credit calculation
2. Blockchain-style immutable ledger with audit trail
3. Attribution flow tracking
4. Contributor credit management
5. Ledger integrity verification
"""

import json
import time
import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import uuid
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContributionType(Enum):
    """Types of contributions that can be tracked."""
    RESEARCH = "research"
    DEVELOPMENT = "development"
    INNOVATION = "innovation"
    FOUNDATIONAL = "foundational"
    COLLABORATIVE = "collaborative"
    LEGACY = "legacy"

@dataclass
class LedgerEntry:
    """Immutable ledger entry."""
    entry_id: str
    timestamp: str
    contributor_id: str
    contribution_type: str
    description: str
    credit_amount: float
    attribution_chain: List[str]
    metadata: Dict[str, Any]
    signature: str
    previous_hash: str
    current_hash: str

@dataclass
class AttributionChain:
    """Attribution chain linking contributions."""
    chain_id: str
    child_contribution_id: str
    parent_contribution_id: str
    share_percentage: float
    generation: int
    timestamp: str
    metadata: Dict[str, Any]

class DigitalLedgerSystem:
    """Simplified digital ledger system for real-time attribution tracking."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.db_path = "research_data/digital_ledger.db"
        self.ledger_path = "research_data/immutable_ledger.json"
        
        # Initialize database and ledger
        self.init_database()
        self.init_immutable_ledger()
        
        # Core tracking systems
        self.ledger_entries = []
        self.attribution_chains = {}
        self.contributor_registry = {}
        
        # Real-time tracking
        self.lock = threading.Lock()
        
        logger.info("🚀 KOBA42 Digital Ledger System initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for digital ledger system."""
        return {
            'ledger': {
                'block_time': 60,  # seconds
                'max_block_size': 1000,
                'difficulty': 4
            },
            'attribution': {
                'parent_share_base': 0.15,
                'geometric_decay_factor': 0.7,
                'max_generations': 5
            }
        }
    
    def init_database(self):
        """Initialize digital ledger database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Immutable ledger entries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    contributor_id TEXT,
                    contribution_type TEXT,
                    description TEXT,
                    credit_amount REAL,
                    attribution_chain TEXT,
                    metadata TEXT,
                    signature TEXT,
                    previous_hash TEXT,
                    current_hash TEXT,
                    block_number INTEGER
                )
            ''')
            
            # Attribution chains
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attribution_chains (
                    chain_id TEXT PRIMARY KEY,
                    child_contribution_id TEXT,
                    parent_contribution_id TEXT,
                    share_percentage REAL,
                    generation INTEGER,
                    timestamp TEXT,
                    metadata TEXT
                )
            ''')
            
            # Contributor registry
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contributor_registry (
                    contributor_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    reputation_score REAL,
                    verification_status TEXT,
                    total_credits REAL,
                    last_updated TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Digital ledger database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
    
    def init_immutable_ledger(self):
        """Initialize immutable ledger file."""
        try:
            initial_ledger = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'system': 'KOBA42_Digital_Ledger'
                },
                'genesis_block': {
                    'block_number': 0,
                    'timestamp': datetime.now().isoformat(),
                    'previous_hash': '0' * 64,
                    'merkle_root': '0' * 64,
                    'entries': []
                },
                'blocks': [],
                'total_entries': 0,
                'total_credits': 0.0
            }
            
            with open(self.ledger_path, 'w') as f:
                json.dump(initial_ledger, f, indent=2)
            
            logger.info("✅ Immutable ledger initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize immutable ledger: {e}")
    
    def calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def create_ledger_entry(self, contributor_id: str, contribution_type: str, 
                           description: str, credit_amount: float, 
                           attribution_chain: List[str] = None, 
                           metadata: Dict[str, Any] = None) -> str:
        """Create a new immutable ledger entry."""
        try:
            with self.lock:
                # Generate entry ID
                entry_id = f"entry_{hashlib.md5(f'{contributor_id}{time.time()}'.encode()).hexdigest()[:12]}"
                
                # Get previous hash
                with open(self.ledger_path, 'r') as f:
                    ledger = json.load(f)
                
                previous_hash = ledger['blocks'][-1]['current_hash'] if ledger['blocks'] else '0' * 64
                
                # Create entry
                entry = LedgerEntry(
                    entry_id=entry_id,
                    timestamp=datetime.now().isoformat(),
                    contributor_id=contributor_id,
                    contribution_type=contribution_type,
                    description=description,
                    credit_amount=credit_amount,
                    attribution_chain=attribution_chain or [],
                    metadata=metadata or {},
                    signature=self.calculate_signature(contributor_id, description, credit_amount),
                    previous_hash=previous_hash,
                    current_hash=''
                )
                
                # Calculate current hash
                entry_data = f"{entry.entry_id}{entry.timestamp}{entry.contributor_id}{entry.contribution_type}{entry.description}{entry.credit_amount}{entry.previous_hash}"
                entry.current_hash = self.calculate_hash(entry_data)
                
                # Store in database
                self.store_ledger_entry(entry)
                
                # Add to immutable ledger
                self.add_to_immutable_ledger(entry)
                
                # Update contributor registry
                self.update_contributor_registry(contributor_id, credit_amount)
                
                # Calculate attribution flows
                if attribution_chain:
                    self.calculate_attribution_flows(entry_id, attribution_chain, credit_amount)
                
                logger.info(f"✅ Created ledger entry: {entry_id}")
                return entry_id
                
        except Exception as e:
            logger.error(f"❌ Failed to create ledger entry: {e}")
            return None
    
    def calculate_signature(self, contributor_id: str, description: str, credit_amount: float) -> str:
        """Calculate digital signature for ledger entry."""
        data = f"{contributor_id}{description}{credit_amount}koba42_secret_key"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def store_ledger_entry(self, entry: LedgerEntry):
        """Store ledger entry in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ledger_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.entry_id, entry.timestamp, entry.contributor_id, entry.contribution_type,
                entry.description, entry.credit_amount, json.dumps(entry.attribution_chain),
                json.dumps(entry.metadata), entry.signature, entry.previous_hash,
                entry.current_hash, len(self.ledger_entries)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store ledger entry: {e}")
    
    def add_to_immutable_ledger(self, entry: LedgerEntry):
        """Add entry to immutable ledger."""
        try:
            with open(self.ledger_path, 'r') as f:
                ledger = json.load(f)
            
            # Add to current block
            if not ledger['blocks'] or len(ledger['blocks'][-1]['entries']) >= self.config['ledger']['max_block_size']:
                # Create new block
                new_block = {
                    'block_number': len(ledger['blocks']),
                    'timestamp': datetime.now().isoformat(),
                    'previous_hash': ledger['blocks'][-1]['current_hash'] if ledger['blocks'] else '0' * 64,
                    'merkle_root': '0' * 64,  # Simplified for now
                    'entries': []
                }
                ledger['blocks'].append(new_block)
            
            # Add entry to current block
            current_block = ledger['blocks'][-1]
            current_block['entries'].append(asdict(entry))
            
            # Update block hash
            block_data = f"{current_block['block_number']}{current_block['timestamp']}{current_block['previous_hash']}{len(current_block['entries'])}"
            current_block['current_hash'] = self.calculate_hash(block_data)
            
            # Update ledger metadata
            ledger['total_entries'] += 1
            ledger['total_credits'] += entry.credit_amount
            
            # Save updated ledger
            with open(self.ledger_path, 'w') as f:
                json.dump(ledger, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Failed to add to immutable ledger: {e}")
    
    def update_contributor_registry(self, contributor_id: str, credit_amount: float):
        """Update contributor registry with new credits."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current contributor data
            cursor.execute('SELECT * FROM contributor_registry WHERE contributor_id = ?', (contributor_id,))
            result = cursor.fetchone()
            
            if result:
                current_credits = result[5] or 0.0
                new_total_credits = current_credits + credit_amount
                
                cursor.execute('''
                    UPDATE contributor_registry 
                    SET total_credits = ?, last_updated = ?
                    WHERE contributor_id = ?
                ''', (new_total_credits, datetime.now().isoformat(), contributor_id))
            else:
                # Create new contributor entry
                cursor.execute('''
                    INSERT INTO contributor_registry VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    contributor_id, contributor_id, '', 1.0, 'unverified',
                    credit_amount, datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update contributor registry: {e}")
    
    def calculate_attribution_flows(self, entry_id: str, attribution_chain: List[str], credit_amount: float):
        """Calculate attribution flows for parent contributions."""
        try:
            for parent_id in attribution_chain:
                share_percentage = self.config['attribution']['parent_share_base']
                parent_credit = credit_amount * share_percentage
                
                # Create attribution chain entry
                chain_id = f"chain_{hashlib.md5(f'{entry_id}{parent_id}{time.time()}'.encode()).hexdigest()[:12]}"
                
                chain = AttributionChain(
                    chain_id=chain_id,
                    child_contribution_id=entry_id,
                    parent_contribution_id=parent_id,
                    share_percentage=share_percentage,
                    generation=0,
                    timestamp=datetime.now().isoformat(),
                    metadata={'credit_amount': parent_credit}
                )
                
                # Store attribution chain
                self.store_attribution_chain(chain)
                
                # Update parent contributor credits
                self.update_contributor_registry(parent_id, parent_credit)
                
                logger.info(f"✅ Created attribution flow: {entry_id} → {parent_id} ({parent_credit:.2f} credits)")
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate attribution flows: {e}")
    
    def store_attribution_chain(self, chain: AttributionChain):
        """Store attribution chain in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO attribution_chains VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                chain.chain_id, chain.child_contribution_id, chain.parent_contribution_id,
                chain.share_percentage, chain.generation, chain.timestamp,
                json.dumps(chain.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store attribution chain: {e}")
    
    def get_ledger_summary(self) -> Dict[str, Any]:
        """Get comprehensive ledger summary."""
        try:
            with open(self.ledger_path, 'r') as f:
                ledger = json.load(f)
            
            # Get contributor statistics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*), SUM(total_credits) FROM contributor_registry')
            contributor_stats = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM ledger_entries')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM attribution_chains')
            total_chains = cursor.fetchone()[0]
            
            conn.close()
            
            summary = {
                'ledger_metadata': ledger['metadata'],
                'total_blocks': len(ledger['blocks']),
                'total_entries': total_entries,
                'total_credits': ledger['total_credits'],
                'total_contributors': contributor_stats[0] or 0,
                'total_attribution_chains': total_chains,
                'last_block_hash': ledger['blocks'][-1]['current_hash'] if ledger['blocks'] else None,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get ledger summary: {e}")
            return {}
    
    def get_contributor_credits(self, contributor_id: str) -> Dict[str, Any]:
        """Get detailed credit information for a contributor."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get contributor info
            cursor.execute('SELECT * FROM contributor_registry WHERE contributor_id = ?', (contributor_id,))
            contributor = cursor.fetchone()
            
            if not contributor:
                return {}
            
            # Get all entries for this contributor
            cursor.execute('''
                SELECT * FROM ledger_entries 
                WHERE contributor_id = ? 
                ORDER BY timestamp DESC
            ''', (contributor_id,))
            entries = cursor.fetchall()
            
            # Get attribution chains where this contributor is a parent
            cursor.execute('''
                SELECT * FROM attribution_chains 
                WHERE parent_contribution_id = ?
            ''', (contributor_id,))
            attribution_chains = cursor.fetchall()
            
            conn.close()
            
            contributor_info = {
                'contributor_id': contributor[0],
                'name': contributor[1],
                'email': contributor[2],
                'reputation_score': contributor[3],
                'verification_status': contributor[4],
                'total_credits': contributor[5],
                'last_updated': contributor[6],
                'entries': len(entries),
                'attribution_flows': len(attribution_chains),
                'recent_entries': entries[:10] if entries else []
            }
            
            return contributor_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get contributor credits: {e}")
            return {}
    
    def verify_ledger_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the immutable ledger."""
        try:
            with open(self.ledger_path, 'r') as f:
                ledger = json.load(f)
            
            integrity_report = {
                'verified': True,
                'total_blocks': len(ledger['blocks']),
                'total_entries': ledger['total_entries'],
                'total_credits': ledger['total_credits'],
                'block_verification': [],
                'errors': []
            }
            
            previous_hash = '0' * 64
            
            for i, block in enumerate(ledger['blocks']):
                # Verify block hash
                block_data = f"{block['block_number']}{block['timestamp']}{block['previous_hash']}{len(block['entries'])}"
                calculated_hash = self.calculate_hash(block_data)
                
                if calculated_hash != block['current_hash']:
                    integrity_report['verified'] = False
                    integrity_report['errors'].append(f"Block {i} hash mismatch")
                
                # Verify previous hash chain
                if block['previous_hash'] != previous_hash:
                    integrity_report['verified'] = False
                    integrity_report['errors'].append(f"Block {i} previous hash mismatch")
                
                previous_hash = block['current_hash']
                
                integrity_report['block_verification'].append({
                    'block_number': block['block_number'],
                    'hash_valid': calculated_hash == block['current_hash'],
                    'previous_hash_valid': block['previous_hash'] == previous_hash,
                    'entry_count': len(block['entries'])
                })
            
            return integrity_report
            
        except Exception as e:
            logger.error(f"❌ Failed to verify ledger integrity: {e}")
            return {'verified': False, 'errors': [str(e)]}
