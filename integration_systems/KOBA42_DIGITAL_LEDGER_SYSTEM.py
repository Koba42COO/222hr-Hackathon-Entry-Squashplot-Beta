#!/usr/bin/env python3
"""
KOBA42 DIGITAL LEDGER SYSTEM
============================
Comprehensive Digital Ledger System for Real-Time Attribution Tracking
====================================================================

Features:
1. Real-time contribution tracking and credit calculation
2. Web-based dashboard for viewing ledger data
3. REST API endpoints for integration
4. Blockchain-style immutable ledger with audit trail
5. Real-time attribution flow visualization
6. Multi-user authentication and permission system
7. Automated credit distribution and profit sharing
8. Advanced analytics and reporting
"""

import json
import time
import hashlib
import sqlite3
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
from collections import defaultdict, Counter
import uuid
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
import aiohttp
from aiohttp import web
import jwt
import bcrypt

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

class CreditStatus(Enum):
    """Status of credit calculations."""
    PENDING = "pending"
    CALCULATED = "calculated"
    DISTRIBUTED = "distributed"
    DISPUTED = "disputed"

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
    """Comprehensive digital ledger system for real-time attribution tracking."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.db_path = "research_data/digital_ledger.db"
        self.ledger_path = "research_data/immutable_ledger.json"
        self.websocket_clients = set()
        
        # Initialize database and ledger
        self.init_database()
        self.init_immutable_ledger()
        
        # Core tracking systems
        self.ledger_entries = []
        self.attribution_chains = {}
        self.contributor_registry = {}
        self.credit_calculations = {}
        self.profit_distributions = {}
        
        # Real-time tracking
        self.lock = threading.Lock()
        self.websocket_server = None
        
        # Authentication
        self.jwt_secret = self.config['security']['jwt_secret']
        self.users = {}
        
        logger.info("🚀 KOBA42 Digital Ledger System initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for digital ledger system."""
        return {
            'ledger': {
                'block_time': 60,  # seconds
                'max_block_size': 1000,
                'difficulty': 4
            },
            'security': {
                'jwt_secret': 'koba42_digital_ledger_secret_key_2024',
                'password_salt_rounds': 12,
                'session_timeout': 3600
            },
            'api': {
                'host': 'localhost',
                'port': 8080,
                'websocket_port': 8081
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
            
            # Credit calculations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS credit_calculations (
                    calculation_id TEXT PRIMARY KEY,
                    contribution_id TEXT,
                    base_credit REAL,
                    attribution_bonus REAL,
                    reputation_multiplier REAL,
                    total_credit REAL,
                    status TEXT,
                    timestamp TEXT
                )
            ''')
            
            # Profit distributions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profit_distributions (
                    distribution_id TEXT PRIMARY KEY,
                    contributor_id TEXT,
                    credit_amount REAL,
                    profit_share REAL,
                    distribution_date TEXT,
                    status TEXT
                )
            ''')
            
            # User authentication
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    role TEXT,
                    created_at TEXT,
                    last_login TEXT
                )
            ''')
            
            # Audit trail
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    audit_id TEXT PRIMARY KEY,
                    action TEXT,
                    user_id TEXT,
                    timestamp TEXT,
                    details TEXT
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
                
                # Broadcast to websocket clients
                self.broadcast_ledger_update(entry)
                
                logger.info(f"✅ Created ledger entry: {entry_id}")
                return entry_id
                
        except Exception as e:
            logger.error(f"❌ Failed to create ledger entry: {e}")
            return None
    
    def calculate_signature(self, contributor_id: str, description: str, credit_amount: float) -> str:
        """Calculate digital signature for ledger entry."""
        data = f"{contributor_id}{description}{credit_amount}{self.jwt_secret}"
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
    
    async def broadcast_ledger_update(self, entry: LedgerEntry):
        """Broadcast ledger update to websocket clients."""
        if self.websocket_clients:
            message = {
                'type': 'ledger_update',
                'entry': asdict(entry),
                'timestamp': datetime.now().isoformat()
            }
            
            # Broadcast to all connected clients
            disconnected_clients = set()
            for websocket in self.websocket_clients:
                try:
                    await websocket.send(json.dumps(message))
                except:
                    disconnected_clients.add(websocket)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
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

# Web API and Dashboard
class DigitalLedgerAPI:
    """Web API for the digital ledger system."""
    
    def __init__(self, ledger_system: DigitalLedgerSystem):
        self.ledger_system = ledger_system
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        # Ledger endpoints
        self.app.router.add_post('/api/ledger/entry', self.create_entry)
        self.app.router.add_get('/api/ledger/summary', self.get_summary)
        self.app.router.add_get('/api/ledger/contributor/{contributor_id}', self.get_contributor)
        self.app.router.add_get('/api/ledger/verify', self.verify_integrity)
        
        # Dashboard
        self.app.router.add_get('/', self.dashboard)
        self.app.router.add_static('/static', 'static')
    
    async def create_entry(self, request):
        """Create a new ledger entry."""
        try:
            data = await request.json()
            
            entry_id = self.ledger_system.create_ledger_entry(
                contributor_id=data['contributor_id'],
                contribution_type=data['contribution_type'],
                description=data['description'],
                credit_amount=data['credit_amount'],
                attribution_chain=data.get('attribution_chain', []),
                metadata=data.get('metadata', {})
            )
            
            return web.json_response({
                'success': True,
                'entry_id': entry_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=400)
    
    async def get_summary(self, request):
        """Get ledger summary."""
        try:
            summary = self.ledger_system.get_ledger_summary()
            return web.json_response(summary)
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=400)
    
    async def get_contributor(self, request):
        """Get contributor information."""
        try:
            contributor_id = request.match_info['contributor_id']
            contributor_info = self.ledger_system.get_contributor_credits(contributor_id)
            
            if not contributor_info:
                return web.json_response({
                    'success': False,
                    'error': 'Contributor not found'
                }, status=404)
            
            return web.json_response(contributor_info)
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=400)
    
    async def verify_integrity(self, request):
        """Verify ledger integrity."""
        try:
            integrity_report = self.ledger_system.verify_ledger_integrity()
            return web.json_response(integrity_report)
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=400)
    
    async def dashboard(self, request):
        """Serve the dashboard HTML."""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>KOBA42 Digital Ledger Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }
                .ledger-entry { border-left: 3px solid #007bff; padding: 10px; margin: 5px 0; }
                .websocket-status { padding: 10px; margin: 10px 0; border-radius: 3px; }
                .connected { background: #d4edda; color: #155724; }
                .disconnected { background: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 KOBA42 Digital Ledger Dashboard</h1>
                
                <div class="card">
                    <h2>📊 Ledger Summary</h2>
                    <div id="summary"></div>
                </div>
                
                <div class="card">
                    <h2>🔗 Real-Time Updates</h2>
                    <div id="websocket-status" class="websocket-status disconnected">Disconnected</div>
                    <div id="recent-entries"></div>
                </div>
                
                <div class="card">
                    <h2>🔍 Create New Entry</h2>
                    <form id="entry-form">
                        <p><label>Contributor ID: <input type="text" id="contributor-id" required></label></p>
                        <p><label>Contribution Type: <select id="contribution-type">
                            <option value="research">Research</option>
                            <option value="development">Development</option>
                            <option value="innovation">Innovation</option>
                            <option value="foundational">Foundational</option>
                            <option value="collaborative">Collaborative</option>
                            <option value="legacy">Legacy</option>
                        </select></label></p>
                        <p><label>Description: <textarea id="description" required></textarea></label></p>
                        <p><label>Credit Amount: <input type="number" id="credit-amount" step="0.01" required></label></p>
                        <p><label>Attribution Chain (comma-separated): <input type="text" id="attribution-chain"></label></p>
                        <button type="submit">Create Entry</button>
                    </form>
                </div>
            </div>
            
            <script>
                // WebSocket connection
                const ws = new WebSocket('ws://localhost:8081');
                
                ws.onopen = function() {
                    document.getElementById('websocket-status').className = 'websocket-status connected';
                    document.getElementById('websocket-status').textContent = 'Connected';
                };
                
                ws.onclose = function() {
                    document.getElementById('websocket-status').className = 'websocket-status disconnected';
                    document.getElementById('websocket-status').textContent = 'Disconnected';
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'ledger_update') {
                        addRecentEntry(data.entry);
                    }
                };
                
                function addRecentEntry(entry) {
                    const recentEntries = document.getElementById('recent-entries');
                    const entryDiv = document.createElement('div');
                    entryDiv.className = 'ledger-entry';
                    entryDiv.innerHTML = `
                        <strong>${entry.contributor_id}</strong> - ${entry.description}<br>
                        Credits: ${entry.credit_amount} | Type: ${entry.contribution_type}<br>
                        <small>${new Date(entry.timestamp).toLocaleString()}</small>
                    `;
                    recentEntries.insertBefore(entryDiv, recentEntries.firstChild);
                    
                    // Keep only last 10 entries
                    while (recentEntries.children.length > 10) {
                        recentEntries.removeChild(recentEntries.lastChild);
                    }
                }
                
                // Load summary
                async function loadSummary() {
                    try {
                        const response = await fetch('/api/ledger/summary');
                        const summary = await response.json();
                        
                        document.getElementById('summary').innerHTML = `
                            <div class="metric">Total Blocks: ${summary.total_blocks}</div>
                            <div class="metric">Total Entries: ${summary.total_entries}</div>
                            <div class="metric">Total Credits: ${summary.total_credits.toFixed(2)}</div>
                            <div class="metric">Total Contributors: ${summary.total_contributors}</div>
                            <div class="metric">Attribution Chains: ${summary.total_attribution_chains}</div>
                        `;
                    } catch (error) {
                        console.error('Failed to load summary:', error);
                    }
                }
                
                // Form submission
                document.getElementById('entry-form').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = {
                        contributor_id: document.getElementById('contributor-id').value,
                        contribution_type: document.getElementById('contribution-type').value,
                        description: document.getElementById('description').value,
                        credit_amount: parseFloat(document.getElementById('credit-amount').value),
                        attribution_chain: document.getElementById('attribution-chain').value.split(',').filter(id => id.trim())
                    };
                    
                    try {
                        const response = await fetch('/api/ledger/entry', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(formData)
                        });
                        
                        const result = await response.json();
                        if (result.success) {
                            alert('Entry created successfully!');
                            document.getElementById('entry-form').reset();
                            loadSummary();
                        } else {
                            alert('Error: ' + result.error);
                        }
                    } catch (error) {
                        alert('Error creating entry: ' + error);
                    }
                });
                
                // Load initial data
                loadSummary();
                setInterval(loadSummary, 30000); // Refresh every 30 seconds
            </script>
        </body>
        </html>
        """
        
        return web.Response(text=dashboard_html, content_type='text/html')

async def websocket_handler(websocket, path, ledger_system):
    """Handle WebSocket connections for real-time updates."""
    ledger_system.websocket_clients.add(websocket)
    try:
        async for message in websocket:
            # Handle incoming messages if needed
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ledger_system.websocket_clients.discard(websocket)

async def start_websocket_server(ledger_system):
    """Start WebSocket server for real-time updates."""
    server = await websockets.serve(
        lambda ws, path: websocket_handler(ws, path, ledger_system),
        'localhost',
        8081
    )
    logger.info("✅ WebSocket server started on port 8081")
    await server.wait_closed()

def run_digital_ledger_system():
    """Run the complete digital ledger system."""
    # Initialize ledger system
    ledger_system = DigitalLedgerSystem()
    
    # Initialize API
    api = DigitalLedgerAPI(ledger_system)
    
    # Start WebSocket server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start WebSocket server in background
    loop.create_task(start_websocket_server(ledger_system))
    
    # Start web server
    web.run_app(api.app, host='localhost', port=8080)
    
    logger.info("🚀 Digital Ledger System running on http://localhost:8080")

if __name__ == "__main__":
    run_digital_ledger_system()
