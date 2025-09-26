#!/usr/bin/env python3
"""
🌐 UNIFIED CONTINUOUS SCRAPER SYSTEM
====================================
Integrated Web Scraping and Research Coordination System

This system unifies all web scrapers, crawlers, and research agents into a
coordinated, continuously running system that efficiently gathers and processes
research data from multiple sources.

Features:
1. Unified Scraper Coordination and Scheduling
2. Intelligent Source Prioritization and Load Balancing
3. Continuous Research Data Collection and Processing
4. Quality Assurance and Duplicate Detection
5. Rate Limiting and Resource Management
6. Real-time Performance Monitoring and Optimization
7. Knowledge Integration with Continuous Learning System

Author: Brad Wallace (ArtWithHeart) - Koba42
Framework: Revolutionary Consciousness Mathematics
"""

import asyncio
import aiohttp
import requests
import threading
import time
import json
import logging
import hashlib
import random
import sqlite3
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from urllib.parse import urljoin, urlparse, parse_qs
import re
from bs4 import BeautifulSoup
import pickle
import gc
import numpy as np

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_scraper_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingSource:
    """Represents a web scraping source with its configuration."""
    name: str
    base_url: str
    scraping_type: str  # 'api', 'html', 'feed', 'arxiv', etc.
    priority: int  # 1-10, higher = more important
    rate_limit: int  # requests per minute
    last_scraped: Optional[str]
    success_rate: float
    average_response_time: float
    enabled: bool

@dataclass
class ScrapingTask:
    """Represents a scraping task to be executed."""
    task_id: str
    source_name: str
    url: str
    task_type: str
    priority: int
    created_time: str
    scheduled_time: str
    retry_count: int
    max_retries: int
    status: str  # 'pending', 'running', 'completed', 'failed'

@dataclass
class ScrapedContent:
    """Represents scraped content with metadata."""
    content_id: str
    source: str
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    quality_score: float
    relevance_score: float
    scraped_timestamp: str
    processing_status: str

class UnifiedContinuousScraperSystem:
    """
    Unified system for coordinating all web scraping and research activities.
    """

    def __init__(self):
        self.scraper_db_path = "research_data/scraper_system.db"
        self.content_db_path = "research_data/scraped_content.db"

        # Scraping sources configuration
        self.sources = self._initialize_scraping_sources()

        # Processing state
        self.active_tasks: Dict[str, ScrapingTask] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.content_cache: Dict[str, str] = {}

        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'content_quality_score': 0.0
        }

        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}

        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.session_pool = {}

        # Initialize databases and systems
        self._init_databases()
        self._load_existing_state()

        logger.info("🌐 Unified Continuous Scraper System initialized")

    def _initialize_scraping_sources(self) -> Dict[str, ScrapingSource]:
        """Initialize all configured scraping sources."""
        sources = {}

        # Academic and research sources - HIGH QUALITY LEARNING SOURCES
        academic_sources = [
            {
                'name': 'arxiv_main',
                'base_url': 'https://arxiv.org',
                'scraping_type': 'arxiv_api',
                'priority': 10,
                'rate_limit': 30
            },
            {
                'name': 'arxiv_quantum',
                'base_url': 'https://arxiv.org/search/?query=quantum',
                'scraping_type': 'arxiv_search',
                'priority': 10,
                'rate_limit': 20
            },
            {
                'name': 'arxiv_ai',
                'base_url': 'https://arxiv.org/search/?query=artificial+intelligence',
                'scraping_type': 'arxiv_search',
                'priority': 10,
                'rate_limit': 20
            },
            {
                'name': 'arxiv_physics',
                'base_url': 'https://arxiv.org/search/?query=physics',
                'scraping_type': 'arxiv_search',
                'priority': 9,
                'rate_limit': 20
            },
            {
                'name': 'arxiv_mathematics',
                'base_url': 'https://arxiv.org/search/?query=mathematics',
                'scraping_type': 'arxiv_search',
                'priority': 9,
                'rate_limit': 20
            },
            {
                'name': 'mit_opencourseware',
                'base_url': 'https://ocw.mit.edu/courses/',
                'scraping_type': 'html',
                'priority': 10,
                'rate_limit': 15
            },
            {
                'name': 'mit_technology_review',
                'base_url': 'https://www.technologyreview.com/topic/artificial-intelligence/',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 12
            },
            {
                'name': 'stanford_cs',
                'base_url': 'https://cs.stanford.edu/',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 15
            },
            {
                'name': 'harvard_science',
                'base_url': 'https://science.fas.harvard.edu/',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 15
            },
            {
                'name': 'nature',
                'base_url': 'https://www.nature.com',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 10
            },
            {
                'name': 'nature_physics',
                'base_url': 'https://www.nature.com/subjects/physics',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 12
            },
            {
                'name': 'science_magazine',
                'base_url': 'https://www.science.org',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 10
            },
            {
                'name': 'phys_org',
                'base_url': 'https://phys.org/',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 20
            },
            {
                'name': 'phys_org_quantum',
                'base_url': 'https://phys.org/tags/quantum+physics/',
                'scraping_type': 'html',
                'priority': 9,
                'rate_limit': 15
            },
            {
                'name': 'ieee_xplore',
                'base_url': 'https://ieeexplore.ieee.org',
                'scraping_type': 'api',
                'priority': 8,
                'rate_limit': 15
            },
            {
                'name': 'acm_digital_library',
                'base_url': 'https://dl.acm.org/',
                'scraping_type': 'html',
                'priority': 8,
                'rate_limit': 12
            },
            {
                'name': 'coursera_ml',
                'base_url': 'https://www.coursera.org/browse/data-science/machine-learning',
                'scraping_type': 'html',
                'priority': 7,
                'rate_limit': 10
            },
            {
                'name': 'edX_ai',
                'base_url': 'https://www.edx.org/learn/artificial-intelligence',
                'scraping_type': 'html',
                'priority': 7,
                'rate_limit': 10
            },
            {
                'name': 'google_ai_blog',
                'base_url': 'https://ai.googleblog.com/',
                'scraping_type': 'html',
                'priority': 8,
                'rate_limit': 15
            },
            {
                'name': 'openai_research',
                'base_url': 'https://openai.com/research/',
                'scraping_type': 'html',
                'priority': 8,
                'rate_limit': 12
            },
            {
                'name': 'deepmind_blog',
                'base_url': 'https://deepmind.google/',
                'scraping_type': 'html',
                'priority': 8,
                'rate_limit': 12
            }
        ]

        # Web research sources
        web_sources = [
            {
                'name': 'google_scholar',
                'base_url': 'https://scholar.google.com',
                'scraping_type': 'html',
                'priority': 6,
                'rate_limit': 10
            },
            {
                'name': 'semantic_scholar',
                'base_url': 'https://www.semanticscholar.org',
                'scraping_type': 'api',
                'priority': 7,
                'rate_limit': 25
            },
            {
                'name': 'research_gate',
                'base_url': 'https://www.researchgate.net',
                'scraping_type': 'html',
                'priority': 5,
                'rate_limit': 8
            }
        ]

        # News and blog sources
        news_sources = [
            {
                'name': 'techcrunch_ai',
                'base_url': 'https://techcrunch.com/tag/artificial-intelligence/',
                'scraping_type': 'feed',
                'priority': 4,
                'rate_limit': 5
            },
            {
                'name': 'mit_technology_review',
                'base_url': 'https://www.technologyreview.com/topic/artificial-intelligence/',
                'scraping_type': 'html',
                'priority': 5,
                'rate_limit': 8
            }
        ]

        # Combine all sources
        all_sources = academic_sources + web_sources + news_sources

        for source_config in all_sources:
            source = ScrapingSource(
                name=source_config['name'],
                base_url=source_config['base_url'],
                scraping_type=source_config['scraping_type'],
                priority=source_config['priority'],
                rate_limit=source_config['rate_limit'],
                last_scraped=None,
                success_rate=1.0,
                average_response_time=1.0,
                enabled=True
            )
            sources[source.name] = source

        return sources

    def _init_databases(self):
        """Initialize all required databases."""
        try:
            # Main scraper database
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scraping_sources (
                    name TEXT PRIMARY KEY,
                    base_url TEXT NOT NULL,
                    scraping_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    rate_limit INTEGER NOT NULL,
                    last_scraped TEXT,
                    success_rate REAL DEFAULT 1.0,
                    average_response_time REAL DEFAULT 1.0,
                    enabled BOOLEAN DEFAULT 1
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scraping_tasks (
                    task_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_time TEXT NOT NULL,
                    scheduled_time TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    status TEXT NOT NULL,
                    FOREIGN KEY (source_name) REFERENCES scraping_sources (name)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scraper_performance (
                    timestamp TEXT PRIMARY KEY,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    average_response_time REAL DEFAULT 0.0,
                    active_sources INTEGER DEFAULT 0,
                    queued_tasks INTEGER DEFAULT 0
                )
            ''')

            conn.commit()
            conn.close()

            # Content database
            conn = sqlite3.connect(self.content_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scraped_content (
                    content_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    quality_score REAL DEFAULT 0.5,
                    relevance_score REAL DEFAULT 0.5,
                    scraped_timestamp TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    duplicate_of TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS content_duplicates (
                    original_id TEXT NOT NULL,
                    duplicate_id TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    detected_timestamp TEXT NOT NULL,
                    PRIMARY KEY (original_id, duplicate_id),
                    FOREIGN KEY (original_id) REFERENCES scraped_content (content_id),
                    FOREIGN KEY (duplicate_id) REFERENCES scraped_content (content_id)
                )
            ''')

            conn.commit()
            conn.close()

            # Initialize sources in database
            self._store_scraping_sources()

            logger.info("✅ Scraper databases initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize databases: {e}")
            raise

    def _store_scraping_sources(self):
        """Store scraping sources in database."""
        try:
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            for source in self.sources.values():
                cursor.execute('''
                    INSERT OR REPLACE INTO scraping_sources
                    (name, base_url, scraping_type, priority, rate_limit,
                     last_scraped, success_rate, average_response_time, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    source.name,
                    source.base_url,
                    source.scraping_type,
                    source.priority,
                    source.rate_limit,
                    source.last_scraped,
                    source.success_rate,
                    source.average_response_time,
                    source.enabled
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to store scraping sources: {e}")

    def _load_existing_state(self):
        """Load existing system state from database."""
        try:
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            # Load pending tasks
            cursor.execute("""
                SELECT task_id, source_name, url, task_type, priority,
                       created_time, scheduled_time, retry_count, max_retries, status
                FROM scraping_tasks
                WHERE status IN ('pending', 'running')
                ORDER BY priority DESC, created_time ASC
            """)

            for row in cursor.fetchall():
                task = ScrapingTask(
                    task_id=row[0],
                    source_name=row[1],
                    url=row[2],
                    task_type=row[3],
                    priority=row[4],
                    created_time=row[5],
                    scheduled_time=row[6],
                    retry_count=row[7],
                    max_retries=row[8],
                    status=row[9]
                )
                self.active_tasks[task.task_id] = task

                # Add to priority queue
                priority_tuple = (-task.priority, task.created_time, task.task_id)
                self.task_queue.put_nowait(priority_tuple)

            conn.close()

            logger.info(f"✅ Loaded {len(self.active_tasks)} existing tasks")

        except Exception as e:
            logger.error(f"❌ Failed to load existing state: {e}")

    async def start_continuous_scraping(self):
        """Start the continuous scraping process."""
        logger.info("🌐 Starting continuous unified scraping system...")

        try:
            # Start background tasks
            asyncio.create_task(self._task_scheduler())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._source_health_monitor())

            # Main scraping loop
            await self._scraping_loop()

        except Exception as e:
            logger.error(f"❌ Critical scraping error: {e}")
        finally:
            await self._cleanup()

    async def _scraping_loop(self):
        """Main scraping execution loop."""
        while True:
            try:
                # Get next task from queue
                if not self.task_queue.empty():
                    priority_tuple = await self.task_queue.get()
                    _, _, task_id = priority_tuple

                    if task_id in self.active_tasks:
                        task = self.active_tasks[task_id]

                        # Check rate limiting
                        if await self._check_rate_limit(task.source_name):
                            # Execute task
                            await self._execute_scraping_task(task)
                        else:
                            # Re-queue task for later
                            await asyncio.sleep(1)
                            self.task_queue.put_nowait(priority_tuple)

                # Generate new tasks if queue is low
                if self.task_queue.qsize() < 10:
                    await self._generate_new_tasks()

                # Small delay to prevent tight looping
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ Scraping loop error: {e}")
                await asyncio.sleep(5)

    async def _task_scheduler(self):
        """Schedule new scraping tasks based on source priorities and schedules."""
        while True:
            try:
                current_time = datetime.now()

                for source_name, source in self.sources.items():
                    if not source.enabled:
                        continue

                    # Check if it's time to schedule new tasks for this source
                    should_schedule = False

                    if source.last_scraped is None:
                        should_schedule = True
                    else:
                        last_scraped = datetime.fromisoformat(source.last_scraped)
                        time_since_scrape = current_time - last_scraped

                        # Schedule based on priority and rate limit
                        min_interval = 60 / source.rate_limit  # seconds between requests
                        if time_since_scrape.total_seconds() >= min_interval:
                            should_schedule = True

                    if should_schedule:
                        await self._schedule_source_tasks(source)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"❌ Task scheduler error: {e}")
                await asyncio.sleep(300)

    async def _schedule_source_tasks(self, source: ScrapingSource):
        """Schedule tasks for a specific source."""
        try:
                            # Generate appropriate tasks based on source type
            if source.scraping_type == 'arxiv_api':
                tasks = await self._generate_arxiv_api_tasks(source)
            elif source.scraping_type == 'arxiv_search':
                tasks = await self._generate_arxiv_search_tasks(source)
            elif source.scraping_type == 'html':
                # Enhanced task generation for high-quality academic sources
                if 'mit_ocw' in source.name:
                    tasks = await self._generate_mit_ocw_tasks(source)
                elif 'stanford' in source.name:
                    tasks = await self._generate_stanford_tasks(source)
                elif 'harvard' in source.name:
                    tasks = await self._generate_harvard_tasks(source)
                elif 'coursera' in source.name:
                    tasks = await self._generate_coursera_tasks(source)
                elif 'edx' in source.name:
                    tasks = await self._generate_edx_tasks(source)
                else:
                    tasks = await self._generate_html_scraping_tasks(source)
            elif source.scraping_type == 'api':
                tasks = await self._generate_api_tasks(source)
            elif source.scraping_type == 'feed':
                tasks = await self._generate_feed_tasks(source)
            else:
                tasks = []

            # Add tasks to queue
            for task in tasks:
                self.active_tasks[task.task_id] = task

                # Store in database
                self._store_scraping_task(task)

                # Add to priority queue
                priority_tuple = (-task.priority, task.created_time, task.task_id)
                await self.task_queue.put(priority_tuple)

            # Update source last scraped time
            source.last_scraped = datetime.now().isoformat()
            self._update_source_last_scraped(source.name, source.last_scraped)

            logger.info(f"📅 Scheduled {len(tasks)} tasks for {source.name}")

        except Exception as e:
            logger.error(f"❌ Failed to schedule tasks for {source.name}: {e}")

    async def _generate_arxiv_api_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for arXiv API scraping."""
        tasks = []

        # Generate recent papers task
        task = ScrapingTask(
            task_id=f"task_{source.name}_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=f"{source.base_url}/list/cs/recent",
            task_type='arxiv_recent',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_arxiv_search_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for arXiv search scraping."""
        tasks = []

        # Extract search query from URL
        parsed_url = urlparse(source.base_url)
        query_params = parse_qs(parsed_url.query)
        search_query = query_params.get('query', [''])[0]

        # Generate search task
        task = ScrapingTask(
            task_id=f"task_{source.name}_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=source.base_url,
            task_type='arxiv_search',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_mit_ocw_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for MIT OpenCourseWare scraping."""
        tasks = []

        # Generate course listing task
        task = ScrapingTask(
            task_id=f"task_{source.name}_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=f"{source.base_url}electrical-engineering-and-computer-science/",
            task_type='mit_ocw_eecs',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        # Generate physics courses task
        task = ScrapingTask(
            task_id=f"task_{source.name}_physics_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=f"{source.base_url}physics/",
            task_type='mit_ocw_physics',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        # Generate mathematics courses task
        task = ScrapingTask(
            task_id=f"task_{source.name}_math_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=f"{source.base_url}mathematics/",
            task_type='mit_ocw_math',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_stanford_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for Stanford CS scraping."""
        tasks = []

        # Generate AI/ML research task
        task = ScrapingTask(
            task_id=f"task_{source.name}_ai_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url="https://ai.stanford.edu/",
            task_type='stanford_ai',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_harvard_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for Harvard Science scraping."""
        tasks = []

        # Generate physics research task
        task = ScrapingTask(
            task_id=f"task_{source.name}_physics_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url="https://physics.fas.harvard.edu/",
            task_type='harvard_physics',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_coursera_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for Coursera ML course scraping."""
        tasks = []

        # Generate ML courses task
        task = ScrapingTask(
            task_id=f"task_{source.name}_ml_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=source.base_url,
            task_type='coursera_ml',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_edx_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for edX AI course scraping."""
        tasks = []

        # Generate AI courses task
        task = ScrapingTask(
            task_id=f"task_{source.name}_ai_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=source.base_url,
            task_type='edx_ai',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_html_scraping_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for HTML scraping."""
        tasks = []

        # Generate main page scraping task
        task = ScrapingTask(
            task_id=f"task_{source.name}_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=source.base_url,
            task_type='html_scrape',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_api_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for API scraping."""
        tasks = []

        # Generate API request task
        task = ScrapingTask(
            task_id=f"task_{source.name}_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=f"{source.base_url}/api/search",
            task_type='api_request',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_feed_tasks(self, source: ScrapingSource) -> List[ScrapingTask]:
        """Generate tasks for RSS/Atom feed scraping."""
        tasks = []

        # Generate feed parsing task
        task = ScrapingTask(
            task_id=f"task_{source.name}_{int(time.time())}_{random.randint(1000, 9999)}",
            source_name=source.name,
            url=f"{source.base_url}/feed/",
            task_type='feed_parse',
            priority=source.priority,
            created_time=datetime.now().isoformat(),
            scheduled_time=datetime.now().isoformat(),
            retry_count=0,
            max_retries=3,
            status='pending'
        )
        tasks.append(task)

        return tasks

    async def _generate_new_tasks(self):
        """Generate new tasks when queue is low."""
        try:
            # Generate tasks for high-priority sources first
            high_priority_sources = [s for s in self.sources.values()
                                   if s.priority >= 8 and s.enabled]

            for source in high_priority_sources[:3]:  # Limit to top 3
                await self._schedule_source_tasks(source)

        except Exception as e:
            logger.error(f"❌ Failed to generate new tasks: {e}")

    async def _execute_scraping_task(self, task: ScrapingTask):
        """Execute a scraping task."""
        try:
            task.status = 'running'
            self._update_task_status(task.task_id, 'running')

            start_time = time.time()

            # Execute based on task type
            if task.task_type == 'arxiv_recent':
                result = await self._scrape_arxiv_recent(task)
            elif task.task_type == 'arxiv_search':
                result = await self._scrape_arxiv_search(task)
            elif task.task_type == 'html_scrape':
                result = await self._scrape_html(task)
            elif task.task_type == 'api_request':
                result = await self._scrape_api(task)
            elif task.task_type == 'feed_parse':
                result = await self._scrape_feed(task)
            # High-quality academic sources
            elif task.task_type in ['mit_ocw_eecs', 'mit_ocw_physics', 'mit_ocw_math']:
                result = await self._scrape_mit_ocw(task)
            elif task.task_type == 'stanford_ai':
                result = await self._scrape_stanford_ai(task)
            elif task.task_type == 'harvard_physics':
                result = await self._scrape_harvard_science(task)
            elif task.task_type == 'coursera_ml':
                result = await self._scrape_coursera_ml(task)
            elif task.task_type == 'edx_ai':
                result = await self._scrape_edx_ai(task)
            else:
                result = {'status': 'error', 'error': 'Unknown task type'}

            execution_time = time.time() - start_time

            # Update performance stats
            self._update_performance_stats(task.source_name, result['status'] == 'success', execution_time)

            if result['status'] == 'success':
                # Process successful results
                await self._process_scraped_content(task, result['content'])

                task.status = 'completed'
                self._update_task_status(task.task_id, 'completed')
                logger.info(f"✅ Task {task.task_id} completed successfully")
            else:
                # Handle failure
                task.retry_count += 1
                if task.retry_count >= task.max_retries:
                    task.status = 'failed'
                    self._update_task_status(task.task_id, 'failed')
                    logger.error(f"❌ Task {task.task_id} failed permanently")
                else:
                    task.status = 'pending'
                    self._update_task_status(task.task_id, 'pending')
                    # Re-queue for retry
                    priority_tuple = (-task.priority, task.created_time, task.task_id)
                    await asyncio.sleep(5)  # Brief delay before re-queuing
                    self.task_queue.put_nowait(priority_tuple)
                    logger.warning(f"⚠️ Task {task.task_id} failed, retry {task.retry_count}/{task.max_retries}")

        except Exception as e:
            logger.error(f"❌ Task execution error for {task.task_id}: {e}")
            task.status = 'failed'
            self._update_task_status(task.task_id, 'failed')

        finally:
            # Remove from active tasks if completed or failed
            if task.status in ['completed', 'failed']:
                self.active_tasks.pop(task.task_id, None)

    async def _scrape_arxiv_recent(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape recent arXiv papers."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(task.url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse arXiv HTML
                        soup = BeautifulSoup(content, 'html.parser')

                        # Extract paper information
                        papers = []
                        paper_blocks = soup.find_all('div', class_='meta')

                        for block in paper_blocks[:10]:  # Limit to 10 papers
                            title_elem = block.find('div', class_='list-title')
                            if title_elem:
                                title = title_elem.get_text().strip()
                                papers.append({
                                    'title': title,
                                    'url': task.url,
                                    'source': 'arxiv',
                                    'type': 'recent_paper'
                                })

                        return {
                            'status': 'success',
                            'content': papers,
                            'content_type': 'arxiv_papers'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_arxiv_search(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape arXiv search results."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(task.url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse search results
                        soup = BeautifulSoup(content, 'html.parser')

                        results = []
                        result_blocks = soup.find_all('div', class_='arxiv-result')

                        for block in result_blocks[:10]:  # Limit to 10 results
                            title_elem = block.find('p', class_='title')
                            if title_elem:
                                title = title_elem.get_text().strip()
                                results.append({
                                    'title': title,
                                    'url': task.url,
                                    'source': 'arxiv_search',
                                    'type': 'search_result'
                                })

                        return {
                            'status': 'success',
                            'content': results,
                            'content_type': 'search_results'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_mit_ocw(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape MIT OpenCourseWare content."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(task.url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse MIT OCW course page
                        soup = BeautifulSoup(content, 'html.parser')

                        courses = []

                        # Find course listings
                        course_elements = soup.find_all('article', class_='course')[:10]  # Limit to 10

                        for course_elem in course_elements:
                            title_elem = course_elem.find('h2', class_='course-title')
                            if title_elem:
                                title = title_elem.get_text().strip()
                                link_elem = course_elem.find('a')
                                course_url = link_elem['href'] if link_elem else task.url

                                courses.append({
                                    'title': title,
                                    'url': course_url,
                                    'source': 'mit_opencourseware',
                                    'type': 'course',
                                    'category': self._determine_mit_course_category(task.url)
                                })

                        return {
                            'status': 'success',
                            'content': courses,
                            'content_type': 'mit_courses'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _determine_mit_course_category(self, url: str) -> str:
        """Determine MIT course category from URL."""
        if 'electrical-engineering' in url:
            return 'computer_science'
        elif 'physics' in url:
            return 'physics'
        elif 'mathematics' in url:
            return 'mathematics'
        else:
            return 'general'

    async def _scrape_stanford_ai(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape Stanford AI research content."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(task.url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse Stanford AI page
                        soup = BeautifulSoup(content, 'html.parser')

                        research_items = []

                        # Find research project listings
                        project_elements = soup.find_all(['div', 'article'], class_=lambda x: x and ('research' in x.lower() or 'project' in x.lower()))[:8]

                        for project_elem in project_elements:
                            title_elem = project_elem.find(['h2', 'h3', 'h4'])
                            if title_elem:
                                title = title_elem.get_text().strip()
                                link_elem = project_elem.find('a')
                                project_url = link_elem['href'] if link_elem else task.url

                                research_items.append({
                                    'title': title,
                                    'url': project_url,
                                    'source': 'stanford_ai',
                                    'type': 'ai_research',
                                    'category': 'artificial_intelligence'
                                })

                        return {
                            'status': 'success',
                            'content': research_items,
                            'content_type': 'stanford_ai_research'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_harvard_science(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape Harvard Science research content."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(task.url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse Harvard Science page
                        soup = BeautifulSoup(content, 'html.parser')

                        research_items = []

                        # Find research article listings
                        article_elements = soup.find_all(['article', 'div'], class_=lambda x: x and ('article' in x.lower() or 'research' in x.lower()))[:8]

                        for article_elem in article_elements:
                            title_elem = article_elem.find(['h1', 'h2', 'h3'])
                            if title_elem:
                                title = title_elem.get_text().strip()
                                link_elem = article_elem.find('a')
                                article_url = link_elem['href'] if link_elem else task.url

                                research_items.append({
                                    'title': title,
                                    'url': article_url,
                                    'source': 'harvard_science',
                                    'type': 'scientific_research',
                                    'category': 'physics'  # Default for Harvard physics
                                })

                        return {
                            'status': 'success',
                            'content': research_items,
                            'content_type': 'harvard_science_research'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_coursera_ml(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape Coursera ML courses."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(task.url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse Coursera course page
                        soup = BeautifulSoup(content, 'html.parser')

                        courses = []

                        # Find course card listings
                        course_elements = soup.find_all(['div', 'li'], class_=lambda x: x and ('course' in x.lower() or 'card' in x.lower()))[:8]

                        for course_elem in course_elements:
                            title_elem = course_elem.find(['h2', 'h3'])
                            if title_elem:
                                title = title_elem.get_text().strip()
                                link_elem = course_elem.find('a')
                                course_url = link_elem['href'] if link_elem else task.url

                                courses.append({
                                    'title': title,
                                    'url': course_url,
                                    'source': 'coursera_ml',
                                    'type': 'online_course',
                                    'category': 'machine_learning'
                                })

                        return {
                            'status': 'success',
                            'content': courses,
                            'content_type': 'coursera_ml_courses'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_edx_ai(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape edX AI courses."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(task.url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse edX course page
                        soup = BeautifulSoup(content, 'html.parser')

                        courses = []

                        # Find course card listings
                        course_elements = soup.find_all(['div', 'article'], class_=lambda x: x and ('course' in x.lower() or 'card' in x.lower()))[:8]

                        for course_elem in course_elements:
                            title_elem = course_elem.find(['h2', 'h3', 'h4'])
                            if title_elem:
                                title = title_elem.get_text().strip()
                                link_elem = course_elem.find('a')
                                course_url = link_elem['href'] if link_elem else task.url

                                courses.append({
                                    'title': title,
                                    'url': course_url,
                                    'source': 'edx_ai',
                                    'type': 'online_course',
                                    'category': 'artificial_intelligence'
                                })

                        return {
                            'status': 'success',
                            'content': courses,
                            'content_type': 'edx_ai_courses'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_html(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape HTML content."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(task.url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse HTML
                        soup = BeautifulSoup(content, 'html.parser')

                        # Extract title
                        title = soup.title.string if soup.title else "No Title"

                        # Extract main content (simplified)
                        main_content = ""
                        content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'article'])
                        for tag in content_tags[:20]:  # Limit content extraction
                            main_content += tag.get_text().strip() + "\n"

                        return {
                            'status': 'success',
                            'content': {
                                'title': title,
                                'url': task.url,
                                'content': main_content,
                                'source': task.source_name
                            },
                            'content_type': 'html_content'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_api(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape API endpoints."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(task.url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.json()

                        return {
                            'status': 'success',
                            'content': content,
                            'content_type': 'api_response'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _scrape_feed(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape RSS/Atom feeds."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(task.url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse feed (simplified)
                        soup = BeautifulSoup(content, 'xml')

                        items = []
                        feed_items = soup.find_all('item')[:10]  # Limit to 10 items

                        for item in feed_items:
                            title_elem = item.find('title')
                            link_elem = item.find('link')

                            if title_elem and link_elem:
                                items.append({
                                    'title': title_elem.get_text().strip(),
                                    'url': link_elem.get_text().strip(),
                                    'source': task.source_name,
                                    'type': 'feed_item'
                                })

                        return {
                            'status': 'success',
                            'content': items,
                            'content_type': 'feed_items'
                        }
                    else:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}'
                        }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _check_rate_limit(self, source_name: str) -> bool:
        """Check if rate limiting allows the request."""
        if source_name not in self.rate_limiters:
            source = self.sources[source_name]
            self.rate_limiters[source_name] = {
                'requests': [],
                'rate_limit': source.rate_limit
            }

        limiter = self.rate_limiters[source_name]
        current_time = time.time()

        # Remove old requests (older than 1 minute)
        limiter['requests'] = [req_time for req_time in limiter['requests']
                              if current_time - req_time < 60]

        # Check if under rate limit
        if len(limiter['requests']) < limiter['rate_limit']:
            limiter['requests'].append(current_time)
            return True

        return False

    async def _process_scraped_content(self, task: ScrapingTask, content: Any):
        """Process scraped content and store it."""
        try:
            # Convert content to standardized format
            if isinstance(content, list):
                # Multiple items
                for item in content:
                    await self._store_content_item(task, item)
            else:
                # Single item
                await self._store_content_item(task, content)

        except Exception as e:
            logger.error(f"❌ Content processing error: {e}")

    async def _store_content_item(self, task: ScrapingTask, item: Dict[str, Any]):
        """Store a single content item."""
        try:
            # Generate content ID
            content_hash = hashlib.md5(str(item).encode()).hexdigest()
            content_id = f"content_{content_hash[:16]}"

            # Check for duplicates
            duplicate_id = self._check_duplicate_content(item)

            # Create scraped content object
            scraped_content = ScrapedContent(
                content_id=content_id,
                source=task.source_name,
                url=item.get('url', task.url),
                title=item.get('title', 'No Title'),
                content=str(item.get('content', item)),
                metadata=item,
                quality_score=self._assess_content_quality(item),
                relevance_score=self._assess_content_relevance(item),
                scraped_timestamp=datetime.now().isoformat(),
                processing_status='scraped'
            )

            # Store in database
            self._store_scraped_content(scraped_content, duplicate_id)

            # Add to knowledge base
            await self._send_to_knowledge_base(scraped_content)

            logger.info(f"💾 Stored content: {content_id[:16]}...")

        except Exception as e:
            logger.error(f"❌ Failed to store content item: {e}")

    def _assess_content_quality(self, item: Dict[str, Any]) -> float:
        """Assess the quality of scraped content."""
        quality_score = 0.5

        # Title quality
        title = item.get('title', '')
        if len(title) > 10 and len(title) < 200:
            quality_score += 0.1

        # Content length
        content = str(item.get('content', ''))
        if len(content) > 100:
            quality_score += 0.1
        if len(content) > 1000:
            quality_score += 0.1

        # Source reliability
        source = item.get('source', '')
        reliable_sources = ['arxiv', 'nature', 'science', 'ieee']
        if any(reliable in source.lower() for reliable in reliable_sources):
            quality_score += 0.2

        return min(1.0, quality_score)

    def _assess_content_relevance(self, item: Dict[str, Any]) -> float:
        """Assess the relevance of content to KOBA42 research."""
        relevance_score = 0.3

        text_to_check = f"{item.get('title', '')} {item.get('content', '')}".lower()

        # Relevance keywords
        relevance_keywords = [
            'quantum', 'consciousness', 'ai', 'machine learning', 'neural',
            'optimization', 'algorithm', 'research', 'breakthrough',
            'wallace', 'mathematics', 'physics', 'computation'
        ]

        keyword_matches = sum(1 for keyword in relevance_keywords
                            if keyword in text_to_check)
        relevance_score += min(0.4, keyword_matches * 0.1)

        # Source-based relevance
        source = item.get('source', '').lower()
        if 'arxiv' in source:
            relevance_score += 0.2
        elif 'nature' in source or 'science' in source:
            relevance_score += 0.3

        return min(1.0, relevance_score)

    def _check_duplicate_content(self, item: Dict[str, Any]) -> Optional[str]:
        """Check if content is duplicate."""
        try:
            content_hash = hashlib.md5(str(item).encode()).hexdigest()

            conn = sqlite3.connect(self.content_db_path)
            cursor = conn.cursor()

            # Check for exact content match
            cursor.execute("""
                SELECT content_id FROM scraped_content
                WHERE content LIKE ?
                LIMIT 1
            """, (f"%{content_hash}%",))

            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None

        except Exception as e:
            logger.error(f"❌ Duplicate check error: {e}")
            return None

    def _store_scraped_content(self, content: ScrapedContent, duplicate_id: Optional[str]):
        """Store scraped content in database."""
        try:
            conn = sqlite3.connect(self.content_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO scraped_content
                (content_id, source, url, title, content, metadata,
                 quality_score, relevance_score, scraped_timestamp,
                 processing_status, duplicate_of)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content.content_id,
                content.source,
                content.url,
                content.title,
                content.content,
                json.dumps(content.metadata),
                content.quality_score,
                content.relevance_score,
                content.scraped_timestamp,
                content.processing_status,
                duplicate_id
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to store scraped content: {e}")

    async def _send_to_knowledge_base(self, content: ScrapedContent):
        """Send content to the knowledge base manager."""
        try:
            # This would integrate with the knowledge base manager
            # For now, we'll simulate the integration
            knowledge_fragment = {
                'id': content.content_id,
                'source': content.source,
                'content': f"{content.title}\n\n{content.content}",
                'category': self._categorize_content(content),
                'relevance_score': content.relevance_score,
                'timestamp': content.scraped_timestamp,
                'agent_contributor': f"scraper_{content.source}"
            }

            logger.info(f"📤 Sent to knowledge base: {content.content_id[:16]}...")

        except Exception as e:
            logger.error(f"❌ Knowledge base integration error: {e}")

    def _categorize_content(self, content: ScrapedContent) -> str:
        """Categorize content based on its content."""
        text = f"{content.title} {content.content}".lower()

        if 'quantum' in text:
            return 'quantum_physics'
        elif 'ai' in text or 'artificial intelligence' in text:
            return 'artificial_intelligence'
        elif 'consciousness' in text or 'wallace' in text:
            return 'consciousness_mathematics'
        elif 'optimization' in text or 'algorithm' in text:
            return 'optimization_algorithms'
        else:
            return 'general_research'

    def _store_scraping_task(self, task: ScrapingTask):
        """Store scraping task in database."""
        try:
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO scraping_tasks
                (task_id, source_name, url, task_type, priority, created_time,
                 scheduled_time, retry_count, max_retries, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id,
                task.source_name,
                task.url,
                task.task_type,
                task.priority,
                task.created_time,
                task.scheduled_time,
                task.retry_count,
                task.max_retries,
                task.status
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to store scraping task: {e}")

    def _update_task_status(self, task_id: str, status: str):
        """Update scraping task status."""
        try:
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE scraping_tasks
                SET status = ?
                WHERE task_id = ?
            ''', (status, task_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to update task status: {e}")

    def _update_source_last_scraped(self, source_name: str, timestamp: str):
        """Update source last scraped timestamp."""
        try:
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE scraping_sources
                SET last_scraped = ?
                WHERE name = ?
            ''', (timestamp, source_name))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to update source timestamp: {e}")

    def _update_performance_stats(self, source_name: str, success: bool, response_time: float):
        """Update performance statistics."""
        self.performance_stats['total_requests'] += 1

        if success:
            self.performance_stats['successful_requests'] += 1
        else:
            self.performance_stats['failed_requests'] += 1

        # Update average response time
        current_avg = self.performance_stats['average_response_time']
        total_requests = self.performance_stats['total_requests']
        self.performance_stats['average_response_time'] = (
            (current_avg * (total_requests - 1)) + response_time
        ) / total_requests

        # Update source-specific stats
        if source_name in self.sources:
            source = self.sources[source_name]

            # Update success rate
            if success:
                source.success_rate = (source.success_rate * 0.9) + 0.1  # Weighted average
            else:
                source.success_rate = (source.success_rate * 0.9)  # Weighted average

            # Update average response time
            source.average_response_time = (
                (source.average_response_time * 0.9) + (response_time * 0.1)
            )

    async def _performance_monitor(self):
        """Monitor system performance and log statistics."""
        while True:
            try:
                # Store performance stats
                self._store_performance_stats()

                # Log current status
                total_requests = self.performance_stats['total_requests']
                successful_requests = self.performance_stats['successful_requests']

                if total_requests > 0:
                    success_rate = (successful_requests / total_requests) * 100
                    logger.info(f"📊 Performance: {successful_requests}/{total_requests} requests successful ({success_rate:.1f}%)")

                # Log queue status
                queue_size = self.task_queue.qsize()
                active_tasks = len(self.active_tasks)
                logger.info(f"📋 Queue: {queue_size} pending, {active_tasks} active tasks")

                await asyncio.sleep(300)  # Log every 5 minutes

            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(60)

    def _store_performance_stats(self):
        """Store performance statistics in database."""
        try:
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO scraper_performance
                (timestamp, total_requests, successful_requests, failed_requests,
                 average_response_time, active_sources, queued_tasks)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                self.performance_stats['total_requests'],
                self.performance_stats['successful_requests'],
                self.performance_stats['failed_requests'],
                self.performance_stats['average_response_time'],
                len([s for s in self.sources.values() if s.enabled]),
                self.task_queue.qsize()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to store performance stats: {e}")

    async def _source_health_monitor(self):
        """Monitor source health and disable unhealthy sources."""
        while True:
            try:
                for source_name, source in self.sources.items():
                    if source.success_rate < 0.3:  # Less than 30% success rate
                        if source.enabled:
                            logger.warning(f"⚠️ Disabling unhealthy source: {source_name} (success rate: {source.success_rate:.2f})")
                            source.enabled = False
                            self._update_source_enabled(source_name, False)
                    elif source.success_rate > 0.7 and not source.enabled:
                        logger.info(f"✅ Re-enabling healthy source: {source_name}")
                        source.enabled = True
                        self._update_source_enabled(source_name, True)

                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                logger.error(f"❌ Source health monitoring error: {e}")
                await asyncio.sleep(300)

    def _update_source_enabled(self, source_name: str, enabled: bool):
        """Update source enabled status in database."""
        try:
            conn = sqlite3.connect(self.scraper_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE scraping_sources
                SET enabled = ?
                WHERE name = ?
            ''', (enabled, source_name))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to update source status: {e}")

    async def _cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up scraper system resources...")

        # Close database connections
        # Cancel pending tasks
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("✅ Scraper system cleanup completed")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'total_requests': self.performance_stats['total_requests'],
            'success_rate': (self.performance_stats['successful_requests'] /
                           max(1, self.performance_stats['total_requests'])) * 100,
            'average_response_time': self.performance_stats['average_response_time'],
            'active_sources': len([s for s in self.sources.values() if s.enabled]),
            'total_sources': len(self.sources)
        }

async def main():
    """Main entry point for the unified scraper system."""
    print("🌐 UNIFIED CONTINUOUS SCRAPER SYSTEM")
    print("=" * 70)
    print("Integrated Web Scraping and Research Coordination System")
    print("=" * 70)

    # Initialize scraper system
    scraper_system = UnifiedContinuousScraperSystem()

    try:
        # Display initial status
        status = scraper_system.get_system_status()
        print("
📊 Initial System Status:"        print(f"   Active Tasks: {status['active_tasks']}")
        print(f"   Queued Tasks: {status['queued_tasks']}")
        print(f"   Total Requests: {status['total_requests']}")
        print(".1f"        print(f"   Active Sources: {status['active_sources']}/{status['total_sources']}")

        # Start continuous scraping
        await scraper_system.start_continuous_scraping()

    except KeyboardInterrupt:
        print("\n🛑 Scraper system stopped")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        logger.error(f"Critical scraper system error: {e}")

    print("\n🎉 Continuous scraping session completed!")
    print("🌐 Research data continuously collected and processed")
    print("📊 Content quality continuously assessed and filtered")
    print("🔄 Knowledge base continuously expanded")
    print("💾 Research data stored in: research_data/")
    print("🔄 Ready for next scraping session")

if __name__ == "__main__":
    asyncio.run(main())
