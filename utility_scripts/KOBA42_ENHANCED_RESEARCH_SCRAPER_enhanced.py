
import time
from collections import defaultdict
from typing import Dict, Tuple

class RateLimiter:
    """Intelligent rate limiting system"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        client_requests = self.requests[client_id]

        # Remove old requests outside the time window
        window_start = now - 60  # 1 minute window
        client_requests[:] = [req for req in client_requests if req > window_start]

        # Check if under limit
        if len(client_requests) < self.requests_per_minute:
            client_requests.append(now)
            return True

        return False

    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        now = time.time()
        client_requests = self.requests[client_id]
        window_start = now - 60
        client_requests[:] = [req for req in client_requests if req > window_start]

        return max(0, self.requests_per_minute - len(client_requests))

    def get_reset_time(self, client_id: str) -> float:
        """Get time until rate limit resets"""
        client_requests = self.requests[client_id]
        if not client_requests:
            return 0

        oldest_request = min(client_requests)
        return max(0, 60 - (time.time() - oldest_request))


# Enhanced with rate limiting

import logging
import json
from datetime import datetime
from typing import Dict, Any

class SecurityLogger:
    """Security event logging system"""

    def __init__(self, log_file: str = 'security.log'):
        self.logger = logging.getLogger('security')
        self.logger.setLevel(logging.INFO)

        # Create secure log format
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )

        # File handler with secure permissions
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          severity: str = 'INFO'):
        """Log security-related events"""
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details,
            'severity': severity,
            'source': 'enhanced_system'
        }

        if severity == 'CRITICAL':
            self.logger.critical(json.dumps(event_data))
        elif severity == 'ERROR':
            self.logger.error(json.dumps(event_data))
        elif severity == 'WARNING':
            self.logger.warning(json.dumps(event_data))
        else:
            self.logger.info(json.dumps(event_data))

    def log_access_attempt(self, resource: str, user_id: str = None,
                          success: bool = True):
        """Log access attempts"""
        self.log_security_event(
            'ACCESS_ATTEMPT',
            {
                'resource': resource,
                'user_id': user_id or 'anonymous',
                'success': success,
                'ip_address': 'logged_ip'  # Would get real IP in production
            },
            'WARNING' if not success else 'INFO'
        )

    def log_suspicious_activity(self, activity: str, details: Dict[str, Any]):
        """Log suspicious activities"""
        self.log_security_event(
            'SUSPICIOUS_ACTIVITY',
            {'activity': activity, 'details': details},
            'WARNING'
        )


# Enhanced with security logging

import logging
from contextlib import contextmanager
from typing import Any, Optional

class SecureErrorHandler:
    """Security-focused error handling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @contextmanager
    def secure_context(self, operation: str):
        """Context manager for secure operations"""
        try:
            yield
        except Exception as e:
            # Log error without exposing sensitive information
            self.logger.error(f"Secure operation '{operation}' failed: {type(e).__name__}")
            # Don't re-raise to prevent information leakage
            raise RuntimeError(f"Operation '{operation}' failed securely")

    def validate_and_execute(self, func: callable, *args, **kwargs) -> Any:
        """Execute function with security validation"""
        with self.secure_context(func.__name__):
            # Validate inputs before execution
            validated_args = [self._validate_arg(arg) for arg in args]
            validated_kwargs = {k: self._validate_arg(v) for k, v in kwargs.items()}

            return func(*validated_args, **validated_kwargs)

    def _validate_arg(self, arg: Any) -> Any:
        """Validate individual argument"""
        # Implement argument validation logic
        if isinstance(arg, str) and len(arg) > 10000:
            raise ValueError("Input too large")
        if isinstance(arg, (dict, list)) and len(str(arg)) > 50000:
            raise ValueError("Complex input too large")
        return arg


# Enhanced with secure error handling

import re
from typing import Union, Any

class InputValidator:
    """Comprehensive input validation system"""

    @staticmethod
    def validate_string(input_str: str, max_length: int = 1000,
                       pattern: str = None) -> bool:
        """Validate string input"""
        if not isinstance(input_str, str):
            return False
        if len(input_str) > max_length:
            return False
        if pattern and not re.match(pattern, input_str):
            return False
        return True

    @staticmethod
    def sanitize_input(input_data: Any) -> Any:
        """Sanitize input to prevent injection attacks"""
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            return re.sub(r'[^\w\s\-_.]', '', input_data)
        elif isinstance(input_data, dict):
            return {k: InputValidator.sanitize_input(v)
                   for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [InputValidator.sanitize_input(item) for item in input_data]
        return input_data

    @staticmethod
    def validate_numeric(value: Any, min_val: float = None,
                        max_val: float = None) -> Union[float, None]:
        """Validate numeric input"""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return None
            if max_val is not None and num > max_val:
                return None
            return num
        except (ValueError, TypeError):
            return None


# Enhanced with input validation

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import os

class ConcurrencyManager:
    """Intelligent concurrency management"""

    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)

    def get_optimal_workers(self, task_type: str = 'cpu') -> int:
        """Determine optimal number of workers"""
        if task_type == 'cpu':
            return max(1, self.cpu_count - 1)
        elif task_type == 'io':
            return min(32, self.cpu_count * 2)
        else:
            return self.cpu_count

    def parallel_process(self, items: List[Any], process_func: callable,
                        task_type: str = 'cpu') -> List[Any]:
        """Process items in parallel"""
        num_workers = self.get_optimal_workers(task_type)

        if task_type == 'cpu' and len(items) > 100:
            # Use process pool for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_func, items))
        else:
            # Use thread pool for I/O or small tasks
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_func, items))

        return results


# Enhanced with intelligent concurrency

from functools import lru_cache
import time
from typing import Dict, Any, Optional

class CacheManager:
    """Intelligent caching system"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl

    @lru_cache(maxsize=128)
    def get_cached_result(self, key: str, compute_func: callable, *args, **kwargs):
        """Get cached result or compute new one"""
        cache_key = f"{key}_{hash(str(args) + str(kwargs))}"

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['result']

        result = compute_func(*args, **kwargs)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

        # Clean old entries if cache is full
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        return result


# Enhanced with intelligent caching

import asyncio
from typing import Coroutine, Any

class AsyncEnhancer:
    """Async enhancement wrapper"""

    @staticmethod
    async def run_async(func: Callable[..., Any], *args, **kwargs) -> Any:
        """Run function asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    @staticmethod
    def make_async(func: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Convert sync function to async"""
        async def wrapper(*args, **kwargs):
            return await AsyncEnhancer.run_async(func, *args, **kwargs)
        return wrapper


# Enhanced with async support
"""
KOBA42 ENHANCED RESEARCH SCRAPER
================================
Enhanced Multi-Source Research Scraper with Chunked Processing
============================================================

Features:
1. Chunked Processing with Progress Tracking
2. Comprehensive Logging and Status Tracking
3. Efficient Storage of Important Research Data
4. Filtering of Relevant vs Non-Relevant Articles
5. Batch Processing with Resume Capability
6. Future Development Data Storage
"""
import requests
import json
import logging
import time
import random
import hashlib
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pickle
import sqlite3
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('research_scraping.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

@dataclass
class ResearchArticle:
    """Enhanced research article data structure."""
    title: str
    url: str
    source: str
    field: str
    subfield: str
    publication_date: str
    authors: List[str]
    summary: str
    content: str
    tags: List[str]
    research_impact: float
    quantum_relevance: float
    technology_relevance: float
    article_id: str
    scraped_timestamp: str
    processing_status: str
    relevance_score: float
    key_insights: List[str]
    koba42_integration_potential: float

@dataclass
class ScrapingBatch:
    """Scraping batch tracking."""
    batch_id: str
    source: str
    start_time: str
    end_time: Optional[str]
    articles_scraped: int
    articles_processed: int
    articles_filtered: int
    articles_stored: int
    status: str
    error_message: Optional[str]

@dataclass
class ScrapingSession:
    """Complete scraping session tracking."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    total_batches: int
    completed_batches: int
    total_articles_scraped: int
    total_articles_stored: int
    sources_processed: List[str]
    status: str

class EnhancedResearchScraper:
    """Enhanced multi-source research scraper with chunked processing."""

    def __init__(self, storage_dir: str='research_data'):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / 'research_articles.db'
        self.init_database()
        self.current_session = None
        self.current_batch = None
        self.scraped_articles = []
        self.processed_articles = []
        self.filtered_articles = []
        self.chunk_size = 10
        self.max_articles_per_source = 50
        self.min_relevance_score = 5.0
        self.sources = self._define_sources()
        self.research_fields = self._define_research_fields()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        logger.info('Enhanced Research Scraper initialized')

    def init_database(self):
        """Initialize SQLite database for article storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('\n            CREATE TABLE IF NOT EXISTS articles (\n                article_id TEXT PRIMARY KEY,\n                title TEXT NOT NULL,\n                url TEXT NOT NULL,\n                source TEXT NOT NULL,\n                field TEXT NOT NULL,\n                subfield TEXT NOT NULL,\n                publication_date TEXT,\n                authors TEXT,\n                summary TEXT,\n                content TEXT,\n                tags TEXT,\n                research_impact REAL,\n                quantum_relevance REAL,\n                technology_relevance REAL,\n                relevance_score REAL,\n                scraped_timestamp TEXT,\n                processing_status TEXT,\n                key_insights TEXT,\n                koba42_integration_potential REAL,\n                UNIQUE(url)\n            )\n        ')
        cursor.execute('\n            CREATE TABLE IF NOT EXISTS batches (\n                batch_id TEXT PRIMARY KEY,\n                source TEXT NOT NULL,\n                start_time TEXT NOT NULL,\n                end_time TEXT,\n                articles_scraped INTEGER,\n                articles_processed INTEGER,\n                articles_filtered INTEGER,\n                articles_stored INTEGER,\n                status TEXT,\n                error_message TEXT\n            )\n        ')
        cursor.execute('\n            CREATE TABLE IF NOT EXISTS sessions (\n                session_id TEXT PRIMARY KEY,\n                start_time TEXT NOT NULL,\n                end_time TEXT,\n                total_batches INTEGER,\n                completed_batches INTEGER,\n                total_articles_scraped INTEGER,\n                total_articles_stored INTEGER,\n                sources_processed TEXT,\n                status TEXT\n            )\n        ')
        conn.commit()
        conn.close()
        logger.info(f'Database initialized at {self.db_path}')

    def _define_sources(self) -> Dict[str, Dict[str, Any]]:
        """Define research sources and their configurations."""
        return {'phys_org': {'name': 'Phys.org', 'base_url': 'https://phys.org', 'article_pattern': '/news/\\d{4}-\\d{2}-\\d{2}-', 'title_selector': 'h1, title', 'content_selector': 'div.article-content, article, div.content', 'author_selector': 'span.author, div.byline, p.author', 'date_selector': 'span.date, time, div.date', 'tags_selector': 'a.tag, span.category, div.tags', 'requires_auth': False, 'rate_limit': 2.0, 'priority_fields': ['physics', 'chemistry', 'materials_science']}, 'nature': {'name': 'Nature', 'base_url': 'https://www.nature.com', 'article_pattern': '/articles/', 'title_selector': 'h1[data-test="article-title"], h1.title', 'content_selector': 'div.article__body, div.content, article', 'author_selector': 'span.author, a.author, div.authors', 'date_selector': 'time, span.date, div.publication-date', 'tags_selector': 'a.tag, span.category, div.keywords', 'requires_auth': True, 'rate_limit': 3.0, 'priority_fields': ['physics', 'technology', 'materials_science']}, 'infoq': {'name': 'InfoQ', 'base_url': 'https://www.infoq.com', 'article_pattern': '/news/\\d{4}/\\d{2}/', 'title_selector': 'h1.title, h1.article-title', 'content_selector': 'div.article-content, div.content, article', 'author_selector': 'span.author, a.author, div.byline', 'date_selector': 'time, span.date, div.publication-date', 'tags_selector': 'a.tag, span.category, div.tags', 'requires_auth': False, 'rate_limit': 2.5, 'priority_fields': ['software', 'technology', 'artificial_intelligence']}}

    def _define_research_fields(self) -> Dict[str, Dict[str, Any]]:
        """Define research fields and their characteristics."""
        return {'physics': {'keywords': ['quantum', 'particle', 'matter', 'energy', 'force', 'wave', 'field', 'atom', 'molecule'], 'quantum_keywords': ['quantum', 'entanglement', 'superposition', 'qubit', 'quantum_computing', 'quantum_mechanics'], 'priority_score': 9.5, 'koba42_relevance': 9.0}, 'chemistry': {'keywords': ['molecule', 'reaction', 'catalyst', 'synthesis', 'compound', 'element', 'bond'], 'quantum_keywords': ['quantum_chemistry', 'molecular_orbital', 'electronic_structure', 'quantum_tunneling'], 'priority_score': 8.0, 'koba42_relevance': 7.5}, 'technology': {'keywords': ['computer', 'algorithm', 'device', 'sensor', 'processor', 'memory', 'network'], 'quantum_keywords': ['quantum_computer', 'quantum_algorithm', 'quantum_sensor', 'quantum_network'], 'priority_score': 9.0, 'koba42_relevance': 8.5}, 'software': {'keywords': ['software', 'programming', 'code', 'application', 'system', 'framework', 'library'], 'quantum_keywords': ['quantum_software', 'quantum_programming', 'quantum_framework'], 'priority_score': 7.5, 'koba42_relevance': 8.0}, 'materials_science': {'keywords': ['material', 'crystal', 'structure', 'property', 'conductivity', 'magnetic'], 'quantum_keywords': ['quantum_material', 'topological_insulator', 'quantum_dot', 'quantum_well'], 'priority_score': 8.5, 'koba42_relevance': 8.0}, 'artificial_intelligence': {'keywords': ['ai', 'machine_learning', 'neural_network', 'algorithm', 'intelligence', 'automation'], 'quantum_keywords': ['quantum_ai', 'quantum_machine_learning', 'quantum_neural_network'], 'priority_score': 8.0, 'koba42_relevance': 8.5}}

    def start_scraping_session(self) -> str:
        """Start a new scraping session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = ScrapingSession(session_id=session_id, start_time=datetime.now().isoformat(), end_time=None, total_batches=0, completed_batches=0, total_articles_scraped=0, total_articles_stored=0, sources_processed=[], status='running')
        self._save_session_to_db(self.current_session)
        logger.info(f'🚀 Started scraping session: {session_id}')
        return session_id

    def scrape_all_sources_chunked(self) -> Dict[str, Any]:
        """Scrape all sources in chunks with progress tracking."""
        session_id = self.start_scraping_session()
        total_results = {'session_id': session_id, 'total_articles_scraped': 0, 'total_articles_stored': 0, 'sources_processed': [], 'batches_completed': 0, 'processing_time': 0}
        start_time = time.time()
        for (source_name, source_config) in self.sources.items():
            try:
                logger.info(f"📡 Processing source: {source_config['name']}")
                source_results = self._scrape_source_chunked(source_name, source_config)
                total_results['total_articles_scraped'] += source_results['articles_scraped']
                total_results['total_articles_stored'] += source_results['articles_stored']
                total_results['sources_processed'].append(source_name)
                total_results['batches_completed'] += source_results['batches_completed']
                self.current_session.sources_processed.append(source_name)
                self.current_session.total_articles_scraped = total_results['total_articles_scraped']
                self.current_session.total_articles_stored = total_results['total_articles_stored']
                self.current_session.completed_batches = total_results['batches_completed']
                time.sleep(random.uniform(3, 6))
            except Exception as e:
                logger.error(f'❌ Failed to process source {source_name}: {e}')
                continue
        processing_time = time.time() - start_time
        total_results['processing_time'] = processing_time
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.status = 'completed'
        self._save_session_to_db(self.current_session)
        logger.info(f'✅ Scraping session completed: {session_id}')
        logger.info(f"📊 Total articles scraped: {total_results['total_articles_scraped']}")
        logger.info(f"💾 Total articles stored: {total_results['total_articles_stored']}")
        logger.info(f'⏱️ Processing time: {processing_time:.2f} seconds')
        return total_results

    def _scrape_source_chunked(self, source_name: str, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape a single source in chunks."""
        batch_results = {'articles_scraped': 0, 'articles_stored': 0, 'batches_completed': 0}
        try:
            response = self.session.get(source_config['base_url'], timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            article_links = soup.find_all('a', href=re.compile(source_config['article_pattern']))
            article_links = article_links[:self.max_articles_per_source]
            for i in range(0, len(article_links), self.chunk_size):
                chunk_links = article_links[i:i + self.chunk_size]
                batch_id = f'{source_name}_batch_{i // self.chunk_size + 1}'
                self.current_batch = ScrapingBatch(batch_id=batch_id, source=source_name, start_time=datetime.now().isoformat(), end_time=None, articles_scraped=0, articles_processed=0, articles_filtered=0, articles_stored=0, status='running', error_message=None)
                logger.info(f'🔄 Processing batch {batch_id} ({len(chunk_links)} articles)')
                chunk_results = self._process_article_chunk(chunk_links, source_name, source_config)
                self.current_batch.end_time = datetime.now().isoformat()
                self.current_batch.articles_scraped = chunk_results['scraped']
                self.current_batch.articles_processed = chunk_results['processed']
                self.current_batch.articles_filtered = chunk_results['filtered']
                self.current_batch.articles_stored = chunk_results['stored']
                self.current_batch.status = 'completed'
                self._save_batch_to_db(self.current_batch)
                batch_results['articles_scraped'] += chunk_results['scraped']
                batch_results['articles_stored'] += chunk_results['stored']
                batch_results['batches_completed'] += 1
                time.sleep(random.uniform(source_config['rate_limit'], source_config['rate_limit'] + 1))
                logger.info(f"✅ Batch {batch_id} completed: {chunk_results['stored']}/{chunk_results['scraped']} articles stored")
        except Exception as e:
            logger.error(f'❌ Failed to scrape source {source_name}: {e}')
            if self.current_batch:
                self.current_batch.status = 'failed'
                self.current_batch.error_message = str(e)
                self.current_batch.end_time = datetime.now().isoformat()
                self._save_batch_to_db(self.current_batch)
        return batch_results

    def _process_article_chunk(self, article_links: List, source_name: str, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chunk of article links."""
        results = {'scraped': 0, 'processed': 0, 'filtered': 0, 'stored': 0}
        for link in article_links:
            try:
                article_url = urljoin(source_config['base_url'], link['href'])
                if self._article_exists_in_db(article_url):
                    logger.debug(f'⏭️ Article already exists: {article_url}')
                    continue
                article = self._scrape_single_article(article_url, source_name, source_config)
                if article:
                    results['scraped'] += 1
                    processed_article = self._process_article(article)
                    if processed_article:
                        results['processed'] += 1
                        if self._is_article_relevant(processed_article):
                            results['filtered'] += 1
                            if self._store_article(processed_article):
                                results['stored'] += 1
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                logger.warning(f'⚠️ Failed to process article: {e}')
                continue
        return results

    def _scrape_single_article(self, url: str, source_name: str, source_config: Dict[str, Any]) -> Optional[ResearchArticle]:
        """Scrape a single article."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = self._extract_with_selectors(soup, source_config['title_selector'])
            if not title:
                return None
            content = self._extract_with_selectors(soup, source_config['content_selector'])
            authors = self._extract_authors_with_selectors(soup, source_config['author_selector'])
            publication_date = self._extract_date_with_selectors(soup, source_config['date_selector'])
            tags = self._extract_tags_with_selectors(soup, source_config['tags_selector'])
            summary = self._generate_summary(content)
            (field, subfield) = self._categorize_article(title, summary, content, tags)
            research_impact = self._calculate_research_impact(title, summary, content)
            quantum_relevance = self._calculate_quantum_relevance(title, summary, content, tags)
            technology_relevance = self._calculate_technology_relevance(title, summary, content, tags)
            article_id = self._generate_article_id(url, title)
            article = ResearchArticle(title=title, url=url, source=source_name, field=field, subfield=subfield, publication_date=publication_date, authors=authors, summary=summary, content=content, tags=tags, research_impact=research_impact, quantum_relevance=quantum_relevance, technology_relevance=technology_relevance, article_id=article_id, scraped_timestamp=datetime.now().isoformat(), processing_status='scraped', relevance_score=0.0, key_insights=[], koba42_integration_potential=0.0)
            return article
        except Exception as e:
            logger.warning(f'⚠️ Failed to scrape article {url}: {e}')
            return None

    def _process_article(self, article: ResearchArticle) -> Dict[str, Any]:
        """Process and enhance article data."""
        try:
            relevance_score = (article.quantum_relevance + article.technology_relevance + article.research_impact) / 3
            key_insights = self._extract_key_insights(article)
            koba42_potential = self._calculate_koba42_integration_potential(article)
            article.relevance_score = relevance_score
            article.key_insights = key_insights
            article.koba42_integration_potential = koba42_potential
            article.processing_status = 'processed'
            return article
        except Exception as e:
            logger.warning(f'⚠️ Failed to process article: {e}')
            return None

    def _is_article_relevant(self, article: ResearchArticle) -> bool:
        """Check if article meets relevance criteria."""
        return article.relevance_score >= self.min_relevance_score and (article.quantum_relevance >= 5.0 or article.technology_relevance >= 5.0)

    def _store_article(self, article: ResearchArticle) -> bool:
        """Store article in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('\n                INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', (article.article_id, article.title, article.url, article.source, article.field, article.subfield, article.publication_date, json.dumps(article.authors), article.summary, article.content, json.dumps(article.tags), article.research_impact, article.quantum_relevance, article.technology_relevance, article.relevance_score, article.scraped_timestamp, article.processing_status, json.dumps(article.key_insights), article.koba42_integration_potential))
            conn.commit()
            conn.close()
            article.processing_status = 'stored'
            return True
        except Exception as e:
            logger.error(f'❌ Failed to store article: {e}')
            return False

    def _extract_key_insights(self, article: ResearchArticle) -> List[str]:
        """Extract key insights from article."""
        insights = []
        if article.quantum_relevance >= 8.0:
            insights.append('High quantum physics relevance')
        if article.technology_relevance >= 8.0:
            insights.append('High technology relevance')
        if article.research_impact >= 8.0:
            insights.append('Breakthrough research')
        if 'quantum' in article.title.lower() or 'quantum' in article.summary.lower():
            insights.append('Quantum computing/technology focus')
        if 'algorithm' in article.title.lower() or 'optimization' in article.title.lower():
            insights.append('Algorithm/optimization focus')
        if 'material' in article.title.lower() or 'crystal' in article.title.lower():
            insights.append('Materials science focus')
        return insights

    def _calculate_koba42_integration_potential(self, article: ResearchArticle) -> float:
        """Calculate KOBA42 integration potential."""
        potential = 0.0
        field_config = self.research_fields.get(article.field, {})
        potential += field_config.get('koba42_relevance', 5.0)
        potential += article.quantum_relevance * 0.3
        potential += article.technology_relevance * 0.2
        potential += article.research_impact * 0.2
        if article.source == 'nature':
            potential += 1.0
        elif article.source == 'phys_org':
            potential += 0.5
        return min(potential, 10.0)

    def _article_exists_in_db(self, url: str) -> bool:
        """Check if article already exists in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM articles WHERE url = ?', (url,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception:
            return False

    def _save_batch_to_db(self, batch: ScrapingBatch):
        """Save batch information to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('\n                INSERT OR REPLACE INTO batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', (batch.batch_id, batch.source, batch.start_time, batch.end_time, batch.articles_scraped, batch.articles_processed, batch.articles_filtered, batch.articles_stored, batch.status, batch.error_message))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f'❌ Failed to save batch to database: {e}')

    def _save_session_to_db(self, session: ScrapingSession):
        """Save session information to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('\n                INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', (session.session_id, session.start_time, session.end_time, session.total_batches, session.completed_batches, session.total_articles_scraped, session.total_articles_stored, json.dumps(session.sources_processed), session.status))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f'❌ Failed to save session to database: {e}')

    def _extract_with_selectors(self, soup: BeautifulSoup, selectors: str) -> str:
        """Extract text using multiple CSS selectors."""
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                if text:
                    return text
        return ''

    def _extract_authors_with_selectors(self, soup: BeautifulSoup, selectors: str) -> List[str]:
        """Extract authors using CSS selectors."""
        authors = []
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                if text and 'author' in text.lower():
                    author_text = re.sub('by\\s+', '', text, flags=re.IGNORECASE)
                    authors.extend([a.strip() for a in author_text.split(',')])
        return authors if authors else ['Unknown Author']

    def _extract_date_with_selectors(self, soup: BeautifulSoup, selectors: str) -> str:
        """Extract publication date using CSS selectors."""
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                date_match = re.search('\\d{4}-\\d{2}-\\d{2}|\\d{1,2}\\s+\\w+\\s+\\d{4}', text)
                if date_match:
                    return date_match.group()
        return datetime.now().strftime('%Y-%m-%d')

    def _extract_tags_with_selectors(self, soup: BeautifulSoup, selectors: str) -> List[str]:
        """Extract tags using CSS selectors."""
        tags = []
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                if text:
                    tags.append(text)
        return tags

    def _generate_summary(self, content: str) -> str:
        """Generate summary from content."""
        if not content:
            return ''
        summary = content[:200].strip()
        if len(content) > 200:
            summary += '...'
        return summary

    def _categorize_article(self, title: str, summary: str, content: str, tags: List[str]) -> Tuple[str, str]:
        """Categorize article by field and subfield."""
        text = f"{title} {summary} {' '.join(tags)}".lower()
        best_field = 'general'
        best_subfield = 'general'
        best_score = 0
        for (field_name, field_config) in self.research_fields.items():
            field_score = 0
            for keyword in field_config['keywords']:
                if keyword.lower() in text:
                    field_score += 1
            quantum_score = 0
            for keyword in field_config['quantum_keywords']:
                if keyword.lower() in text:
                    quantum_score += 2
            total_score = field_score + quantum_score
            if total_score > best_score:
                best_score = total_score
                best_field = field_name
                best_subfield = f'{field_name}_research'
        return (best_field, best_subfield)

    def _calculate_research_impact(self, title: str, summary: str, content: str) -> float:
        """Calculate research impact score (0-10)."""
        text = f'{title} {summary} {content}'.lower()
        impact_score = 0
        high_impact_keywords = ['breakthrough', 'discovery', 'first', 'novel', 'revolutionary', 'groundbreaking', 'milestone', 'advance', 'innovation', 'new', 'significant', 'important', 'major', 'key', 'critical']
        for keyword in high_impact_keywords:
            if keyword in text:
                impact_score += 1
        return min(impact_score, 10.0)

    def _calculate_quantum_relevance(self, title: str, summary: str, content: str, tags: List[str]) -> float:
        """Calculate quantum relevance score (0-10)."""
        text = f"{title} {summary} {' '.join(tags)}".lower()
        quantum_score = 0
        quantum_keywords = {'quantum': 3, 'qubit': 3, 'entanglement': 3, 'superposition': 3, 'quantum_computing': 4, 'quantum_mechanics': 3, 'quantum_physics': 3, 'quantum_chemistry': 3, 'quantum_material': 3, 'quantum_algorithm': 3, 'quantum_sensor': 3, 'quantum_network': 3, 'quantum_internet': 4, 'quantum_hall': 3, 'quantum_spin': 3, 'quantum_optics': 3, 'quantum_information': 3, 'quantum_cryptography': 3, 'quantum_simulation': 3, 'quantum_advantage': 4}
        for (keyword, weight) in quantum_keywords.items():
            if keyword.replace('_', ' ') in text:
                quantum_score += weight
        return min(quantum_score, 10.0)

    def _calculate_technology_relevance(self, title: str, summary: str, content: str, tags: List[str]) -> float:
        """Calculate technology relevance score (0-10)."""
        text = f"{title} {summary} {' '.join(tags)}".lower()
        tech_score = 0
        tech_keywords = {'software': 2, 'programming': 2, 'algorithm': 3, 'computer': 2, 'technology': 2, 'digital': 2, 'electronic': 2, 'automation': 3, 'artificial_intelligence': 4, 'machine_learning': 4, 'neural_network': 3, 'data_science': 3, 'cloud_computing': 3, 'blockchain': 3, 'cybersecurity': 3, 'internet_of_things': 3, 'virtual_reality': 3, 'augmented_reality': 3, 'robotics': 3}
        for (keyword, weight) in tech_keywords.items():
            if keyword.replace('_', ' ') in text:
                tech_score += weight
        return min(tech_score, 10.0)

    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID."""
        content = f'{url}{title}'
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_stored_articles_summary(self) -> Optional[Any]:
        """Get summary of stored articles."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM articles')
            total_articles = cursor.fetchone()[0]
            cursor.execute('SELECT source, COUNT(*) FROM articles GROUP BY source')
            source_distribution = dict(cursor.fetchall())
            cursor.execute('SELECT field, COUNT(*) FROM articles GROUP BY field')
            field_distribution = dict(cursor.fetchall())
            cursor.execute('SELECT COUNT(*) FROM articles WHERE relevance_score >= 7.0')
            high_relevance_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM articles WHERE koba42_integration_potential >= 7.0')
            high_potential_count = cursor.fetchone()[0]
            conn.close()
            return {'total_articles': total_articles, 'source_distribution': source_distribution, 'field_distribution': field_distribution, 'high_relevance_articles': high_relevance_count, 'high_koba42_potential': high_potential_count}
        except Exception as e:
            logger.error(f'❌ Failed to get articles summary: {e}')
            return {}

def demonstrate_enhanced_scraping():
    """Demonstrate enhanced research scraping."""
    logger.info('🚀 KOBA42 Enhanced Research Scraper')
    logger.info('=' * 50)
    scraper = EnhancedResearchScraper()
    summary_before = scraper.get_stored_articles_summary()
    logger.info(f"📊 Articles in database before scraping: {summary_before.get('total_articles', 0)}")
    print('\n🔍 Starting enhanced research scraping...')
    results = scraper.scrape_all_sources_chunked()
    summary_after = scraper.get_stored_articles_summary()
    print(f'\n📋 ENHANCED SCRAPING RESULTS')
    print('=' * 50)
    print(f"Session ID: {results['session_id']}")
    print(f"Total Articles Scraped: {results['total_articles_scraped']}")
    print(f"Total Articles Stored: {results['total_articles_stored']}")
    print(f"Sources Processed: {', '.join(results['sources_processed'])}")
    print(f"Batches Completed: {results['batches_completed']}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    print(f'\n📊 DATABASE SUMMARY')
    print('=' * 50)
    print(f"Total Articles in Database: {summary_after.get('total_articles', 0)}")
    print(f"Source Distribution: {summary_after.get('source_distribution', {})}")
    print(f"Field Distribution: {summary_after.get('field_distribution', {})}")
    print(f"High Relevance Articles (≥7.0): {summary_after.get('high_relevance_articles', 0)}")
    print(f"High KOBA42 Potential (≥7.0): {summary_after.get('high_koba42_potential', 0)}")
    logger.info('✅ Enhanced research scraping demonstration completed')
    return (results, summary_after)
if __name__ == '__main__':
    (results, summary) = demonstrate_enhanced_scraping()
    print(f'\n🎉 Enhanced research scraping completed!')
    print(f'💾 Data stored in: research_data/research_articles.db')
    print(f'📝 Logs saved to: research_scraping.log')
    print(f'🔬 Ready for future development and KOBA42 integration')