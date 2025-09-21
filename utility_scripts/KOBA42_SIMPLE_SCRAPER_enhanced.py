
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
KOBA42 SIMPLE SCRAPER
======================
Simple Research Article Scraper for Testing and Analysis
=======================================================

This scraper is designed to capture articles for analysis and testing
the comprehensive exploration system.
"""
import requests
import json
import logging
import time
import random
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Dict, Any, List
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleResearchScraper:
    """Simple research scraper for testing and analysis."""

    def __init__(self):
        self.db_path = 'research_data/research_articles.db'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        logger.info('Simple Research Scraper initialized')

    def scrape_sample_articles(self) -> Dict[str, Any]:
        """Scrape sample articles for testing."""
        logger.info('🔍 Starting simple research scraping...')
        sample_articles = [{'title': 'Breakthrough in Quantum Computing: New Algorithm Achieves Quantum Advantage', 'url': 'https://example.com/quantum-breakthrough-1', 'source': 'phys_org', 'field': 'physics', 'subfield': 'quantum_physics', 'publication_date': '2024-01-15', 'authors': ['Dr. Sarah Chen', 'Prof. Michael Rodriguez'], 'summary': 'Researchers have developed a novel quantum algorithm that demonstrates quantum advantage for optimization problems, marking a significant breakthrough in quantum computing technology.', 'content': 'A team of physicists has made a significant breakthrough in quantum computing technology. The new algorithm leverages quantum entanglement and superposition to solve complex optimization problems that would take classical computers years to complete. This revolutionary development opens new possibilities for quantum computing applications in cryptography, materials science, and artificial intelligence.', 'tags': ['quantum computing', 'quantum algorithm', 'breakthrough', 'optimization', 'quantum advantage'], 'research_impact': 9.2, 'quantum_relevance': 9.8, 'technology_relevance': 8.5}, {'title': 'Novel Machine Learning Framework for Quantum Chemistry Simulations', 'url': 'https://example.com/ml-quantum-chemistry-1', 'source': 'nature', 'field': 'chemistry', 'subfield': 'quantum_chemistry', 'publication_date': '2024-01-12', 'authors': ['Dr. Elena Petrova', 'Dr. James Wilson'], 'summary': 'A new machine learning approach accelerates quantum chemistry calculations by orders of magnitude while maintaining accuracy, enabling faster drug discovery and materials design.', 'content': 'Scientists have developed an innovative machine learning framework that dramatically accelerates quantum chemistry simulations. This cutting-edge technology combines artificial intelligence with quantum mechanics to predict molecular properties with unprecedented speed and accuracy. The breakthrough has immediate applications in pharmaceutical research, materials science, and chemical engineering.', 'tags': ['machine learning', 'quantum chemistry', 'simulation', 'drug discovery', 'materials science'], 'research_impact': 8.8, 'quantum_relevance': 7.5, 'technology_relevance': 9.2}, {'title': 'Revolutionary Quantum Internet Protocol Achieves Secure Communication', 'url': 'https://example.com/quantum-internet-1', 'source': 'infoq', 'field': 'technology', 'subfield': 'quantum_networking', 'publication_date': '2024-01-10', 'authors': ['Dr. Alex Thompson', 'Dr. Maria Garcia'], 'summary': 'Researchers demonstrate a new quantum internet protocol that enables secure quantum communication over unprecedented distances, bringing quantum internet closer to reality.', 'content': 'A groundbreaking quantum internet protocol has been developed that enables secure quantum communication over unprecedented distances. This revolutionary technology uses quantum entanglement to create unhackable communication channels, marking a major milestone in the development of quantum internet infrastructure.', 'tags': ['quantum internet', 'quantum communication', 'cryptography', 'entanglement', 'security'], 'research_impact': 8.5, 'quantum_relevance': 9.0, 'technology_relevance': 8.8}, {'title': 'Advanced AI Algorithm Discovers New Quantum Materials', 'url': 'https://example.com/ai-quantum-materials-1', 'source': 'phys_org', 'field': 'materials_science', 'subfield': 'quantum_materials', 'publication_date': '2024-01-08', 'authors': ['Dr. Wei Zhang', 'Prof. Lisa Anderson'], 'summary': 'Artificial intelligence has identified previously unknown quantum materials with exceptional properties, accelerating the discovery of next-generation quantum technologies.', 'content': 'Artificial intelligence has made a remarkable breakthrough in materials science by discovering new quantum materials with exceptional properties. This pioneering research combines machine learning algorithms with quantum physics to predict and identify materials that could revolutionize quantum computing, sensing, and communication technologies.', 'tags': ['artificial intelligence', 'quantum materials', 'machine learning', 'materials discovery', 'quantum technology'], 'research_impact': 8.2, 'quantum_relevance': 8.8, 'technology_relevance': 8.8}, {'title': 'Innovative Software Framework for Quantum Programming', 'url': 'https://example.com/quantum-software-1', 'source': 'infoq', 'field': 'software', 'subfield': 'quantum_programming', 'publication_date': '2024-01-05', 'authors': ['Dr. Emily Johnson', 'Dr. Carlos Mendez'], 'summary': 'A comprehensive quantum software development kit enables developers to write and test quantum algorithms easily, democratizing quantum computing access.', 'content': 'A comprehensive quantum software development kit has been released that democratizes access to quantum computing. This innovative framework provides developers with intuitive tools to write, test, and optimize quantum algorithms, accelerating the development of quantum applications across various industries.', 'tags': ['quantum software', 'SDK', 'quantum programming', 'development tools', 'quantum computing'], 'research_impact': 7.5, 'quantum_relevance': 7.8, 'technology_relevance': 8.5}]
        results = {'articles_scraped': len(sample_articles), 'articles_stored': 0, 'processing_time': 0}
        start_time = time.time()
        for article_data in sample_articles:
            try:
                article_id = self._generate_article_id(article_data['url'], article_data['title'])
                relevance_score = (article_data['quantum_relevance'] + article_data['technology_relevance'] + article_data['research_impact']) / 3
                key_insights = self._extract_key_insights(article_data)
                koba42_potential = self._calculate_koba42_potential(article_data)
                if self._store_article(article_data, article_id, relevance_score, key_insights, koba42_potential):
                    results['articles_stored'] += 1
                    logger.info(f"✅ Stored article: {article_data['title'][:50]}...")
                time.sleep(random.uniform(0.5, 1.0))
            except Exception as e:
                logger.warning(f'⚠️ Failed to process article: {e}')
                continue
        results['processing_time'] = time.time() - start_time
        logger.info(f'✅ Simple scraping completed')
        logger.info(f"📊 Articles scraped: {results['articles_scraped']}")
        logger.info(f"💾 Articles stored: {results['articles_stored']}")
        return results

    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID."""
        content = f'{url}{title}'
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_key_insights(self, article_data: Dict[str, Any]) -> List[str]:
        """Extract key insights from article."""
        insights = []
        if article_data['quantum_relevance'] >= 8.0:
            insights.append('High quantum physics relevance')
        if article_data['technology_relevance'] >= 8.0:
            insights.append('High technology relevance')
        if article_data['research_impact'] >= 8.0:
            insights.append('Breakthrough research')
        text = f"{article_data['title']} {article_data['summary']}".lower()
        if 'quantum' in text:
            insights.append('Quantum computing/technology focus')
        if 'algorithm' in text or 'optimization' in text:
            insights.append('Algorithm/optimization focus')
        if 'material' in text or 'crystal' in text:
            insights.append('Materials science focus')
        if 'software' in text or 'programming' in text:
            insights.append('Software/programming focus')
        if 'breakthrough' in text or 'revolutionary' in text:
            insights.append('Breakthrough/revolutionary research')
        return insights

    def _calculate_koba42_potential(self, article_data: Dict[str, Any]) -> float:
        """Calculate KOBA42 integration potential."""
        potential = 0.0
        field_potentials = {'physics': 9.0, 'chemistry': 7.5, 'technology': 8.5, 'software': 8.0, 'materials_science': 8.0}
        potential += field_potentials.get(article_data['field'], 5.0)
        potential += article_data['quantum_relevance'] * 0.4
        potential += article_data['technology_relevance'] * 0.3
        potential += article_data['research_impact'] * 0.3
        source_bonuses = {'nature': 1.0, 'phys_org': 0.5, 'infoq': 0.3}
        potential += source_bonuses.get(article_data['source'], 0.0)
        text = f"{article_data['title']} {article_data['summary']}".lower()
        if 'breakthrough' in text or 'revolutionary' in text or 'novel' in text:
            potential += 1.0
        return min(potential, 10.0)

    def _store_article(self, article_data: Dict[str, Any], article_id: str, relevance_score: float, key_insights: List[str], koba42_potential: float) -> bool:
        """Store article in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('\n                INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ', (article_id, article_data['title'], article_data['url'], article_data['source'], article_data['field'], article_data['subfield'], article_data['publication_date'], json.dumps(article_data['authors']), article_data['summary'], article_data['content'], json.dumps(article_data['tags']), article_data['research_impact'], article_data['quantum_relevance'], article_data['technology_relevance'], relevance_score, datetime.now().isoformat(), 'stored', json.dumps(key_insights), koba42_potential))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f'❌ Failed to store article: {e}')
            return False

def demonstrate_simple_scraping():
    """Demonstrate simple research scraping."""
    logger.info('🚀 KOBA42 Simple Research Scraper')
    logger.info('=' * 50)
    scraper = SimpleResearchScraper()
    print('\n🔍 Starting simple research scraping...')
    results = scraper.scrape_sample_articles()
    print(f'\n📋 SIMPLE SCRAPING RESULTS')
    print('=' * 50)
    print(f"Articles Scraped: {results['articles_scraped']}")
    print(f"Articles Stored: {results['articles_stored']}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    try:
        conn = sqlite3.connect(scraper.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM articles')
        total_stored = cursor.fetchone()[0]
        if total_stored > 0:
            cursor.execute('\n                SELECT title, source, field, relevance_score, koba42_integration_potential \n                FROM articles ORDER BY relevance_score DESC LIMIT 5\n            ')
            top_articles = cursor.fetchall()
            print(f'\n📊 TOP STORED ARTICLES')
            print('=' * 50)
            for (i, article) in enumerate(top_articles, 1):
                print(f'\n{i}. {article[0][:60]}...')
                print(f'   Source: {article[1]}')
                print(f'   Field: {article[2]}')
                print(f'   Relevance Score: {article[3]:.2f}')
                print(f'   KOBA42 Potential: {article[4]:.2f}')
        conn.close()
    except Exception as e:
        logger.error(f'❌ Error checking database: {e}')
    logger.info('✅ Simple research scraping demonstration completed')
    return results
if __name__ == '__main__':
    results = demonstrate_simple_scraping()
    print(f'\n🎉 Simple research scraping completed!')
    print(f'💾 Data stored in: research_data/research_articles.db')
    print(f'🔬 Ready for comprehensive exploration and analysis')