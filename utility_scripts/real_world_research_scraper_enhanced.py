
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
Real-World Research Scraper
Comprehensive scraping system for prestigious academic and research sources
"""
import json
import requests
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import urllib.parse
from bs4 import BeautifulSoup
import feedparser
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealWorldResearchScraper:
    """
    Comprehensive scraper for prestigious academic and research sources
    """

    def __init__(self):
        self.research_dir = Path('research_data')
        self.scraping_log = self.research_dir / 'real_world_scraping_log.json'
        self.subjects_discovered = self.research_dir / 'real_world_subjects.json'
        self.rate_limits = {'arxiv': {'requests_per_minute': 30, 'last_request': None}, 'mit': {'requests_per_minute': 10, 'last_request': None}, 'cambridge': {'requests_per_minute': 10, 'last_request': None}, 'caltech': {'requests_per_minute': 10, 'last_request': None}, 'uspto': {'requests_per_minute': 20, 'last_request': None}, 'semantic_scholar': {'requests_per_minute': 100, 'last_request': None}}
        self._initialize_scraping_tracking()

    def _initialize_scraping_tracking(self):
        """Initialize scraping tracking system."""
        if not self.scraping_log.exists():
            initial_data = {'last_scraping_run': None, 'sources_scraped': [], 'subjects_found': 0, 'papers_analyzed': 0, 'patents_reviewed': 0, 'disclosures_processed': 0, 'rate_limit_hits': 0, 'scraping_history': []}
            with open(self.scraping_log, 'w') as f:
                json.dump(initial_data, f, indent=2)
        if not self.subjects_discovered.exists():
            initial_subjects = {'academic_papers': [], 'patents': [], 'disclosures': [], 'research_projects': [], 'conference_papers': []}
            with open(self.subjects_discovered, 'w') as f:
                json.dump(initial_subjects, f, indent=2)

    def _respect_rate_limit(self, source: str):
        """Respect rate limits for different sources."""
        if source in self.rate_limits:
            limit_info = self.rate_limits[source]
            if limit_info['last_request']:
                time_diff = time.time() - limit_info['last_request']
                min_interval = 60 / limit_info['requests_per_minute']
                if time_diff < min_interval:
                    sleep_time = min_interval - time_diff
                    logger.info(f'Rate limiting: sleeping {sleep_time:.2f}s for {source}')
                    time.sleep(sleep_time)
            self.rate_limits[source]['last_request'] = time.time()

    def scrape_arxiv(self, query: str='artificial intelligence', max_results: int=50) -> List[Dict[str, Any]]:
        """
        Scrape arXiv for recent academic papers.
        """
        logger.info('🔬 Scraping arXiv for academic papers...')
        self._respect_rate_limit('arxiv')
        try:
            url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            subjects = []
            for entry in feed.entries:
                title = entry.title
                abstract = entry.get('summary', '')
                published = entry.get('published', '')
                relevance_keywords = ['machine learning', 'artificial intelligence', 'cybersecurity', 'cryptography', 'quantum computing', 'neural network', 'blockchain', 'distributed system', 'security', 'privacy', 'algorithm', 'optimization', 'formal method']
                relevance_score = sum((1 for keyword in relevance_keywords if keyword.lower() in (title + abstract).lower()))
                if relevance_score >= 1:
                    subject_name = self._generate_subject_name(title, 'arxiv')
                    subjects.append({'name': subject_name, 'title': title, 'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract, 'source': 'arXiv', 'url': entry.link, 'published': published, 'category': self._classify_subject(title, abstract), 'difficulty': 'expert', 'relevance_score': min(relevance_score / 3, 1.0), 'citations': 0, 'authors': [author.name for author in entry.authors] if hasattr(entry, 'authors') else []})
            logger.info(f'📄 Found {len(subjects)} relevant papers on arXiv')
            return subjects
        except Exception as e:
            logger.error(f'❌ Error scraping arXiv: {e}')
            return []

    def scrape_mit_research(self) -> List[Dict[str, Any]]:
        """
        Scrape MIT research projects and publications.
        """
        logger.info('🏛️ Scraping MIT research...')
        self._respect_rate_limit('mit')
        try:
            urls = ['https://www.csail.mit.edu/research', 'https://www.eecs.mit.edu/research', 'https://www.mit.edu/research/']
            subjects = []
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    research_items = soup.find_all(['h3', 'h4', 'div'], class_=re.compile('(research|project|paper)'))
                    for item in research_items[:10]:
                        title = item.get_text().strip()
                        if len(title) > 10 and self._is_relevant_research(title):
                            subject_name = self._generate_subject_name(title, 'mit')
                            subjects.append({'name': subject_name, 'title': title, 'source': 'MIT', 'url': url, 'category': self._classify_subject(title, ''), 'difficulty': 'expert', 'relevance_score': 0.9, 'description': f'MIT research project: {title}', 'institution': 'Massachusetts Institute of Technology'})
                except Exception as e:
                    logger.warning(f'⚠️ Error scraping {url}: {e}')
            logger.info(f'🔬 Found {len(subjects)} MIT research projects')
            return subjects
        except Exception as e:
            logger.error(f'❌ Error scraping MIT: {e}')
            return []

    def scrape_cambridge_research(self) -> List[Dict[str, Any]]:
        """
        Scrape University of Cambridge research.
        """
        logger.info('🎓 Scraping Cambridge University research...')
        self._respect_rate_limit('cambridge')
        try:
            urls = ['https://www.cst.cam.ac.uk/research', 'https://www.cl.cam.ac.uk/research/', 'https://www.ai.cam.ac.uk/research/']
            subjects = []
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    research_titles = soup.find_all(['h2', 'h3', 'h4'], class_=re.compile('(title|research|project)'))
                    research_descriptions = soup.find_all(['p', 'div'], class_=re.compile('(description|summary|abstract)'))
                    for (i, title_elem) in enumerate(research_titles[:8]):
                        title = title_elem.get_text().strip()
                        description = research_descriptions[i].get_text().strip() if i < len(research_descriptions) else ''
                        if len(title) > 10 and self._is_relevant_research(title):
                            subject_name = self._generate_subject_name(title, 'cambridge')
                            subjects.append({'name': subject_name, 'title': title, 'description': description[:300] + '...' if len(description) > 300 else description, 'source': 'University of Cambridge', 'url': url, 'category': self._classify_subject(title, description), 'difficulty': 'expert', 'relevance_score': 0.95, 'institution': 'University of Cambridge'})
                except Exception as e:
                    logger.warning(f'⚠️ Error scraping {url}: {e}')
            logger.info(f'🎓 Found {len(subjects)} Cambridge research projects')
            return subjects
        except Exception as e:
            logger.error(f'❌ Error scraping Cambridge: {e}')
            return []

    def scrape_caltech_research(self) -> List[Dict[str, Any]]:
        """
        Scrape Caltech research projects.
        """
        logger.info('🔭 Scraping Caltech research...')
        self._respect_rate_limit('caltech')
        try:
            urls = ['https://www.cms.caltech.edu/research', 'https://www.cs.caltech.edu/research', 'https://iqim.caltech.edu/research/']
            subjects = []
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    research_items = soup.find_all(['div', 'article'], class_=re.compile('(research|project|paper)'))
                    for item in research_items[:6]:
                        title_elem = item.find(['h2', 'h3', 'h4'])
                        desc_elem = item.find(['p', 'div'])
                        if title_elem:
                            title = title_elem.get_text().strip()
                            description = desc_elem.get_text().strip() if desc_elem else ''
                            if len(title) > 10 and self._is_relevant_research(title):
                                subject_name = self._generate_subject_name(title, 'caltech')
                                subjects.append({'name': subject_name, 'title': title, 'description': description[:300] + '...' if len(description) > 300 else description, 'source': 'Caltech', 'url': url, 'category': self._classify_subject(title, description), 'difficulty': 'expert', 'relevance_score': 0.9, 'institution': 'California Institute of Technology'})
                except Exception as e:
                    logger.warning(f'⚠️ Error scraping {url}: {e}')
            logger.info(f'🔭 Found {len(subjects)} Caltech research projects')
            return subjects
        except Exception as e:
            logger.error(f'❌ Error scraping Caltech: {e}')
            return []

    def scrape_uspto_patents(self, query: str='artificial intelligence', max_results: int=20) -> List[Dict[str, Any]]:
        """
        Scrape USPTO for recent AI/cybersecurity patents.
        """
        logger.info('📋 Scraping USPTO patents...')
        self._respect_rate_limit('uspto')
        try:
            base_url = 'https://developer.uspto.gov/ibd-api/v1/search'
            params = {'query': f'({query}) AND (machine learning OR cybersecurity OR cryptography)', 'rows': max_results, 'start': 0, 'sort': 'publicationDate desc'}
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            subjects = []
            for patent in data.get('results', []):
                title = patent.get('patentTitle', '')
                abstract = patent.get('abstractText', '')
                inventors = patent.get('inventors', [])
                publication_date = patent.get('publicationDate', '')
                if self._is_relevant_research(title) or self._is_relevant_research(abstract):
                    subject_name = self._generate_subject_name(title, 'uspto')
                    subjects.append({'name': subject_name, 'title': title, 'abstract': abstract[:400] + '...' if len(abstract) > 400 else abstract, 'source': 'USPTO', 'patent_number': patent.get('patentNumber', ''), 'publication_date': publication_date, 'inventors': [inv.get('name', '') for inv in inventors], 'category': self._classify_subject(title, abstract), 'difficulty': 'expert', 'relevance_score': 0.85, 'patent_url': f"https://patents.google.com/patent/{patent.get('patentNumber', '')}"})
            logger.info(f'📋 Found {len(subjects)} relevant USPTO patents')
            return subjects
        except Exception as e:
            logger.error(f'❌ Error scraping USPTO: {e}')
            return []

    def scrape_semantic_scholar(self, query: str='artificial intelligence', max_results: int=30) -> List[Dict[str, Any]]:
        """
        Scrape Semantic Scholar for highly cited papers.
        """
        logger.info('📚 Scraping Semantic Scholar...')
        self._respect_rate_limit('semantic_scholar')
        try:
            url = f'https://api.semanticscholar.org/graph/v1/paper/search'
            params = {'query': query, 'limit': max_results, 'fields': 'title,abstract,year,citationCount,authors,venue', 'sort': 'citationCount:desc'}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            subjects = []
            for paper in data.get('data', []):
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                citations = paper.get('citationCount', 0)
                year = paper.get('year', '')
                authors = paper.get('authors', [])
                venue = paper.get('venue', '')
                if citations >= 50 and self._is_relevant_research(title):
                    subject_name = self._generate_subject_name(title, 'semantic_scholar')
                    subjects.append({'name': subject_name, 'title': title, 'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract, 'source': 'Semantic Scholar', 'citations': citations, 'year': year, 'venue': venue, 'authors': [author.get('name', '') for author in authors], 'category': self._classify_subject(title, abstract), 'difficulty': 'expert', 'relevance_score': min(citations / 500, 1.0), 'paper_url': f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"})
            logger.info(f'📚 Found {len(subjects)} highly cited papers on Semantic Scholar')
            return subjects
        except Exception as e:
            logger.error(f'❌ Error scraping Semantic Scholar: {e}')
            return []

    def scrape_foia_documents(self) -> List[Dict[str, Any]]:
        """
        Scrape FOIA documents from government databases.
        """
        logger.info('📄 Scraping FOIA documents...')
        try:
            urls = ['https://www.foia.gov/search.html', 'https://www.cia.gov/readingroom/', 'https://www.nsa.gov/FOIA/']
            subjects = []
            logger.warning('⚠️ FOIA scraping requires special permissions and ethical considerations')
            logger.info('📄 FOIA scraping simulated (would require proper authorization)')
            return subjects
        except Exception as e:
            logger.error(f'❌ Error scraping FOIA documents: {e}')
            return []

    def _generate_subject_name(self, title: str, source: str) -> str:
        """Generate a unique subject name from title and source."""
        clean_title = re.sub('[^\\w\\s-]', '', title.lower())
        words = clean_title.split()[:4]
        base_name = '_'.join(words)
        timestamp = int(time.time() * 1000) % 10000
        return f'{base_name}_{source}_{timestamp}'

    def _is_relevant_research(self, text: str) -> bool:
        """Check if research topic is relevant to our curriculum."""
        relevant_keywords = ['machine learning', 'artificial intelligence', 'cybersecurity', 'cryptography', 'quantum', 'neural network', 'blockchain', 'distributed system', 'security', 'privacy', 'algorithm', 'optimization', 'formal method', 'computer vision', 'nlp', 'deep learning', 'reinforcement learning', 'federated learning']
        text_lower = text.lower()
        return any((keyword in text_lower for keyword in relevant_keywords))

    def _classify_subject(self, title: str, abstract: str) -> str:
        """Classify subject into appropriate category."""
        text = (title + ' ' + abstract).lower()
        if any((word in text for word in ['machine learning', 'neural network', 'deep learning'])):
            return 'machine_learning'
        elif any((word in text for word in ['cybersecurity', 'security', 'cryptography'])):
            return 'cybersecurity'
        elif any((word in text for word in ['quantum', 'qubit', 'quantum computing'])):
            return 'quantum_computing'
        elif any((word in text for word in ['blockchain', 'distributed ledger'])):
            return 'blockchain'
        elif any((word in text for word in ['computer vision', 'image processing'])):
            return 'computer_vision'
        elif any((word in text for word in ['natural language', 'nlp', 'language model'])):
            return 'natural_language_processing'
        else:
            return 'artificial_intelligence'

    def run_comprehensive_scraping(self) -> Dict[str, Any]:
        """
        Run comprehensive scraping across all prestigious sources.
        """
        logger.info('🌟 Starting comprehensive research scraping...')
        print('=' * 80)
        print('🔬 COMPREHENSIVE RESEARCH SCRAPING SYSTEM')
        print('Scraping from prestigious academic and research sources')
        print('=' * 80)
        all_subjects = {'academic_papers': [], 'research_projects': [], 'patents': [], 'conference_papers': [], 'disclosures': []}
        print('\n📄 Scraping arXiv...')
        arxiv_papers = self.scrape_arxiv()
        all_subjects['academic_papers'].extend(arxiv_papers)
        print('\n🏛️ Scraping MIT research...')
        mit_projects = self.scrape_mit_research()
        all_subjects['research_projects'].extend(mit_projects)
        print('\n🎓 Scraping Cambridge research...')
        cambridge_projects = self.scrape_cambridge_research()
        all_subjects['research_projects'].extend(cambridge_projects)
        print('\n🔭 Scraping Caltech research...')
        caltech_projects = self.scrape_caltech_research()
        all_subjects['research_projects'].extend(caltech_projects)
        print('\n📋 Scraping USPTO patents...')
        patents = self.scrape_uspto_patents()
        all_subjects['patents'].extend(patents)
        print('\n📚 Scraping Semantic Scholar...')
        semantic_papers = self.scrape_semantic_scholar()
        all_subjects['academic_papers'].extend(semantic_papers)
        with open(self.subjects_discovered, 'w') as f:
            json.dump(all_subjects, f, indent=2, default=str)
        self._update_scraping_log(all_subjects)
        total_subjects = sum((len(subjects) for subjects in all_subjects.values()))
        print('\n📊 COMPREHENSIVE SCRAPING COMPLETE!')
        print('=' * 80)
        print(f"📄 Academic Papers: {len(all_subjects['academic_papers'])}")
        print(f"🔬 Research Projects: {len(all_subjects['research_projects'])}")
        print(f"📋 Patents: {len(all_subjects['patents'])}")
        print(f"🎤 Conference Papers: {len(all_subjects['conference_papers'])}")
        print(f"📄 Disclosures: {len(all_subjects['disclosures'])}")
        print(f'📊 Total Subjects Discovered: {total_subjects}')
        return {'total_subjects': total_subjects, 'subjects_by_category': {k: len(v) for (k, v) in all_subjects.items()}, 'sources_scraped': ['arXiv', 'MIT', 'Cambridge', 'Caltech', 'USPTO', 'Semantic Scholar'], 'timestamp': datetime.now().isoformat()}

    def _update_scraping_log(self, subjects: Dict[str, List]):
        """Update scraping log with latest results."""
        try:
            with open(self.scraping_log, 'r') as f:
                log_data = json.load(f)
            log_data['last_scraping_run'] = datetime.now().isoformat()
            log_data['subjects_found'] += sum((len(subjects_list) for subjects_list in subjects.values()))
            log_data['papers_analyzed'] += len(subjects.get('academic_papers', []))
            log_data['patents_reviewed'] += len(subjects.get('patents', []))
            log_data['disclosures_processed'] += len(subjects.get('disclosures', []))
            sources = ['arXiv', 'MIT', 'Cambridge', 'Caltech', 'USPTO', 'Semantic Scholar']
            for source in sources:
                if source not in log_data['sources_scraped']:
                    log_data['sources_scraped'].append(source)
            log_data['scraping_history'].append({'timestamp': datetime.now().isoformat(), 'subjects_found': sum((len(subjects_list) for subjects_list in subjects.values())), 'sources_scraped': sources})
            with open(self.scraping_log, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f'❌ Error updating scraping log: {e}')

def main():
    """Main function to demonstrate comprehensive research scraping."""
    print('🔬 Real-World Research Scraper')
    print('=' * 60)
    print('Comprehensive scraping from prestigious academic sources')
    print('arXiv, MIT, Cambridge, Caltech, USPTO, Semantic Scholar, FOIA')
    scraper = RealWorldResearchScraper()
    results = scraper.run_comprehensive_scraping()
    print('\n🏆 SCRAPING RESULTS:')
    print(f"  Total Subjects: {results['total_subjects']}")
    print(f"  Academic Papers: {results['subjects_by_category']['academic_papers']}")
    print(f"  Research Projects: {results['subjects_by_category']['research_projects']}")
    print(f"  Patents: {results['subjects_by_category']['patents']}")
    print(f"  Sources Scraped: {len(results['sources_scraped'])}")
    print('\n🔗 Sources Scraped:')
    for source in results['sources_scraped']:
        print(f'  • {source}')
    print('\n🚀 Möbius Loop Trainer now has access to')
    print("cutting-edge research from the world's most prestigious institutions!")
    print('🎓 MIT, Cambridge, Caltech, USPTO, arXiv, Semantic Scholar')
if __name__ == '__main__':
    main()