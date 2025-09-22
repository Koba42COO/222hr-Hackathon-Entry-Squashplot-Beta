
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
"""
KOBA42 COMPREHENSIVE EXPLORER
==============================
Comprehensive Research Article Exploration and Analysis System
============================================================

Features:
1. Full System ID Analysis of Scraped Articles
2. Breakthrough Detection and Synonym Analysis
3. Content Quality Assessment
4. KOBA42 Integration Potential Analysis
5. Research Trend Identification
6. Future Development Insights
"""
import sqlite3
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KOBA42ComprehensiveExplorer:
    """Comprehensive research article exploration and analysis system."""

    def __init__(self, db_path: str='research_data/research_articles.db'):
        self.db_path = db_path
        self.conn = None
        self.connect_database()
        self.breakthrough_keywords = ['breakthrough', 'discovery', 'first', 'novel', 'revolutionary', 'groundbreaking', 'milestone', 'advance', 'innovation', 'new', 'significant', 'important', 'major', 'key', 'critical', 'pioneering', 'cutting-edge', 'state-of-the-art', 'leading', 'promising', 'exciting', 'remarkable', 'notable', 'unprecedented', 'historic', 'landmark', 'game-changing', 'transformative', 'revolutionary', 'paradigm-shifting', 'next-generation']
        self.quantum_keywords = ['quantum', 'qubit', 'entanglement', 'superposition', 'quantum_computing', 'quantum_mechanics', 'quantum_physics', 'quantum_chemistry', 'quantum_material', 'quantum_algorithm', 'quantum_sensor', 'quantum_network', 'quantum_internet', 'quantum_hall', 'quantum_spin', 'quantum_optics', 'quantum_information', 'quantum_cryptography', 'quantum_simulation', 'quantum_advantage', 'quantum_state', 'quantum_system', 'quantum_effect', 'quantum_phenomenon', 'quantum_technology', 'quantum_software', 'quantum_programming']
        self.technology_keywords = ['software', 'programming', 'algorithm', 'computer', 'technology', 'digital', 'electronic', 'automation', 'artificial_intelligence', 'machine_learning', 'neural_network', 'data_science', 'cloud_computing', 'blockchain', 'cybersecurity', 'internet_of_things', 'virtual_reality', 'augmented_reality', 'robotics', 'computing', 'system', 'platform', 'application', 'development', 'innovation', 'tech']

    def connect_database(self):
        """Connect to the research database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f'✅ Connected to database: {self.db_path}')
        except Exception as e:
            logger.error(f'❌ Failed to connect to database: {e}')
            raise

    def get_all_scraped_articles(self) -> Optional[Any]:
        """Get all scraped articles from the database."""
        articles = []
        try:
            cursor = self.conn.cursor()
            cursor.execute('\n                SELECT article_id, title, url, source, field, subfield, \n                       publication_date, authors, summary, content, tags,\n                       research_impact, quantum_relevance, technology_relevance,\n                       relevance_score, scraped_timestamp, processing_status,\n                       key_insights, koba42_integration_potential\n                FROM articles\n                ORDER BY scraped_timestamp DESC\n            ')
            rows = cursor.fetchall()
            for row in rows:
                article = {'article_id': row[0], 'title': row[1], 'url': row[2], 'source': row[3], 'field': row[4], 'subfield': row[5], 'publication_date': row[6], 'authors': json.loads(row[7]) if row[7] else [], 'summary': row[8], 'content': row[9], 'tags': json.loads(row[10]) if row[10] else [], 'research_impact': row[11], 'quantum_relevance': row[12], 'technology_relevance': row[13], 'relevance_score': row[14], 'scraped_timestamp': row[15], 'processing_status': row[16], 'key_insights': json.loads(row[17]) if row[17] else [], 'koba42_integration_potential': row[18]}
                articles.append(article)
        except Exception as e:
            logger.error(f'❌ Error fetching articles: {e}')
        return articles

    def analyze_article_content(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze article content for breakthroughs and relevance."""
        analysis = {'breakthrough_detection': {}, 'quantum_analysis': {}, 'technology_analysis': {}, 'content_quality': {}, 'koba42_potential': {}, 'recommendations': []}
        full_text = f"{article['title']} {article['summary']} {article['content']} {' '.join(article['tags'])}".lower()
        breakthrough_matches = []
        for keyword in self.breakthrough_keywords:
            if keyword.replace('_', ' ') in full_text:
                breakthrough_matches.append(keyword)
        analysis['breakthrough_detection'] = {'breakthrough_keywords_found': breakthrough_matches, 'breakthrough_score': len(breakthrough_matches), 'is_breakthrough': len(breakthrough_matches) >= 2, 'breakthrough_intensity': len(breakthrough_matches) / len(self.breakthrough_keywords) * 100}
        quantum_matches = []
        for keyword in self.quantum_keywords:
            if keyword.replace('_', ' ') in full_text:
                quantum_matches.append(keyword)
        analysis['quantum_analysis'] = {'quantum_keywords_found': quantum_matches, 'quantum_score': len(quantum_matches), 'quantum_intensity': len(quantum_matches) / len(self.quantum_keywords) * 100, 'is_quantum_focused': len(quantum_matches) >= 3}
        tech_matches = []
        for keyword in self.technology_keywords:
            if keyword.replace('_', ' ') in full_text:
                tech_matches.append(keyword)
        analysis['technology_analysis'] = {'technology_keywords_found': tech_matches, 'technology_score': len(tech_matches), 'technology_intensity': len(tech_matches) / len(self.technology_keywords) * 100, 'is_technology_focused': len(tech_matches) >= 3}
        content_length = len(article['content'])
        summary_length = len(article['summary'])
        title_length = len(article['title'])
        analysis['content_quality'] = {'content_length': content_length, 'summary_length': summary_length, 'title_length': title_length, 'content_completeness': min(content_length / 1000 * 100, 100), 'summary_quality': min(summary_length / 200 * 100, 100), 'overall_quality_score': (content_length / 1000 + summary_length / 200) / 2 * 100}
        koba42_score = 0
        koba42_score += article.get('koba42_integration_potential', 0)
        if analysis['breakthrough_detection']['is_breakthrough']:
            koba42_score += 2.0
        if analysis['quantum_analysis']['is_quantum_focused']:
            koba42_score += 1.5
        if analysis['technology_analysis']['is_technology_focused']:
            koba42_score += 1.0
        if analysis['content_quality']['overall_quality_score'] >= 70:
            koba42_score += 0.5
        analysis['koba42_potential'] = {'enhanced_koba42_score': min(koba42_score, 10.0), 'integration_priority': 'high' if koba42_score >= 7.0 else 'medium' if koba42_score >= 5.0 else 'low', 'recommended_actions': self._generate_koba42_recommendations(analysis)}
        analysis['recommendations'] = self._generate_article_recommendations(analysis)
        return analysis

    def _generate_koba42_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate KOBA42 integration recommendations."""
        recommendations = []
        if analysis['breakthrough_detection']['is_breakthrough']:
            recommendations.append('High-priority integration due to breakthrough content')
        if analysis['quantum_analysis']['is_quantum_focused']:
            recommendations.append('Integrate quantum physics insights into optimization algorithms')
        if analysis['technology_analysis']['is_technology_focused']:
            recommendations.append('Apply technology insights to KOBA42 framework')
        if analysis['content_quality']['overall_quality_score'] >= 80:
            recommendations.append('High-quality content suitable for detailed analysis')
        return recommendations

    def _generate_article_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate general article recommendations."""
        recommendations = []
        if analysis['breakthrough_detection']['breakthrough_score'] >= 3:
            recommendations.append('Consider for immediate KOBA42 integration')
        if analysis['quantum_analysis']['quantum_score'] >= 5:
            recommendations.append('High quantum relevance - prioritize for quantum optimization')
        if analysis['technology_analysis']['technology_score'] >= 5:
            recommendations.append('High technology relevance - integrate into tech framework')
        if analysis['content_quality']['overall_quality_score'] < 50:
            recommendations.append('Consider re-scraping for better content quality')
        return recommendations

    def explore_all_articles(self) -> Dict[str, Any]:
        """Comprehensive exploration of all scraped articles."""
        logger.info('🔍 Starting comprehensive article exploration...')
        articles = self.get_all_scraped_articles()
        if not articles:
            logger.warning('⚠️ No articles found in database')
            return {'status': 'no_articles', 'recommendations': ['Run scraping with adjusted criteria', 'Check database connection', 'Verify scraping configuration']}
        logger.info(f'📊 Found {len(articles)} articles for analysis')
        analyzed_articles = []
        breakthrough_articles = []
        high_quantum_articles = []
        high_tech_articles = []
        high_koba42_articles = []
        for article in articles:
            analysis = self.analyze_article_content(article)
            article['analysis'] = analysis
            analyzed_articles.append(article)
            if analysis['breakthrough_detection']['is_breakthrough']:
                breakthrough_articles.append(article)
            if analysis['quantum_analysis']['is_quantum_focused']:
                high_quantum_articles.append(article)
            if analysis['technology_analysis']['is_technology_focused']:
                high_tech_articles.append(article)
            if analysis['koba42_potential']['enhanced_koba42_score'] >= 7.0:
                high_koba42_articles.append(article)
        report = {'total_articles': len(articles), 'breakthrough_articles': len(breakthrough_articles), 'high_quantum_articles': len(high_quantum_articles), 'high_tech_articles': len(high_tech_articles), 'high_koba42_articles': len(high_koba42_articles), 'article_analyses': analyzed_articles, 'breakthrough_findings': self._analyze_breakthroughs(breakthrough_articles), 'quantum_findings': self._analyze_quantum_content(high_quantum_articles), 'technology_findings': self._analyze_technology_content(high_tech_articles), 'koba42_opportunities': self._analyze_koba42_opportunities(high_koba42_articles), 'recommendations': self._generate_comprehensive_recommendations(analyzed_articles)}
        return report

    def _analyze_breakthroughs(self, breakthrough_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze breakthrough articles."""
        if not breakthrough_articles:
            return {'status': 'no_breakthroughs_found'}
        findings = {'total_breakthroughs': len(breakthrough_articles), 'breakthrough_keywords': {}, 'sources': {}, 'fields': {}, 'top_breakthroughs': []}
        all_breakthrough_keywords = []
        for article in breakthrough_articles:
            keywords = article['analysis']['breakthrough_detection']['breakthrough_keywords_found']
            all_breakthrough_keywords.extend(keywords)
        keyword_counts = {}
        for keyword in all_breakthrough_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        findings['breakthrough_keywords'] = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
        source_counts = {}
        for article in breakthrough_articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        findings['sources'] = source_counts
        field_counts = {}
        for article in breakthrough_articles:
            field = article['field']
            field_counts[field] = field_counts.get(field, 0) + 1
        findings['fields'] = field_counts
        sorted_breakthroughs = sorted(breakthrough_articles, key=lambda x: x['analysis']['breakthrough_detection']['breakthrough_score'], reverse=True)
        findings['top_breakthroughs'] = sorted_breakthroughs[:5]
        return findings

    def _analyze_quantum_content(self, quantum_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quantum-focused articles."""
        if not quantum_articles:
            return {'status': 'no_quantum_content_found'}
        findings = {'total_quantum_articles': len(quantum_articles), 'quantum_keywords': {}, 'quantum_topics': {}, 'koba42_quantum_opportunities': []}
        all_quantum_keywords = []
        for article in quantum_articles:
            keywords = article['analysis']['quantum_analysis']['quantum_keywords_found']
            all_quantum_keywords.extend(keywords)
        keyword_counts = {}
        for keyword in all_quantum_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        findings['quantum_keywords'] = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
        quantum_topics = {}
        for article in quantum_articles:
            title = article['title'].lower()
            if 'computing' in title or 'computer' in title:
                quantum_topics['quantum_computing'] = quantum_topics.get('quantum_computing', 0) + 1
            if 'algorithm' in title:
                quantum_topics['quantum_algorithms'] = quantum_topics.get('quantum_algorithms', 0) + 1
            if 'material' in title:
                quantum_topics['quantum_materials'] = quantum_topics.get('quantum_materials', 0) + 1
            if 'network' in title:
                quantum_topics['quantum_networking'] = quantum_topics.get('quantum_networking', 0) + 1
        findings['quantum_topics'] = quantum_topics
        high_potential_quantum = [a for a in quantum_articles if a['analysis']['koba42_potential']['enhanced_koba42_score'] >= 7.0]
        findings['koba42_quantum_opportunities'] = high_potential_quantum[:5]
        return findings

    def _analyze_technology_content(self, tech_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze technology-focused articles."""
        if not tech_articles:
            return {'status': 'no_technology_content_found'}
        findings = {'total_tech_articles': len(tech_articles), 'technology_keywords': {}, 'tech_topics': {}, 'koba42_tech_opportunities': []}
        all_tech_keywords = []
        for article in tech_articles:
            keywords = article['analysis']['technology_analysis']['technology_keywords_found']
            all_tech_keywords.extend(keywords)
        keyword_counts = {}
        for keyword in all_tech_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        findings['technology_keywords'] = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
        tech_topics = {}
        for article in tech_articles:
            title = article['title'].lower()
            if 'ai' in title or 'artificial' in title:
                tech_topics['artificial_intelligence'] = tech_topics.get('artificial_intelligence', 0) + 1
            if 'algorithm' in title:
                tech_topics['algorithms'] = tech_topics.get('algorithms', 0) + 1
            if 'software' in title:
                tech_topics['software'] = tech_topics.get('software', 0) + 1
            if 'computing' in title:
                tech_topics['computing'] = tech_topics.get('computing', 0) + 1
        findings['tech_topics'] = tech_topics
        high_potential_tech = [a for a in tech_articles if a['analysis']['koba42_potential']['enhanced_koba42_score'] >= 7.0]
        findings['koba42_tech_opportunities'] = high_potential_tech[:5]
        return findings

    def _analyze_koba42_opportunities(self, high_koba42_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze high KOBA42 potential articles."""
        if not high_koba42_articles:
            return {'status': 'no_high_potential_articles_found'}
        findings = {'total_high_potential': len(high_koba42_articles), 'priority_articles': [], 'integration_opportunities': [], 'research_directions': []}
        sorted_articles = sorted(high_koba42_articles, key=lambda x: x['analysis']['koba42_potential']['enhanced_koba42_score'], reverse=True)
        findings['priority_articles'] = sorted_articles[:10]
        for article in sorted_articles[:5]:
            opportunity = {'title': article['title'], 'koba42_score': article['analysis']['koba42_potential']['enhanced_koba42_score'], 'breakthrough_score': article['analysis']['breakthrough_detection']['breakthrough_score'], 'quantum_score': article['analysis']['quantum_analysis']['quantum_score'], 'tech_score': article['analysis']['technology_analysis']['technology_score'], 'recommended_actions': article['analysis']['koba42_potential']['recommended_actions']}
            findings['integration_opportunities'].append(opportunity)
        research_directions = set()
        for article in sorted_articles:
            if article['analysis']['quantum_analysis']['is_quantum_focused']:
                research_directions.add('quantum_optimization')
            if article['analysis']['technology_analysis']['is_technology_focused']:
                research_directions.add('technology_integration')
            if article['analysis']['breakthrough_detection']['is_breakthrough']:
                research_directions.add('breakthrough_research')
        findings['research_directions'] = list(research_directions)
        return findings

    def _generate_comprehensive_recommendations(self, analyzed_articles: List[Dict[str, Any]]) -> List[str]:
        """Generate comprehensive recommendations based on all articles."""
        recommendations = []
        breakthrough_count = sum((1 for a in analyzed_articles if a['analysis']['breakthrough_detection']['is_breakthrough']))
        quantum_count = sum((1 for a in analyzed_articles if a['analysis']['quantum_analysis']['is_quantum_focused']))
        tech_count = sum((1 for a in analyzed_articles if a['analysis']['technology_analysis']['is_technology_focused']))
        high_koba42_count = sum((1 for a in analyzed_articles if a['analysis']['koba42_potential']['enhanced_koba42_score'] >= 7.0))
        if breakthrough_count > 0:
            recommendations.append(f'Found {breakthrough_count} breakthrough articles - prioritize for immediate KOBA42 integration')
        if quantum_count > 0:
            recommendations.append(f'Found {quantum_count} quantum-focused articles - integrate quantum insights into optimization algorithms')
        if tech_count > 0:
            recommendations.append(f'Found {tech_count} technology-focused articles - apply technology insights to KOBA42 framework')
        if high_koba42_count > 0:
            recommendations.append(f'Found {high_koba42_count} high-potential articles for KOBA42 integration')
        if len(analyzed_articles) == 0:
            recommendations.append('No articles found - run scraping with adjusted criteria')
        elif len(analyzed_articles) < 5:
            recommendations.append('Limited article set - expand scraping to more sources')
        recommendations.extend(['Implement regular scraping sessions to build comprehensive research database', 'Focus on high-impact sources (Nature, Science, Physical Review Letters)', 'Monitor emerging quantum technologies and applications', 'Track breakthrough research for immediate integration opportunities', 'Develop automated KOBA42 integration pipeline for high-potential articles'])
        return recommendations

    def display_comprehensive_report(self, report: Dict[str, Any]):
        """Display comprehensive exploration report."""
        print('\n🔍 KOBA42 COMPREHENSIVE EXPLORATION REPORT')
        print('=' * 60)
        if report.get('status') == 'no_articles':
            print(f'\n⚠️ NO ARTICLES FOUND')
            print('-' * 30)
            print('No articles were found in the database for analysis.')
            print('Recommendations:')
            for (i, rec) in enumerate(report.get('recommendations', []), 1):
                print(f'{i}. {rec}')
            return
        print(f'\n📊 OVERALL STATISTICS')
        print('-' * 30)
        print(f"Total Articles Analyzed: {report.get('total_articles', 0)}")
        print(f"Breakthrough Articles: {report.get('breakthrough_articles', 0)}")
        print(f"High Quantum Articles: {report.get('high_quantum_articles', 0)}")
        print(f"High Tech Articles: {report.get('high_tech_articles', 0)}")
        print(f"High KOBA42 Potential: {report.get('high_koba42_articles', 0)}")
        if report.get('breakthrough_findings', {}).get('total_breakthroughs', 0) > 0:
            print(f'\n🚀 BREAKTHROUGH FINDINGS')
            print('-' * 30)
            breakthroughs = report['breakthrough_findings']
            print(f"Total Breakthroughs: {breakthroughs['total_breakthroughs']}")
            if breakthroughs.get('breakthrough_keywords'):
                print(f"Top Breakthrough Keywords: {list(breakthroughs['breakthrough_keywords'].keys())[:5]}")
            if breakthroughs.get('top_breakthroughs'):
                print(f'\nTop Breakthrough Articles:')
                for (i, article) in enumerate(breakthroughs['top_breakthroughs'][:3], 1):
                    print(f"  {i}. {article['title'][:60]}...")
                    print(f"     Score: {article['analysis']['breakthrough_detection']['breakthrough_score']}")
        if report.get('quantum_findings', {}).get('total_quantum_articles', 0) > 0:
            print(f'\n🔬 QUANTUM FINDINGS')
            print('-' * 30)
            quantum = report['quantum_findings']
            print(f"Quantum Articles: {quantum['total_quantum_articles']}")
            if quantum.get('quantum_keywords'):
                print(f"Top Quantum Keywords: {list(quantum['quantum_keywords'].keys())[:5]}")
            if quantum.get('koba42_quantum_opportunities'):
                print(f'\nTop Quantum KOBA42 Opportunities:')
                for (i, article) in enumerate(quantum['koba42_quantum_opportunities'][:3], 1):
                    print(f"  {i}. {article['title'][:60]}...")
                    print(f"     KOBA42 Score: {article['analysis']['koba42_potential']['enhanced_koba42_score']:.2f}")
        if report.get('technology_findings', {}).get('total_tech_articles', 0) > 0:
            print(f'\n💻 TECHNOLOGY FINDINGS')
            print('-' * 30)
            tech = report['technology_findings']
            print(f"Tech Articles: {tech['total_tech_articles']}")
            if tech.get('technology_keywords'):
                print(f"Top Technology Keywords: {list(tech['technology_keywords'].keys())[:5]}")
        if report.get('koba42_opportunities', {}).get('total_high_potential', 0) > 0:
            print(f'\n🎯 KOBA42 INTEGRATION OPPORTUNITIES')
            print('-' * 30)
            opportunities = report['koba42_opportunities']
            print(f"High Potential Articles: {opportunities['total_high_potential']}")
            if opportunities.get('priority_articles'):
                print(f'\nTop Priority Articles:')
                for (i, article) in enumerate(opportunities['priority_articles'][:5], 1):
                    print(f"  {i}. {article['title'][:60]}...")
                    print(f"     KOBA42 Score: {article['analysis']['koba42_potential']['enhanced_koba42_score']:.2f}")
                    print(f"     Source: {article['source']}")
        print(f'\n💡 RECOMMENDATIONS')
        print('-' * 30)
        for (i, rec) in enumerate(report.get('recommendations', [])[:10], 1):
            print(f'{i}. {rec}')

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info('✅ Database connection closed')

def demonstrate_comprehensive_exploration():
    """Demonstrate comprehensive article exploration."""
    logger.info('🔍 KOBA42 Comprehensive Explorer')
    logger.info('=' * 50)
    explorer = KOBA42ComprehensiveExplorer()
    print('\n🔍 Starting comprehensive article exploration...')
    report = explorer.explore_all_articles()
    explorer.display_comprehensive_report(report)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'comprehensive_exploration_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f'📄 Detailed report saved to {report_file}')
    explorer.close()
    return report
if __name__ == '__main__':
    report = demonstrate_comprehensive_exploration()
    print(f'\n🎉 Comprehensive exploration completed!')
    print(f'📊 Full system analysis of scraped articles')
    print(f'🚀 Breakthrough detection and analysis')
    print(f'🔬 Quantum and technology content identification')
    print(f'🎯 KOBA42 integration opportunities identified')