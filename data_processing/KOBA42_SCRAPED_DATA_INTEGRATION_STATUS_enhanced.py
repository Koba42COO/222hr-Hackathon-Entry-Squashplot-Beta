
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
"""
KOBA42 SCRAPED DATA INTEGRATION STATUS
======================================
Comprehensive Status Report of All Scraped Data Integration
=========================================================

This script provides a complete overview of:
1. What data we have scraped and stored
2. What has been processed and analyzed
3. What integration opportunities exist
4. What might be missing or need updating
"""
import sqlite3
import json
from datetime import datetime, timedelta

def generate_scraped_data_integration_status():
    """Generate comprehensive status report of scraped data integration."""
    print('🔍 KOBA42 SCRAPED DATA INTEGRATION STATUS REPORT')
    print('=' * 70)
    print('Comprehensive Analysis of All Scraped Data and Integration')
    print('=' * 70)
    research_db = 'research_data/research_articles.db'
    explorations_db = 'research_data/agentic_explorations.db'
    ledger_db = 'research_data/digital_ledger.db'
    print('\n📊 RESEARCH ARTICLES DATABASE STATUS')
    print('-' * 50)
    try:
        conn = sqlite3.connect(research_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM articles')
        total_articles = cursor.fetchone()[0]
        print(f'Total Articles Stored: {total_articles}')
        cursor.execute('SELECT source, COUNT(*) as count FROM articles GROUP BY source ORDER BY count DESC')
        sources = cursor.fetchall()
        print('\nArticles by Source:')
        for (source, count) in sources:
            print(f'  {source}: {count} articles')
        cursor.execute('SELECT field, COUNT(*) as count FROM articles GROUP BY field ORDER BY count DESC LIMIT 10')
        fields = cursor.fetchall()
        print('\nTop 10 Fields:')
        for (field, count) in fields:
            print(f'  {field}: {count} articles')
        cursor.execute('SELECT MAX(scraped_timestamp) FROM articles')
        last_scraped = cursor.fetchone()[0]
        print(f'\nLast Scraping Activity: {last_scraped}')
        cursor.execute('SELECT COUNT(*) FROM articles WHERE koba42_integration_potential > 0.7')
        high_potential = cursor.fetchone()[0]
        print(f'Articles with High Integration Potential (>0.7): {high_potential}')
        cursor.execute('SELECT COUNT(*) FROM articles WHERE quantum_relevance > 0.8')
        quantum_relevant = cursor.fetchone()[0]
        print(f'Articles with High Quantum Relevance (>0.8): {quantum_relevant}')
        conn.close()
    except Exception as e:
        print(f'Error accessing research articles database: {e}')
    print('\n🤖 AGENTIC EXPLORATIONS STATUS')
    print('-' * 50)
    try:
        conn = sqlite3.connect(explorations_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM agentic_explorations')
        total_explorations = cursor.fetchone()[0]
        print(f'Total Agentic Explorations: {total_explorations}')
        cursor.execute('SELECT COUNT(*) FROM agentic_explorations WHERE f2_optimization_analysis IS NOT NULL OR ml_improvement_analysis IS NOT NULL OR cpu_enhancement_analysis IS NOT NULL OR weighting_analysis IS NOT NULL')
        analyzed_explorations = cursor.fetchone()[0]
        print(f'Explorations with AI Analysis: {analyzed_explorations}')
        cursor.execute('SELECT field, COUNT(*) as count FROM agentic_explorations GROUP BY field ORDER BY count DESC')
        exploration_fields = cursor.fetchall()
        print('\nExplorations by Field:')
        for (field, count) in exploration_fields:
            print(f'  {field}: {count} explorations')
        cursor.execute("SELECT COUNT(*) FROM agentic_explorations WHERE implementation_priority = 'high'")
        high_priority = cursor.fetchone()[0]
        print(f'High Priority Implementation Opportunities: {high_priority}')
        cursor.execute("SELECT COUNT(*) FROM agentic_explorations WHERE potential_impact = 'high'")
        high_impact = cursor.fetchone()[0]
        print(f'High Impact Opportunities: {high_impact}')
        conn.close()
    except Exception as e:
        print(f'Error accessing agentic explorations database: {e}')
    print('\n📜 DIGITAL LEDGER INTEGRATION STATUS')
    print('-' * 50)
    try:
        conn = sqlite3.connect(ledger_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM ledger_entries')
        total_entries = cursor.fetchone()[0]
        print(f'Total Ledger Entries: {total_entries}')
        cursor.execute('SELECT contribution_type, COUNT(*) as count FROM ledger_entries GROUP BY contribution_type ORDER BY count DESC')
        contribution_types = cursor.fetchall()
        print('\nLedger Entries by Contribution Type:')
        for (contrib_type, count) in contribution_types:
            print(f'  {contrib_type}: {count} entries')
        cursor.execute('SELECT SUM(credit_amount) FROM ledger_entries')
        total_credits = cursor.fetchone()[0] or 0
        print(f'Total Credits Distributed: {total_credits:.2f}')
        cursor.execute('SELECT COUNT(*) FROM attribution_chains')
        total_chains = cursor.fetchone()[0]
        print(f'Total Attribution Chains: {total_chains}')
        conn.close()
    except Exception as e:
        print(f'Error accessing digital ledger database: {e}')
    print('\n🔍 INTEGRATION GAPS ANALYSIS')
    print('-' * 50)
    try:
        conn_research = sqlite3.connect(research_db)
        conn_explorations = sqlite3.connect(explorations_db)
        cursor_research = conn_research.cursor()
        cursor_explorations = conn_explorations.cursor()
        cursor_research.execute('SELECT article_id FROM articles')
        article_ids = [row[0] for row in cursor_research.fetchall()]
        cursor_explorations.execute('SELECT paper_id FROM agentic_explorations')
        explored_ids = [row[0] for row in cursor_explorations.fetchall()]
        unexplored = set(article_ids) - set(explored_ids)
        print(f'Articles Without Agentic Exploration: {len(unexplored)}')
        if len(unexplored) > 0:
            print('Sample Unexplored Articles:')
            cursor_research.execute('SELECT title, field FROM articles WHERE article_id IN (?) LIMIT 5', (list(unexplored)[:5],))
            for (title, field) in cursor_research.fetchall():
                print(f'  {title[:50]}... ({field})')
        conn_research.close()
        conn_explorations.close()
    except Exception as e:
        print(f'Error analyzing integration gaps: {e}')
    print('\n⏰ RECENT ACTIVITY ANALYSIS')
    print('-' * 50)
    try:
        conn = sqlite3.connect(research_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles WHERE scraped_timestamp > datetime('now', '-1 day')")
        recent_24h = cursor.fetchone()[0]
        print(f'Articles Scraped in Last 24 Hours: {recent_24h}')
        cursor.execute("SELECT COUNT(*) FROM articles WHERE scraped_timestamp > datetime('now', '-7 days')")
        recent_week = cursor.fetchone()[0]
        print(f'Articles Scraped in Last Week: {recent_week}')
        cursor.execute("SELECT COUNT(*) FROM articles WHERE scraped_timestamp > datetime('now', '-30 days')")
        recent_month = cursor.fetchone()[0]
        print(f'Articles Scraped in Last Month: {recent_month}')
        conn.close()
    except Exception as e:
        print(f'Error analyzing recent activity: {e}')
    print('\n🎯 INTEGRATION OPPORTUNITIES')
    print('-' * 50)
    try:
        conn = sqlite3.connect(research_db)
        cursor = conn.cursor()
        cursor.execute('\n            SELECT field, COUNT(*) as count, AVG(koba42_integration_potential) as avg_potential \n            FROM articles \n            WHERE koba42_integration_potential > 0.7 \n            GROUP BY field \n            ORDER BY avg_potential DESC \n            LIMIT 5\n        ')
        high_potential_fields = cursor.fetchall()
        print('High Integration Potential by Field:')
        for (field, count, avg_potential) in high_potential_fields:
            print(f'  {field}: {count} articles (avg potential: {avg_potential:.3f})')
        cursor.execute('\n            SELECT title, quantum_relevance, koba42_integration_potential \n            FROM articles \n            WHERE quantum_relevance > 0.8 \n            ORDER BY koba42_integration_potential DESC \n            LIMIT 5\n        ')
        quantum_articles = cursor.fetchall()
        print('\nTop Quantum-Relevant Articles:')
        for (title, quantum_rel, integration_pot) in quantum_articles:
            print(f'  {title[:60]}... (Q: {quantum_rel:.2f}, I: {integration_pot:.2f})')
        conn.close()
    except Exception as e:
        print(f'Error analyzing integration opportunities: {e}')
    print('\n💡 INTEGRATION RECOMMENDATIONS')
    print('-' * 50)
    recommendations = ['1. Run fresh scraping for latest research (last scraping: 2025-08-29)', '2. Process unexplored articles with agentic exploration', '3. Integrate high-potential articles into digital ledger', '4. Focus on quantum-relevant articles for immediate integration', '5. Expand scraping to include more recent sources', '6. Implement automated integration pipeline for new articles', '7. Create summary reports for high-impact discoveries', '8. Prioritize articles with high KOBA42 integration potential']
    for recommendation in recommendations:
        print(recommendation)
    print('\n📈 SUMMARY STATISTICS')
    print('-' * 50)
    try:
        conn_research = sqlite3.connect(research_db)
        conn_explorations = sqlite3.connect(explorations_db)
        conn_ledger = sqlite3.connect(ledger_db)
        cursor = conn_research.cursor()
        cursor.execute('SELECT COUNT(*) FROM articles')
        total_articles = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM articles WHERE koba42_integration_potential > 0.7')
        high_potential = cursor.fetchone()[0]
        cursor = conn_explorations.cursor()
        cursor.execute('SELECT COUNT(*) FROM agentic_explorations')
        total_explorations = cursor.fetchone()[0]
        cursor = conn_ledger.cursor()
        cursor.execute('SELECT COUNT(*) FROM ledger_entries')
        total_entries = cursor.fetchone()[0]
        cursor.execute('SELECT SUM(credit_amount) FROM ledger_entries')
        total_credits = cursor.fetchone()[0] or 0
        print(f'Total Research Articles: {total_articles}')
        print(f'High Integration Potential: {high_potential}')
        print(f'Agentic Explorations: {total_explorations}')
        print(f'Digital Ledger Entries: {total_entries}')
        print(f'Total Credits Distributed: {total_credits:.2f}')
        print(f'Integration Coverage: {total_explorations / total_articles * 100:.1f}%')
        conn_research.close()
        conn_explorations.close()
        conn_ledger.close()
    except Exception as e:
        print(f'Error generating summary statistics: {e}')
    print('\n🎉 SCRAPED DATA INTEGRATION STATUS REPORT COMPLETE')
    print('=' * 70)
    print('The system has comprehensive data but could benefit from:')
    print('• Fresh scraping for latest research')
    print('• Processing of unexplored articles')
    print('• Enhanced integration of high-potential discoveries')
    print('• Automated pipeline for continuous integration')
    print('=' * 70)
if __name__ == '__main__':
    generate_scraped_data_integration_status()