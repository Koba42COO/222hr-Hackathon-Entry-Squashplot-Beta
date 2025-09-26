
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
"""
Quantum Domain Scraping Configuration
TASK-022: Quantum Email & 5D Entanglement Cloud

This system maintains a comprehensive list of domains for quantum research scraping,
ensuring we capture the latest breakthroughs in quantum computing and consciousness mathematics.
"""
import json
import time
from typing import Dict, List, Any

class QuantumDomainScrapingConfig:
    """Quantum Domain Scraping Configuration System"""

    def __init__(self):
        self.config_id = f'quantum-domain-scraping-{int(time.time())}'
        self.config_version = '1.0.0'
        self.primary_domains = {'nature.com': {'priority': 'critical', 'categories': ['quantum_physics', 'materials_science', 'quantum_computing'], 'subdomains': ['www.nature.com', 'www.nature.com/articles', 'www.nature.com/nature', 'www.nature.com/nature-materials', 'www.nature.com/nature-physics', 'www.nature.com/nature-computing'], 'last_updated': time.time(), 'status': 'active'}, 'science.org': {'priority': 'critical', 'categories': ['quantum_physics', 'quantum_computing', 'consciousness_research'], 'subdomains': ['www.science.org', 'www.science.org/doi', 'www.science.org/advances'], 'last_updated': time.time(), 'status': 'active'}, 'phys.org': {'priority': 'high', 'categories': ['quantum_physics', 'quantum_computing', 'consciousness_research'], 'subdomains': ['phys.org', 'phys.org/news', 'phys.org/quantum'], 'last_updated': time.time(), 'status': 'active'}, 'arxiv.org': {'priority': 'critical', 'categories': ['quantum_physics', 'quantum_computing', 'consciousness_mathematics'], 'subdomains': ['arxiv.org', 'arxiv.org/abs', 'arxiv.org/pdf'], 'last_updated': time.time(), 'status': 'active'}, 'quantamagazine.org': {'priority': 'high', 'categories': ['quantum_physics', 'consciousness_research', 'quantum_computing'], 'subdomains': ['www.quantamagazine.org', 'www.quantamagazine.org/quantum'], 'last_updated': time.time(), 'status': 'active'}}
        self.secondary_domains = {'ieee.org': {'priority': 'medium', 'categories': ['quantum_computing', 'quantum_engineering'], 'status': 'active'}, 'acm.org': {'priority': 'medium', 'categories': ['quantum_computing', 'quantum_algorithms'], 'status': 'active'}, 'springer.com': {'priority': 'high', 'categories': ['quantum_physics', 'consciousness_research'], 'status': 'active'}, 'wiley.com': {'priority': 'medium', 'categories': ['quantum_physics', 'quantum_computing'], 'status': 'active'}}
        self.consciousness_domains = {'consciousness.org': {'priority': 'critical', 'categories': ['consciousness_research', 'consciousness_mathematics'], 'status': 'active'}, 'consciousness-studies.org': {'priority': 'high', 'categories': ['consciousness_research', 'consciousness_mathematics'], 'status': 'active'}}
        self.quantum_computing_domains = {'ibm.com/quantum': {'priority': 'critical', 'categories': ['quantum_computing', 'quantum_algorithms'], 'status': 'active'}, 'quantum.microsoft.com': {'priority': 'critical', 'categories': ['quantum_computing', 'quantum_algorithms'], 'status': 'active'}, 'quantum.google.com': {'priority': 'critical', 'categories': ['quantum_computing', 'quantum_algorithms'], 'status': 'active'}}

    def add_domain(self, domain: str, priority: str='medium', categories: List[str]=None):
        """Add a new domain to the scraping configuration"""
        if categories is None:
            categories = ['quantum_research']
        if domain not in self.primary_domains and domain not in self.secondary_domains:
            if priority in ['critical', 'high']:
                self.primary_domains[domain] = {'priority': priority, 'categories': categories, 'last_updated': time.time(), 'status': 'active'}
            else:
                self.secondary_domains[domain] = {'priority': priority, 'categories': categories, 'status': 'active'}
            print(f'✅ Added domain: {domain} with priority: {priority}')
        else:
            print(f'⚠️  Domain {domain} already exists in configuration')

    def get_all_domains(self) -> Optional[Any]:
        """Get all domains in the configuration"""
        return {'primary_domains': self.primary_domains, 'secondary_domains': self.secondary_domains, 'consciousness_domains': self.consciousness_domains, 'quantum_computing_domains': self.quantum_computing_domains}

    def get_domains_by_category(self, category: str) -> Optional[Any]:
        """Get domains by category"""
        domains = []
        for (domain, config) in self.primary_domains.items():
            if category in config['categories']:
                domains.append(domain)
        for (domain, config) in self.secondary_domains.items():
            if category in config['categories']:
                domains.append(domain)
        for (domain, config) in self.consciousness_domains.items():
            if category in config['categories']:
                domains.append(domain)
        for (domain, config) in self.quantum_computing_domains.items():
            if category in config['categories']:
                domains.append(domain)
        return domains

    def save_configuration(self, filename: str=None):
        """Save the configuration to a JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f'quantum_domain_scraping_config_{timestamp}.json'
        config_data = {'config_id': self.config_id, 'config_version': self.config_version, 'timestamp': time.time(), 'domains': self.get_all_domains()}
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f'💾 Configuration saved to: {filename}')
        return filename

def demonstrate_quantum_domain_scraping_config():
    """Demonstrate the quantum domain scraping configuration system"""
    print('🚀 QUANTUM DOMAIN SCRAPING CONFIGURATION SYSTEM')
    print('=' * 55)
    config_system = QuantumDomainScrapingConfig()
    print(f'\n📊 CONFIGURATION OVERVIEW:')
    print(f'Config ID: {config_system.config_id}')
    print(f'Config Version: {config_system.config_version}')
    print(f'Primary Domains: {len(config_system.primary_domains)}')
    print(f'Secondary Domains: {len(config_system.secondary_domains)}')
    print(f'Consciousness Domains: {len(config_system.consciousness_domains)}')
    print(f'Quantum Computing Domains: {len(config_system.quantum_computing_domains)}')
    print(f'\n🔍 PRIMARY DOMAINS:')
    for (domain, config) in config_system.primary_domains.items():
        print(f'  {domain}:')
        print(f"    Priority: {config['priority']}")
        print(f"    Categories: {', '.join(config['categories'])}")
        print(f"    Status: {config['status']}")
    print(f'\n🔍 QUANTUM COMPUTING DOMAINS:')
    for (domain, config) in config_system.quantum_computing_domains.items():
        print(f'  {domain}:')
        print(f"    Priority: {config['priority']}")
        print(f"    Categories: {', '.join(config['categories'])}")
        print(f"    Status: {config['status']}")
    print(f'\n🔍 CONSCIOUSNESS DOMAINS:')
    for (domain, config) in config_system.consciousness_domains.items():
        print(f'  {domain}:')
        print(f"    Priority: {config['priority']}")
        print(f"    Categories: {', '.join(config['categories'])}")
        print(f"    Status: {config['status']}")
    print(f'\n➕ TESTING DOMAIN ADDITION:')
    config_system.add_domain('quantum-research.org', 'high', ['quantum_physics', 'consciousness_research'])
    print(f'\n🎯 DOMAINS BY CATEGORY:')
    quantum_physics_domains = config_system.get_domains_by_category('quantum_physics')
    print(f'  Quantum Physics: {len(quantum_physics_domains)} domains')
    for domain in quantum_physics_domains:
        print(f'    - {domain}')
    consciousness_domains = config_system.get_domains_by_category('consciousness_research')
    print(f'  Consciousness Research: {len(consciousness_domains)} domains')
    for domain in consciousness_domains:
        print(f'    - {domain}')
    config_file = config_system.save_configuration()
    print(f'\n🎉 CONFIGURATION SYSTEM READY!')
    print(f"Total domains configured: {len(config_system.get_all_domains()['primary_domains']) + len(config_system.get_all_domains()['secondary_domains']) + len(config_system.get_all_domains()['consciousness_domains']) + len(config_system.get_all_domains()['quantum_computing_domains'])}")
    return config_system
if __name__ == '__main__':
    config_system = demonstrate_quantum_domain_scraping_config()