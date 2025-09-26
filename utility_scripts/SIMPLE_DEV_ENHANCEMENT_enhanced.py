
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
🏗️ SIMPLE DEV ENVIRONMENT ENHANCEMENT
=====================================

Quick implementation of essential missing components
"""
import os
import json
from pathlib import Path
from datetime import datetime

class SimpleDevEnhancer:
    """Simple dev environment enhancement"""

    def __init__(self, root_path='/Users/coo-koba42/dev'):
        self.root_path = Path(root_path)
        self.created_files = []

    def run_quick_enhancement(self):
        """Run quick enhancement of essential components"""
        print('🏗️ SIMPLE DEV ENVIRONMENT ENHANCEMENT')
        print('=' * 50)
        self.create_directories()
        self.create_config_files()
        self.create_documentation()
        self.create_dev_tools()
        self.generate_report()

    def create_directories(self):
        """Create missing directories"""
        print('📁 Creating missing directories...')
        directories = ['src', 'tests', 'scripts', 'config', 'docs', 'tools', 'examples', 'build', 'dist', 'logs']
        for directory in directories:
            dir_path = self.root_path / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f'✅ Created: {directory}/')

    def create_config_files(self):
        """Create essential configuration files"""
        print('⚙️ Creating configuration files...')
        pyproject_content = '[build-system]\nrequires = ["setuptools>=61.0", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "consciousness-dev-environment"\nversion = "1.0.0"\ndescription = "Advanced consciousness-driven development environment"\nreadme = "README.md"\nrequires-python = ">=3.8"\ndependencies = [\n    "numpy>=1.21.0",\n    "scipy>=1.7.0",\n    "torch>=1.11.0",\n    "fastapi>=0.75.0",\n    "pytest>=7.0.0",\n]\n\n[tool.black]\nline-length = 88\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        self.write_file('pyproject.toml', pyproject_content)
        editorconfig_content = 'root = true\n[*]\nindent_style = space\nindent_size = 4\nend_of_line = lf\ncharset = utf-8\ntrim_trailing_whitespace = true\ninsert_final_newline = true\n\n[*.py]\nmax_line_length = 88\n'
        self.write_file('.editorconfig', editorconfig_content)
        dev_req_content = '-r requirements.txt\nblack>=22.0.0\nflake8>=4.0.0\nmypy>=0.950\npre-commit>=2.17.0\npytest-cov>=3.0.0\n'
        self.write_file('dev-requirements.txt', dev_req_content)

    def create_documentation(self):
        """Create essential documentation"""
        print('📚 Creating documentation...')
        readme_content = '# 🧠 Consciousness Development Environment\n\nAn advanced, consciousness-driven development environment featuring quantum computing, AI integration, and revolutionary learning systems.\n\n## 🚀 Quick Start\n\n1. **Setup environment**\n   ```bash\n   pip install -r dev-requirements.txt\n   ```\n\n2. **Run tests**\n   ```bash\n   pytest tests/\n   ```\n\n3. **Start development**\n   ```bash\n   python -m uvicorn src.main:app --reload\n   ```\n\n## 📖 Documentation\n\n- [Architecture Guide](docs/architecture.md)\n- [API Documentation](docs/api.md)\n- [Contributing Guide](CONTRIBUTING.md)\n\n## 🧪 Testing\n\n```bash\npytest tests/ -v --cov=src\n```\n\n## 📄 License\n\nMIT License\n'
        self.write_file('README.md', readme_content)
        contributing_content = '# 🤝 Contributing\n\n## Development Setup\n\n1. Install dependencies: `pip install -r dev-requirements.txt`\n2. Install pre-commit: `pre-commit install`\n3. Run tests: `pytest tests/`\n\n## Code Standards\n\n- Black for formatting\n- Flake8 for linting\n- MyPy for type checking\n- 100% test coverage required\n\n## Commit Convention\n\n- `feat:` for new features\n- `fix:` for bug fixes\n- `docs:` for documentation\n- `test:` for testing\n'
        self.write_file('CONTRIBUTING.md', contributing_content)
        license_content = 'MIT License\n\nCopyright (c) YYYY STREET NAME Environment\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'
        self.write_file('LICENSE', license_content)

    def create_dev_tools(self):
        """Create development tools and scripts"""
        print('🛠️ Creating development tools...')
        makefile_content = '.PHONY: help install test lint format clean\n\nhelp:\n\t@echo "Available commands:"\n\t@grep -E \'^[a-zA-Z_-]+:.*?## .*$$\' $(MAKEFILE_LIST) | sort | awk \'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\\n", $$1, $$2}\'\n\ninstall: ## Install dependencies\n\tpip install -r requirements.txt\n\ndev-install: ## Install development dependencies\n\tpip install -r dev-requirements.txt\n\ntest: ## Run tests\n\tpytest tests/ -v --cov=src\n\nlint: ## Run linting\n\tflake8 src\n\nformat: ## Format code\n\tblack src tests\n\nclean: ## Clean up\n\tfind . -type f -name "*.pyc" -delete\n\tfind . -type d -name "__pycache__" -delete\n'
        self.write_file('Makefile', makefile_content)
        conftest_content = 'import pytest\nfrom pathlib import Path\n\user@domain.com(scope="session")\ndef test_data_dir():\n    return Path(__file__).parent / "test_data"\n\user@domain.com(scope="session")\ndef sample_data():\n    return {\n        "measurements": [0.8, 0.9, 0.7, 0.95, 0.85],\n        "baseline": 0.8,\n        "threshold": 0.75\n    }\n'
        self.write_file('tests/conftest.py', conftest_content)
        test_content = 'import pytest\n\ndef test_sample_function(sample_data):\n    """Test basic functionality"""\n    assert len(sample_data["measurements"]) == 5\n    assert all(isinstance(x, float) for x in sample_data["measurements"])\n\ndef test_baseline_calculation(sample_data):\n    """Test baseline calculation"""\n    measurements = sample_data["measurements"]\n    avg = sum(measurements) / len(measurements)\n    assert avg > sample_data["baseline"]\n'
        self.write_file('tests/test_sample.py', test_content)
        setup_script = '#!/bin/bash\n# Development Environment Setup Script\n\necho "🚀 Setting up development environment..."\n\n# Create virtual environment\nif [ ! -d ".venv" ]; then\n    python -m venv .venv\nfi\n\n# Activate and install dependencies\nsource .venv/bin/activate\npip install -r dev-requirements.txt\n\necho "✅ Setup complete!"\necho "Run \'source .venv/bin/activate\' to activate the environment"\n'
        self.create_script('scripts/setup-dev.sh', setup_script)

    def write_file(self, file_path: str, content: str):
        """Write content to file"""
        full_path = self.root_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        self.created_files.append(file_path)
        print(f'📝 Created: {file_path}')

    def create_script(self, file_path: str, content: str):
        """Create executable script"""
        full_path = self.root_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        full_path.chmod(493)
        self.created_files.append(file_path)
        print(f'📜 Created script: {file_path}')

    def generate_report(self):
        """Generate completion report"""
        print('\n📋 ENHANCEMENT COMPLETE')
        print('=' * 30)
        print(f'📁 Files Created: {len(self.created_files)}')
        print(f"📂 Directories Created: {len([f for f in self.created_files if '/' in f])}")
        print('\n🏆 CREATED COMPONENTS:')
        for file in self.created_files:
            print(f'  ✅ {file}')
        print('\n🚀 NEXT STEPS:')
        print("  1. Run 'make dev-install' to install dependencies")
        print("  2. Execute 'make test' to run tests")
        print("  3. Use 'make help' to see all available commands")
        print("  4. Start development with 'python -m uvicorn src.main:app --reload'")
        report = {'timestamp': datetime.now().isoformat(), 'files_created': self.created_files, 'enhancement_complete': True, 'next_steps': ['make dev-install', 'make test', 'make help', 'python -m uvicorn src.main:app --reload']}
        report_file = f"dev_enhancement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\n📄 Report saved: {report_file}')

def main():
    """Main function"""
    enhancer = SimpleDevEnhancer()
    enhancer.run_quick_enhancement()
if __name__ == '__main__':
    main()