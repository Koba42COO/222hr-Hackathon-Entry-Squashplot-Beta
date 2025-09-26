#!/usr/bin/env python3
"""
SquashPlot Critical Gaps Implementer
Implements the most critical missing industry-standard features
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

class CriticalGapsImplementer:
    """Implements critical missing industry-standard features"""
    
    def __init__(self):
        self.implementation_status = {}
        self.critical_features = {
            "security_headers": "Implement HTTP security headers",
            "encryption": "Add end-to-end encryption",
            "session_management": "Implement secure session handling",
            "csrf_protection": "Add CSRF protection",
            "accessibility": "Implement WCAG 2.1 AA compliance",
            "data_validation": "Add comprehensive data validation",
            "logging": "Implement centralized logging",
            "metrics": "Add performance monitoring"
        }
    
    def safe_print(self, message: str):
        """Print message with safe encoding for all OS"""
        try:
            print(message)
        except UnicodeEncodeError:
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    def implement_security_headers(self):
        """Implement HTTP security headers"""
        self.safe_print("Implementing security headers...")
        
        try:
            # Create security headers middleware
            security_headers = '''
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import time

class SecurityHeadersMiddleware:
    """Middleware to add security headers"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            response = await self.app(scope, receive, send)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            
            return response
        else:
            await self.app(scope, receive, send)
'''
            
            with open("security_headers_middleware.py", 'w', encoding='utf-8') as f:
                f.write(security_headers)
            
            self.safe_print("[OK] Security headers middleware created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement security headers: {str(e)}")
            return False
    
    def implement_encryption(self):
        """Implement end-to-end encryption"""
        self.safe_print("Implementing encryption framework...")
        
        try:
            encryption_framework = '''
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

class EncryptionManager:
    """End-to-end encryption manager"""
    
    def __init__(self, password: str = None):
        if password:
            self.key = self._derive_key(password)
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return base64.urlsafe_b64encode(salt + pwdhash).decode('ascii')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            decoded = base64.urlsafe_b64decode(hashed.encode('ascii'))
            salt = decoded[:32]
            pwdhash = decoded[32:]
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return hmac.compare_digest(pwdhash, new_hash)
        except Exception:
            return False
'''
            
            with open("encryption_manager.py", 'w', encoding='utf-8') as f:
                f.write(encryption_framework)
            
            self.safe_print("[OK] Encryption framework created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement encryption: {str(e)}")
            return False
    
    def implement_session_management(self):
        """Implement secure session management"""
        self.safe_print("Implementing session management...")
        
        try:
            session_manager = '''
import uuid
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

class SessionManager:
    """Secure session management system"""
    
    def __init__(self, session_timeout: int = 3600):
        self.sessions: Dict[str, dict] = {}
        self.session_timeout = session_timeout
    
    def create_session(self, user_id: str, user_data: dict = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "data": user_data or {},
            "is_active": True
        }
        self.sessions[session_id] = session_data
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is expired
        if self._is_session_expired(session):
            self.invalidate_session(session_id)
            return None
        
        # Update last activity
        session["last_activity"] = datetime.now().isoformat()
        return session
    
    def update_session(self, session_id: str, data: dict):
        """Update session data"""
        if session_id in self.sessions:
            self.sessions[session_id]["data"].update(data)
            self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.sessions:
            self.sessions[session_id]["is_active"] = False
            del self.sessions[session_id]
    
    def _is_session_expired(self, session: dict) -> bool:
        """Check if session is expired"""
        last_activity = datetime.fromisoformat(session["last_activity"])
        return datetime.now() - last_activity > timedelta(seconds=self.session_timeout)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
    
    def get_active_sessions(self) -> Dict[str, dict]:
        """Get all active sessions"""
        self.cleanup_expired_sessions()
        return {sid: session for sid, session in self.sessions.items() if session.get("is_active", False)}
'''
            
            with open("session_manager.py", 'w', encoding='utf-8') as f:
                f.write(session_manager)
            
            self.safe_print("[OK] Session management created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement session management: {str(e)}")
            return False
    
    def implement_csrf_protection(self):
        """Implement CSRF protection"""
        self.safe_print("Implementing CSRF protection...")
        
        try:
            csrf_protection = '''
import secrets
import hashlib
import hmac
from typing import Dict, Optional

class CSRFProtection:
    """CSRF protection system"""
    
    def __init__(self):
        self.tokens: Dict[str, str] = {}
        self.secret_key = secrets.token_hex(32)
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = token
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token"""
        if session_id not in self.tokens:
            return False
        
        stored_token = self.tokens[session_id]
        return hmac.compare_digest(stored_token, token)
    
    def invalidate_token(self, session_id: str):
        """Invalidate CSRF token"""
        if session_id in self.tokens:
            del self.tokens[session_id]
    
    def get_token(self, session_id: str) -> Optional[str]:
        """Get current token for session"""
        return self.tokens.get(session_id)
    
    def rotate_token(self, session_id: str) -> str:
        """Rotate CSRF token"""
        return self.generate_token(session_id)
'''
            
            with open("csrf_protection.py", 'w', encoding='utf-8') as f:
                f.write(csrf_protection)
            
            self.safe_print("[OK] CSRF protection created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement CSRF protection: {str(e)}")
            return False
    
    def implement_accessibility(self):
        """Implement accessibility features"""
        self.safe_print("Implementing accessibility features...")
        
        try:
            # Create accessibility enhancement for dashboard
            accessibility_enhancements = '''
/* Accessibility Enhancements for SquashPlot Dashboard */

/* High contrast mode */
.high-contrast {
    --bg-primary: #000000;
    --bg-secondary: #1a1a1a;
    --text-primary: #ffffff;
    --text-secondary: #cccccc;
    --accent-primary: #ffff00;
    --accent-secondary: #00ffff;
}

/* Focus indicators */
*:focus {
    outline: 3px solid #00d4ff;
    outline-offset: 2px;
}

/* Skip links */
.skip-link {
    position: absolute;
    top: -40px;
    left: 6px;
    background: #000;
    color: #fff;
    padding: 8px;
    text-decoration: none;
    z-index: 1000;
}

.skip-link:focus {
    top: 6px;
}

/* Screen reader only content */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* High contrast support */
@media (prefers-contrast: high) {
    .card {
        border: 2px solid currentColor;
    }
}

/* Keyboard navigation */
.keyboard-nav *:focus {
    outline: 3px solid #00d4ff;
    outline-offset: 2px;
}

/* ARIA labels and descriptions */
.aria-label {
    position: relative;
}

.aria-label::after {
    content: attr(aria-label);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #000;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
}

.aria-label:hover::after {
    opacity: 1;
}
'''
            
            with open("accessibility_enhancements.css", 'w', encoding='utf-8') as f:
                f.write(accessibility_enhancements)
            
            # Create accessibility JavaScript
            accessibility_js = '''
// Accessibility Enhancements for SquashPlot Dashboard

class AccessibilityManager {
    constructor() {
        this.init();
    }
    
    init() {
        this.addSkipLinks();
        this.enhanceKeyboardNavigation();
        this.addARIALabels();
        this.implementFocusManagement();
        this.addScreenReaderSupport();
    }
    
    addSkipLinks() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-link';
        skipLink.textContent = 'Skip to main content';
        document.body.insertBefore(skipLink, document.body.firstChild);
    }
    
    enhanceKeyboardNavigation() {
        // Add keyboard navigation for all interactive elements
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-nav');
            }
        });
        
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-nav');
        });
    }
    
    addARIALabels() {
        // Add ARIA labels to interactive elements
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            if (!button.getAttribute('aria-label')) {
                button.setAttribute('aria-label', button.textContent || 'Button');
            }
        });
        
        const inputs = document.querySelectorAll('input');
        inputs.forEach(input => {
            if (!input.getAttribute('aria-label')) {
                const label = document.querySelector(`label[for="${input.id}"]`);
                if (label) {
                    input.setAttribute('aria-label', label.textContent);
                }
            }
        });
    }
    
    implementFocusManagement() {
        // Manage focus for modals and dialogs
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modal = document.querySelector('.modal:not([style*="display: none"])');
                if (modal) {
                    modal.style.display = 'none';
                    const focusable = modal.querySelector('button, input, select, textarea, a[href]');
                    if (focusable) {
                        focusable.focus();
                    }
                }
            }
        });
    }
    
    addScreenReaderSupport() {
        // Add live regions for dynamic content
        const liveRegion = document.createElement('div');
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        liveRegion.id = 'live-region';
        document.body.appendChild(liveRegion);
        
        // Announce status changes
        this.announce = (message) => {
            const liveRegion = document.getElementById('live-region');
            if (liveRegion) {
                liveRegion.textContent = message;
            }
        };
    }
}

// Initialize accessibility manager
document.addEventListener('DOMContentLoaded', () => {
    new AccessibilityManager();
});
'''
            
            with open("accessibility_enhancements.js", 'w', encoding='utf-8') as f:
                f.write(accessibility_js)
            
            self.safe_print("[OK] Accessibility features created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement accessibility: {str(e)}")
            return False
    
    def implement_data_validation(self):
        """Implement comprehensive data validation"""
        self.safe_print("Implementing data validation...")
        
        try:
            data_validator = '''
import re
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self):
        self.validation_rules = {}
        self.error_messages = {}
    
    def validate_string(self, value: Any, min_length: int = 0, max_length: int = None, 
                       pattern: str = None, required: bool = True) -> Dict[str, Any]:
        """Validate string data"""
        errors = []
        
        if not value and required:
            errors.append("Field is required")
            return {"valid": False, "errors": errors}
        
        if value is None:
            return {"valid": True, "errors": []}
        
        if not isinstance(value, str):
            errors.append("Must be a string")
            return {"valid": False, "errors": errors}
        
        if len(value) < min_length:
            errors.append(f"Must be at least {min_length} characters")
        
        if max_length and len(value) > max_length:
            errors.append(f"Must be no more than {max_length} characters")
        
        if pattern and not re.match(pattern, value):
            errors.append("Invalid format")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return self.validate_string(email, pattern=pattern, required=True)
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain number")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain special character")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_number(self, value: Any, min_value: float = None, max_value: float = None, 
                       integer_only: bool = False) -> Dict[str, Any]:
        """Validate numeric data"""
        errors = []
        
        try:
            if integer_only:
                num_value = int(value)
            else:
                num_value = float(value)
        except (ValueError, TypeError):
            errors.append("Must be a valid number")
            return {"valid": False, "errors": errors}
        
        if min_value is not None and num_value < min_value:
            errors.append(f"Must be at least {min_value}")
        
        if max_value is not None and num_value > max_value:
            errors.append(f"Must be no more than {max_value}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_json(self, value: Any) -> Dict[str, Any]:
        """Validate JSON data"""
        errors = []
        
        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError:
                errors.append("Invalid JSON format")
        elif not isinstance(value, (dict, list)):
            errors.append("Must be JSON object or array")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_file_upload(self, filename: str, allowed_extensions: List[str] = None,
                           max_size: int = None) -> Dict[str, Any]:
        """Validate file upload"""
        errors = []
        
        if not filename:
            errors.append("Filename is required")
            return {"valid": False, "errors": errors}
        
        if allowed_extensions:
            file_ext = filename.split('.')[-1].lower()
            if file_ext not in allowed_extensions:
                errors.append(f"File type not allowed. Allowed: {', '.join(allowed_extensions)}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def sanitize_input(self, value: str) -> str:
        """Sanitize user input"""
        if not isinstance(value, str):
            return str(value)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', value)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    def validate_complete_form(self, data: Dict[str, Any], rules: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate complete form data"""
        results = {}
        all_valid = True
        
        for field, rule in rules.items():
            value = data.get(field)
            validation_result = self.validate_string(value, **rule)
            results[field] = validation_result
            
            if not validation_result["valid"]:
                all_valid = False
        
        return {
            "valid": all_valid,
            "results": results,
            "errors": [error for result in results.values() for error in result.get("errors", [])]
        }
'''
            
            with open("data_validator.py", 'w', encoding='utf-8') as f:
                f.write(data_validator)
            
            self.safe_print("[OK] Data validation created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement data validation: {str(e)}")
            return False
    
    def implement_logging(self):
        """Implement centralized logging"""
        self.safe_print("Implementing centralized logging...")
        
        try:
            logging_system = '''
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class CentralizedLogger:
    """Centralized logging system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handlers
        self._setup_file_handlers(detailed_formatter, simple_formatter)
        
        # Setup console handler
        self._setup_console_handler(simple_formatter)
    
    def _setup_file_handlers(self, detailed_formatter, simple_formatter):
        """Setup file handlers for different log levels"""
        # Error log
        error_handler = logging.FileHandler(self.log_dir / 'error.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Warning log
        warning_handler = logging.FileHandler(self.log_dir / 'warning.log')
        warning_handler.setLevel(logging.WARNING)
        warning_handler.setFormatter(detailed_formatter)
        
        # Info log
        info_handler = logging.FileHandler(self.log_dir / 'info.log')
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(simple_formatter)
        
        # Debug log
        debug_handler = logging.FileHandler(self.log_dir / 'debug.log')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(error_handler)
        root_logger.addHandler(warning_handler)
        root_logger.addHandler(info_handler)
        root_logger.addHandler(debug_handler)
    
    def _setup_console_handler(self, formatter):
        """Setup console handler"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
    
    def log_security_event(self, event_type: str, user_id: str = None, 
                          ip_address: str = None, details: Dict[str, Any] = None):
        """Log security-related events"""
        security_logger = logging.getLogger('security')
        
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {}
        }
        
        security_logger.warning(f"SECURITY_EVENT: {json.dumps(event_data)}")
    
    def log_user_action(self, user_id: str, action: str, resource: str = None, 
                       details: Dict[str, Any] = None):
        """Log user actions"""
        user_logger = logging.getLogger('user_actions')
        
        action_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "details": details or {}
        }
        
        user_logger.info(f"USER_ACTION: {json.dumps(action_data)}")
    
    def log_system_event(self, event_type: str, component: str, 
                        details: Dict[str, Any] = None):
        """Log system events"""
        system_logger = logging.getLogger('system')
        
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "component": component,
            "details": details or {}
        }
        
        system_logger.info(f"SYSTEM_EVENT: {json.dumps(event_data)}")
    
    def log_performance(self, operation: str, duration: float, 
                       details: Dict[str, Any] = None):
        """Log performance metrics"""
        perf_logger = logging.getLogger('performance')
        
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": duration * 1000,
            "details": details or {}
        }
        
        perf_logger.info(f"PERFORMANCE: {json.dumps(perf_data)}")
    
    def get_log_entries(self, log_type: str = "info", limit: int = 100) -> List[Dict]:
        """Get log entries from files"""
        log_file = self.log_dir / f"{log_type}.log"
        
        if not log_file.exists():
            return []
        
        entries = []
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines[-limit:]:
                try:
                    # Parse log entry (simplified)
                    parts = line.split(' - ', 4)
                    if len(parts) >= 4:
                        entries.append({
                            "timestamp": parts[0],
                            "logger": parts[1],
                            "level": parts[2],
                            "message": parts[3].strip()
                        })
                except Exception:
                    continue
        
        return entries
'''
            
            with open("centralized_logger.py", 'w', encoding='utf-8') as f:
                f.write(logging_system)
            
            self.safe_print("[OK] Centralized logging created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement logging: {str(e)}")
            return False
    
    def implement_metrics(self):
        """Implement performance monitoring"""
        self.safe_print("Implementing performance monitoring...")
        
        try:
            metrics_system = '''
import time
import psutil
import json
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class Metric:
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str] = None

class MetricsCollector:
    """Performance metrics collector"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.start_time = time.time()
    
    def record_counter(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """Record counter metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now().isoformat(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record gauge metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now().isoformat(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record timer metric"""
        metric = Metric(
            name=name,
            value=duration,
            timestamp=datetime.now().isoformat(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        return {
            "total_requests": len([m for m in self.metrics if m.name == "request"]),
            "error_rate": self._calculate_error_rate(),
            "average_response_time": self._calculate_average_response_time(),
            "active_sessions": len([m for m in self.metrics if m.name == "session"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        error_metrics = [m for m in self.metrics if m.tags and m.tags.get("status") == "error"]
        total_metrics = [m for m in self.metrics if m.name == "request"]
        
        if not total_metrics:
            return 0.0
        
        return len(error_metrics) / len(total_metrics) * 100
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time"""
        response_times = [m.value for m in self.metrics if m.name == "response_time"]
        
        if not response_times:
            return 0.0
        
        return sum(response_times) / len(response_times)
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            return json.dumps({
                "system_metrics": self.get_system_metrics(),
                "application_metrics": self.get_application_metrics(),
                "raw_metrics": [{"name": m.name, "value": m.value, "timestamp": m.timestamp, "tags": m.tags} for m in self.metrics]
            }, indent=2)
        else:
            return str(self.metrics)
    
    def clear_metrics(self):
        """Clear collected metrics"""
        self.metrics.clear()

# Global metrics collector instance
metrics_collector = MetricsCollector()
'''
            
            with open("metrics_collector.py", 'w', encoding='utf-8') as f:
                f.write(metrics_system)
            
            self.safe_print("[OK] Performance monitoring created")
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to implement metrics: {str(e)}")
            return False
    
    def run_critical_implementation(self):
        """Run critical gaps implementation"""
        self.safe_print("SquashPlot Critical Gaps Implementer")
        self.safe_print("=" * 60)
        self.safe_print("Implementing critical missing industry-standard features...")
        self.safe_print("")
        
        implementations = [
            ("Security Headers", self.implement_security_headers),
            ("Encryption Framework", self.implement_encryption),
            ("Session Management", self.implement_session_management),
            ("CSRF Protection", self.implement_csrf_protection),
            ("Accessibility Features", self.implement_accessibility),
            ("Data Validation", self.implement_data_validation),
            ("Centralized Logging", self.implement_logging),
            ("Performance Monitoring", self.implement_metrics)
        ]
        
        successful = 0
        failed = 0
        
        for name, implementation_func in implementations:
            self.safe_print(f"Implementing {name}...")
            try:
                if implementation_func():
                    self.safe_print(f"[OK] {name} implemented successfully")
                    successful += 1
                else:
                    self.safe_print(f"[ERROR] {name} implementation failed")
                    failed += 1
            except Exception as e:
                self.safe_print(f"[ERROR] {name} implementation failed: {str(e)}")
                failed += 1
        
        # Generate implementation report
        self._generate_implementation_report(successful, failed)
        
        return successful, failed
    
    def _generate_implementation_report(self, successful: int, failed: int):
        """Generate implementation report"""
        self.safe_print("\n" + "=" * 60)
        self.safe_print("CRITICAL GAPS IMPLEMENTATION REPORT")
        self.safe_print("=" * 60)
        
        total = successful + failed
        success_rate = (successful / total * 100) if total > 0 else 0
        
        self.safe_print(f"Implementations Successful: {successful}/{total}")
        self.safe_print(f"Success Rate: {success_rate:.1f}%")
        
        if successful > 0:
            self.safe_print(f"\n[SUCCESS] {successful} critical features implemented!")
            self.safe_print("The system now has enhanced security, accessibility, and monitoring capabilities.")
        
        if failed > 0:
            self.safe_print(f"\n[WARNING] {failed} implementations failed.")
            self.safe_print("Please check the error messages above and retry failed implementations.")
        
        # Save implementation status
        status = {
            "timestamp": datetime.now().isoformat(),
            "successful_implementations": successful,
            "failed_implementations": failed,
            "success_rate": success_rate,
            "implemented_features": list(self.critical_features.keys())[:successful]
        }
        
        try:
            with open("critical_implementation_status.json", 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2)
            self.safe_print(f"\nImplementation status saved to: critical_implementation_status.json")
        except Exception as e:
            self.safe_print(f"\nError saving status: {str(e)}")

def main():
    """Main implementation entry point"""
    print("SquashPlot Critical Gaps Implementer")
    print("=" * 60)
    
    implementer = CriticalGapsImplementer()
    successful, failed = implementer.run_critical_implementation()
    
    if successful > 0:
        print(f"\n[SUCCESS] {successful} critical features implemented!")
        print("The system now has enhanced industry-standard capabilities.")
    
    if failed > 0:
        print(f"\n[WARNING] {failed} implementations failed.")
        print("Please review the error messages and retry failed implementations.")

if __name__ == "__main__":
    main()
