
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
