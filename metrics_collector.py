
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
