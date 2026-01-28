"""
Metrics tracking for monitoring bot performance and health
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """
    Thread-safe metrics tracker for bot operations
    """
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    # Thread lock for thread-safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def record_query(self, success: bool, response_time: float, from_cache: bool = False) -> None:
        """
        Record a query execution
        
        Args:
            success: Whether the query was successful
            response_time: Time taken to respond in seconds
            from_cache: Whether response was served from cache
        """
        with self._lock:
            self.total_queries += 1
            
            if success:
                self.successful_queries += 1
            else:
                self.failed_queries += 1
            
            if from_cache:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            self.total_response_time += response_time
            self.response_times.append(response_time)
            
            # Keep only last 1000 response times to avoid memory bloat
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
    
    def record_error(self, error_type: str) -> None:
        """
        Record an error occurrence
        
        Args:
            error_type: Type/name of the error
        """
        with self._lock:
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
    
    def get_average_response_time(self) -> float:
        """
        Get average response time
        
        Returns:
            Average response time in seconds
        """
        with self._lock:
            if self.total_queries == 0:
                return 0.0
            return self.total_response_time / self.total_queries
    
    def get_cache_hit_rate(self) -> float:
        """
        Get cache hit rate as percentage
        
        Returns:
            Cache hit rate (0-100)
        """
        with self._lock:
            total_cache_ops = self.cache_hits + self.cache_misses
            if total_cache_ops == 0:
                return 0.0
            return (self.cache_hits / total_cache_ops) * 100
    
    def get_success_rate(self) -> float:
        """
        Get query success rate as percentage
        
        Returns:
            Success rate (0-100)
        """
        with self._lock:
            if self.total_queries == 0:
                return 0.0
            return (self.successful_queries / self.total_queries) * 100
    
    def get_uptime_seconds(self) -> float:
        """
        Get bot uptime in seconds
        
        Returns:
            Uptime in seconds
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> dict:
        """
        Convert metrics to dictionary format
        
        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            return {
                'total_queries': self.total_queries,
                'successful_queries': self.successful_queries,
                'failed_queries': self.failed_queries,
                'success_rate': f"{self.get_success_rate():.2f}%",
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': f"{self.get_cache_hit_rate():.2f}%",
                'average_response_time': f"{self.get_average_response_time():.3f}s",
                'uptime_seconds': f"{self.get_uptime_seconds():.0f}s",
                'errors_by_type': dict(self.errors_by_type),
            }
    
    def log_summary(self) -> None:
        """Log a summary of current metrics"""
        summary = self.to_dict()
        logger.info(f"Metrics Summary: {summary}")
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self.total_queries = 0
            self.successful_queries = 0
            self.failed_queries = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_response_time = 0.0
            self.response_times.clear()
            self.errors_by_type.clear()
            self.start_time = datetime.now()
        logger.info("Metrics reset")


# Global metrics instance
metrics = Metrics()


class MetricsContext:
    """Context manager for tracking operation metrics"""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.success = False
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        
        if exc_type is None:
            self.success = True
            logger.debug(f"Completed {self.operation_name} in {elapsed_time:.3f}s")
        else:
            self.success = False
            error_name = exc_type.__name__ if exc_type else "Unknown"
            metrics.record_error(error_name)
            logger.error(f"Failed {self.operation_name} after {elapsed_time:.3f}s: {error_name}")
        
        return False  # Don't suppress exceptions
