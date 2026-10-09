"""
src/telemetry.py
================
Thread-safe real-time telemetry tracker for TrendRadar-AI ingestion and pipeline processing.
Tracks ingestion volume, processing throughput, error rates, system memory, and vector index size.
"""

import threading
import time
from typing import Dict, Any, Optional, List
from collections import deque
import logging

logger = logging.getLogger("TrendRadar.Telemetry")

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class TelemetryTracker:
    """
    Thread-safe telemetry collector for monitoring real-time stream metrics.
    """

    def __init__(self, max_latency_samples: int = 1000):
        self._lock = threading.Lock()
        self.max_latency_samples = max_latency_samples
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self.posts_received: int = 0
            self.posts_processed: int = 0
            self.posts_rejected: int = 0
            self.errors: int = 0
            self.error_counts: Dict[str, int] = {}
            self.latest_timestamp: Optional[str] = None
            self.source_counts: Dict[str, int] = {}
            self._latencies: deque = deque(maxlen=self.max_latency_samples)
            self.vector_index_size: int = 0
            self.start_time: float = time.time()

    def record_post(
        self,
        source: str,
        status: str = "processed",
        latency_ms: float = 0.0,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Record a post event in a thread-safe manner.
        :param source: e.g. 'reddit', 'hackernews', 'rss', 'replay'
        :param status: 'processed', 'rejected', or 'received'
        :param latency_ms: Ingestion/processing latency in milliseconds
        :param timestamp: ISO UTC timestamp string of the post if available
        """
        with self._lock:
            self.posts_received += 1
            src_key = source.lower()
            self.source_counts[src_key] = self.source_counts.get(src_key, 0) + 1

            if status == "processed":
                self.posts_processed += 1
            elif status == "rejected":
                self.posts_rejected += 1

            if latency_ms > 0:
                self._latencies.append(latency_ms)

            if timestamp:
                self.latest_timestamp = timestamp

    def record_error(self, error_type: str = "general") -> None:
        """Record an ingestion or pipeline processing error."""
        with self._lock:
            self.errors += 1
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def set_vector_index_size(self, size: int) -> None:
        """Update current vector index size."""
        with self._lock:
            self.vector_index_size = max(0, size)

    def get_memory_use_mb(self) -> float:
        """Retrieve current process resident set size (RSS) in MB."""
        if _HAS_PSUTIL:
            try:
                process = psutil.Process()
                return round(process.memory_info().rss / (1024 * 1024), 2)
            except Exception as e:
                logger.debug(f"Failed to query psutil memory: {e}")
                return 0.0
        return 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of current telemetry metrics."""
        with self._lock:
            avg_latency = (
                round(sum(self._latencies) / len(self._latencies), 2)
                if self._latencies else 0.0
            )
            return {
                "posts_received": self.posts_received,
                "posts_processed": self.posts_processed,
                "posts_rejected": self.posts_rejected,
                "errors": self.errors,
                "error_counts": dict(self.error_counts),
                "latest_timestamp": self.latest_timestamp,
                "source_counts": dict(self.source_counts),
                "latency_ms": avg_latency,
                "latency_samples_count": len(self._latencies),
                "memory_use_mb": self.get_memory_use_mb(),
                "vector_index_size": self.vector_index_size,
                "uptime_seconds": round(time.time() - self.start_time, 2)
            }


# Global telemetry singleton
_GLOBAL_TELEMETRY = TelemetryTracker()


def get_telemetry() -> TelemetryTracker:
    """Access the global TelemetryTracker instance."""
    return _GLOBAL_TELEMETRY
