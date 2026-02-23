"""
core/metrics.py

Real-time reliability metrics — the data an agent scoring function needs
to decide whether to use Agent Memory.

Gary Tan's three factors:
  1. Liveness   — is it responding right now?
  2. Reliability — has it worked before?
  3. Confidence  — how often does it return accurate results?

This module tracks all three in real time and exposes them
as machine-readable endpoints.
"""

import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Optional


# ─────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────

@dataclass
class OperationRecord:
    """Single operation outcome record."""
    timestamp:    float
    operation:    str      # interact | store | retrieve | consolidate
    success:      bool
    duration_ms:  float
    agent_id:     str
    error:        Optional[str] = None


@dataclass
class MetricsSnapshot:
    """Point-in-time reliability snapshot — what an agent scoring function reads."""
    timestamp:             str
    liveness:              bool     # Is the service responding right now?
    uptime_pct_24h:        float    # Uptime last 24h (0.0 - 1.0)
    uptime_pct_30d:        float    # Uptime last 30d (0.0 - 1.0)
    success_rate_1h:       float    # Success rate last hour
    success_rate_24h:      float    # Success rate last 24h
    latency_p50_ms:        float    # Median latency
    latency_p95_ms:        float    # 95th percentile latency
    latency_p99_ms:        float    # 99th percentile latency
    total_operations:      int      # All time
    operations_last_1h:    int
    operations_last_24h:   int
    confidence_score:      float    # Composite 0.0 - 1.0
    active_agents:         int
    total_memories_stored: int


# ─────────────────────────────────────────
# METRICS COLLECTOR
# ─────────────────────────────────────────

class MetricsCollector:
    """
    Thread-safe metrics collector.
    Wraps around Agent Memory operations to track outcomes.

    Usage:
        metrics = MetricsCollector()

        # Wrap any operation
        with metrics.track("interact", agent_id="my_agent"):
            result = agent.interact(...)

        # Get snapshot
        snapshot = metrics.snapshot()
    """

    MAX_RECORDS = 10_000  # Rolling window

    def __init__(self, persist_path: str = "./data/metrics.json"):
        self._records: list[OperationRecord] = []
        self._lock = Lock()
        self._start_time = time.time()
        self._persist_path = Path(persist_path)
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def record(self, operation: str, agent_id: str, success: bool, duration_ms: float, error: str = None):
        """Record an operation outcome."""
        rec = OperationRecord(
            timestamp   = time.time(),
            operation   = operation,
            success     = success,
            duration_ms = duration_ms,
            agent_id    = agent_id,
            error       = error,
        )
        with self._lock:
            self._records.append(rec)
            # Rolling window — drop oldest
            if len(self._records) > self.MAX_RECORDS:
                self._records = self._records[-self.MAX_RECORDS:]
        self._persist()

    def track(self, operation: str, agent_id: str = "system"):
        """Context manager for tracking operations."""
        return _TrackContext(self, operation, agent_id)

    def snapshot(self, active_agents: int = 0, total_memories: int = 0) -> MetricsSnapshot:
        """Generate current reliability snapshot."""
        now = time.time()
        one_hour_ago  = now - 3600
        one_day_ago   = now - 86400
        thirty_days_ago = now - 86400 * 30

        with self._lock:
            records_1h  = [r for r in self._records if r.timestamp >= one_hour_ago]
            records_24h = [r for r in self._records if r.timestamp >= one_day_ago]
            records_30d = [r for r in self._records if r.timestamp >= thirty_days_ago]
            all_records = self._records[:]

        def success_rate(recs):
            if not recs:
                return 1.0  # No data = assume operational
            return sum(1 for r in recs if r.success) / len(recs)

        def percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        latencies_1h = [r.duration_ms for r in records_1h if r.success]

        sr_1h  = success_rate(records_1h)
        sr_24h = success_rate(records_24h)
        sr_30d = success_rate(records_30d)

        # Confidence = weighted average of success rates + latency factor
        lat_factor = 1.0
        if latencies_1h:
            p99 = percentile(latencies_1h, 99)
            lat_factor = max(0.5, 1.0 - (p99 / 5000))  # Penalize if p99 > 5s

        confidence = (sr_1h * 0.5 + sr_24h * 0.3 + sr_30d * 0.2) * lat_factor

        return MetricsSnapshot(
            timestamp           = datetime.now().isoformat(),
            liveness            = True,  # If this code runs, service is live
            uptime_pct_24h      = round(sr_24h, 4),
            uptime_pct_30d      = round(sr_30d, 4),
            success_rate_1h     = round(sr_1h, 4),
            success_rate_24h    = round(sr_24h, 4),
            latency_p50_ms      = round(percentile(latencies_1h, 50), 1),
            latency_p95_ms      = round(percentile(latencies_1h, 95), 1),
            latency_p99_ms      = round(percentile(latencies_1h, 99), 1),
            total_operations    = len(all_records),
            operations_last_1h  = len(records_1h),
            operations_last_24h = len(records_24h),
            confidence_score    = round(confidence, 4),
            active_agents       = active_agents,
            total_memories_stored = total_memories,
        )

    def to_dict(self, **kwargs) -> dict:
        return asdict(self.snapshot(**kwargs))

    def _persist(self):
        """Save recent records to disk for persistence across restarts."""
        try:
            recent = self._records[-1000:]  # Save last 1000
            data = [asdict(r) for r in recent]
            with open(self._persist_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load(self):
        """Load persisted records on startup."""
        try:
            if self._persist_path.exists():
                with open(self._persist_path) as f:
                    data = json.load(f)
                self._records = [OperationRecord(**r) for r in data]
        except Exception:
            self._records = []


class _TrackContext:
    """Context manager returned by MetricsCollector.track()."""
    def __init__(self, collector: MetricsCollector, operation: str, agent_id: str):
        self._collector = collector
        self._operation = operation
        self._agent_id  = agent_id
        self._start     = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self._start) * 1000
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self._collector.record(self._operation, self._agent_id, success, duration_ms, error)
        return False  # Don't suppress exceptions


# Global singleton — shared across server.py and mcp_server.py
_global_metrics = MetricsCollector()

def get_metrics() -> MetricsCollector:
    return _global_metrics
