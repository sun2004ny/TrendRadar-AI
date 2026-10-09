"""
src/shadow_mode.py
==================
Shadow Mode Execution and Telemetry Logging for TrendRadar-AI.
Allows testing ingestion, clustering, and trajectory evaluation in production
without exposing unvetted alerts to operators or end users.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import get_config, ROOT_DIR

logger = logging.getLogger("TrendRadar.ShadowMode")


class ShadowLogger:
    """
    Thread-safe logger that records shadow candidates and suppresses user-facing alerts.
    """

    def __init__(self, log_path: Optional[Path] = None, config: Optional[Any] = None):
        cfg = config or get_config()
        shadow_cfg = getattr(cfg, "shadow_mode", {}) if hasattr(cfg, "shadow_mode") else {}
        if isinstance(shadow_cfg, dict):
            self.enabled = bool(shadow_cfg.get("enabled", False))
            path_str = shadow_cfg.get("log_path", "logs/shadow_log.jsonl")
        else:
            self.enabled = False
            path_str = "logs/shadow_log.jsonl"

        self.log_path = Path(log_path) if log_path else (ROOT_DIR / path_str)
        self._lock = threading.Lock()

    def is_enabled(self) -> bool:
        """Check whether shadow mode is enabled."""
        return self.enabled

    def should_suppress_alerts(self) -> bool:
        """In shadow mode, user-facing alerts must be completely suppressed."""
        return self.enabled

    def log_candidate(self, candidate_data: Dict[str, Any]) -> None:
        """
        Record an emerging topic candidate to the shadow JSONL log.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trajectory_id": str(candidate_data.get("trajectory_id", candidate_data.get("id", "UNKNOWN"))),
            "day": str(candidate_data.get("day", candidate_data.get("window_id", "UNKNOWN"))),
            "keywords": candidate_data.get("keywords", candidate_data.get("c_tf_idf_keywords", [])),
            "post_count": int(candidate_data.get("post_count", candidate_data.get("volume", 0))),
            "emergence_score": float(candidate_data.get("emergence_score", candidate_data.get("score", 0.0))),
            "coherence": float(candidate_data.get("coherence", 0.0)),
            "velocity": float(candidate_data.get("velocity", 0.0)),
            "acceleration": float(candidate_data.get("acceleration", 0.0)),
            "novelty_status": str(candidate_data.get("novelty_status", "passed")),
            "novelty_score": float(candidate_data.get("novelty_score", 0.0)),
            "max_prior_sim": float(candidate_data.get("max_prior_sim", 0.0)),
            "representative_post": str(candidate_data.get("representative_post", candidate_data.get("sample_post", "")))[:200],
            "shadow_mode": True
        }

        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.debug(f"Shadow candidate recorded: {record['trajectory_id']}")

    def read_records(self) -> List[Dict[str, Any]]:
        """Read and parse all logged shadow records."""
        if not self.log_path.exists():
            return []
        records = []
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return records


# Global shadow logger singleton
_GLOBAL_SHADOW_LOGGER = ShadowLogger()


def get_shadow_logger() -> ShadowLogger:
    """Return the global ShadowLogger instance."""
    return _GLOBAL_SHADOW_LOGGER
