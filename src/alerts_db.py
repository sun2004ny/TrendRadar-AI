"""
src/alerts_db.py
================
SQLite Alert Lifecycle Management for TrendRadar-AI.
Enforces WAL mode, strict state machine transitions, deduplication, and full audit trails.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import logging

from contextlib import contextmanager

from src.config import ROOT_DIR

logger = logging.getLogger("TrendRadar.AlertsDB")

# Valid state machine states
VALID_STATUSES: Set[str] = {
    "NEW",
    "ACKNOWLEDGED",
    "WATCHING",
    "RESOLVED",
    "DISMISSED"
}

# Strict allowed transitions
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    "NEW": {"ACKNOWLEDGED", "DISMISSED"},
    "ACKNOWLEDGED": {"WATCHING", "RESOLVED", "DISMISSED"},
    "WATCHING": {"ACKNOWLEDGED", "RESOLVED", "DISMISSED"},
    "RESOLVED": {"NEW"},       # Allows reopening an alert if new emergence signal occurs
    "DISMISSED": {"NEW"}      # Allows un-dismissing / reopening
}


class AlertsDB:
    """
    SQLite database managing the lifecycle and audit trail of TrendRadar-AI alerts.
    Enforces WAL mode for high-concurrency readers and writers.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else ROOT_DIR / "data" / "alerts.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Opens a SQLite connection with WAL mode and foreign keys enabled, ensuring clean closure."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            yield conn
        finally:
            conn.close()


    def _init_db(self) -> None:
        """Initializes tables and indexes."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    trajectory_id TEXT NOT NULL,
                    checkpoint_timestamp TEXT NOT NULL,
                    day TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT,
                    keywords TEXT,
                    post_count INTEGER DEFAULT 0,
                    coherence REAL DEFAULT 0.0,
                    velocity REAL DEFAULT 0.0,
                    acceleration REAL DEFAULT 0.0,
                    emergence_score REAL DEFAULT 0.0,
                    novelty_score REAL DEFAULT 0.0,
                    sample_posts TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    history TEXT
                );
                """)
                conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_trajectory_checkpoint 
                ON alerts (trajectory_id, checkpoint_timestamp);
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_day ON alerts (day);
                """)
                conn.commit()

    def create_or_get_alert(self, alert_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Creates a new alert in state 'NEW' or retrieves existing alert if deduplication matches.
        Deduplication rule: unique on (trajectory_id, checkpoint_timestamp).
        :return: (alert_dict, was_created)
        """
        tid = str(alert_data.get("trajectory_id", "UNKNOWN"))
        ckpt_ts = str(alert_data.get("checkpoint_timestamp", alert_data.get("day", "UNKNOWN")))

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM alerts WHERE trajectory_id = ? AND checkpoint_timestamp = ?",
                    (tid, ckpt_ts)
                )
                existing = cur.fetchone()
                if existing:
                    return self._row_to_dict(existing), False

                now = datetime.now(timezone.utc).isoformat()
                aid = str(alert_data.get("alert_id") or f"ALERT_{tid}_{int(datetime.now(timezone.utc).timestamp())}")
                status = alert_data.get("status", "NEW").upper()
                if status not in VALID_STATUSES:
                    status = "NEW"

                kws = alert_data.get("keywords", [])
                keywords_str = "; ".join(kws) if isinstance(kws, list) else str(kws)

                samples = alert_data.get("sample_posts", [])
                samples_json = json.dumps(samples if isinstance(samples, list) else [str(samples)])

                initial_history = json.dumps([{
                    "from_status": None,
                    "to_status": status,
                    "timestamp": now,
                    "note": alert_data.get("note", "Alert generated by emergence pipeline")
                }])

                conn.execute("""
                INSERT INTO alerts (
                    alert_id, trajectory_id, checkpoint_timestamp, day, status,
                    title, keywords, post_count, coherence, velocity, acceleration,
                    emergence_score, novelty_score, sample_posts, created_at, updated_at, history
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    aid,
                    tid,
                    ckpt_ts,
                    str(alert_data.get("day", ckpt_ts)),
                    status,
                    str(alert_data.get("title", f"Emerging Topic: {tid}")),
                    keywords_str,
                    int(alert_data.get("post_count", 0)),
                    float(alert_data.get("coherence", 0.0)),
                    float(alert_data.get("velocity", 0.0)),
                    float(alert_data.get("acceleration", 0.0)),
                    float(alert_data.get("emergence_score", 0.0)),
                    float(alert_data.get("novelty_score", 0.0)),
                    samples_json,
                    now,
                    now,
                    initial_history
                ))
                conn.commit()

                cur = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (aid,))
                row = cur.fetchone()
                return self._row_to_dict(row), True

    def update_status(self, alert_id: str, new_status: str, note: str = "") -> Dict[str, Any]:
        """
        Transitions an alert's lifecycle status, validating the state transition.
        Raises ValueError if transition is invalid or alert not found.
        """
        new_status_clean = new_status.upper().strip()
        if new_status_clean not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {VALID_STATUSES}")

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Alert with id '{alert_id}' not found.")

                current_status = row["status"]
                if current_status == new_status_clean:
                    return self._row_to_dict(row)

                allowed_next = VALID_TRANSITIONS.get(current_status, set())
                if new_status_clean not in allowed_next:
                    raise ValueError(
                        f"Illegal status transition: cannot transition alert from '{current_status}' to '{new_status_clean}'. "
                        f"Allowed transitions from '{current_status}': {sorted(list(allowed_next))}"
                    )

                now = datetime.now(timezone.utc).isoformat()
                history_list = []
                if row["history"]:
                    try:
                        history_list = json.loads(row["history"])
                    except json.JSONDecodeError:
                        pass

                history_list.append({
                    "from_status": current_status,
                    "to_status": new_status_clean,
                    "timestamp": now,
                    "note": note
                })

                conn.execute("""
                UPDATE alerts
                SET status = ?, updated_at = ?, history = ?
                WHERE alert_id = ?
                """, (new_status_clean, now, json.dumps(history_list), alert_id))
                conn.commit()

                cur = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
                return self._row_to_dict(cur.fetchone())

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single alert by ID."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cur.fetchone()
            return self._row_to_dict(row) if row else None

    def get_alerts(
        self,
        status: Optional[str] = None,
        day: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query alerts with optional status or day filters."""
        query = "SELECT * FROM alerts"
        params: List[Any] = []
        conditions: List[str] = []

        if status:
            conditions.append("status = ?")
            params.append(status.upper().strip())
        if day:
            conditions.append("day = ?")
            params.append(str(day))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cur = conn.execute(query, tuple(params))
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve all pending, non-resolved and non-dismissed alerts."""
        with self._get_connection() as conn:
            cur = conn.execute("""
            SELECT * FROM alerts
            WHERE status NOT IN ('RESOLVED', 'DISMISSED')
            ORDER BY emergence_score DESC, created_at DESC
            LIMIT ?
            """, (limit,))
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def count_by_status(self) -> Dict[str, int]:
        """Returns tally of alerts across all lifecycle states."""
        counts = {st: 0 for st in VALID_STATUSES}
        with self._get_connection() as conn:
            cur = conn.execute("SELECT status, COUNT(*) as c FROM alerts GROUP BY status")
            for r in cur.fetchall():
                st = r["status"]
                if st in counts:
                    counts[st] = r["c"]
        return counts

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Converts sqlite3.Row to standard dictionary with parsed JSON fields."""
        d = dict(row)
        if "sample_posts" in d and d["sample_posts"]:
            try:
                d["sample_posts"] = json.loads(d["sample_posts"])
            except Exception:
                d["sample_posts"] = [str(d["sample_posts"])]
        if "history" in d and d["history"]:
            try:
                d["history"] = json.loads(d["history"])
            except Exception:
                d["history"] = []
        return d


# Global database singleton instance
_GLOBAL_ALERTS_DB = AlertsDB()


def get_alerts_db() -> AlertsDB:
    """Returns the global AlertsDB instance."""
    return _GLOBAL_ALERTS_DB
