"""
tests/test_alerts_db.py
=======================
Tests for Phase 7 Alert Lifecycle Management:
1. Creation and initial state 'NEW'.
2. Deduplication on (trajectory_id, checkpoint_timestamp).
3. State machine validation (allowed transitions vs illegal transitions).
4. Full audit trail and history updates.
5. Durability across database reconnects.
6. Multi-threaded concurrent access in WAL mode.
"""

import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.alerts_db import AlertsDB, VALID_STATUSES


class TestAlertsDB(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.tmpdir.name) / "test_alerts.db"
        self.db = AlertsDB(db_path=self.db_path)

    def tearDown(self):
        try:
            self.tmpdir.cleanup()
        except Exception:
            pass


    def test_alert_creation_and_initial_state(self):
        """Verify new alert is created in 'NEW' state with an initial history entry."""
        alert_data = {
            "alert_id": "ALT_1001",
            "trajectory_id": "TRAJ_50",
            "checkpoint_timestamp": "D_12",
            "day": "D_12",
            "title": "Unusual Spike in Tech Disruption",
            "keywords": ["ai", "disruption", "model"],
            "post_count": 22,
            "coherence": 0.78,
            "velocity": 2.4,
            "acceleration": 0.8,
            "emergence_score": 0.89,
            "sample_posts": ["Post 1 text", "Post 2 text"]
        }

        alert, was_created = self.db.create_or_get_alert(alert_data)
        self.assertTrue(was_created)
        self.assertEqual(alert["alert_id"], "ALT_1001")
        self.assertEqual(alert["status"], "NEW")
        self.assertEqual(alert["trajectory_id"], "TRAJ_50")
        self.assertEqual(alert["post_count"], 22)
        self.assertEqual(len(alert["history"]), 1)
        self.assertEqual(alert["history"][0]["to_status"], "NEW")

    def test_deduplication_on_trajectory_and_checkpoint(self):
        """Verify identical (trajectory_id, checkpoint_timestamp) returns existing alert."""
        alert_data = {
            "alert_id": "ALT_FIRST",
            "trajectory_id": "TRAJ_99",
            "checkpoint_timestamp": "D_15_1200",
            "day": "D_15",
            "title": "Initial Alert",
            "post_count": 10
        }
        alt1, created1 = self.db.create_or_get_alert(alert_data)
        self.assertTrue(created1)

        # Attempt to insert same trajectory and checkpoint again
        duplicate_data = {
            "alert_id": "ALT_SECOND",
            "trajectory_id": "TRAJ_99",
            "checkpoint_timestamp": "D_15_1200",
            "day": "D_15",
            "title": "Duplicate Alert",
            "post_count": 12
        }
        alt2, created2 = self.db.create_or_get_alert(duplicate_data)
        self.assertFalse(created2)
        self.assertEqual(alt2["alert_id"], "ALT_FIRST")
        self.assertEqual(alt2["id"], alt1["id"])

    def test_state_machine_valid_transitions(self):
        """Verify standard lifecycle transitions work smoothly."""
        alert, _ = self.db.create_or_get_alert({
            "alert_id": "ALT_LIFECYCLE",
            "trajectory_id": "TRAJ_LIFE",
            "checkpoint_timestamp": "D_05",
            "day": "D_05"
        })
        self.assertEqual(alert["status"], "NEW")

        # NEW -> ACKNOWLEDGED
        a1 = self.db.update_status("ALT_LIFECYCLE", "ACKNOWLEDGED", note="Investigating analyst assigned")
        self.assertEqual(a1["status"], "ACKNOWLEDGED")

        # ACKNOWLEDGED -> WATCHING
        a2 = self.db.update_status("ALT_LIFECYCLE", "WATCHING", note="Adding to watchlist for next 24h")
        self.assertEqual(a2["status"], "WATCHING")

        # WATCHING -> RESOLVED
        a3 = self.db.update_status("ALT_LIFECYCLE", "RESOLVED", note="Confirmed false alarm / chatter decayed")
        self.assertEqual(a3["status"], "RESOLVED")

        # RESOLVED -> NEW (reopen)
        a4 = self.db.update_status("ALT_LIFECYCLE", "NEW", note="Topic re-emerged with higher velocity")
        self.assertEqual(a4["status"], "NEW")

        # NEW -> DISMISSED
        a5 = self.db.update_status("ALT_LIFECYCLE", "DISMISSED", note="Operator dismisses topic")
        self.assertEqual(a5["status"], "DISMISSED")

        # Check full history trail
        self.assertEqual(len(a5["history"]), 6)  # Initial creation + 5 transitions

    def test_state_machine_invalid_transitions_raise_value_error(self):
        """Verify illegal transitions are rejected with descriptive ValueError."""
        self.db.create_or_get_alert({
            "alert_id": "ALT_ILLEGAL",
            "trajectory_id": "TRAJ_ILL",
            "checkpoint_timestamp": "D_08",
            "day": "D_08"
        })

        # NEW -> RESOLVED directly is prohibited (must acknowledge first)
        with self.assertRaises(ValueError) as ctx:
            self.db.update_status("ALT_ILLEGAL", "RESOLVED")
        self.assertIn("Illegal status transition", str(ctx.exception))

        # NEW -> WATCHING directly is prohibited
        with self.assertRaises(ValueError) as ctx:
            self.db.update_status("ALT_ILLEGAL", "WATCHING")
        self.assertIn("Illegal status transition", str(ctx.exception))

        # Invalid random status
        with self.assertRaises(ValueError) as ctx:
            self.db.update_status("ALT_ILLEGAL", "BOGUS_STATUS")
        self.assertIn("Invalid status", str(ctx.exception))

    def test_durability_across_reconnect(self):
        """Verify SQLite data persists when a new instance connects to existing database."""
        self.db.create_or_get_alert({
            "alert_id": "ALT_DURABLE",
            "trajectory_id": "TRAJ_DUR",
            "checkpoint_timestamp": "D_20",
            "day": "D_20",
            "post_count": 45
        })
        self.db.update_status("ALT_DURABLE", "ACKNOWLEDGED", note="Operator check")

        # Open fresh instance against same path
        new_db = AlertsDB(db_path=self.db_path)
        alert = new_db.get_alert("ALT_DURABLE")
        self.assertIsNotNone(alert)
        self.assertEqual(alert["status"], "ACKNOWLEDGED")
        self.assertEqual(alert["post_count"], 45)
        self.assertEqual(len(alert["history"]), 2)

    def test_concurrent_wal_writes_and_reads(self):
        """Verify multiple threads can read and write concurrently without locking errors."""
        total_threads = 6
        alerts_per_thread = 20

        def worker(thread_idx: int):
            for i in range(alerts_per_thread):
                aid = f"ALT_TH_{thread_idx}_{i}"
                tid = f"TRAJ_{thread_idx}_{i}"
                self.db.create_or_get_alert({
                    "alert_id": aid,
                    "trajectory_id": tid,
                    "checkpoint_timestamp": f"D_{thread_idx}_{i}",
                    "day": f"D_{thread_idx}",
                    "post_count": i * 5
                })
                # Transition half of them
                if i % 2 == 0:
                    self.db.update_status(aid, "ACKNOWLEDGED", note=f"Thread {thread_idx} ack")

        with ThreadPoolExecutor(max_workers=total_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(total_threads)]
            for f in futures:
                f.result()

        counts = self.db.count_by_status()
        total_created = total_threads * alerts_per_thread
        self.assertEqual(counts["NEW"] + counts["ACKNOWLEDGED"], total_created)
        self.assertEqual(counts["ACKNOWLEDGED"], total_threads * (alerts_per_thread // 2))


if __name__ == "__main__":
    unittest.main()
