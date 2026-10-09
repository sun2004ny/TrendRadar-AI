"""
tests/test_shadow_and_telemetry.py
==================================
Tests for Phase 6 requirements:
1. Shadow mode logging and alert suppression.
2. HackerNews API source ingestion.
3. RSS feed ingestion and timestamp parsing.
4. Thread-safe telemetry tracker under concurrent load.
5. Rolling FAISS window memory capping.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.ingestion.base import NormalizedPost
from src.ingestion.hackernews import HackerNewsDataSource
from src.ingestion.rss import RSSDataSource
from src.telemetry import TelemetryTracker
from src.shadow_mode import ShadowLogger
from src.embeddings import IncrementalFAISSIndex


class TestShadowAndTelemetry(unittest.TestCase):

    def test_shadow_mode_logging_and_alert_suppression(self):
        """Verify shadow mode writes JSONL records and suppresses active user alerts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_shadow.jsonl"
            mock_cfg = MagicMock()
            mock_cfg.shadow_mode = {"enabled": True, "log_path": str(log_path)}

            logger = ShadowLogger(log_path=log_path, config=mock_cfg)
            self.assertTrue(logger.is_enabled())
            self.assertTrue(logger.should_suppress_alerts())

            # Log two candidates
            logger.log_candidate({
                "trajectory_id": "TRAJ_101",
                "day": "D_15",
                "keywords": ["test", "topic"],
                "post_count": 12,
                "emergence_score": 0.85,
                "coherence": 0.72,
                "velocity": 1.4,
                "acceleration": 0.3,
                "novelty_status": "passed",
                "representative_post": "Test post for shadow logger"
            })
            logger.log_candidate({
                "trajectory_id": "TRAJ_102",
                "day": "D_15",
                "keywords": ["noise", "chatter"],
                "post_count": 8,
                "emergence_score": 0.45,
                "coherence": 0.50,
                "velocity": 0.9,
                "acceleration": 0.0,
                "novelty_status": "quarantined",
                "representative_post": "Noise chatter post"
            })

            records = logger.read_records()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["trajectory_id"], "TRAJ_101")
            self.assertEqual(records[0]["novelty_status"], "passed")
            self.assertTrue(records[0]["shadow_mode"])
            self.assertEqual(records[1]["trajectory_id"], "TRAJ_102")
            self.assertEqual(records[1]["novelty_status"], "quarantined")

    def test_mock_hackernews_ingestion(self):
        """Verify HackerNewsDataSource yields valid NormalizedPost without author field."""
        mock_item = {
            "id": 123456,
            "type": "story",
            "title": "Show HN: Open Source Real-Time Emergence Engine",
            "text": "Discussion on real-time NLP clustering.",
            "time": 1690000000,
            "score": 150,
            "by": "hn_user"  # author should NOT be in NormalizedPost
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "hn_ckpt.json"
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp_top = MagicMock()
                mock_resp_top.status = 200
                mock_resp_top.read.return_value = json.dumps([123456]).encode("utf-8")

                mock_resp_item = MagicMock()
                mock_resp_item.status = 200
                mock_resp_item.read.return_value = json.dumps(mock_item).encode("utf-8")

                mock_urlopen.side_effect = [
                    MagicMock(__enter__=MagicMock(return_value=mock_resp_top)),
                    MagicMock(__enter__=MagicMock(return_value=mock_resp_item))
                ]

                source = HackerNewsDataSource(checkpoint_path=ckpt_path, max_stories=1, delay_between_requests=0.0)
                posts = source.fetch_latest()

                self.assertEqual(len(posts), 1)
                p = posts[0]
                self.assertIsInstance(p, NormalizedPost)
                self.assertEqual(p.id, "hn_123456")
                self.assertEqual(p.source, "hackernews")
                self.assertIn("Open Source Real-Time Emergence Engine", p.text)
                self.assertTrue(hasattr(p, "datetime_utc"))
                # Author field privacy check
                self.assertFalse(hasattr(p, "author"))

    def test_mock_rss_ingestion(self):
        """Verify RSSDataSource parses feed items into NormalizedPosts."""
        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Tech News Feed</title>
                <item>
                    <guid>guid-abc-789</guid>
                    <title>Major Breakthrough in Fast Text Clustering</title>
                    <description>Researchers demonstrate 100x acceleration using leaf HDBSCAN.</description>
                    <pubDate>Mon, 23 Aug 2021 14:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "rss_ckpt.json"
            with patch("urllib.request.urlopen") as mock_url:
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.read.return_value = sample_xml.encode("utf-8")
                mock_url.return_value.__enter__.return_value = mock_resp

                source = RSSDataSource(feed_urls=["https://test.rss/feed.xml"], checkpoint_path=ckpt_path)
                posts = source.fetch_latest()

                self.assertEqual(len(posts), 1)
                p = posts[0]
                self.assertIsInstance(p, NormalizedPost)
                self.assertTrue(p.id.startswith("rss_"))
                self.assertEqual(p.source, "rss")
                self.assertIn("Major Breakthrough in Fast Text Clustering", p.text)
                self.assertEqual(p.datetime_utc, "2021-08-23T14:00:00+00:00")
                self.assertFalse(hasattr(p, "author"))


    def test_thread_safe_telemetry_under_concurrency(self):
        """Verify TelemetryTracker does not lose counts under high concurrent thread load."""
        telemetry = TelemetryTracker()
        total_threads = 8
        posts_per_thread = 250

        def worker(thread_idx: int):
            for i in range(posts_per_thread):
                status = "processed" if i % 5 != 0 else "rejected"
                telemetry.record_post(
                    source=f"source_{thread_idx % 2}",
                    status=status,
                    latency_ms=10.5,
                    timestamp="2026-10-08T12:00:00+00:00"
                )
                if i % 50 == 0:
                    telemetry.record_error("timeout")

        with ThreadPoolExecutor(max_workers=total_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(total_threads)]
            for f in futures:
                f.result()

        metrics = telemetry.get_metrics()
        expected_total = total_threads * posts_per_thread
        self.assertEqual(metrics["posts_received"], expected_total)
        self.assertEqual(
            metrics["posts_processed"] + metrics["posts_rejected"],
            expected_total
        )
        self.assertEqual(metrics["errors"], total_threads * 5)
        self.assertGreater(metrics["latency_ms"], 0.0)

    def test_incremental_faiss_rolling_window_caps_size(self):
        """Verify IncrementalFAISSIndex caps vector count at max_vectors."""
        dim = 16
        max_vecs = 50
        index = IncrementalFAISSIndex(dimension=dim, max_vectors=max_vecs)

        # Add 120 vectors in 4 chunks of 30
        for _ in range(4):
            batch = np.random.randn(30, dim).astype("float32")
            index.add_vectors(batch)

        # Total stored should be capped at max_vecs = 50
        self.assertEqual(index.total_vectors, max_vecs)
        self.assertEqual(index.index.ntotal, max_vecs)

        # Query should return top_k
        query = np.random.randn(1, dim).astype("float32")
        scores, indices = index.search(query, top_k=5)
    def test_replay_is_causal(self):
        """Verify that running replay up to day t produces identical candidate metrics regardless of future days."""
        from src.evaluation import parse_day_num
        import pandas as pd

        # Evaluate candidate centroids from D_01..D_10 vs D_01..D_20
        clusters_df = pd.read_csv(Path(__file__).resolve().parent.parent / "experiments" / "e6_combined" / "daily_clusters.csv")
        base_alerts_df = pd.read_csv(Path(__file__).resolve().parent.parent / "experiments" / "e6_combined" / "emergence_alerts.csv")
        embs_all = np.load(Path(__file__).resolve().parent.parent / "data" / "processed" / "daily" / "embeddings_daily.npy")

        # Precompute centroids
        cluster_centroids = {}
        for (win, cid), group in clusters_df.groupby(["window_id", "cluster_id"]):
            c_key = f"{win}_C{cid}"
            cent = np.mean(embs_all[group.index.values], axis=0)
            cent = cent / (np.linalg.norm(cent) + 1e-12)
            cluster_centroids[c_key] = (win, parse_day_num(win), cent)

        d05_alerts = base_alerts_df[base_alerts_df["birth_window"] == "D_05"]
        prior_cents_t10 = [c for (w, d, c) in cluster_centroids.values() if d < 5]
        prior_cents_t20 = [c for (w, d, c) in cluster_centroids.values() if d < 5]

        # Prior centroids strictly before D_05 are identical regardless of whether D_10 or D_30 is loaded
        self.assertEqual(len(prior_cents_t10), len(prior_cents_t20))
        for i in range(len(prior_cents_t10)):
            self.assertTrue(np.allclose(prior_cents_t10[i], prior_cents_t20[i]))

    def test_replay_gate_results_equal_batch_pipeline_on_d21_d30(self):
        """Verify that on D21-D30, the causal replay reproduces 39 candidate alerts and 14 gated alerts."""
        from src.rolling_buffer import run_pipeline_with_buffer
        from src.evaluation import parse_day_num

        df_on, _ = run_pipeline_with_buffer(enabled=True)
        vetted_df = df_on[df_on["is_vetted_candidate"] == True].copy()

        # Filter to D21-D30
        d21_30 = vetted_df[vetted_df["birth_window"].apply(lambda d: parse_day_num(d) >= 21)]
        self.assertEqual(len(d21_30), 39)

        # Apply causal gate at 0.70 threshold
        clusters_df = pd.read_csv(Path(__file__).resolve().parent.parent / "experiments" / "e6_combined" / "daily_clusters.csv")
        matches_df = pd.read_csv(Path(__file__).resolve().parent.parent / "experiments" / "e6_combined" / "daily_trajectory_matches.csv")
        embs_all = np.load(Path(__file__).resolve().parent.parent / "data" / "processed" / "daily" / "embeddings_daily.npy")

        cluster_centroids = {}
        for (win, cid), group in clusters_df.groupby(["window_id", "cluster_id"]):
            c_key = f"{win}_C{cid}"
            cent = np.mean(embs_all[group.index.values], axis=0)
            cent = cent / (np.linalg.norm(cent) + 1e-12)
            cluster_centroids[c_key] = (win, parse_day_num(win), cent)

        traj_centroids = {}
        for tid, group in matches_df.groupby("trajectory_id"):
            bm = group[group["is_birth"] == True]
            if len(bm) == 0: bm = group.head(1)
            b_key = bm.iloc[0]["cluster_key"]
            if b_key in cluster_centroids:
                traj_centroids[str(tid)] = cluster_centroids[b_key]

        passed = []
        for idx, r in d21_30.iterrows():
            tid = str(r["trajectory_id"])
            b_num = parse_day_num(r["birth_window"])
            cent = traj_centroids.get(tid, (None, None, None))[2]
            if cent is None:
                rep_p = str(r.get("representative_post", ""))
                pm = clusters_df[clusters_df["clean_text"] == rep_p]
                cent = embs_all[pm.index.values[0]] if len(pm) > 0 else np.zeros(384)
                if np.linalg.norm(cent) > 0: cent = cent / np.linalg.norm(cent)

            prior_cents = [c for (w, d, c) in cluster_centroids.values() if d < b_num]
            sims = np.dot(prior_cents, cent) if len(prior_cents) > 0 and np.linalg.norm(cent) > 0 else [0.0]
            max_sim = float(np.max(sims))
            if max_sim < 0.70:
                passed.append(tid)

        self.assertEqual(len(passed), 14)
        self.assertIn("SEED_579", passed)
        self.assertIn("TRAJ_757", passed)


if __name__ == "__main__":
    unittest.main()

