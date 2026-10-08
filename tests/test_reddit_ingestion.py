"""
tests/test_reddit_ingestion.py - Unit Tests for Live Reddit Ingestion (Phase 5)

Tests:
1. Rate-limit response (HTTP 429) triggers exponential backoff.
2. Duplicate posts are dropped across queries and windows.
3. Restart resumes from saved checkpoint without repeating or losing posts.
4. Bad/missing credentials fail clearly without crashing.
5. Daily request budget limiter enforces quota constraints.
6. Incremental FAISS vector indexing dynamically stores and searches embeddings.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import tempfile
import json
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import prawcore

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.reddit import RedditLiveDataSource, DailyRequestBudgetLimiter
from src.embeddings import IncrementalFAISSIndex


class MockSubmission:
    """Mock PRAW submission object."""
    def __init__(self, id_str, title, selftext="", created_utc=None, score=10, author="test_user"):
        self.id = id_str
        self.title = title
        self.selftext = selftext
        self.created_utc = created_utc or time.time()
        self.score = score
        self.num_comments = 5
        self.author = author


class TestRedditLiveIngestion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temp_dir.name) / "checkpoint.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rate_limit_triggers_exponential_backoff(self):
        """HTTP 429 TooManyRequests triggers exponential backoff and succeeds on retry."""
        mock_sub = MagicMock()
        mock_sub.new.side_effect = [
            prawcore.TooManyRequests(MagicMock(status_code=429, headers={})),
            [MockSubmission("post_429", "Recovered after rate limit backoff")]
        ]

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_sub

        ds = RedditLiveDataSource(
            subreddits=["test_sub"],
            checkpoint_path=self.checkpoint_path,
            reddit_client=mock_reddit
        )

        with patch("time.sleep") as mock_sleep:
            posts = ds.fetch_with_backoff("test_sub", limit=10, max_retries=3, base_delay=0.1)

            # Backoff was invoked
            self.assertEqual(mock_sleep.call_count, 1)
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0].id, "post_429")

    def test_duplicates_are_dropped(self):
        """Duplicate post IDs across streams are dropped."""
        s1 = MockSubmission("dup_01", "Unique post one", created_utc=100.0)
        s2 = MockSubmission("dup_02", "Unique post two", created_utc=101.0)
        s3 = MockSubmission("dup_01", "Repeated post one", created_utc=100.0)

        mock_sub = MagicMock()
        mock_sub.new.return_value = [s1, s2, s3]

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_sub

        ds = RedditLiveDataSource(
            subreddits=["test_sub"],
            checkpoint_path=self.checkpoint_path,
            reddit_client=mock_reddit
        )

        batches = list(ds.yield_batches(batch_size=10))
        all_posts = [p for b in batches for p in b]

        self.assertEqual(len(all_posts), 2)
        post_ids = [p.id for p in all_posts]
        self.assertEqual(post_ids, ["dup_01", "dup_02"])
        self.assertIn("dup_01", ds.seen_ids)

    def test_restart_resumes_from_checkpoint(self):
        """Restart loads checkpoint and ignores posts prior to last_seen_timestamp."""
        # 1. Run initial session that saves checkpoint at ts=200.0
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "last_seen_timestamp": 200.0,
                "last_seen_id": "old_post",
                "seen_ids": ["old_post"]
            }, f)

        # 2. Simulate new posts: one old (ts=150.0) and one new (ts=250.0)
        s_old = MockSubmission("old_p2", "Older post", created_utc=150.0)
        s_new = MockSubmission("new_p1", "New post after restart", created_utc=250.0)

        mock_sub = MagicMock()
        mock_sub.new.return_value = [s_old, s_new]

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_sub

        ds = RedditLiveDataSource(
            subreddits=["test_sub"],
            checkpoint_path=self.checkpoint_path,
            reddit_client=mock_reddit
        )

        self.assertEqual(ds.last_seen_timestamp, 200.0)

        batches = list(ds.yield_batches(batch_size=10))
        all_posts = [p for b in batches for p in b]

        # Only the post newer than checkpoint was yielded
        self.assertEqual(len(all_posts), 1)
        self.assertEqual(all_posts[0].id, "new_p1")
        self.assertEqual(ds.last_seen_timestamp, 250.0)

    def test_bad_credentials_fail_clearly_without_crashing(self):
        """Unauthenticated / bad credentials mode logs and exits gracefully."""
        # Point to dummy env file with placeholder credentials
        dummy_env = Path(self.temp_dir.name) / ".env.dummy"
        with open(dummy_env, "w", encoding="utf-8") as f:
            f.write("REDDIT_CLIENT_ID=your_reddit_client_id_here\n")
            f.write("REDDIT_CLIENT_SECRET=your_reddit_client_secret_here\n")
            f.write("REDDIT_USER_AGENT=trendradar:test\n")

        ds = RedditLiveDataSource(
            env_path=dummy_env,
            checkpoint_path=self.checkpoint_path
        )

        self.assertFalse(ds.is_authenticated)
        # Calling yield_batches must not crash and yield 0 posts
        batches = list(ds.yield_batches(batch_size=10))
        self.assertEqual(len(batches), 0)

    def test_daily_budget_limiter(self):
        """DailyRequestBudgetLimiter enforces daily quota bounds."""
        limiter = DailyRequestBudgetLimiter(max_requests_per_day=2)
        self.assertTrue(limiter.can_request())
        limiter.record_request()

        self.assertTrue(limiter.can_request())
        limiter.record_request()

        # 3rd request exceeds budget
        self.assertFalse(limiter.can_request())

    def test_incremental_faiss_vector_index(self):
        """FAISS incremental index adds embeddings, persists, and performs nearest neighbor search."""
        idx_path = Path(self.temp_dir.name) / "test_faiss.index"
        meta_path = Path(self.temp_dir.name) / "test_faiss_meta.json"

        index = IncrementalFAISSIndex(dim=384, index_path=idx_path, meta_path=meta_path)
        vecs = np.random.randn(4, 384).astype(np.float32)
        pids = ["P_A", "P_B", "P_C", "P_D"]

        index.add(vecs, pids)
        self.assertEqual(len(index), 4)

        # Search
        distances, result_ids = index.search(vecs[:1], top_k=2)
        self.assertEqual(result_ids[0][0], "P_A")
        self.assertAlmostEqual(distances[0][0], 1.0, places=4)

        # Save and reload
        index.save()
        new_index = IncrementalFAISSIndex(dim=384, index_path=idx_path, meta_path=meta_path)
        self.assertEqual(len(new_index), 4)
        _, reload_ids = new_index.search(vecs[1:2], top_k=1)
        self.assertEqual(reload_ids[0][0], "P_B")


if __name__ == "__main__":
    unittest.main()
