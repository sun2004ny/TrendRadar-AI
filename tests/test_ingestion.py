"""
tests/test_ingestion.py - Unit & Integration Tests for Data Source Layer (Phase 4)

Tests:
1. Integration test: Run pipeline through HistoricalDataSource and compare to golden file (must be identical).
2. Validation tests: Invalid posts (empty text, missing timestamp, non-UTC time) are rejected and counted, not crashed on.
3. Batched streaming: HistoricalDataSource yields validated batches of NormalizedPost objects.
4. Preprocessing: Pipeline functions accept NormalizedPost instances without raw DataFrame column names.
"""

import unittest
from pathlib import Path
import json
import tempfile
import pandas as pd
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from src.config import ROOT_DIR
from src.ingestion.base import NormalizedPost, DataSource
from src.ingestion.historical import HistoricalDataSource
from src.rolling_buffer import run_pipeline_with_buffer, RollingSemanticBuffer
from src.preprocessing import clean_normalized_post, preprocess_normalized_posts


class TestNormalizedPostModel(unittest.TestCase):
    """Verifies strict validation invariants of NormalizedPost."""

    def test_valid_post_instantiation(self):
        utc_now = datetime.now(timezone.utc)
        post = NormalizedPost(
            id="p_001",
            raw_id="p_001",
            text="Valid technical discussion text about machine learning",
            timestamp=utc_now,
            source="reddit"
        )
        self.assertEqual(post.id, "p_001")
        self.assertEqual(post.raw_id, "p_001")
        self.assertEqual(post.text, "Valid technical discussion text about machine learning")
        self.assertEqual(post.timestamp, utc_now)
        self.assertEqual(post.timestamp.tzinfo, timezone.utc)
        self.assertEqual(post.source, "reddit")
        self.assertFalse(hasattr(post, "author"))

        # Test C4 requirement: delete raw text after processing
        post.delete_raw_text()
        self.assertEqual(post.text, "")

    def test_rejection_empty_text(self):
        """Empty or whitespace-only text must be rejected."""
        utc_now = datetime.now(timezone.utc)
        with self.assertRaises(ValidationError):
            NormalizedPost(
                id="p_empty",
                raw_id="p_empty",
                text="",
                timestamp=utc_now,
                source="reddit"
            )

        with self.assertRaises(ValidationError):
            NormalizedPost(
                id="p_spaces",
                raw_id="p_spaces",
                text="     ",
                timestamp=utc_now,
                source="reddit"
            )

    def test_rejection_missing_timestamp(self):
        """Missing or None timestamp must be rejected."""
        with self.assertRaises(ValidationError):
            NormalizedPost(
                id="p_notime",
                raw_id="p_notime",
                text="Valid text with missing time",
                timestamp=None,
                source="reddit"
            )

    def test_rejection_non_utc_timestamp(self):
        """Naive timestamp or non-UTC timezone must be rejected."""
        # Naive datetime (no tzinfo)
        naive_dt = datetime(2021, 8, 15, 12, 0, 0)
        with self.assertRaises(ValidationError):
            NormalizedPost(
                id="p_naive",
                raw_id="p_naive",
                text="Valid text with naive datetime",
                timestamp=naive_dt,
                source="reddit"
            )

        # Non-UTC timezone (EST = UTC-5)
        est_tz = timezone(timedelta(hours=-5))
        non_utc_dt = datetime(2021, 8, 15, 12, 0, 0, tzinfo=est_tz)
        with self.assertRaises(ValidationError):
            NormalizedPost(
                id="p_est",
                raw_id="p_est",
                text="Valid text with EST datetime",
                timestamp=non_utc_dt,
                source="reddit"
            )

    def test_timestamp_epoch_conversion(self):
        """Numeric epoch timestamps are automatically converted to UTC datetime."""
        epoch_ts = 1628710477  # 2021-08-11 19:34:37 UTC
        post = NormalizedPost(
            id="p_epoch",
            raw_id="p_epoch",
            text="Valid text with epoch timestamp",
            timestamp=epoch_ts,
            source="reddit"
        )
        self.assertIsNotNone(post.timestamp.tzinfo)
        self.assertEqual(post.timestamp.utcoffset(), timedelta(0))
        self.assertEqual(post.timestamp.year, 2021)


class TestDataSourceRobustness(unittest.TestCase):
    """Verifies that DataSource rejects invalid rows, counts them, and never crashes."""

    def test_invalid_posts_rejected_and_counted_without_crash(self):
        """
        Creates a CSV with a mix of valid and invalid posts:
        - Row 1: Valid
        - Row 2: Empty text (Invalid)
        - Row 3: Missing timestamp (Invalid)
        - Row 4: Naive non-UTC timestamp string without timezone (Invalid)
        - Row 5: Valid
        """
        records = [
            {
                "id": "valid_01",
                "clean_text": "First valid technical question about distributed databases",
                "datetime_utc": "2021-08-11 12:00:00+00:00",
                "author": "user1"
            },
            {
                "id": "invalid_empty_text",
                "clean_text": "   ",
                "datetime_utc": "2021-08-11 12:05:00+00:00",
                "author": "user2"
            },
            {
                "id": "invalid_missing_time",
                "clean_text": "Third post with completely missing timestamp information",
                "datetime_utc": None,
                "author": "user3"
            },
            {
                "id": "invalid_naive_time",
                "clean_text": "Fourth post with non-UTC naive timestamp string",
                "datetime_utc": "2021-08-11 12:15:00",  # missing +00:00 or timezone
                "author": "user4"
            },
            {
                "id": "valid_02",
                "clean_text": "Fifth valid post discussing neural network quantization",
                "datetime_utc": "2021-08-11 12:20:00+00:00",
                "author": "user5"
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
            temp_p = Path(f.name)
            pd.DataFrame(records).to_csv(temp_p, index=False)

        try:
            ds = HistoricalDataSource(file_path=temp_p, source_name="test_mixed")
            batches = list(ds.yield_batches(batch_size=10))

            all_ingested = [p for b in batches for p in b]

            # Ingestion must not crash
            self.assertEqual(len(all_ingested), 2)
            self.assertEqual(all_ingested[0].id, "valid_01")
            self.assertEqual(all_ingested[1].id, "valid_02")

            # Must count valid and invalid rejections accurately
            self.assertEqual(ds.valid_posts_count, 2)
            self.assertEqual(ds.invalid_posts_count, 3)

        finally:
            if temp_p.exists():
                temp_p.unlink()


class TestPipelineIntegrationWithHistoricalDataSource(unittest.TestCase):
    """
    Integration test: run pipeline through HistoricalDataSource and compare to golden file.
    Results must be identical.
    """

    def setUp(self):
        self.golden_path = ROOT_DIR / "tests" / "golden_metrics.json"
        self.assertTrue(self.golden_path.exists(), f"Golden file missing at {self.golden_path}")
        with open(self.golden_path, "r", encoding="utf-8") as f:
            self.golden = json.load(f)

    def test_pipeline_through_historical_data_source_matches_golden_results(self):
        """
        When running the pipeline through HistoricalDataSource with rolling_buffer disabled,
        output metrics MUST match the Phase 2 golden baseline exactly:
        - 862 trajectories
        - 102 vetted candidates
        - 517 quarantined micro-seeds
        """
        data_source = HistoricalDataSource()
        df, metrics = run_pipeline_with_buffer(enabled=False, data_source=data_source)

        self.assertEqual(
            metrics["total_trajectories"],
            self.golden["total_trajectories"],
            f"Trajectory count mismatch: {metrics['total_trajectories']} vs {self.golden['total_trajectories']}"
        )
        self.assertEqual(
            metrics["vetted_candidates"],
            self.golden["vetted_candidates"],
            f"Vetted candidate mismatch: {metrics['vetted_candidates']} vs {self.golden['vetted_candidates']}"
        )
        self.assertEqual(
            metrics["micro_seeds"],
            self.golden["micro_seeds"],
            f"Micro-seed mismatch: {metrics['micro_seeds']} vs {self.golden['micro_seeds']}"
        )
        self.assertEqual(metrics["sparse_seeds_promoted"], 0)
        self.assertFalse(metrics["case_d_detected"])

    def test_preprocessing_accepts_normalized_posts(self):
        """
        Verify that preprocessing functions operate directly on NormalizedPost objects
        without referencing raw DataFrame column names.
        """
        utc_now = datetime.now(timezone.utc)
        raw_posts = [
            NormalizedPost(
                id="p1",
                raw_id="p1",
                text="What is the best way to optimize PyTorch inference latency on GPUs?",
                timestamp=utc_now,
                source="reddit"
            ),
            NormalizedPost(
                id="p2",
                raw_id="p2",
                text="too short",  # < 15 chars, should be dropped
                timestamp=utc_now + timedelta(minutes=5),
                source="reddit"
            ),
            NormalizedPost(
                id="p3",
                raw_id="p3",
                text="Redditors of reddit, how do you handle memory fragmentation in large clusters?",
                timestamp=utc_now + timedelta(minutes=10),
                source="reddit"
            )
        ]

        cleaned_posts = preprocess_normalized_posts(raw_posts)

        # p2 was filtered out (< 15 chars)
        self.assertEqual(len(cleaned_posts), 2)
        self.assertEqual(cleaned_posts[0].id, "p1")
        self.assertEqual(cleaned_posts[1].id, "p3")

        # Interrogative boilerplate stripped
        self.assertNotIn("What is", cleaned_posts[0].text)
        self.assertIn("optimize PyTorch inference", cleaned_posts[0].text)

        self.assertNotIn("Redditors of reddit", cleaned_posts[1].text)
        self.assertIn("handle memory fragmentation", cleaned_posts[1].text)


if __name__ == "__main__":
    unittest.main()
