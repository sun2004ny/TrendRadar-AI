import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import ROOT_DIR, get_config
from src.novelty import NoveltyGate, apply_novelty_gate


class TestNoveltyGate(unittest.TestCase):

    def setUp(self):
        self.config = get_config()

    def test_flag_off_identical_to_golden_results(self):
        """When novelty gate is disabled, alerts DataFrame is returned identical and unmodified."""
        raw_df = pd.DataFrame([
            {
                "trajectory_id": "TRAJ_001",
                "birth_window": "D_05",
                "is_vetted_candidate": True,
                "emergence_score": 0.85,
                "lifecycle_status": "[BURST]"
            },
            {
                "trajectory_id": "TRAJ_002",
                "birth_window": "D_10",
                "is_vetted_candidate": True,
                "emergence_score": 0.70,
                "lifecycle_status": "[EMERGING]"
            }
        ])

        gate = NoveltyGate(enabled=False)
        out_df, metrics = gate.apply(raw_df)

        self.assertFalse(metrics["novelty_gate_enabled"])
        self.assertEqual(len(out_df), len(raw_df))
        self.assertEqual(list(out_df["is_vetted_candidate"]), [True, True])
        self.assertEqual(list(out_df["emergence_score"]), [0.85, 0.70])

    def test_no_future_data_leakage(self):
        """
        Verify causal isolation:
        An alert on day D_05 must NEVER compare against centroids from D_06 or later.
        If a future day D_06 has an identical centroid, the D_05 alert's max_prior_sim must still be 0.
        """
        vec_base = np.array([1.0] + [0.0] * 383)

        # Mock clusters: day D_01, day D_05, day D_06
        clusters_df = pd.DataFrame([
            {"window_id": "D_01", "cluster_id": 0, "clean_text": "text_d1"},
            {"window_id": "D_06", "cluster_id": 0, "clean_text": "text_d6"}  # Future cluster
        ])

        # Embeddings: D_01 orthogonal ([0, 1, 0...]), D_06 identical to target ([1, 0, 0...])
        embeddings = np.zeros((2, 384))
        embeddings[0, 1] = 1.0  # D_01 is orthogonal
        embeddings[1, 0] = 1.0  # D_06 is identical to target

        # Alert born on D_05 with vector [1, 0, 0...]
        alerts_df = pd.DataFrame([{
            "trajectory_id": "TARGET_ALERT",
            "birth_window": "D_05",
            "representative_post": "target post",
            "is_vetted_candidate": True,
            "emergence_score": 0.90
        }])

        matches_df = pd.DataFrame([{
            "trajectory_id": "TARGET_ALERT",
            "cluster_key": "D_05_C0",
            "is_birth": True
        }])

        gate = NoveltyGate(enabled=True, similarity_threshold=0.70)
        # Pass centroid directly or simulate via apply
        out_df, metrics = gate.apply(alerts_df, clusters_df=clusters_df, embeddings=embeddings, matches_df=matches_df)

        # Even though D_06 has a 1.0 identical centroid, it's in the FUTURE relative to D_05
        # Therefore max_prior_sim must be 0.0 (only D_01 existed before D_05, and D_01 has sim 0.0)
        self.assertEqual(out_df.iloc[0]["max_prior_sim"], 0.0)
        self.assertFalse(out_df.iloc[0]["is_background_chatter"])

    def test_recurring_chatter_is_demoted_and_quarantined(self):
        """
        Verify that a recurring topic appearing on day D_10 that is highly similar
        to a topic from day D_02 is flagged as background chatter and quarantined.
        """
        # Cluster on D_02
        clusters_df = pd.DataFrame([
            {"window_id": "D_02", "cluster_id": 0, "clean_text": "recurring chatter from d2"}
        ])
        embeddings = np.zeros((2, 384))
        embeddings[0, 0] = 1.0  # D_02 cluster vector

        # Alert on D_10 with highly similar centroid
        alerts_df = pd.DataFrame([{
            "trajectory_id": "CHATTER_ALERT",
            "birth_window": "D_10",
            "representative_post": "recurring chatter on d10",
            "is_vetted_candidate": True,
            "emergence_score": 0.85
        }])
        matches_df = pd.DataFrame([{
            "trajectory_id": "CHATTER_ALERT",
            "cluster_key": "D_10_C0",
            "is_birth": True
        }])
        # Mock matches cluster key pointing to an embedding similar to D_02
        # Let's assign embedding[1] to D_10_C0
        clusters_extended = pd.DataFrame([
            {"window_id": "D_02", "cluster_id": 0, "clean_text": "recurring chatter from d2"},
            {"window_id": "D_10", "cluster_id": 0, "clean_text": "recurring chatter on d10"}
        ])
        embeddings[1, 0] = 0.95  # 95% similar to D_02
        embeddings[1, 1] = np.sqrt(1 - 0.95**2)

        gate = NoveltyGate(enabled=True, similarity_threshold=0.70, action="quarantine")
        out_df, metrics = gate.apply(
            alerts_df,
            clusters_df=clusters_extended,
            embeddings=embeddings,
            matches_df=matches_df
        )

        row = out_df.iloc[0]
        self.assertTrue(row["is_background_chatter"])
        self.assertGreaterEqual(row["max_prior_sim"], 0.70)
        self.assertFalse(row["is_vetted_candidate"])
        self.assertEqual(row["lifecycle_status"], "[RECURRING_CHATTER]")
        self.assertLess(row["emergence_ranking_score"], 0.85)

    def test_genuinely_new_topic_is_preserved(self):
        """
        Verify that a genuinely new topic with low similarity to past days
        retains candidate status and full emergence ranking score.
        """
        clusters_df = pd.DataFrame([
            {"window_id": "D_02", "cluster_id": 0, "clean_text": "past old topic"},
            {"window_id": "D_10", "cluster_id": 0, "clean_text": "brand new breaking event"}
        ])
        embeddings = np.zeros((2, 384))
        embeddings[0, 0] = 1.0  # past topic on axis 0
        embeddings[1, 1] = 1.0  # new topic on axis 1 (orthogonal, sim = 0.0)

        alerts_df = pd.DataFrame([{
            "trajectory_id": "NEW_EVENT",
            "birth_window": "D_10",
            "representative_post": "brand new breaking event",
            "is_vetted_candidate": True,
            "emergence_score": 0.88
        }])
        matches_df = pd.DataFrame([{
            "trajectory_id": "NEW_EVENT",
            "cluster_key": "D_10_C0",
            "is_birth": True
        }])

        gate = NoveltyGate(enabled=True, similarity_threshold=0.70, action="quarantine")
        out_df, metrics = gate.apply(
            alerts_df,
            clusters_df=clusters_df,
            embeddings=embeddings,
            matches_df=matches_df
        )

        row = out_df.iloc[0]
        self.assertFalse(row["is_background_chatter"])
        self.assertLess(row["max_prior_sim"], 0.70)
        self.assertTrue(row["is_vetted_candidate"])
        self.assertEqual(row["emergence_ranking_score"], 0.88)


if __name__ == "__main__":
    unittest.main()
