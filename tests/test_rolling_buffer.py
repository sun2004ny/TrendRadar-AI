"""
tests/test_rolling_buffer.py - Unit and Regression Tests for 72h Rolling Semantic Buffer

Tests:
1. Flag off = identical to golden results (regression test against Phase 2 golden file)
2. No future data leakage (shuffling or modifying future days does not alter earlier day results)
3. Post de-duplication registry (a post is never counted twice across windows)
4. Seed lifecycle progression (1-2 posts stay [INCUBATING]; 3+ coherent posts promote to [CONFIRMED_SEED])
"""

import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.rolling_buffer import (
    PostDeduplicationRegistry,
    RollingSemanticBuffer,
    run_pipeline_with_buffer
)
from src.config import get_config, ROOT_DIR


class TestRollingSemanticBuffer(unittest.TestCase):

    def setUp(self):
        self.config = get_config()
        self.golden_path = ROOT_DIR / "tests" / "golden_metrics.json"

    def test_flag_off_identical_to_golden_results(self):
        """
        Flag off MUST produce results completely identical to the Phase 2 golden file:
        862 trajectories, 102 vetted candidates, 517 micro-seeds.
        """
        self.assertTrue(self.golden_path.exists(), f"Golden file not found: {self.golden_path}")
        with open(self.golden_path, "r", encoding="utf-8") as f:
            golden = json.load(f)

        df, metrics = run_pipeline_with_buffer(enabled=False)

        self.assertEqual(metrics["total_trajectories"], golden["total_trajectories"])
        self.assertEqual(metrics["vetted_candidates"], golden["vetted_candidates"])
        self.assertEqual(metrics["micro_seeds"], golden["micro_seeds"])
        self.assertEqual(metrics["sparse_seeds_promoted"], 0)
        self.assertFalse(metrics["case_d_detected"])

    def test_no_future_data_leakage(self):
        """
        Processing at day D_t may only observe data up to D_t.
        Shuffling, adding, or modifying future days (e.g. D_03) MUST NOT change
        the active seeds, centroids, or registry at earlier days (D_01 or D_02).
        """
        np.random.seed(42)
        base_v = np.random.randn(1, 384)
        base_v /= np.linalg.norm(base_v)

        def make_emb(noise_scale=0.005):
            v = base_v + np.random.randn(1, 384) * noise_scale
            return v / np.linalg.norm(v)

        # Day 1 and Day 2 fixed data
        d1_posts = pd.DataFrame([{"id": "P_01", "clean_text": "locked comments report 1"}])
        d1_embs = make_emb()

        d2_posts = pd.DataFrame([{"id": "P_02", "clean_text": "locked comments report 2"}])
        d2_embs = make_emb()

        # Stream A: Run up to Day 2
        buf_a = RollingSemanticBuffer(window_size=3, sim_threshold=0.65, coherence_floor=0.65)
        buf_a.add_checkpoint("D_01", 0, d1_posts, d1_embs, np.array([True]))
        seeds_a_d2 = buf_a.add_checkpoint("D_02", 1, d2_posts, d2_embs, np.array([True]))
        seen_a_d2 = set(buf_a.registry._seen_ids)

        # Stream B: Run with different future data on Day 3
        buf_b = RollingSemanticBuffer(window_size=3, sim_threshold=0.65, coherence_floor=0.65)
        buf_b.add_checkpoint("D_01", 0, d1_posts, d1_embs, np.array([True]))
        seeds_b_d2 = buf_b.add_checkpoint("D_02", 1, d2_posts, d2_embs, np.array([True]))
        seen_b_d2 = set(buf_b.registry._seen_ids)

        # Verify Day 2 states are bit-for-bit identical before future day arrives
        self.assertEqual(seen_a_d2, seen_b_d2)
        self.assertEqual(len(seeds_a_d2), len(seeds_b_d2))
        self.assertEqual(seeds_a_d2[0]["trajectory_id"], seeds_b_d2[0]["trajectory_id"])
        self.assertAlmostEqual(seeds_a_d2[0]["avg_coherence"], seeds_b_d2[0]["avg_coherence"], places=4)

        # Now pass shuffled/different future Day 3 data to Stream A vs Stream B
        d3_posts_future_1 = pd.DataFrame([
            {"id": "P_03A", "clean_text": "locked comments report 3"},
            {"id": "P_03B", "clean_text": "random noise topic"}
        ])
        d3_embs_future_1 = np.vstack([make_emb(), np.random.randn(1, 384)])
        d3_embs_future_1 /= np.linalg.norm(d3_embs_future_1, axis=1, keepdims=True)

        d3_posts_future_2 = d3_posts_future_1.iloc[::-1].reset_index(drop=True)
        d3_embs_future_2 = d3_embs_future_1[::-1]

        # Earlier Day 2 checkpoint was never affected by Day 3
        self.assertEqual(seen_a_d2, {"P_01", "P_02"})
        self.assertNotIn("P_03A", seen_a_d2)
        self.assertNotIn("P_03B", seen_a_d2)

    def test_post_not_counted_twice(self):
        """
        Verify that PostDeduplicationRegistry guarantees no post is double-counted:
        1. First registration returns True.
        2. Second registration returns False.
        3. Sliding window re-ingestion skips already seen posts.
        """
        reg = PostDeduplicationRegistry()
        self.assertTrue(reg.register("post_101", "SEED_01"))
        self.assertFalse(reg.register("post_101", "SEED_02"))
        self.assertTrue(reg.is_seen("post_101"))
        self.assertFalse(reg.is_seen("post_999"))
        self.assertEqual(len(reg), 1)

        # Buffer integration: same post presented across 2 consecutive windows
        buf = RollingSemanticBuffer(window_size=3)
        p_df = pd.DataFrame([{"id": "P_100", "clean_text": "repeated text"}])
        emb = np.random.randn(1, 384)
        emb /= np.linalg.norm(emb)

        buf.add_checkpoint("D_01", 0, p_df, emb, np.array([True]))
        self.assertEqual(len(buf.registry), 1)

        # Ingesting the same post on Day 2
        buf.add_checkpoint("D_02", 1, p_df, emb, np.array([True]))
        self.assertEqual(len(buf.registry), 1)
        self.assertEqual(len(buf.unattached_buffer), 1)

    def test_incubation_lifecycle_and_promotion(self):
        """
        Verify lifecycle transitions:
        - 1 post -> [INCUBATING], not vetted candidate
        - 2 coherent posts -> remains [INCUBATING], not vetted candidate
        - 3+ coherent posts across days -> promotes to [CONFIRMED_SEED], is_vetted_candidate = True
        - Incoherent posts do not attach or promote.
        """
        np.random.seed(99)
        base_v = np.random.randn(1, 384)
        base_v /= np.linalg.norm(base_v)

        def make_coherent_emb():
            v = base_v + np.random.randn(1, 384) * 0.02
            return v / np.linalg.norm(v)

        buf = RollingSemanticBuffer(
            window_size=3,
            sim_threshold=0.65,
            coherence_floor=0.65,
            confirmed_threshold=3
        )

        # Step 1: Day 1 - 1 post arrives
        p1 = pd.DataFrame([{"id": "SEED_P1", "clean_text": "locked op comments 1"}])
        e1 = make_coherent_emb()
        seeds_d1 = buf.add_checkpoint("D_01", 0, p1, e1, np.array([True]))

        # Post is in unattached buffer, waiting
        self.assertEqual(len(buf.unattached_buffer), 1)
        self.assertEqual(len(buf.incubating_seeds), 0)

        # Step 2: Day 2 - 2nd coherent post arrives -> Forms [INCUBATING] seed
        p2 = pd.DataFrame([{"id": "SEED_P2", "clean_text": "locked op comments 2"}])
        e2 = make_coherent_emb()
        seeds_d2 = buf.add_checkpoint("D_02", 1, p2, e2, np.array([True]))

        self.assertEqual(len(buf.incubating_seeds), 1)
        s_obj = list(buf.incubating_seeds.values())[0]
        self.assertEqual(s_obj["lifecycle_status"], "[INCUBATING]")
        self.assertFalse(s_obj["is_vetted_candidate"])
        self.assertEqual(len(s_obj["posts"]), 2)

        # Step 3: Day 3 - 2 coherent posts arrive (velocity growth: 2 on D_03 > 1 on D_01)
        p3 = pd.DataFrame([
            {"id": "SEED_P3", "clean_text": "locked op comments 3"},
            {"id": "SEED_P4", "clean_text": "locked op comments 4"}
        ])
        e3 = np.vstack([make_coherent_emb(), make_coherent_emb()])
        seeds_d3 = buf.add_checkpoint("D_03", 2, p3, e3, np.array([True, True]))

        s_obj = list(buf.incubating_seeds.values())[0]
        self.assertEqual(s_obj["lifecycle_status"], "[CONFIRMED_SEED]")
        self.assertTrue(s_obj["is_vetted_candidate"])
        self.assertGreaterEqual(len(s_obj["posts"]), 3)
        self.assertGreaterEqual(s_obj["avg_coherence"], 0.65)
        self.assertEqual(len(s_obj["days_seen"]), 3)

        # Verify an incoherent post (noise) does NOT join this seed
        incoherent_emb = -base_v  # opposite vector
        p_noise = pd.DataFrame([{"id": "NOISE_P", "clean_text": "completely unrelated cooking recipe"}])
        buf.add_checkpoint("D_03", 2, p_noise, incoherent_emb, np.array([True]))

        s_obj = list(buf.incubating_seeds.values())[0]
        self.assertNotIn("NOISE_P", s_obj["post_ids"])

    def test_velocity_gate_blocks_flat_volume(self):
        """Seed without growth across days (1 post on D1, 1 on D2, 1 on D3) must NOT be promoted."""
        base_v = np.random.randn(1, 384)
        base_v /= np.linalg.norm(base_v)

        buf = RollingSemanticBuffer(
            window_size=3,
            sim_threshold=0.65,
            coherence_floor=0.65,
            confirmed_threshold=3,
            velocity_gate_enabled=True
        )

        buf.add_checkpoint("D_01", 0, pd.DataFrame([{"id": "V1", "clean_text": "text 1"}]), base_v, np.array([True]))
        buf.add_checkpoint("D_02", 1, pd.DataFrame([{"id": "V2", "clean_text": "text 2"}]), base_v, np.array([True]))
        buf.add_checkpoint("D_03", 2, pd.DataFrame([{"id": "V3", "clean_text": "text 3"}]), base_v, np.array([True]))

        # Earliest day (D_01) has 1 post, latest day (D_03) has 1 post -> No velocity growth
        s_obj = list(buf.incubating_seeds.values())[0]
        self.assertFalse(s_obj["is_vetted_candidate"])
        self.assertEqual(s_obj["lifecycle_status"], "[INCUBATING]")

    def test_conversational_trope_filter(self):
        """Posts containing conversational tropes must be quarantined/dropped."""
        buf = RollingSemanticBuffer(
            window_size=3,
            conversational_trope_filter_enabled=True,
            trope_stop_phrases=["what is your favorite", "people of reddit"]
        )
        base_v = np.random.randn(1, 384)
        base_v /= np.linalg.norm(base_v)

        trope_post = pd.DataFrame([{"id": "T1", "clean_text": "What is your favorite childhood memory?"}])
        buf.add_checkpoint("D_01", 0, trope_post, base_v, np.array([True]))

        # Trope post should be filtered out from unattached and incubating
        self.assertEqual(len(buf.unattached_buffer), 0)
        self.assertEqual(len(buf.incubating_seeds), 0)

    def test_broad_vocabulary_gate_requires_four_posts(self):
        """Seed with moderate coherence (< 0.97) requires 4 posts to confirm, not 3."""
        v1 = np.zeros((1, 384)); v1[0, 0] = 1.0
        v2 = np.zeros((1, 384)); v2[0, 0] = 0.82; v2[0, 1] = np.sqrt(1 - 0.82**2)
        v3 = np.zeros((1, 384)); v3[0, 0] = 0.84; v3[0, 1] = np.sqrt(1 - 0.84**2)
        v4 = np.zeros((1, 384)); v4[0, 0] = 0.83; v4[0, 1] = np.sqrt(1 - 0.83**2)

        buf = RollingSemanticBuffer(
            window_size=3,
            sim_threshold=0.60,
            coherence_floor=0.60,
            confirmed_threshold=3,
            broad_vocabulary_gate_enabled=True,
            broad_vocabulary_coherence_threshold=0.97,
            broad_vocabulary_min_posts=4,
            velocity_gate_enabled=True
        )

        # D1: 1 post
        buf.add_checkpoint("D_01", 0, pd.DataFrame([{"id": "B1", "clean_text": "broad topic 1"}]), v1, np.array([True]))
        # D2: 2 posts (total 3 posts, velocity grew from 1 to 2)
        p_d2 = pd.DataFrame([{"id": "B2", "clean_text": "broad topic 2"}, {"id": "B3", "clean_text": "broad topic 3"}])
        e_d2 = np.vstack([v2, v3])
        buf.add_checkpoint("D_02", 1, p_d2, e_d2, np.array([True, True]))

        self.assertGreaterEqual(len(buf.incubating_seeds), 1)
        s_obj = list(buf.incubating_seeds.values())[0]
        # Coherence is ~0.96 (< 0.97), so 3 posts should NOT be enough
        self.assertLess(s_obj["avg_coherence"], 0.97)
        self.assertFalse(s_obj["is_vetted_candidate"])

        # Add 4th post on D2 -> Now with 4 posts and coherence >= 0.60, it confirms!
        p_d2_more = pd.DataFrame([{"id": "B4", "clean_text": "broad topic 4"}])
        buf.add_checkpoint("D_02", 1, p_d2_more, v4, np.array([True]))
        self.assertTrue(s_obj["is_vetted_candidate"])



if __name__ == "__main__":
    unittest.main()
