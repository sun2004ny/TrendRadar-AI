import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from src.temporal import track_topic_trajectories


class TestTemporalMatching(unittest.TestCase):

    def test_track_topic_trajectories_with_real_function(self):
        """
        Verify track_topic_trajectories from src.temporal correctly links
        consecutive window clusters with cosine similarity >= threshold.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create 2 windows: W_0 and W_1
            # W_0 has Cluster 0
            # W_1 has Cluster 0 (similar to W_0) and Cluster 1 (new/different)
            clusters_summary = pd.DataFrame([
                {
                    "cluster_key": "W_0_C0",
                    "window_id": "W_0",
                    "week_label": "W_0",
                    "cluster_id": 0,
                    "cluster_label": "vaccine / mandate",
                    "post_count": 10,
                    "avg_semantic_coherence": 0.85,
                    "representative_post": "vaccine mandate post W0"
                },
                {
                    "cluster_key": "W_1_C0",
                    "window_id": "W_1",
                    "week_label": "W_1",
                    "cluster_id": 0,
                    "cluster_label": "vaccine / doses",
                    "post_count": 15,
                    "avg_semantic_coherence": 0.88,
                    "representative_post": "vaccine mandate post W1"
                },
                {
                    "cluster_key": "W_1_C1",
                    "window_id": "W_1",
                    "week_label": "W_1",
                    "cluster_id": 1,
                    "cluster_label": "unrelated / sports",
                    "post_count": 8,
                    "avg_semantic_coherence": 0.70,
                    "representative_post": "unrelated sports post"
                }
            ])

            clustered_topics = pd.DataFrame([
                {"window_id": "W_0", "cluster_id": 0},
                {"window_id": "W_1", "cluster_id": 0},
                {"window_id": "W_1", "cluster_id": 1}
            ])

            # 3 embeddings: vector 0 and 1 are nearly identical (sim ≈ 1.0), vector 2 is orthogonal
            emb = np.array([
                [1.0, 0.0, 0.0],
                [0.98, 0.2, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            clusters_csv = tmp_path / "clusters_summary.csv"
            meta_csv = tmp_path / "clustered_topics.csv"
            emb_npy = tmp_path / "embeddings.npy"
            out_trajs = tmp_path / "trajectories.csv"
            out_matches = tmp_path / "matches.csv"

            clusters_summary.to_csv(clusters_csv, index=False)
            clustered_topics.to_csv(meta_csv, index=False)
            np.save(emb_npy, emb)

            trajs_df = track_topic_trajectories(
                clusters_path=clusters_csv,
                meta_path=meta_csv,
                embeddings_path=emb_npy,
                output_trajectories_path=out_trajs,
                output_matches_path=out_matches,
                match_threshold=0.68
            )

            # W_0_C0 and W_1_C0 should link into 1 trajectory spanning 2 checkpoints
            # W_1_C1 should birth a 2nd trajectory
            self.assertEqual(len(trajs_df), 2)

            linked_traj = trajs_df[trajs_df["active_checkpoints"] == 2]
            self.assertEqual(len(linked_traj), 1)
            self.assertEqual(linked_traj.iloc[0]["timeline_path"], "W_0 -> W_1")
            self.assertEqual(linked_traj.iloc[0]["total_volume"], 25)  # 10 + 15


if __name__ == "__main__":
    unittest.main()
