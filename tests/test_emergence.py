import unittest
from pathlib import Path
import tempfile
import pandas as pd

from src.emergence import score_topic_trajectories


class TestEmergence(unittest.TestCase):

    def test_score_topic_trajectories_with_real_function(self):
        """
        Verify score_topic_trajectories from src.emergence correctly applies:
        1. Burst detection for novel high-volume shockwaves (vol >= 15, birth > W0).
        2. Emerging classification for multi-checkpoint trajectories with net growth.
        3. Low-coherence rejection (< 0.62).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trajs_csv = tmp_path / "trajectories.csv"
            alerts_csv = tmp_path / "alerts.csv"

            trajs_data = pd.DataFrame([
                # 1. Burst event: birth_idx > 0, vol >= 15, high coherence
                {
                    "trajectory_id": "TRAJ_BURST",
                    "topic_name": "onlyfans / ban",
                    "birth_window": "W_2",
                    "birth_window_index": 2,
                    "latest_window": "W_2",
                    "timeline_path": "W_2",
                    "volume_path": "20",
                    "total_volume": 20,
                    "avg_coherence": 0.85,
                    "min_coherence": 0.85,
                    "representative_post": "Onlyfans announced policy change"
                },
                # 2. Emerging multi-checkpoint event: net growth, momentum >= 1.0
                {
                    "trajectory_id": "TRAJ_EMERGING",
                    "topic_name": "vaccine / mandate",
                    "birth_window": "W_1",
                    "birth_window_index": 1,
                    "latest_window": "W_3",
                    "timeline_path": "W_1 -> W_2 -> W_3",
                    "volume_path": "10 -> 15 -> 22",
                    "total_volume": 47,
                    "avg_coherence": 0.80,
                    "min_coherence": 0.78,
                    "representative_post": "Vaccine mandate debates grow"
                },
                # 3. Low-coherence event (should be rejected)
                {
                    "trajectory_id": "TRAJ_LOW_COH",
                    "topic_name": "vague / chatter",
                    "birth_window": "W_1",
                    "birth_window_index": 1,
                    "latest_window": "W_2",
                    "timeline_path": "W_1 -> W_2",
                    "volume_path": "10 -> 12",
                    "total_volume": 22,
                    "avg_coherence": 0.52,
                    "min_coherence": 0.50,
                    "representative_post": "random unrelated stuff"
                }
            ])

            trajs_data.to_csv(trajs_csv, index=False)

            scored_df = score_topic_trajectories(
                trajectories_path=trajs_csv,
                output_path=alerts_csv
            )

            self.assertEqual(len(scored_df), 3)

            burst_row = scored_df[scored_df["trajectory_id"] == "TRAJ_BURST"].iloc[0]
            self.assertEqual(burst_row["lifecycle_status"], "[BURST]")
            self.assertTrue(burst_row["is_vetted_candidate"])
            self.assertGreater(burst_row["emergence_score"], 0.0)

            emerging_row = scored_df[scored_df["trajectory_id"] == "TRAJ_EMERGING"].iloc[0]
            self.assertEqual(emerging_row["lifecycle_status"], "[EMERGING]")
            self.assertTrue(emerging_row["is_vetted_candidate"])
            self.assertGreater(emerging_row["emergence_score"], 0.0)

            low_coh_row = scored_df[scored_df["trajectory_id"] == "TRAJ_LOW_COH"].iloc[0]
            self.assertEqual(low_coh_row["lifecycle_status"], "[REJECTED (LOW_COHERENCE)]")
            self.assertFalse(low_coh_row["is_vetted_candidate"])
            self.assertEqual(low_coh_row["emergence_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
