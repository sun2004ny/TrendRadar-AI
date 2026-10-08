import json
import unittest
from pathlib import Path
import pandas as pd

from src.config import ROOT_DIR


class TestRegressionGolden(unittest.TestCase):
    """
    Regression Test:
    Ensures refactoring preserves identical pipeline outputs:
    - Exactly 862 trajectories
    - Exactly 102 validated candidates
    - Exactly 517 quarantined micro-seeds
    - Key benchmark trajectories intact (OnlyFans, Vaccine, Hurricane Ida)
    """

    GOLDEN_PATH = ROOT_DIR / "tests" / "golden_metrics.json"

    def setUp(self):
        self.alerts_path = ROOT_DIR / "experiments" / "e6_combined" / "emergence_alerts.csv"
        self.assertTrue(self.alerts_path.exists(), f"Alerts file missing at {self.alerts_path}")
        self.alerts_df = pd.read_csv(self.alerts_path)

    def test_golden_regression_counts(self):
        total_trajectories = len(self.alerts_df)
        vetted_candidates = int((self.alerts_df["is_vetted_candidate"] == True).sum())
        micro_seeds = int((self.alerts_df["lifecycle_status"] == "[MICRO_SEED]").sum())

        current_metrics = {
            "total_trajectories": total_trajectories,
            "vetted_candidates": vetted_candidates,
            "micro_seeds": micro_seeds,
            "benchmark_trajectories": {
                "TRAJ_284": "OnlyFans",
                "TRAJ_456": "Vaccine Mandate",
                "TRAJ_757": "Hurricane Ida"
            }
        }

        # Create golden file if not present
        if not self.GOLDEN_PATH.exists():
            with open(self.GOLDEN_PATH, "w", encoding="utf-8") as f:
                json.dump(current_metrics, f, indent=2)

        # Load and compare against golden file
        with open(self.GOLDEN_PATH, "r", encoding="utf-8") as f:
            golden = json.load(f)

        self.assertEqual(
            total_trajectories,
            golden["total_trajectories"],
            f"Trajectory count changed: {total_trajectories} vs {golden['total_trajectories']}"
        )
        self.assertEqual(
            vetted_candidates,
            golden["vetted_candidates"],
            f"Vetted candidate count changed: {vetted_candidates} vs {golden['vetted_candidates']}"
        )
        self.assertEqual(
            micro_seeds,
            golden["micro_seeds"],
            f"Micro-seed count changed: {micro_seeds} vs {golden['micro_seeds']}"
        )

        # Exact expected numbers: 862, 102, 517
        self.assertEqual(total_trajectories, 862)
        self.assertEqual(vetted_candidates, 102)
        self.assertEqual(micro_seeds, 517)

        # Check key trajectories
        all_tids = set(self.alerts_df["trajectory_id"])
        for tid in golden["benchmark_trajectories"]:
            self.assertIn(tid, all_tids, f"Benchmark trajectory {tid} missing from production alerts!")


if __name__ == "__main__":
    unittest.main()
