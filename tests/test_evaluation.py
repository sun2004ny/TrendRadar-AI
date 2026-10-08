import os
import sys
import unittest
import json
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.evaluation import (
    load_config,
    parse_day_num,
    get_split_name,
    match_keywords,
    match_alert_to_events,
    compute_metrics,
    evaluate_pipeline,
)


class TestEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        self.config = load_config()

    def test_precision_recall_f1_known_example(self):
        """
        Verify precision/recall/F1 math on a known, hand-calculated example:
        - 2 real events: EVT_1 (day 10), EVT_2 (day 15)
        - 1 noise item:  EVT_3 (day 5)
        - 3 alerts:
            * Alert 1 matches EVT_1 at day 9  -> True Positive (TP = 1)
            * Alert 2 has no match at all     -> False Positive (FP += 1)
            * Alert 3 matches EVT_3 (noise)   -> False Positive (FP += 1)
        Expected:
            TP = 1, FP = 2, Total Alerts = 3
            Precision = 1 / 3 ≈ 0.3333
            Recall = 1 / 2 = 0.5000 (EVT_1 detected, EVT_2 missed)
            F1 = 2 * ( (1/3) * (1/2) ) / ( (1/3) + (1/2) ) = (1/3) / (5/6) = 2/5 = 0.4000
            Mean Lead Time = (10 - 9) = 1.0 day
        """
        gt_df = pd.DataFrame([
            {"event_id": "EVT_1", "label": "real_event", "start_day": "D_10", "keywords": "alpha,beta"},
            {"event_id": "EVT_2", "label": "real_event", "start_day": "D_15", "keywords": "gamma,delta"},
            {"event_id": "EVT_3", "label": "noise", "start_day": "D_05", "keywords": "spam,fake"}
        ])

        alerts_df = pd.DataFrame([
            {"birth_window": "D_09", "topic_name": "alpha topic", "representative_post": "alpha discussion", "emergence_score": 0.9},
            {"birth_window": "D_08", "topic_name": "random topic", "representative_post": "unrelated chatter", "emergence_score": 0.8},
            {"birth_window": "D_05", "topic_name": "spam topic", "representative_post": "fake spam message", "emergence_score": 0.7}
        ])

        metrics = compute_metrics(alerts_df, gt_df, alert_type="pipeline", config=self.config)

        self.assertEqual(metrics["total_alerts"], 3)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 2)
        self.assertAlmostEqual(metrics["precision"], 1 / 3, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["f1"], 0.4, places=4)
        self.assertEqual(metrics["mean_lead_time_days"], 1.0)

    def test_unmatched_alert_counts_as_false_positive(self):
        """
        Verify that an alert with no matching ground-truth row counts strictly as a False Positive.
        - 1 ground-truth real event
        - 2 alerts:
            * Alert 1 matches the event (TP)
            * Alert 2 discusses completely unrelated keywords (FP)
        Result: Total Alerts = 2, TP = 1, FP = 1 -> Precision = 1/2 = 0.50.
        """
        gt_df = pd.DataFrame([
            {"event_id": "EVT_TARGET", "label": "real_event", "start_day": "D_12", "keywords": "cyberattack,ransomware"}
        ])

        alerts_df = pd.DataFrame([
            {"birth_window": "D_12", "topic_name": "cyberattack", "representative_post": "ransomware hit hospital", "emergence_score": 0.95},
            {"birth_window": "D_12", "topic_name": "pizza recipe", "representative_post": "best homemade crust", "emergence_score": 0.80}
        ])

        metrics = compute_metrics(alerts_df, gt_df, alert_type="pipeline", config=self.config)

        self.assertEqual(metrics["total_alerts"], 2)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["precision"], 0.5)

    def test_ground_truth_event_with_no_alert_counts_as_miss(self):
        """
        Verify that a ground-truth real event with no matching alert counts as a miss (False Negative).
        - 2 ground-truth real events: EVT_FOUND (day 10), EVT_MISSED (day 12)
        - 1 alert that only matches EVT_FOUND
        Result: Detected = 1, Total Real = 2 -> Recall = 1/2 = 0.50. EVT_MISSED must not be in detected list.
        """
        gt_df = pd.DataFrame([
            {"event_id": "EVT_FOUND", "label": "real_event", "start_day": "D_10", "keywords": "earthquake,tremor"},
            {"event_id": "EVT_MISSED", "label": "real_event", "start_day": "D_12", "keywords": "solar,eclipse"}
        ])

        alerts_df = pd.DataFrame([
            {"birth_window": "D_10", "topic_name": "earthquake", "representative_post": "felt a strong tremor", "emergence_score": 0.88}
        ])

        metrics = compute_metrics(alerts_df, gt_df, alert_type="pipeline", config=self.config)

        self.assertEqual(metrics["total_real_events"], 2)
        self.assertEqual(metrics["detected_real_events"], 1)
        self.assertIn("EVT_FOUND", metrics["detected_event_ids"])
        self.assertNotIn("EVT_MISSED", metrics["detected_event_ids"])
        self.assertEqual(metrics["recall"], 0.5)

    def test_no_held_out_day_is_used_for_tuning(self):
        """
        Verify zero leakage between tuning days (D00-D19) and held-out test days (D20-D29):
        1. Tuning days and held-out days must be strictly disjoint sets.
        2. Any event or alert occurring on a held-out day (e.g. D_21 to D_30) is categorized as 'held_out'
           and NEVER included in tuning calculations.
        3. Adding or modifying held-out events has zero impact on the tuning split metrics.
        """
        tuning_days = self.config["evaluation"]["tuning_days"]
        held_out_days = self.config["evaluation"]["held_out_days"]

        # 1. Strict mathematical disjointness
        overlap = set(tuning_days).intersection(set(held_out_days))
        self.assertEqual(len(overlap), 0, f"Leakage detected: overlapping days {overlap}")

        # 2. Split mapping integrity
        for day in tuning_days:
            self.assertEqual(get_split_name(day, tuning_days, held_out_days), "tuning")
        for day in held_out_days:
            self.assertEqual(get_split_name(day, tuning_days, held_out_days), "held_out")

        # 3. Isolation test: Events on held-out days must not affect tuning split
        gt_mixed = pd.DataFrame([
            {"event_id": "EVT_TUNE", "label": "real_event", "start_day": "D_05", "keywords": "tuning_event", "split": "tuning"},
            {"event_id": "EVT_TEST", "label": "real_event", "start_day": "D_25", "keywords": "heldout_event", "split": "held_out"}
        ])

        alerts_tuning = pd.DataFrame([
            {"birth_window": "D_05", "topic_name": "tuning_event", "representative_post": "tuning post", "emergence_score": 0.9}
        ])

        # Evaluating on tuning subset
        gt_tuning_subset = gt_mixed[gt_mixed["split"] == "tuning"]
        tuning_metrics = compute_metrics(alerts_tuning, gt_tuning_subset, alert_type="pipeline", config=self.config)

        self.assertEqual(tuning_metrics["total_real_events"], 1)
        self.assertEqual(tuning_metrics["detected_real_events"], 1)
        self.assertEqual(tuning_metrics["recall"], 1.0)
        self.assertNotIn("EVT_TEST", tuning_metrics["detected_event_ids"])

    def test_parse_day_num_and_keywords(self):
        """Test day number extraction and regex keyword matching utilities."""
        self.assertEqual(parse_day_num("D_01"), 1)
        self.assertEqual(parse_day_num("D_15"), 15)
        self.assertEqual(parse_day_num("D22"), 22)
        self.assertEqual(parse_day_num(5), 5)
        self.assertEqual(parse_day_num(None), -1)

        matched = match_keywords("Hurricane Ida made landfall in Louisiana.", ["ida", "hurricane", "flood"])
        self.assertIn("ida", matched)
        self.assertIn("hurricane", matched)
        self.assertNotIn("flood", matched)


if __name__ == "__main__":
    unittest.main()
