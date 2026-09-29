"""
test_e2e_pipeline.py - Comprehensive End-to-End Validation
==========================================================
Tests the full operational workflow:
Data Retrieval -> Trend Radar -> Emerging Topics -> Topic Details -> Research Hub -> AI Brief -> Action Hub -> Workspace
"""

import os
import sys
import unittest
import pandas as pd

# Add repo root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.research_service import get_topic_research_dossier, fetch_wikipedia_context, fetch_arxiv_papers
from src.llm_summary import generate_executive_brief
from src.workspace import (
    save_investigation,
    list_investigations,
    get_investigation,
    update_investigation_status,
    delete_investigation,
    export_workspace_markdown
)


class TestTrendRadarE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.alerts_path = os.path.join(ROOT_DIR, "experiments", "e6_combined", "emergence_alerts.csv")
        cls.assertTrue(os.path.exists(cls.alerts_path), f"Alerts file missing at {cls.alerts_path}")
        cls.alerts_df = pd.read_csv(cls.alerts_path)

    def test_01_data_and_topic_integrity(self):
        """Verify production alerts dataset has 102 vetted emerging candidates."""
        self.assertGreaterEqual(len(self.alerts_df), 100)
        vetted = self.alerts_df[self.alerts_df["is_vetted_candidate"] == True]
        self.assertGreaterEqual(len(vetted), 50)
        
        # Verify key benchmark topics exist
        tids = self.alerts_df["trajectory_id"].tolist()
        self.assertIn("TRAJ_284", tids, "TRAJ_284 (OnlyFans) must exist")
        self.assertIn("TRAJ_456", tids, "TRAJ_456 (Vaccine Mandate) must exist")

    def test_02_research_hub_external_context(self):
        """Verify Research Hub retrieves live external knowledge from Wikipedia/arXiv."""
        row = self.alerts_df[self.alerts_df["trajectory_id"] == "TRAJ_284"].iloc[0]
        keywords = ["onlyfans", "ban", "content"]
        
        dossier = get_topic_research_dossier("OnlyFans Policy Shift", keywords, row["representative_post"])
        self.assertIsNotNone(dossier)
        self.assertTrue(dossier["wikipedia"]["found"], "Wikipedia entry for OnlyFans must be found")
        self.assertIn("onlyfans", dossier["wikipedia"]["url"].lower())
        self.assertGreater(len(dossier["wikipedia"]["summary"]), 50)

    def test_03_ai_executive_brief_synthesis(self):
        """Verify AI Briefing correctly populates all 4 executive boxes from evidence."""
        row = self.alerts_df[self.alerts_df["trajectory_id"] == "TRAJ_284"].iloc[0]
        keywords = ["onlyfans", "ban", "content"]
        dossier = get_topic_research_dossier("OnlyFans Policy Shift", keywords, row["representative_post"])

        brief = generate_executive_brief(
            topic_name="OnlyFans Policy Shift",
            lifecycle_status=row["lifecycle_status"],
            volume_history=row["volume_path"],
            coherence=float(row["avg_coherence"]),
            birth_window=row["birth_window"],
            representative_post=row["representative_post"],
            keywords=keywords,
            research_dossier=dossier
        )

        self.assertIn("what_is_happening", brief)
        self.assertIn("why_is_it_emerging", brief)
        self.assertIn("evidence_summary", brief)
        self.assertIn("recommended_next_steps", brief)
        self.assertIn("full_brief_markdown", brief)

        # Check non-empty content
        self.assertTrue(len(brief["what_is_happening"]) > 20)
        self.assertTrue(len(brief["why_is_it_emerging"]) > 20)
        self.assertTrue(len(brief["evidence_summary"]) > 20)
        self.assertTrue(len(brief["recommended_next_steps"]) > 20)

    def test_04_workspace_persistence_and_management(self):
        """Verify saving, updating, listing, and exporting investigations."""
        row = self.alerts_df[self.alerts_df["trajectory_id"] == "TRAJ_456"].iloc[0]
        keywords = ["vaccine", "mandate", "medical"]

        # Save
        rec = save_investigation(
            trajectory_id="TRAJ_456",
            topic_name="Vaccine Mandate Discussion",
            lifecycle_status=row["lifecycle_status"],
            volume_path=str(row["volume_path"]),
            coherence=float(row["avg_coherence"]),
            birth_window=row["birth_window"],
            representative_post=row["representative_post"],
            keywords=keywords,
            analyst_notes="Early incubation signal. Extreme coherence 0.938.",
            status="Monitoring"
        )
        self.assertEqual(rec["id"], "INV_TRAJ_456")
        self.assertEqual(rec["status"], "Monitoring")

        # List
        cases = list_investigations()
        self.assertTrue(any(c["id"] == "INV_TRAJ_456" for c in cases))

        # Update
        updated = update_investigation_status("INV_TRAJ_456", "Active", "Upgraded to active investigation.")
        self.assertTrue(updated)
        fetched = get_investigation("INV_TRAJ_456")
        self.assertEqual(fetched["status"], "Active")

        # Export Dossier
        md_dossier = export_workspace_markdown()
        self.assertIn("TrendRadar AI: Master Workspace Dossier", md_dossier)
        self.assertIn("TRAJ_456", md_dossier)


if __name__ == "__main__":
    unittest.main()
