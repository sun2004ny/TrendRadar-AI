"""
tests/test_explainability.py
============================
Tests for Phase 8 Explainability:
1. Deterministic alert evidence generation and threshold comparisons.
2. Decision factor logic across volume, coherence, and novelty gates.
3. Strictly bounded LLM executive briefing.
4. Graceful deterministic fallback when LLM API is unavailable or errors out.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from src.explainability import generate_alert_evidence
from src.llm_summary import synthesize_bounded_brief


class TestExplainability(unittest.TestCase):

    def setUp(self):
        self.sample_alert = {
            "trajectory_id": "TRAJ_284",
            "day": "D_11",
            "title": "OnlyFans Policy Change",
            "post_count": 24,
            "coherence": 0.684,
            "velocity": 2.45,
            "acceleration": 0.82,
            "emergence_score": 0.92,
            "novelty_score": 0.48,
            "novelty_status": "passed",
            "keywords": ["onlyfans", "ban", "creators", "policy"],
            "sample_posts": ["With Onlyfans banning sexually explicit content, how does it affect creators?"]
        }

    def test_generate_alert_evidence_structure(self):
        """Verify evidence dict contains all required fields, metrics, and comparisons."""
        evidence = generate_alert_evidence(self.sample_alert)

        self.assertEqual(evidence["trajectory_id"], "TRAJ_284")
        self.assertEqual(evidence["day"], "D_11")
        self.assertEqual(evidence["title"], "OnlyFans Policy Change")

        # Metrics
        m = evidence["metrics"]
        self.assertEqual(m["post_count"], 24)
        self.assertAlmostEqual(m["coherence"], 0.684, places=3)
        self.assertAlmostEqual(m["velocity"], 2.45, places=2)
        self.assertAlmostEqual(m["novelty_score"], 0.48, places=2)

        # Thresholds
        t = evidence["thresholds"]
        self.assertIn("coherence_floor", t)
        self.assertIn("velocity_threshold", t)
        self.assertIn("novelty_threshold", t)

        # Decision factors
        df = evidence["decision_factors"]
        self.assertIsInstance(df, list)
        self.assertGreaterEqual(len(df), 3)

        # Confirm velocity factor mentions exceeding threshold
        self.assertTrue(any("velocity (2.45x) exceeded" in s.lower() for s in df))
        # Confirm coherence factor mentions exceeding baseline floor
        self.assertTrue(any("semantic coherence (0.684) exceeded" in s.lower() for s in df))
        # Confirm novelty gate factor
        self.assertTrue(any("novelty gate verified novel discourse" in s.lower() for s in df))

        # Deterministic text
        self.assertIn("TRAJ_284", evidence["deterministic_explanation"])
        self.assertIn("24 posts", evidence["deterministic_explanation"])

    def test_synthesize_bounded_brief_deterministic_fallback(self):
        """Verify brief synthesis falls back gracefully to deterministic brief without API key."""
        evidence = generate_alert_evidence(self.sample_alert)
        brief = synthesize_bounded_brief(evidence, api_key=None)

        self.assertIn("what_is_happening", brief)
        self.assertIn("why_is_it_emerging", brief)
        self.assertIn("evidence_summary", brief)
        self.assertIn("recommended_next_steps", brief)
        self.assertIn("full_brief_markdown", brief)

        # Grounding check
        self.assertIn("TRAJ_284", brief["what_is_happening"])
        self.assertIn("2.45x", brief["evidence_summary"])
        self.assertIn("0.684", brief["evidence_summary"])

    def test_synthesize_bounded_brief_mocked_llm_call(self):
        """Verify mock LLM call passes strictly bounded evidence without hallucination."""
        evidence = generate_alert_evidence(self.sample_alert)

        mock_llm_response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "what_is_happening": "Creators are discussing the OnlyFans policy ban.",
                            "why_is_it_emerging": "Discussion volume surged with 2.45x velocity.",
                            "evidence_summary": "24 posts with 0.684 coherence.",
                            "recommended_next_steps": "Monitor next checkpoint for policy updates."
                        })
                    }]
                }
            }]
        }

        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_llm_response).encode("utf-8")
            mock_url.return_value.__enter__.return_value = mock_resp

            brief = synthesize_bounded_brief(evidence, api_key="fake-key-for-test")
            self.assertEqual(brief["what_is_happening"], "Creators are discussing the OnlyFans policy ban.")
            self.assertEqual(brief["why_is_it_emerging"], "Discussion volume surged with 2.45x velocity.")

            # Verify prompt contents in mock call
            call_args = mock_url.call_args[0][0]
            payload = json.loads(call_args.data.decode("utf-8"))
            prompt_sent = payload["contents"][0]["parts"][0]["text"]
            self.assertIn("TRAJ_284", prompt_sent)
            self.assertIn("2.45x", prompt_sent)
            self.assertIn("0.684", prompt_sent)
            self.assertIn("STRICTLY VERIFIED", prompt_sent)

    def test_synthesize_bounded_brief_llm_error_fallback(self):
        """Verify LLM network failure falls back cleanly to deterministic brief without throwing."""
        evidence = generate_alert_evidence(self.sample_alert)

        with patch("urllib.request.urlopen", side_effect=RuntimeError("Connection refused")):
            brief = synthesize_bounded_brief(evidence, api_key="broken-key")
            self.assertIn("what_is_happening", brief)
            self.assertIn("TRAJ_284", brief["what_is_happening"])


if __name__ == "__main__":
    unittest.main()
