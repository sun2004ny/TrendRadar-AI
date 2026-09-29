import unittest

class TestTemporalMatching(unittest.TestCase):
    def test_dual_continuity_gate(self):
        # Cosine must be >= 0.68 AND (Lexical Jaccard >= 0.20 OR Entity Jaccard >= 0.20)
        cosine_sim = 0.72
        lex_jaccard = 0.00
        ent_jaccard = 0.00
        # Drift pseudo-chain fails
        passes = (cosine_sim >= 0.68) and (lex_jaccard >= 0.20 or ent_jaccard >= 0.20)
        self.assertFalse(passes)

        # Lexically consistent link passes
        lex_jaccard = 0.25
        passes = (cosine_sim >= 0.68) and (lex_jaccard >= 0.20 or ent_jaccard >= 0.20)
        self.assertTrue(passes)

if __name__ == '__main__':
    unittest.main()
