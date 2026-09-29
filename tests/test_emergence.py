import unittest

class TestTwoTierIncubation(unittest.TestCase):
    def test_quarantined_micro_seed(self):
        # 3 posts, coherence 0.95, but no multi-checkpoint confirmation and no verified anchor
        volume = 3
        coherence = 0.95
        distinct_checkpoints = 1
        has_anchor = False

        if 3 <= volume < 7:
            if distinct_checkpoints >= 2 or (has_anchor and coherence >= 0.85):
                status = "CONFIRMED_SEED"
                score = 0.50
            elif coherence >= 0.65:
                status = "MICRO_SEED"
                score = 0.0  # Quarantined!
            else:
                status = "STATIC_MICRO"
                score = 0.0

        self.assertEqual(status, "MICRO_SEED")
        self.assertEqual(score, 0.0)

    def test_confirmed_seed_promotion(self):
        volume = 3
        coherence = 0.938
        distinct_checkpoints = 1
        has_anchor = True  # e.g., verified vaccine mandate entity

        if 3 <= volume < 7:
            if distinct_checkpoints >= 2 or (has_anchor and coherence >= 0.85):
                status = "CONFIRMED_SEED"
                score = 0.52
            elif coherence >= 0.65:
                status = "MICRO_SEED"
                score = 0.0
            else:
                status = "STATIC_MICRO"
                score = 0.0

        self.assertEqual(status, "CONFIRMED_SEED")
        self.assertGreater(score, 0.0)

if __name__ == '__main__':
    unittest.main()
