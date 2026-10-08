import unittest
from src.clustering import extract_clean_ctf_idf


class TestClustering(unittest.TestCase):

    def test_extract_clean_ctf_idf(self):
        """Verify c-TF-IDF keyword extraction distinguishes clusters from stopwords."""
        cluster_texts = {
            0: [
                "OnlyFans announced sudden ban on sexually explicit content",
                "Explicit creator content prohibited under new OnlyFans policy rules",
                "Content creators react to OnlyFans adult ban"
            ],
            1: [
                "COVID vaccine mandate announced for healthcare workers",
                "Pfizer vaccine doses approved for immunization mandate",
                "Hospital staff face mandatory vaccine requirement"
            ]
        }

        keywords_map = extract_clean_ctf_idf(cluster_texts, top_k=3)

        self.assertIn(0, keywords_map)
        self.assertIn(1, keywords_map)

        # Cluster 0 should contain onlyfans / content / explicit / ban
        kws_0 = [w.lower() for w in keywords_map[0]]
        self.assertTrue(any(w in kws_0 for w in ["onlyfans", "content", "explicit", "ban"]))

        # Cluster 1 should contain vaccine / mandate / pfizer / covid
        kws_1 = [w.lower() for w in keywords_map[1]]
        self.assertTrue(any(w in kws_1 for w in ["vaccine", "mandate", "pfizer", "covid"]))

        # Boilerplate tropes like 'what', 'the', 'is' must never appear
        for kws in keywords_map.values():
            for w in kws:
                self.assertNotIn(w, ["what", "the", "people", "redditors"])


if __name__ == "__main__":
    unittest.main()
