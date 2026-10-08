import unittest
from src.preprocessing import clean_reddit_text, strip_interrogative_templates


class TestPreprocessing(unittest.TestCase):

    def test_strip_interrogative_templates(self):
        """Verify real regex template stripping from src.preprocessing."""
        raw_text = "What is the most painful way to die?"
        stripped = strip_interrogative_templates(raw_text)
        self.assertNotIn("What is", stripped)
        self.assertIn("painful way to die", stripped)

    def test_redditors_of_reddit(self):
        """Verify removal of AskReddit boilerplates."""
        raw_text = "Redditors of reddit who have been in a coma, what did it feel like?"
        stripped = strip_interrogative_templates(raw_text)
        self.assertNotIn("Redditors of reddit", stripped)

    def test_clean_reddit_text_urls_and_tags(self):
        """Verify stripping of urls, html entities, and [deleted] tags."""
        self.assertEqual(clean_reddit_text("[deleted]"), "")
        self.assertEqual(clean_reddit_text("[removed]"), "")

        url_text = "Check this link https://reddit.com/r/news immediately"
        cleaned = clean_reddit_text(url_text)
        self.assertNotIn("https://", cleaned)
        self.assertIn("Check this link", cleaned)


if __name__ == "__main__":
    unittest.main()
