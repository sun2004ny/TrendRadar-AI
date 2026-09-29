import unittest
import re

def strip_interrogative_templates(text: str) -> str:
    patterns = [
        r"^(what|how|why|who|where|when)(\s+is|\s+are|\s+do|\s+does|\s+did|\s+was|\s+were|\s+would|\s+could|\s+should)?\b",
        r"^(people|redditors|men|women|doctors|teachers|users)\s+of\s+reddit\b",
        r"^(what\s+is|what\'s)\s+(the\s+most|your\s+favorite|something|a|an)\b",
        r"^(if\s+you\s+could|have\s+you\s+ever)\b"
    ]
    cleaned = text.strip()
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned

class TestPreprocessing(unittest.TestCase):
    def test_strip_interrogative_templates(self):
        raw_text = "What is the most painful way to die?"
        stripped = strip_interrogative_templates(raw_text)
        self.assertNotIn("What is", stripped)
        self.assertTrue("painful way to die" in stripped)

    def test_redditors_of_reddit(self):
        raw_text = "Redditors of reddit who have been in a coma, what did it feel like?"
        stripped = strip_interrogative_templates(raw_text)
        self.assertNotIn("Redditors of reddit", stripped)

if __name__ == "__main__":
    unittest.main()
