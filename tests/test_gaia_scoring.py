import unittest

from gaia_scoring import check_gaia, extract_final_answer


class GaiaScoringTests(unittest.TestCase):
    def test_extract_final_answer_prefers_explicit_marker(self):
        response = "Reasoning here\nFINAL ANSWER: 42.0"
        self.assertEqual(extract_final_answer(response), "42.0")

    def test_extract_final_answer_without_marker_prefers_last_line(self):
        response = "Reasoning here\n42.0"
        self.assertEqual(extract_final_answer(response), "42.0")

    def test_extract_final_answer_strips_answer_prefix(self):
        response = "Answer: Paris"
        self.assertEqual(extract_final_answer(response), "Paris")

    def test_check_gaia_numeric_tolerance(self):
        self.assertTrue(check_gaia("FINAL ANSWER: 12.0004", "12"))

    def test_check_gaia_string_normalization(self):
        self.assertTrue(check_gaia("FINAL ANSWER: New   York", "newyork"))

    def test_check_gaia_list_normalization(self):
        self.assertTrue(check_gaia("FINAL ANSWER: Alice, Bob", "alice,bob"))

if __name__ == "__main__":
    unittest.main()
