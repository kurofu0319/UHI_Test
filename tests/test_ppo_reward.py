import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from output_parsing import parse_prediction  # noqa: E402
from reward import compute_ppo_reward  # noqa: E402


class PPORewardTest(unittest.TestCase):
    def test_unicode_minus_matches_ascii_minus(self):
        for text in ("-0.45", "-.45", "-5", "-0.1236", "-6.0"):
            with self.subTest(text=text):
                raw_text = text.replace("-", "\u2212")
                expected = parse_prediction(text)
                actual = parse_prediction(raw_text)
                self.assertEqual(actual.raw_text, raw_text)
                self.assertEqual(actual.matched_text, raw_text)
                for field in (
                    "parsed_value", "is_numeric", "numbers_found",
                    "decimal_places", "was_rounded", "in_valid_range",
                    "is_valid", "invalid_reason",
                ):
                    self.assertEqual(getattr(actual, field), getattr(expected, field))

    def test_unicode_minus_reward_uses_signed_prediction(self):
        parsed = parse_prediction("Result: \u22120.45")
        result = compute_ppo_reward(
            -0.5, parsed, y_bar=0.17, penalty_lambda=10
        )

        self.assertTrue(parsed.is_valid)
        self.assertAlmostEqual(parsed.parsed_value, -0.45)
        self.assertAlmostEqual(result.prediction_error, 0.05)
        self.assertEqual(result.invalid_penalty, 0.0)

    def test_valid_prediction_uses_absolute_error_without_penalty(self):
        parsed = parse_prediction("Result: 0.200")
        result = compute_ppo_reward(
            0.5, parsed, y_bar=0.17, penalty_lambda=10
        )

        self.assertTrue(parsed.is_valid)
        self.assertAlmostEqual(result.prediction_error, 0.3)
        self.assertEqual(result.invalid_penalty, 0.0)
        self.assertAlmostEqual(result.reward, -0.13)

    def test_missing_number_receives_explicit_invalid_penalty(self):
        parsed = parse_prediction("Result:")
        result = compute_ppo_reward(
            0.5, parsed, y_bar=0.17, penalty_lambda=10
        )

        self.assertFalse(parsed.is_valid)
        self.assertEqual(parsed.invalid_reason, "missing_numeric_value")
        self.assertEqual(result.prediction_error, 0.0)
        self.assertAlmostEqual(result.invalid_penalty, 1.7)
        self.assertAlmostEqual(result.reward, -1.53)

    def test_out_of_range_number_uses_only_invalid_penalty(self):
        parsed = parse_prediction("6.0", valid_min=-5, valid_max=5)
        result = compute_ppo_reward(
            0.5, parsed, y_bar=0.17, penalty_lambda=10
        )

        self.assertFalse(parsed.is_valid)
        self.assertEqual(parsed.invalid_reason, "out_of_range")
        self.assertEqual(result.prediction_error, 0.0)
        self.assertAlmostEqual(result.invalid_penalty, 1.7)
        self.assertAlmostEqual(result.reward, -1.53)

    def test_spurious_large_number_cannot_create_unbounded_reward(self):
        parsed = parse_prediction(
            "Result: 9651008000781192053",
            valid_min=-5,
            valid_max=5,
        )
        result = compute_ppo_reward(
            0.5, parsed, y_bar=0.15, penalty_lambda=10
        )

        self.assertFalse(parsed.is_valid)
        self.assertEqual(parsed.invalid_reason, "out_of_range")
        self.assertEqual(result.prediction_error, 0.0)
        self.assertAlmostEqual(result.invalid_penalty, 1.5)
        self.assertAlmostEqual(result.reward, -1.35)

    def test_first_number_is_used_when_generation_continues(self):
        parsed = parse_prediction(
            "0.546\n[Analysis:] The predicted UHI intensity for March 2019 is"
        )
        self.assertTrue(parsed.is_valid)
        self.assertAlmostEqual(parsed.parsed_value, 0.546)
        self.assertEqual(parsed.numbers_found, 2)

    def test_extra_precision_is_rounded_before_reward(self):
        parsed = parse_prediction("0.1236", max_decimal_places=3)
        result = compute_ppo_reward(
            0.2, parsed, y_bar=0.17, penalty_lambda=10
        )

        self.assertTrue(parsed.is_valid)
        self.assertTrue(parsed.was_rounded)
        self.assertEqual(parsed.decimal_places, 4)
        self.assertAlmostEqual(parsed.parsed_value, 0.124)
        self.assertAlmostEqual(result.prediction_error, 0.076)


if __name__ == "__main__":
    unittest.main()
