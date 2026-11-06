import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from derive_ppo_reward_baseline import summarize_inference  # noqa: E402
from PPOTrainer import _validate_reward_baseline_report  # noqa: E402


class PPORewardBaselineTest(unittest.TestCase):
    def _write_summary(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "train_inference.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_baseline_is_sample_weighted_training_mae(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            summary = self._write_summary(
                directory,
                [
                    {"city_idx": 1, "samples": 2, "mae": 0.1, "mse": 0.04, "rmse": 0.2},
                    {"city_idx": 2, "samples": 2, "mae": 0.3, "mse": 0.16, "rmse": 0.4},
                ],
            )

            report = summarize_inference(
                summary, split="train", expected_cities=2, expected_samples_per_city=2
            )

            self.assertAlmostEqual(report["y_bar"], 0.2)
            self.assertEqual(report["statistic"], "sample-weighted training MAE")
            self.assertEqual(report["reward_symbol"], "e_bar")
            self.assertAlmostEqual(report["metrics"]["mse"], 0.1)
            self.assertAlmostEqual(report["metrics"]["rmse"], 0.1**0.5)
            self.assertEqual(report["split"]["samples"], 4)
            self.assertEqual(report["split"]["target_start"], "2004-01")
            self.assertEqual(report["split"]["target_end"], "2017-12")
            self.assertFalse(report["split"]["held_out_test_cities_used"])

    def test_wrong_per_city_sample_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            summary = self._write_summary(
                directory,
                [
                    {"city_idx": 1, "samples": 1, "mae": 0.1, "mse": 0.04, "rmse": 0.2},
                    {"city_idx": 2, "samples": 2, "mae": 0.3, "mse": 0.16, "rmse": 0.4},
                ],
            )

            with self.assertRaisesRegex(ValueError, "each train city"):
                summarize_inference(
                    summary,
                    split="train",
                    expected_cities=2,
                    expected_samples_per_city=2,
                )

    def test_report_validator_rejects_test_city_leakage(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "baseline.json"
            report = {
                "statistic": "sample-weighted training MAE",
                "y_bar": 0.17,
                "metrics": {"mae": 0.17},
                "split": {"held_out_test_cities_used": True},
            }
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "test cities"):
                _validate_reward_baseline_report(path, 0.17)

    def test_report_validator_rejects_validation_statistic(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "baseline.json"
            report = {
                "statistic": "sample-weighted validation MAE",
                "y_bar": 0.17,
                "metrics": {"mae": 0.17},
                "split": {"held_out_test_cities_used": False},
            }
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "training MAE"):
                _validate_reward_baseline_report(path, 0.17)

    def test_report_validator_accepts_matching_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "baseline.json"
            report = {
                "statistic": "sample-weighted training MAE",
                "y_bar": 0.17,
                "metrics": {"mae": 0.17},
                "split": {"held_out_test_cities_used": False},
            }
            path.write_text(json.dumps(report), encoding="utf-8")

            validated = _validate_reward_baseline_report(path, 0.17)
            self.assertEqual(validated, report)


if __name__ == "__main__":
    unittest.main()
