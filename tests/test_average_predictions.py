import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "average_predictions.py"


def _write_checkpoint(root, predictions, drop_last=False):
    city_dir = root / "nested" / "city_results"
    city_dir.mkdir(parents=True)
    rows = []
    for city_id in (1, 2):
        for month in (1, 2):
            rows.append(
                {
                    "city_idx": city_id,
                    "target_year": 2018,
                    "target_month": month,
                    "target_year_month": f"2018-{month:02d}",
                    "true_uhi": city_id + month / 10,
                    "predicted_uhi": predictions + city_id + month / 10,
                }
            )
    if drop_last:
        rows.pop()
    frame = pd.DataFrame(rows)
    for city_id, city in frame.groupby("city_idx"):
        city.to_csv(city_dir / f"city_{city_id}_predictions.csv", index=False)


def _write_uneven_checkpoint(root):
    city_dir = root / "city_results"
    city_dir.mkdir(parents=True)
    for city_id, months in ((1, (1,)), (2, (1, 2, 3))):
        city_rows = []
        for month in months:
            city_rows.append(
                {
                    "city_idx": city_id,
                    "target_year": 2018,
                    "target_month": month,
                    "target_year_month": f"2018-{month:02d}",
                    "true_uhi": 0.1 * month,
                    "predicted_uhi": 0.2 * month,
                }
            )
        pd.DataFrame(city_rows).to_csv(
            city_dir / f"city_{city_id}_predictions.csv", index=False
        )


class AveragePredictionsContractTest(unittest.TestCase):
    def test_aligned_inputs_honor_explicit_city_and_sample_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, output = root / "first", root / "second", root / "out"
            _write_checkpoint(first, 0.1)
            _write_checkpoint(second, 0.3)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(first),
                    str(second),
                    "--output",
                    str(output),
                    "--expected-cities",
                    "2",
                    "--expected-samples-per-city",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = pd.read_csv(output / "ensemble_predictions.csv")
            self.assertEqual(len(result), 4)
            self.assertAlmostEqual(result.iloc[0]["predicted_uhi"], 1.3)

    def test_mismatched_checkpoint_keys_are_rejected_before_inner_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            _write_checkpoint(first, 0.1)
            _write_checkpoint(second, 0.3, drop_last=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(first),
                    str(second),
                    "--output",
                    str(root / "out"),
                    "--expected-cities",
                    "2",
                    "--expected-samples-per-city",
                    "2",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Prediction keys differ", result.stderr)

    def test_uneven_per_city_counts_are_rejected_even_when_total_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            _write_uneven_checkpoint(first)
            _write_uneven_checkpoint(second)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(first),
                    str(second),
                    "--output",
                    str(root / "out"),
                    "--expected-cities",
                    "2",
                    "--expected-samples-per-city",
                    "2",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Wrong per-city counts", result.stderr)


if __name__ == "__main__":
    unittest.main()
