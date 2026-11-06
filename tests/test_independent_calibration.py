import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_prediction_intervals.py"


def prediction_rows(city_ids, error):
    rows = []
    for city_id in city_ids:
        for year in range(2018, 2021):
            for month in range(1, 13):
                target = city_id / 100.0 + (year - 2018) / 10.0 + month / 1000.0
                rows.append(
                    {
                        "city_idx": city_id,
                        "target_year": year,
                        "target_month": month,
                        "true_uhi": target,
                        "predicted_uhi": target + error,
                    }
                )
    return rows


class IndependentCalibrationTest(unittest.TestCase):
    def test_workflow_reuses_fitted_baselines(self):
        workflow = (ROOT / "scripts" / "run_independent_calibration.sh").read_text()
        helper = (ROOT / "scripts" / "_baseline_workflows.bash").read_text()
        self.assertIn("baseline/tabular_inference.py", workflow)
        self.assertIn("run_distilled_ensemble_inference.sh", workflow)
        self.assertNotIn("--adapter_path PPO", workflow)
        self.assertIn("run_calibration_timeseries_inference", workflow)
        self.assertNotIn("baseline/tabular_baselines.py", workflow)
        self.assertNotIn("run_regional_timeseries_baselines", workflow)
        inference = helper.split("run_calibration_timeseries_inference()", 1)[1].split(
            "\n)\n\nrun_regional_timeseries_baselines()", 1
        )[0]
        self.assertIn("--skip_train", inference)
        self.assertIn("--load_ckpt_path", inference)
        self.assertIn("--load_checkpoint_path", inference)
        self.assertIn("--load_checkpoints", inference)
        self.assertNotIn("--global_training", inference)

    def test_empirical_band_and_coverage_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path = root / "calibration.csv"
            test_path = root / "test.csv"
            calibration_manifest = root / "calibration_manifest.csv"
            test_manifest = root / "test_manifest.csv"
            output = root / "output"
            pd.DataFrame(prediction_rows([1, 2], 0.2)).to_csv(
                calibration_path, index=False
            )
            pd.DataFrame(prediction_rows([3, 4], 0.1)).to_csv(test_path, index=False)
            pd.DataFrame({"city_id": [1, 2]}).to_csv(
                calibration_manifest, index=False
            )
            pd.DataFrame({"id": [3, 4]}).to_csv(test_manifest, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--calibration_predictions",
                    str(calibration_path),
                    "--test_predictions",
                    str(test_path),
                    "--output_dir",
                    str(output),
                    "--expected_calibration_cities",
                    "2",
                    "--expected_test_cities",
                    "2",
                    "--expected_samples_per_city",
                    "36",
                    "--calibration_city_manifest",
                    str(calibration_manifest),
                    "--test_city_manifest",
                    str(test_manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(
                (output / "prediction_interval_summary.json").read_text()
            )
            self.assertAlmostEqual(summary["q_value"], 0.2)
            self.assertAlmostEqual(summary["test_coverage"], 1.0)
            self.assertEqual(
                set(summary),
                {
                    "quantile", "q_value", "calibration_cities",
                    "calibration_samples", "test_cities", "test_samples",
                    "test_coverage",
                },
            )
            city_coverage = pd.read_csv(output / "test_city_coverage.csv")
            self.assertEqual(city_coverage["city_idx"].tolist(), [3, 4])
            self.assertEqual(city_coverage["coverage"].tolist(), [1.0, 1.0])

    def test_non_finite_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = pd.DataFrame(prediction_rows([1], 0.2))
            calibration.loc[0, "predicted_uhi"] = float("nan")
            calibration.to_csv(root / "calibration.csv", index=False)
            pd.DataFrame(prediction_rows([2], 0.1)).to_csv(root / "test.csv", index=False)
            pd.DataFrame({"city_id": [1]}).to_csv(root / "calibration_manifest.csv", index=False)
            pd.DataFrame({"id": [2]}).to_csv(root / "test_manifest.csv", index=False)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--calibration_predictions",
                    str(root / "calibration.csv"),
                    "--test_predictions",
                    str(root / "test.csv"),
                    "--output_dir",
                    str(root / "output"),
                    "--expected_calibration_cities",
                    "1",
                    "--expected_test_cities",
                    "1",
                    "--calibration_city_manifest",
                    str(root / "calibration_manifest.csv"),
                    "--test_city_manifest",
                    str(root / "test_manifest.csv"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-finite", result.stderr)


if __name__ == "__main__":
    unittest.main()
