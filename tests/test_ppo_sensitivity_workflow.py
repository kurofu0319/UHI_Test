import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "scripts" / "run_ppo_penalty_sensitivity.sh"
SUMMARIZER = ROOT / "scripts" / "summarize_ppo_sensitivity.py"


class PPOSensitivityWorkflowTest(unittest.TestCase):
    def test_workflow_holds_main_ppo_settings_fixed(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"', workflow)
        self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"', workflow)
        self.assertIn('PPO_UPDATE_EPOCHS="${PPO_UPDATE_EPOCHS:-4}"', workflow)
        self.assertIn('PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-3e-6}"', workflow)
        self.assertIn('PPO_MAX_NEW_TOKENS="${PPO_MAX_NEW_TOKENS:-24}"', workflow)
        self.assertIn("--input_format prompt_task_only", workflow)
        self.assertIn("--reward_baseline_report", workflow)
        self.assertIn("ppo_reward_baseline.json", workflow)
        self.assertIn('DISTILLED_MODE="${EVAL_MODE}"', workflow)
        self.assertIn('EVAL_MODE="${EVAL_MODE:-validation}"', workflow)
        self.assertIn("--expected_train_cities", workflow)
        self.assertIn("--expected_samples_per_city", workflow)
        self.assertIn("--train_epochs 2", workflow)
        self.assertIn("run_explanation_distillation.sh", workflow)
        self.assertIn("run_distilled_ensemble_inference.sh", workflow)
        self.assertIn('DISTILLED_EPOCHS="0 1"', workflow)
        self.assertIn('EXPLANATION_POLICY_ADAPTER_DIR="${adapter_dir}/epoch${policy_epoch}"', workflow)
        self.assertNotIn("--train_epochs 1", workflow)
        self.assertNotIn('SENSITIVITY_REWARD_BASELINE:-0.17', workflow)

    @staticmethod
    def _write_run(root: Path, penalty_lambda: int, city_ids=(1, 2)) -> None:
        run_dir = root / f"ppo_lambda_{penalty_lambda}" / "ensemble"
        run_dir.mkdir(parents=True)
        rows = []
        for offset, city_id in enumerate(city_ids):
            absolute_error = 0.1 + offset * 0.1 + penalty_lambda * 0.0001
            for month in range(1, 4):
                rows.append(
                    {
                        "city_idx": city_id,
                        "target_year": 2018,
                        "target_month": month,
                        "true_uhi": 0.5,
                        "predicted_uhi": 0.5 + absolute_error,
                    }
                )
        pd.DataFrame(rows).to_csv(run_dir / "ensemble_predictions.csv", index=False)

    def test_summarizer_accepts_complete_matching_validation_runs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for penalty_lambda in (0, 1, 5, 10, 20):
                self._write_run(root, penalty_lambda)
            output = root / "summary.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(SUMMARIZER),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--expected_cities",
                    "2",
                    "--expected_samples_per_city",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = pd.read_csv(output)
            self.assertEqual(summary["lambda"].tolist(), [0, 1, 5, 10, 20])
            self.assertTrue(summary["cities"].eq(2).all())
            self.assertTrue(summary["samples"].eq(6).all())
            self.assertAlmostEqual(summary.loc[0, "mae"], 0.15)

    def test_summarizer_rejects_different_city_sets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for penalty_lambda in (0, 1, 5, 10, 20):
                city_ids = (1, 3) if penalty_lambda == 5 else (1, 2)
                self._write_run(root, penalty_lambda, city_ids=city_ids)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARIZER),
                    "--root",
                    str(root),
                    "--output",
                    str(root / "summary.csv"),
                    "--expected_cities",
                    "2",
                    "--expected_samples_per_city",
                    "3",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different city set", result.stderr)


if __name__ == "__main__":
    unittest.main()
