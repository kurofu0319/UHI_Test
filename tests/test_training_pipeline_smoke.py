import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class TrainingPipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in (
            "_common.sh", "run_sft.sh", "run_lora_fusion.sh", "run_ppo.sh",
            "run_explanation_distillation.sh", "run_distilled_ensemble_inference.sh",
            "run_main_evaluation.sh", "run_efficiency_measurement.sh",
        ):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        (scripts / "_baseline_workflows.bash").write_text(
            'run_global_tabular_baselines() {\n'
            '  "${PYTHON}" smoke_tabular --output-root "${OUTPUT_ROOT}" --raw-dir "${EFFICIENCY_RECORD_DIR}"\n'
            '}\n'
            'run_global_timeseries_baselines() {\n'
            '  "${PYTHON}" smoke_timeseries --output-root "${OUTPUT_ROOT}" --raw-dir "${EFFICIENCY_RECORD_DIR}"\n'
            '}\n'
        )
        runner = self.root / "python_stub"
        runner.write_text(
            f"#!{sys.executable}\n"
            + (ROOT / "tests/fixtures/training_workflow_stub.py").read_text()
        )
        runner.chmod(0o755)
        base = self.root / "model/deepseek-8b"
        base.mkdir(parents=True)
        (base / "config.json").write_text("{}")
        for name in ("train", "valid", "city_metrics", "explanation_samples"):
            (self.root / "data" / name).mkdir(parents=True)
        for split, ids in (("train", range(1, 501)), ("test", range(501, 601))):
            with (self.root / f"data/city_metrics/city_metrics_{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "Area"])
                writer.writeheader()
                for city in ids:
                    writer.writerow({"id": city, "Area": [125, 250, 600, 1000, 2000][city % 5]})
        for index in range(1, 901):
            (self.root / f"data/explanation_samples/sample_{index}.txt").write_text("test fixture\n")
        self.commands = self.root / "commands.jsonl"
        self.environment = {
            key: val for key, val in os.environ.items()
            if not key.startswith(("SFT_", "PPO_", "FUSION_", "EXPLANATION_", "DISTILL_", "DISTILLED_", "MAIN_", "EFFICIENCY_"))
        }
        self.environment.update({
            "PYTHON": str(runner), "GPU_ID": "0",
            "DATA_DIR": str(self.root / "data"),
            "OUTPUT_ROOT": str(self.root / "outputs"),
            "LOG_DIR": str(self.root / "logs"),
            "SMOKE_COMMANDS": str(self.commands), "SMOKE_REAL_ROOT": str(ROOT),
        })
        for name in ("MODEL", "BASE_MODEL", "MERGED_MODEL", "MERGED_MODEL_DIR"):
            self.environment.pop(name, None)

    def run_workflow(self, name, **overrides):
        result = subprocess.run(
            ["bash", str(self.root / "scripts" / name)],
            cwd=self.root, env={**self.environment, **overrides},
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_readme_main_pipeline_produces_two_members_tables_and_timing(self):
        for name in (
            "run_sft.sh", "run_lora_fusion.sh", "run_ppo.sh",
            "run_explanation_distillation.sh", "run_main_evaluation.sh",
            "run_efficiency_measurement.sh",
        ):
            self.run_workflow(name)
        model = "deepseek-8b_merged__Global500"
        for epoch in (0, 1):
            self.assertTrue((self.root / "model" / f"{model}_PPO_epoch{epoch}_merged/config.json").is_file())
            self.assertTrue((self.root / "model_adapters/explanation_distilled" / model / f"epoch{epoch}/adapter_config.json").is_file())
        output = self.root / "outputs"
        summary = pd.read_csv(output / "main_ensemble/ensemble/ensemble_metrics.csv").iloc[0]
        self.assertEqual(summary["cities"], 100)
        self.assertEqual(summary["samples"], 3600)
        self.assertAlmostEqual(summary["mae"], 0.15)
        self.assertTrue((output / "tables/table1_main_performance.csv").is_file())
        timing = pd.read_csv(output / "efficiency/efficiency_measurements.csv").set_index(["kind", "method", "stage"])
        self.assertAlmostEqual(timing.loc[("training", "UHI-E3LLM", "Total"), "value"], 40.0)
        self.assertAlmostEqual(timing.loc[("inference", "UHI-E3LLM", "inference"), "value"], 240.0)
        calls = [json.loads(line) for line in self.commands.read_text().splitlines()]
        distillations = [call for call in calls if Path(call[0]).name == "explanation_distillation.py"]
        self.assertEqual(len(distillations), 2)
        self.assertEqual([call[call.index("--efficiency_member_epoch") + 1] for call in distillations], ["0", "1"])
        names = [Path(call[0]).name for call in calls]
        self.assertLess(names.index("SFT_trainer.py"), names.index("greedy_adapter_fusion.py"))
        self.assertLess(names.index("greedy_adapter_fusion.py"), names.index("PPOTrainer.py"))
        self.assertLess(names.index("PPOTrainer.py"), names.index("explanation_distillation.py"))

    def test_custom_training_outputs_preserve_main_timing(self):
        self.run_workflow("run_sft.sh")
        self.run_workflow("run_lora_fusion.sh")
        self.run_workflow("run_ppo.sh")
        raw = self.root / "outputs/efficiency/raw"
        originals = {name: (raw / name).read_bytes() for name in ("training_uhi_sft.json", "training_uhi_ppo.json")}
        for scope in ("regional/Africa", "ablations/modules/without_uhi_knowledge", "ablations/components/without_fusion"):
            with self.subTest(scope=scope):
                scoped_root = self.root / "model_adapters" / scope
                self.run_workflow("run_sft.sh", SFT_OUTPUT=str(scoped_root / "sft"), SMOKE_TRAINING_SECONDS="99")
                self.run_workflow("run_ppo.sh", PPO_OUTPUT=str(scoped_root / "ppo"), SMOKE_TRAINING_SECONDS="99")
                for stage in ("sft", "ppo"):
                    record = json.loads((scoped_root / stage / "training_measurement.json").read_text())
                    self.assertEqual(record["value"], 99.0)
                for name, content in originals.items():
                    self.assertEqual((raw / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
