import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_efficiency_measurements import aggregate  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from efficiency import benchmark_model_call  # noqa: E402


class EfficiencyWorkflowTest(unittest.TestCase):
    def test_inference_calls_are_timed_as_independent_repetitions(self):
        calls = []

        def model_call():
            calls.append(len(calls))

        clock = iter([0.0, 0.2, 1.0, 1.4, 2.0, 2.6])
        with patch("efficiency.torch.cuda.is_available", return_value=False), patch(
            "efficiency.synchronize_cuda"
        ), patch(
            "efficiency.time.perf_counter", side_effect=lambda: next(clock)
        ):
            latency = benchmark_model_call(
                model_call,
                sample_count=2,
                warmup_calls=2,
                measured_calls=3,
            )
        self.assertEqual(len(calls), 5)
        self.assertAlmostEqual(latency, 200.0)

    def test_released_inference_protocol_defaults(self):
        expected_gpu_defaults = {
            ROOT / "src/Inference.py": [
                '"--efficiency_batch_size", type=int, default=36',
                '"--efficiency_warmup_calls", type=int, default=5',
                '"--efficiency_measured_calls", type=int, default=5',
            ],
            ROOT / "baseline/armd/main_new.py": [
                "'--efficiency_batch_size', type=int, default=36",
                "'--efficiency_measured_calls', type=int, default=5",
            ],
            ROOT / "baseline/filternet/run.py": [
                "'--efficiency_batch_size', type=int, default=36",
                "'--efficiency_measured_calls', type=int, default=5",
            ],
            ROOT / "baseline/st_mtm/run.py": [
                "'--efficiency_batch_size', type=int, default=36",
                "'--efficiency_measured_calls', type=int, default=5",
            ],
            ROOT / "baseline/tabular_baselines.py": [
                '"--efficiency_batch_size", type=int, default=36',
                '"--efficiency_gpu_batch_size", type=int, default=36',
                '"--efficiency_measured_calls", type=int, default=5',
            ],
        }
        for path, fragments in expected_gpu_defaults.items():
            source = path.read_text(encoding="utf-8")
            for fragment in fragments:
                self.assertIn(fragment, source, f"{path}: {fragment}")

    def _write(self, directory, name, kind, method, stage, value):
        record = {
            "schema_version": 1,
            "kind": kind,
            "method": method,
            "stage": stage,
            "value": value,
            "unit": "seconds" if kind == "training" else "ms_per_sample",
            "hardware": "test hardware",
            "note": "model execution only" if kind == "inference" else "training loop",
            "metadata": (
                {
                    "batch_size": 36,
                    "warmup_calls": 5,
                    "measured_calls": 5,
                    "timing_protocol": "independent_repetitions_mean",
                    "max_new_tokens": 19,
                }
                if kind == "inference" else {}
            ),
        }
        (directory / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")

    def _complete_raw_records(self, directory):
        training = [
            ("uhi_sft", "UHI-E3LLM", "SFT", 10.0),
            ("uhi_ppo", "UHI-E3LLM", "PPO", 20.0),
            ("uhi_distillation_epoch0", "UHI-E3LLM", "Distillation_epoch0", 40.0),
            ("uhi_distillation_epoch1", "UHI-E3LLM", "Distillation_epoch1", 50.0),
            ("armd", "ARMD", "train", 3.0),
            ("filternet", "FilterNet", "train", 4.0),
            ("stmtm_pretrain", "ST-MTM", "pretrain", 5.0),
            ("stmtm_finetune", "ST-MTM", "finetune", 6.0),
            ("xgb", "XGBoost", "train", 7.0),
            ("rf", "RandomForest", "train", 8.0),
            ("lgb", "LightGBM", "train", 9.0),
            ("mlp", "MLP", "train", 10.0),
        ]
        for name, method, stage, value in training:
            self._write(directory, f"training_{name}", "training", method, stage, value)
        for index, method in enumerate(
            [
                "ARMD", "FilterNet", "ST-MTM",
                "XGBoost", "RandomForest", "LightGBM", "MLP",
            ],
            start=1,
        ):
            self._write(
                directory,
                f"inference_{index}",
                "inference",
                method,
                "inference",
                float(index),
            )
        for epoch, value in enumerate((100.0, 140.0)):
            self._write(
                directory,
                f"inference_uhi_epoch{epoch}",
                "inference",
                "UHI-E3LLM",
                f"inference_epoch{epoch}",
                value,
            )

    def test_aggregate_derives_only_documented_sums(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_dir = Path(temp)
            self._complete_raw_records(raw_dir)
            self._write(
                raw_dir, "teacher_generation", "training", "UHI-E3LLM", "teacher_generation", 100.0
            )
            table = aggregate(raw_dir).set_index(["kind", "method", "stage"])
            self.assertEqual(len(table), 24)
            self.assertEqual(
                table.loc[("training", "UHI-E3LLM", "Total"), "value"],
                120.0,
            )
            self.assertEqual(
                table.loc[("training", "UHI-E3LLM", "Distillation"), "value"],
                90.0,
            )
            self.assertEqual(
                table.loc[("training", "UHI-E3LLM", "Distillation_epoch0"), "value"],
                40.0,
            )
            self.assertEqual(
                table.loc[("training", "UHI-E3LLM", "Distillation_epoch1"), "value"],
                50.0,
            )
            self.assertEqual(
                table.loc[("training", "ST-MTM", "train"), "value"],
                11.0,
            )
            self.assertEqual(
                table.loc[("inference", "UHI-E3LLM", "member_mean"), "value"],
                120.0,
            )
            self.assertEqual(
                table.loc[("inference", "UHI-E3LLM", "inference"), "value"],
                240.0,
            )
            self.assertEqual(
                table.loc[("inference", "UHI-E3LLM", "inference_epoch0"), "value"],
                100.0,
            )
            self.assertEqual(
                table.loc[("inference", "UHI-E3LLM", "inference_epoch1"), "value"],
                140.0,
            )
            self.assertNotIn(("training", "UHI-E3LLM", "SFT+PPO"), table.index)

    def test_aggregate_rejects_incomplete_workflow(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_dir = Path(temp)
            self._complete_raw_records(raw_dir)
            next(raw_dir.glob("inference_*.json")).unlink()
            with self.assertRaisesRegex(ValueError, "missing raw records"):
                aggregate(raw_dir)

    def test_single_member_record_cannot_replace_two_distilled_policies(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_dir = Path(temp)
            self._complete_raw_records(raw_dir)
            (raw_dir / "inference_uhi_epoch1.json").unlink()
            self._write(
                raw_dir, "inference_uhi", "inference", "UHI-E3LLM", "inference", 120.0
            )
            with self.assertRaisesRegex(ValueError, "inference_epoch1"):
                aggregate(raw_dir)

    def test_training_total_requires_both_distillation_records(self):
        for epoch in (0, 1):
            with self.subTest(epoch=epoch), tempfile.TemporaryDirectory() as temp:
                raw_dir = Path(temp)
                self._complete_raw_records(raw_dir)
                (raw_dir / f"training_uhi_distillation_epoch{epoch}.json").unlink()
                with self.assertRaisesRegex(ValueError, f"Distillation_epoch{epoch}"):
                    aggregate(raw_dir)

    def test_efficiency_entry_builds_total_and_checks_distillation_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            self._complete_raw_records(raw_dir)
            output = root / "efficiency.csv"
            environment = {
                **os.environ,
                "PYTHON": sys.executable,
                "OUTPUT_ROOT": str(root / "outputs"),
                "LOG_DIR": str(root / "logs"),
                "EFFICIENCY_RECORD_DIR": str(raw_dir),
                "EFFICIENCY_RAW_DIR": str(raw_dir),
                "EFFICIENCY_OUTPUT": str(output),
                "EFFICIENCY_RUN_MAIN_EVALUATION": "0",
            }
            command = ["bash", str(ROOT / "scripts/run_efficiency_measurement.sh")]
            subprocess.run(
                command, env=environment, check=True, capture_output=True, text=True
            )
            import pandas as pd

            table = pd.read_csv(output).set_index(["kind", "method", "stage"])
            self.assertEqual(table.loc[("training", "UHI-E3LLM", "Total"), "value"], 120.0)
            (raw_dir / "training_uhi_distillation_epoch1.json").unlink()
            result = subprocess.run(
                command, env=environment, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("training_uhi_distillation_epoch1.json", result.stderr)

    def test_distillation_timing_files_follow_policy_and_adapter_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            for name in ("_common.sh", "run_explanation_distillation.sh"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
            base = root / "model" / "test"
            base.mkdir(parents=True)
            (base / "config.json").write_text("{}")
            policy_root = root / "policies"
            for epoch in (0, 1):
                adapter = policy_root / f"epoch{epoch}"
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text("{}")
            samples = root / "data" / "explanation_samples"
            samples.mkdir(parents=True)
            (samples / "sample_1.txt").write_text("test fixture")
            recorder = root / "record_commands.py"
            recorder.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                "    print('test python')\n"
                "elif args[0] == '-c':\n"
                "    print('4.46.3 0.11.2 0.19.1 1.0')\n"
                "else:\n"
                "    with open(os.environ['COMMAND_RECORD'], 'a') as stream:\n"
                "        stream.write(json.dumps(args) + '\\n')\n"
                "    output = Path(args[args.index('--output_dir') + 1])\n"
                "    output.mkdir(parents=True, exist_ok=True)\n"
                "    name = 'config.json' if args[0] == 'src/merge_lora.py' else 'adapter_config.json'\n"
                "    (output / name).write_text('{}')\n"
            )
            recorder.chmod(0o755)
            record_path = root / "commands.jsonl"
            distilled_root = root / "distilled"
            environment = {
                **os.environ,
                "PYTHON": str(recorder),
                "BASE_MODEL": "test",
                "DATA_DIR": str(root / "data"),
                "PPO_POLICY_ADAPTER_ROOT": str(policy_root),
                "EXPLANATION_DISTILLED_ADAPTER_ROOT": str(distilled_root),
                "EXPLANATION_DATA_ROOT": str(samples),
                "EXPLANATION_SAMPLE_COUNT": "1",
                "EXPLANATION_POLICY_EPOCHS": "0 1",
                "EXPLANATION_INTERNAL_SINGLE_EPOCH": "0",
                "EXPLANATION_FORCE_REGENERATE": "0",
                "EXPLANATION_DRY_RUN": "0",
                "EXPLANATION_GENERATE_ONLY": "0",
                "EXPLANATION_RUN_INFERENCE": "0",
                "OUTPUT_ROOT": str(root / "outputs"),
                "LOG_DIR": str(root / "logs"),
                "EFFICIENCY_RECORD_DIR": str(root / "raw"),
                "COMMAND_RECORD": str(record_path),
            }
            environment.pop("EXPLANATION_POLICY_ADAPTER_DIR", None)
            command = ["bash", str(root / "scripts/run_explanation_distillation.sh")]
            subprocess.run(
                command, env=environment, check=True, capture_output=True, text=True
            )
            custom_adapter = root / "ablation_adapter"
            environment.update({
                "EXPLANATION_POLICY_ADAPTER_DIR": str(policy_root / "epoch1"),
                "EXPLANATION_POLICY_EPOCH": "1",
                "EXPLANATION_ADAPTER_DIR": str(custom_adapter),
            })
            subprocess.run(
                command, env=environment, check=True, capture_output=True, text=True
            )
            commands = [json.loads(line) for line in record_path.read_text().splitlines()]
            distillations = [
                args for args in commands if args[0] == "src/explanation_distillation.py"
            ]
            self.assertEqual(len(distillations), 3)
            for epoch, args in enumerate(distillations[:2]):
                self.assertEqual(
                    args[args.index("--efficiency_record") + 1],
                    str(root / "raw" / f"training_uhi_distillation_epoch{epoch}.json"),
                )
                self.assertEqual(args[args.index("--efficiency_member_epoch") + 1], str(epoch))
            self.assertEqual(
                distillations[2][distillations[2].index("--efficiency_record") + 1],
                str(custom_adapter / "training_measurement.json"),
            )

    def test_aggregate_rejects_non_36_timing_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_dir = Path(temp)
            self._complete_raw_records(raw_dir)
            path = raw_dir / "inference_4.json"
            record = json.loads(path.read_text())
            record["metadata"]["batch_size"] = 128
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "batch_size must be 36"):
                aggregate(raw_dir)

    def test_aggregate_rejects_different_member_protocols(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_dir = Path(temp)
            self._complete_raw_records(raw_dir)
            path = raw_dir / "inference_uhi_epoch1.json"
            record = json.loads(path.read_text())
            record["metadata"]["max_new_tokens"] = 24
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "different max_new_tokens"):
                aggregate(raw_dir)

    def test_inference_records_mean_of_two_generate_calls(self):
        tree = ast.parse((ROOT / "src/Inference.py").read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "benchmark_inference"
        )
        writer = Mock()
        namespace = {
            "torch": torch,
            "DataLoader": torch.utils.data.DataLoader,
            "benchmark_model_call": benchmark_model_call,
            "write_measurement": writer,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "Inference.py", "exec"), namespace)
        args = SimpleNamespace(
            efficiency_batch_size=36,
            efficiency_warmup_calls=0,
            efficiency_measured_calls=2,
            efficiency_record="member.json",
            efficiency_member_epoch=1,
            max_length=6000,
            max_new_tokens=19,
            temperature=0.1,
            input_format="prompt",
            device="cpu",
            model="policy_epoch1",
            checkpoint="distilled_epoch1",
        )
        inputs = SimpleNamespace(input_ids="inputs", attention_mask="attention")
        inputs.to = lambda device: inputs
        tokenizer = Mock(return_value=inputs, eos_token_id=2)
        model = Mock()
        dataset = [{"y": [float(index)]} for index in range(36)]
        clock = iter([0.0, 3.6, 5.0, 12.2])
        with patch("efficiency.torch.cuda.is_available", return_value=False), patch(
            "efficiency.synchronize_cuda"
        ), patch("efficiency.time.perf_counter", side_effect=lambda: next(clock)):
            namespace["benchmark_inference"](
                args, model, tokenizer, {1: dataset}, lambda batch, index: str(index)
            )
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual(len(tokenizer.call_args.args[0]), 36)
        record = writer.call_args.kwargs
        self.assertAlmostEqual(record["value"], 150.0)
        self.assertEqual(record["stage"], "inference_epoch1")
        self.assertEqual(record["metadata"]["policy_epoch"], 1)
        self.assertEqual(record["metadata"]["measured_calls"], 2)

    def test_tree_benchmark_uses_36_rows_from_complete_test_data(self):
        tree = ast.parse((ROOT / "baseline/tabular_baselines.py").read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_efficiency_records"
        )
        writer = Mock()
        benchmark = Mock(side_effect=lambda call, **kwargs: (call(), 0.5)[1])
        namespace = {
            "Path": Path,
            "detected_hardware": lambda device: "test hardware",
            "benchmark_model_call": benchmark,
            "write_measurement": writer,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "tabular_baselines.py", "exec"), namespace)
        args = SimpleNamespace(
            efficiency_record_dir="raw",
            train_sample_count=84000,
            xgb_device="cuda",
            efficiency_batch_size=36,
            efficiency_gpu_batch_size=36,
            efficiency_warmup_calls=5,
            efficiency_measured_calls=5,
        )
        model = Mock()
        model.get_params.return_value = {"n_jobs": 64, "device": "cuda"}
        namespace["write_efficiency_records"](
            "xgb", model, np.zeros((3600, 12)), 10.0, args
        )
        self.assertEqual(model.predict.call_args.args[0].shape, (36, 12))
        self.assertEqual(benchmark.call_args.kwargs["sample_count"], 36)
        self.assertEqual(writer.call_args.kwargs["metadata"]["batch_size"], 36)
        model.set_params.assert_any_call(n_jobs=64)
        model.set_params.assert_any_call(device="cuda")

    def test_workflow_dispatches_distinct_timing_files_for_both_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            for name in ("_common.sh", "run_distilled_ensemble_inference.sh"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
            for epoch in (0, 1):
                model = root / "model" / f"test_PPO_epoch{epoch}_merged"
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}")
                adapter = root / "adapters" / f"epoch{epoch}"
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text("{}")
            recorder = root / "record_commands.py"
            recorder.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print('test python')\n"
                "else:\n"
                "    with open(os.environ['COMMAND_RECORD'], 'a') as stream:\n"
                "        stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            recorder.chmod(0o755)
            record_path = root / "commands.jsonl"
            environment = {
                **os.environ,
                "PYTHON": str(recorder),
                "BASE_MODEL": "test",
                "DISTILLED_MODEL_PREFIX": "test_PPO_epoch",
                "DISTILLED_MODEL_SUFFIX": "_merged",
                "DISTILLED_ADAPTER_ROOT": str(root / "adapters"),
                "DISTILLED_EPOCHS": "0 1",
                "DISTILLED_EFFICIENCY_RECORD": str(root / "timing.json"),
                "OUTPUT_ROOT": str(root / "outputs"),
                "LOG_DIR": str(root / "logs"),
                "EFFICIENCY_RECORD_DIR": str(root / "raw"),
                "COMMAND_RECORD": str(record_path),
            }
            subprocess.run(
                ["bash", str(root / "scripts/run_distilled_ensemble_inference.sh")],
                env=environment, check=True, capture_output=True, text=True,
            )
            commands = [json.loads(line) for line in record_path.read_text().splitlines()]
            self.assertEqual(len(commands), 3)
            for epoch, command in enumerate(commands[:2]):
                self.assertEqual(command[0], "src/Inference.py")
                self.assertEqual(
                    command[command.index("--efficiency_record") + 1],
                    str(root / f"timing_epoch{epoch}.json"),
                )
                self.assertEqual(
                    command[command.index("--efficiency_member_epoch") + 1], str(epoch)
                )
                self.assertEqual(
                    command[command.index("--efficiency_batch_size") + 1], "36"
                )
            self.assertEqual(commands[2][0], "scripts/average_predictions.py")

    def test_efficiency_plot_accepts_member_and_ensemble_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._complete_raw_records(root)
            measurements = root / "measurements.csv"
            aggregate(root).to_csv(measurements, index=False)
            out_dir = root / "figures"
            subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts/figures/plot_fig7_efficiency.py"),
                    "--measurements", str(measurements), "--out-dir", str(out_dir),
                ],
                check=True, capture_output=True, text=True,
            )
            for stem in ("fig7c_training_time_log_min", "fig7d_inference_time_log_ms"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((out_dir / f"{stem}.{suffix}").stat().st_size, 0)

    def test_figure_workflow_has_no_fixed_efficiency_fallback(self):
        plot = (ROOT / "scripts/figures/plot_fig7_efficiency.py").read_text()
        figure_entry = (ROOT / "scripts/run_figures.sh").read_text()
        self.assertNotIn("manifests/efficiency_measurements.csv", plot)
        self.assertNotIn("manifests/efficiency_measurements.csv", figure_entry)
        self.assertIn("outputs/efficiency/efficiency_measurements.csv", plot)
        self.assertFalse((ROOT / "manifests/efficiency_measurements.csv").exists())


if __name__ == "__main__":
    unittest.main()
