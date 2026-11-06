import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from transformers import BatchEncoding


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Lime_explanation import TabularPredictor
from output_parsing import parse_prediction


class TokenizerStub:
    eos_token_id = 0

    def __init__(self, texts):
        self.texts = texts

    def __call__(self, prompts, **kwargs):
        input_ids = torch.ones((len(prompts), 3), dtype=torch.long)
        return BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        })

    def decode(self, tokens, **kwargs):
        return self.texts[int(tokens[0])]


class ModelStub:
    def __init__(self, tokens):
        self.tokens = tokens
        self.offset = 0
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append((kwargs, torch.is_grad_enabled()))
        count = len(kwargs["input_ids"])
        tokens = torch.tensor(self.tokens[self.offset:self.offset + count])[:, None]
        self.offset += count
        return torch.cat([kwargs["input_ids"], tokens], dim=1)


class LimeEnsembleTest(unittest.TestCase):
    def make_predictor(self, first, second):
        texts = dict(enumerate(first + second))
        models = [
            ModelStub(list(range(len(first)))),
            ModelStub(list(range(len(first), len(first) + len(second)))),
        ]
        args = SimpleNamespace(
            device="cpu", batch_size=2, max_length=1300, max_new_tokens=19,
            valid_min=-5.0, valid_max=5.0, max_decimal_places=3,
        )
        predictor = TabularPredictor(args, models, TokenizerStub(texts))
        predictor._tabular_to_prompt = lambda value: str(value)
        return predictor, models

    def test_mean_matches_parsed_member_predictions_in_every_batch(self):
        first = ["0.1236", "-0.251", "2.000"]
        second = ["0.125", "-0.250", "-2.000"]
        predictor, models = self.make_predictor(first, second)
        actual = predictor.predict(np.arange(3))
        expected = np.mean([
            [parse_prediction(text).parsed_value for text in first],
            [parse_prediction(text).parsed_value for text in second],
        ], axis=0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertAlmostEqual(actual[0], 0.1245)
        self.assertAlmostEqual(actual[1], -0.2505)
        for model in models:
            self.assertEqual(model.offset, 3)
            self.assertEqual(len(model.calls), 2)
            for arguments, gradients_enabled in model.calls:
                self.assertIs(arguments["do_sample"], False)
                self.assertEqual(arguments["max_new_tokens"], 19)
                self.assertFalse(gradients_enabled)

    def test_invalid_member_is_not_silently_dropped(self):
        for bad_output in ("No result", "6.0", "-5.001"):
            with self.subTest(output=bad_output):
                predictor, _ = self.make_predictor(["0.100"], [bad_output])
                with self.assertRaisesRegex(ValueError, "ensemble member 2"):
                    predictor.predict(np.arange(1))

    def test_valid_range_boundaries_are_averaged(self):
        predictor, _ = self.make_predictor(["-5.000"], ["5.000"])
        np.testing.assert_array_equal(predictor.predict(np.arange(1)), [0.0])

    def test_empty_perturbation_batch_does_not_generate(self):
        predictor, models = self.make_predictor([], [])
        self.assertEqual(predictor.predict(np.array([])).shape, (0,))
        self.assertTrue(all(not model.calls for model in models))

    def test_two_members_are_required(self):
        with self.assertRaisesRegex(ValueError, "both distilled ensemble members"):
            TabularPredictor(SimpleNamespace(), [ModelStub([])], TokenizerStub({}))

    def test_case_workflow_passes_both_matching_bases_and_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            for filename in ("_common.sh", "run_case_study_outputs.sh"):
                shutil.copy2(ROOT / "scripts" / filename, scripts / filename)
            (scripts / "run_distilled_ensemble_inference.sh").write_text(
                '#!/usr/bin/env bash\n'
                '"${PYTHON}" distilled_call "${DISTILLED_MODEL_PREFIX}" '
                '"${DISTILLED_MODEL_SUFFIX}" "${DISTILLED_MAX_NEW_TOKENS}"\n'
            )
            runner = root / "python_stub"
            runner.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1:] == ['--version']:\n"
                "    print(sys.version)\n"
                "    sys.exit(0)\n"
                "with open(os.environ['LIME_TEST_COMMANDS'], 'a') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            runner.chmod(0o755)
            commands = root / "commands.jsonl"
            environment = dict(os.environ)
            environment.update({
                "PYTHON": str(runner), "MODEL": "test_model",
                "DATA_DIR": str(root / "data"),
                "OUTPUT_ROOT": str(root / "outputs"),
                "LOG_DIR": str(root / "logs"),
                "EFFICIENCY_RECORD_DIR": str(root / "timing"),
                "DISTILLED_ADAPTER_ROOT": str(root / "adapters"),
                "DISTILLED_MODEL_PREFIX": "test_policy_",
                "DISTILLED_MODEL_SUFFIX": "_base",
                "INFERENCE_MAX_NEW_TOKENS": "29", "RUN_LIME": "1",
                "LIME_TEST_COMMANDS": str(commands),
            })
            result = subprocess.run(
                ["bash", str(scripts / "run_case_study_outputs.sh")],
                cwd=root, env=environment, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in commands.read_text().splitlines()]
            self.assertEqual(calls[:2], [
                ["distilled_call", "test_policy_", "_base", "29"],
                ["distilled_call", "test_policy_", "_base", "29"],
            ])
            lime = next(call for call in calls if call[0] == "src/Lime_explanation.py")
            index = lime.index("--models")
            self.assertEqual(lime[index + 1:index + 3], ["test_policy_0_base", "test_policy_1_base"])
            index = lime.index("--adapter_dirs")
            self.assertEqual(lime[index + 1:index + 3], [
                str(root / "adapters/epoch0"), str(root / "adapters/epoch1"),
            ])
            self.assertEqual(lime[lime.index("--max_new_tokens") + 1], "29")


if __name__ == "__main__":
    unittest.main()
