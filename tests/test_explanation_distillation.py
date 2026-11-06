import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import explanation_distillation as distillation
import generate_explanations as generation
from explanation_distillation import (  # noqa: E402
    ExplanationDataset,
    find_supervised_spans,
    selected_explanation_files,
    weighted_causal_lm_loss,
)
from generate_explanations import (  # noqa: E402
    SECTION_HEADINGS,
    _generate_job,
    build_distillation_text,
    build_prompts,
    load_training_city_ids,
    parse_teacher_content,
    parse_args as parse_generation_args,
    select_candidates,
    validate_city_samples,
)


def _sample_row():
    return {
        "year_info": [2017],
        "month_info": [12],
        "x": [[0.1] * 12],
        "area_info": [751.39],
        "built_height_info": [6.35],
        "built_surface_info": [1492.72],
        "urbanization_level_info": [26.61],
        "compactness_info": [0.14107],
        "urban_land_use_info": [[27.73, 0.74, 2.63, 26.64, 1.15]],
        "location_info": [[120.626, 27.784]],
        "pop_info": [65.94],
        "cloud_info": [[0.83] * 12],
        "solar_info": [[9537774.0] * 12],
        "precipitation_info": [[6.77] * 12],
        "snow_info": [[0.0] * 12],
        "evi_info": [[0.19] * 12],
        "ndvi_info": [[0.35] * 12],
        "pm25_info": [[35.73] * 12],
        "temperature_info": [[290.11] * 12],
        "vapour_info": [[17.26] * 12],
        "wind_info": [[1.70] * 12],
        "y": [0.6644],
    }


def _month_samples():
    samples = []
    year, month = 2004, 1
    while (year, month) <= (2017, 12):
        samples.append({"year_info": [year], "month_info": [month]})
        month += 1
        if month == 13:
            year += 1
            month = 1
    return samples


class ExplanationAPIConfigurationTest(unittest.TestCase):
    def test_chat_completions_generation_uses_model_specific_parameters(self):
        explanation = "\n\n".join(f"{heading} Fixture text." for heading in SECTION_HEADINGS)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=(
                    "[Result:] Predicted UHI intensity for the next month: 0.664 °C\n\n"
                    + explanation
                )
            ))]
        )
        for api_model, expected_parameters in (
            ("gpt-6.1-sol", {"reasoning_effort": "medium"}),
            ("deepseek-reasoner", {
                "temperature": 1.1, "top_p": 0.9, "presence_penalty": 0.1,
            }),
        ):
            with self.subTest(api_model=api_model), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                job = {
                    "city_id": 222,
                    "sample_index": 1,
                    "target_month": "2017-12",
                    "sample": _sample_row(),
                    "content_path": root / "content" / "sample_1.txt",
                    "reasoning_path": root / "reasoning" / "sample_1.txt",
                }
                args = SimpleNamespace(
                    api_key="fixture-key", base_url="https://api.openai.com/v1",
                    timeout=10, api_model=api_model, student_input_format="prompt",
                    reuse_content_dir=None, max_attempts=1, retry_delay=0,
                    temperature=1.1, top_p=0.9, presence_penalty=0.1,
                )
                with patch.object(generation, "_client", return_value=client) as factory:
                    result = _generate_job(job, args)
                factory.assert_called_once_with(
                    "fixture-key", "https://api.openai.com/v1", 10
                )
                self.assertEqual(result["status"], "generated")
                self.assertIn(explanation, job["content_path"].read_text(encoding="utf-8"))
                _, _, teacher_prompt, _ = build_prompts(job["sample"])
                self.assertEqual(client.chat.completions.create.call_args.kwargs, {
                    "model": api_model,
                    "messages": [{"role": "user", "content": teacher_prompt}],
                    **expected_parameters,
                })

    def test_reads_openai_environment_variables(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "fixture-key",
            "OPENAI_BASE_URL": "https://fixture.example/v1",
        }, clear=True), patch.object(sys, "argv", ["generate_explanations.py"]):
            args = parse_generation_args()
        self.assertEqual(args.api_key, "fixture-key")
        self.assertEqual(args.base_url, "https://fixture.example/v1")
        self.assertEqual(args.api_model, "deepseek-reasoner")

    def test_explicit_arguments_override_environment(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "environment-key",
            "OPENAI_BASE_URL": "https://environment.example/v1",
        }, clear=True), patch.object(sys, "argv", [
            "generate_explanations.py",
            "--api-key", "argument-key",
            "--base-url", "https://argument.example/v1",
            "--api-model", "fixture-model",
        ]):
            args = parse_generation_args()
        self.assertEqual(args.api_key, "argument-key")
        self.assertEqual(args.base_url, "https://argument.example/v1")
        self.assertEqual(args.api_model, "fixture-model")

    def test_preserves_default_endpoint_and_teacher_model(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys, "argv", ["generate_explanations.py"]
        ):
            args = parse_generation_args()
        self.assertEqual(args.api_key, "")
        self.assertEqual(args.base_url, "https://api.deepseek.com")
        self.assertEqual(args.api_model, "deepseek-reasoner")


class DistillationTimingTest(unittest.TestCase):
    def test_records_optimizer_execution_without_loading_or_checkpoint_writes(self):
        clock = {"seconds": 0.0}

        def advance(seconds):
            clock["seconds"] += seconds

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 3, 4))
                self.print_trainable_parameters = Mock()

            def forward(self, input_ids, attention_mask):
                advance(2.0)
                return SimpleNamespace(logits=self.logits)

            def save_pretrained(self, path):
                advance(100.0)

        model = TinyModel()
        optimizer = Mock()
        optimizer.zero_grad.side_effect = model.zero_grad
        optimizer.step.side_effect = lambda: advance(3.0)
        tokenizer = SimpleNamespace(
            pad_token="pad", eos_token="eos",
            save_pretrained=Mock(side_effect=lambda path: advance(50.0)),
        )
        sample = {
            "input_ids": torch.tensor([0, 1, 2]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "labels": torch.tensor([0, 1, 2]),
            "loss_weights": torch.ones(3),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record_path = root / "raw" / "training_uhi_distillation_epoch1.json"
            argv = [
                "explanation_distillation.py",
                "--model_path", str(root / "model"),
                "--data_path", str(root / "samples"),
                "--output_dir", str(root / "adapter"),
                "--device", "cpu", "--num_workers", "0",
                "--num_epochs", "2", "--expected_samples", "1",
                "--save_at_steps", "1",
                "--efficiency_record", str(record_path),
                "--efficiency_member_epoch", "1",
            ]
            with patch.object(sys, "argv", argv):
                args = distillation.parse_args()
            with patch.object(distillation, "parse_args", return_value=args), patch.object(
                distillation, "_set_seed"
            ), patch.object(
                distillation.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ), patch.object(
                distillation, "ExplanationDataset", return_value=[sample]
            ), patch.object(
                distillation.AutoModelForCausalLM, "from_pretrained", return_value=model
            ), patch.object(
                distillation, "get_peft_model", side_effect=lambda base, config: base
            ), patch.object(
                distillation.torch.optim, "AdamW", return_value=optimizer
            ), patch.object(
                distillation, "synchronize_cuda"
            ) as synchronize, patch.object(
                distillation.time, "perf_counter", side_effect=lambda: clock["seconds"]
            ), patch("efficiency.torch.cuda.is_available", return_value=False):
                distillation.main()
            record = json.loads(record_path.read_text())
            self.assertEqual(record["stage"], "Distillation_epoch1")
            self.assertEqual(record["value"], 10.0)
            self.assertEqual(record["unit"], "seconds")
            self.assertEqual(record["metadata"]["policy_epoch"], 1)
            self.assertEqual(record["metadata"]["global_steps"], 2)
            self.assertEqual(synchronize.call_count, 4)
            self.assertEqual(clock["seconds"], 310.0)
            config = json.loads((root / "adapter" / "distillation_config.json").read_text())
            self.assertEqual(config["training_seconds"], 10.0)


class ExplanationGenerationTest(unittest.TestCase):
    def test_training_city_ids_are_loaded_uniquely_and_sorted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            city_metrics = Path(temporary_dir) / "city_metrics_train.csv"
            city_metrics.write_text("id\n9\n2\n5\n", encoding="utf-8")
            self.assertEqual(load_training_city_ids(city_metrics, 3), [2, 5, 9])

    def test_training_pool_sampling_is_reproducible_without_replacement(self):
        candidates_a = [
            {"city_id": city_id, "sample_index": sample_index}
            for city_id in range(5)
            for sample_index in range(10)
        ]
        candidates_b = [dict(candidate) for candidate in candidates_a]
        selected_a = select_candidates(candidates_a, sample_count=20, sampling_seed=42)
        selected_b = select_candidates(candidates_b, sample_count=20, sampling_seed=42)
        keys_a = [(item["city_id"], item["sample_index"]) for item in selected_a]
        keys_b = [(item["city_id"], item["sample_index"]) for item in selected_b]
        self.assertEqual(keys_a, keys_b)
        self.assertEqual(len(keys_a), len(set(keys_a)))
        self.assertEqual(
            keys_a,
            random.Random(42).sample(
                [(city_id, sample_index) for city_id in range(5) for sample_index in range(10)],
                20,
            ),
        )

    def test_complete_corrected_training_range_has_168_samples(self):
        ordered = validate_city_samples(
            list(reversed(_month_samples())),
            city_id=222,
            expected_count=168,
            first_target=(2004, 1),
            last_target=(2017, 12),
        )
        self.assertEqual(len(ordered), 168)
        self.assertEqual(ordered[0]["month_info"], [1])
        self.assertEqual(ordered[-1]["month_info"], [12])

    def test_api_prompt_contains_supplied_result_but_student_prompt_does_not(self):
        student_prompt, completion, teacher_prompt, target = build_prompts(_sample_row())
        self.assertTrue(student_prompt.endswith("next month:"))
        self.assertFalse(student_prompt.endswith(" "))
        self.assertEqual(completion, " 0.664 °C\n")
        self.assertIn("next month: 0.664 °C\n", teacher_prompt)
        self.assertIn(
            "Considering the important features related to UHI prediction",
            teacher_prompt,
        )
        self.assertIn("Identify primary Structure Factors", teacher_prompt)
        self.assertIn("Planning-Oriented Recommendations:", teacher_prompt)
        self.assertNotIn("Do not quote the input data directly", teacher_prompt)
        self.assertNotIn("Avoid bullet points and markdown", teacher_prompt)
        self.assertAlmostEqual(target, 0.6644)

    def test_numeric_student_keeps_structured_teacher_context(self):
        student_prompt, completion, teacher_prompt, target = build_prompts(
            _sample_row(), student_input_format="numeric_plain"
        )
        self.assertTrue(student_prompt.startswith("[2017, 12, "))
        self.assertNotIn("Structural Factors", student_prompt)
        self.assertIn("Structural Factors", teacher_prompt)
        self.assertIn("next month: 0.664 °C\n", teacher_prompt)
        self.assertEqual(completion, "0.664\n")
        self.assertAlmostEqual(target, 0.6644)

    def test_teacher_result_header_is_removed_from_training_completion(self):
        explanation = "\n\n".join(f"{heading} Text." for heading in SECTION_HEADINGS)
        response = (
            "[Result:] Predicted UHI intensity for the next month: 0.664 °C\n\n"
            + explanation
        )
        parsed = parse_teacher_content(response, expected_target=0.6644)
        student_prompt, completion, _, _ = build_prompts(_sample_row())
        training_text = build_distillation_text(student_prompt, completion, parsed)
        self.assertIn("next month: 0.664 °C\n\nGeographical Context:", training_text)
        self.assertEqual(training_text.count("Predicted UHI intensity"), 1)

    def test_missing_explanation_section_is_rejected(self):
        response = (
            "[Result:] Predicted UHI intensity for the next month: 0.664 °C\n\n"
            "Geographical Context: Text."
        )
        with self.assertRaises(ValueError):
            parse_teacher_content(response, expected_target=0.6644)

    def test_numeric_student_reuses_identical_teacher_explanation(self):
        explanation = "\n\n".join(
            f"{heading} Fixed teacher text." for heading in SECTION_HEADINGS
        )
        structured_prompt, completion, _, _ = build_prompts(_sample_row())
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source" / "City413" / "sample_1.txt"
            source.parent.mkdir(parents=True)
            source.write_text(
                build_distillation_text(structured_prompt, completion, explanation),
                encoding="utf-8",
            )
            job = {
                "city_id": 413,
                "sample_index": 1,
                "target_month": "2017-12",
                "sample": _sample_row(),
                "content_path": root / "numeric" / "City413" / "sample_1.txt",
                "reasoning_path": root / "reasoning" / "City413" / "sample_1.txt",
            }
            result = _generate_job(
                job,
                SimpleNamespace(
                    student_input_format="numeric_plain",
                    reuse_content_dir=root / "source",
                ),
            )
            numeric_text = job["content_path"].read_text(encoding="utf-8")
            self.assertEqual(result["status"], "reserialized")
            self.assertTrue(numeric_text.startswith("[2017, 12, "))
            self.assertIn(explanation, numeric_text)
            self.assertNotIn("Structural Factors", numeric_text)

    def test_numeric_student_reuses_flat_release_sample_by_selection_rank(self):
        explanation = "\n\n".join(
            f"{heading} Released teacher text." for heading in SECTION_HEADINGS
        )
        structured_prompt, completion, _, _ = build_prompts(_sample_row())
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "released" / "sample_7.txt"
            source.parent.mkdir(parents=True)
            source.write_text(
                build_distillation_text(structured_prompt, completion, explanation),
                encoding="utf-8",
            )
            job = {
                "selection_rank": 7,
                "city_id": 413,
                "sample_index": 1,
                "target_month": "2017-12",
                "sample": _sample_row(),
                "content_path": root / "numeric" / "City413" / "sample_1.txt",
                "reasoning_path": root / "reasoning" / "City413" / "sample_1.txt",
            }
            result = _generate_job(
                job,
                SimpleNamespace(
                    student_input_format="numeric_plain",
                    reuse_content_dir=source.parent,
                ),
            )
            numeric_text = job["content_path"].read_text(encoding="utf-8")
            self.assertEqual(result["status"], "reserialized")
            self.assertTrue(numeric_text.startswith("[2017, 12, "))
            self.assertIn(explanation, numeric_text)


class ExplanationLossTest(unittest.TestCase):
    def test_full_text_scope_supervises_all_non_padding_tokens(self):
        class CharacterTokenizer:
            eos_token_id = 31

            def __call__(self, text, max_length, **_kwargs):
                length = min(len(text), max_length)
                input_ids = torch.arange(max_length).unsqueeze(0)
                attention_mask = torch.zeros((1, max_length), dtype=torch.long)
                attention_mask[:, :length] = 1
                offsets = [(index, index + 1) for index in range(length)]
                offsets.extend([(0, 0)] * (max_length - length))
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "offset_mapping": torch.tensor([offsets]),
                }

        text = "Prompt result explanation"
        encoded, labels, weights, _, counts = ExplanationDataset._encode(
            CharacterTokenizer(),
            text,
            max_length=32,
            result_span=(7, 13),
            explanation_start=14,
            result_loss_weight=1.0,
            explanation_loss_weight=0.001,
            supervision_scope="full_text",
        )

        length = len(text)
        self.assertTrue(torch.all(labels[0, :length] != -100))
        self.assertTrue(torch.all(weights[0, :length] == 1.0))
        self.assertTrue(torch.all(labels[0, length:] == -100))
        self.assertEqual(counts["supervised"], length)
        self.assertEqual(int(encoded["attention_mask"].sum()), length)

    def test_full_text_scope_can_supervise_one_appended_eos(self):
        class CharacterTokenizer:
            eos_token_id = 31

            def __call__(self, text, max_length, **_kwargs):
                length = min(len(text), max_length)
                input_ids = torch.zeros((1, max_length), dtype=torch.long)
                attention_mask = torch.zeros((1, max_length), dtype=torch.long)
                input_ids[:, :length] = torch.arange(length)
                attention_mask[:, :length] = 1
                offsets = [(index, index + 1) for index in range(length)]
                offsets.extend([(0, 0)] * (max_length - length))
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "offset_mapping": torch.tensor([offsets]),
                }

        text = "Prompt result explanation"
        encoded, labels, weights, _, counts = ExplanationDataset._encode(
            CharacterTokenizer(),
            text,
            max_length=32,
            result_span=(7, 13),
            explanation_start=14,
            result_loss_weight=1.0,
            explanation_loss_weight=0.001,
            supervision_scope="full_text",
            append_eos_token=True,
        )

        eos_index = len(text)
        self.assertEqual(encoded["input_ids"][0, eos_index].item(), 31)
        self.assertEqual(encoded["attention_mask"][0, eos_index].item(), 1)
        self.assertEqual(labels[0, eos_index].item(), 31)
        self.assertEqual(weights[0, eos_index].item(), 1.0)
        self.assertEqual(counts["supervised"], len(text) + 1)

    def test_flat_release_samples_are_accepted_without_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            expected = []
            for index in range(1, 4):
                path = root / f"sample_{index}.txt"
                path.write_text(f"sample {index}", encoding="utf-8")
                expected.append(path)
            self.assertEqual(selected_explanation_files(root, None, 3), expected)

    def test_flat_release_samples_must_be_contiguous(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "sample_1.txt").write_text("first", encoding="utf-8")
            (root / "sample_3.txt").write_text("third", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "consecutively numbered"):
                selected_explanation_files(root, None, 3)

    def test_distillation_files_must_match_training_selection_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            content = root / "content"
            first = content / "City2" / "sample_3.txt"
            second = content / "City9" / "sample_7.txt"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            manifest = {
                "source_split": "train",
                "samples": [
                    {
                        "selection_rank": 1,
                        "content_file": "content/City2/sample_3.txt",
                    },
                    {
                        "selection_rank": 2,
                        "content_file": "content/City9/sample_7.txt",
                    },
                ],
            }
            (root / "selection_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(
                selected_explanation_files(content, None, 2),
                [first, second],
            )

    def test_supervised_spans_cover_only_number_and_explanation(self):
        text = (
            "Prompt\n[Result:] Predicted UHI intensity for the next month: 0.664 °C\n\n"
            "Geographical Context: Explanation."
        )
        result_span, explanation_start = find_supervised_spans(text)
        self.assertEqual(text[result_span[0] : result_span[1]], "0.664")
        self.assertEqual(text[explanation_start:].split()[0], "Geographical")

    def test_numeric_list_supervised_spans_cover_result_and_explanation(self):
        text = "[2017, 12, 0.100]\n0.664\n\nGeographical Context: Explanation."
        result_span, explanation_start = find_supervised_spans(text)
        self.assertEqual(text[result_span[0] : result_span[1]], "0.664")
        self.assertEqual(text[explanation_start:].split()[0], "Geographical")

    def test_weighted_loss_matches_manual_weighted_average(self):
        logits = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]], dtype=torch.float32
        )
        labels = torch.tensor([[-100, 1, 0]])
        weights = torch.tensor([[0.0, 1.0, 0.25]])
        loss = weighted_causal_lm_loss(logits, labels, weights)
        first = torch.nn.functional.cross_entropy(logits[:, 0], labels[:, 1])
        second = torch.nn.functional.cross_entropy(logits[:, 1], labels[:, 2])
        expected = (first + 0.25 * second) / 1.25
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
