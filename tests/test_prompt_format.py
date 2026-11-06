import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import (  # noqa: E402
    create_numeric_plain_prompt,
    create_prompt,
    create_result_only_prompt,
    create_task_only_prompt,
    format_numeric_plain_prompt,
    format_numeric_plain_prompt_PPO,
    format_prompt,
    format_prompt_PPO,
    format_task_only_prompt_PPO,
)


def _sample_row():
    return {
        "year_info": [2017],
        "month_info": [4],
        "x": [[
            0.325, 0.172, -0.141, 0.744, 0.381, 0.365,
            0.489, 0.496, 0.502, 0.664, 0.334, 0.763,
        ]],
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
        "y": [0.664],
    }


def _batched_sample(row):
    batched = {}
    sequence_fields = {
        "x",
        "urban_land_use_info",
        "location_info",
        "cloud_info",
        "solar_info",
        "precipitation_info",
        "snow_info",
        "evi_info",
        "ndvi_info",
        "pm25_info",
        "temperature_info",
        "vapour_info",
        "wind_info",
    }
    for field, wrapped_value in row.items():
        value = wrapped_value[0]
        if field in sequence_fields:
            batched[field] = [[torch.tensor([item]) for item in value]]
        else:
            batched[field] = [torch.tensor([value])]
    return batched


class PromptFormatTest(unittest.TestCase):
    def test_publication_prompt_sections_units_and_scaling(self):
        prompt = format_prompt(_sample_row())

        self.assertIn("Introduction\n\n", prompt)
        self.assertIn("Historical UHI Intensity", prompt)
        self.assertIn("Structural Factors", prompt)
        self.assertIn("Liquidity Factors", prompt)
        self.assertIn("Task\n\n", prompt)
        self.assertIn("Prediction\n\n", prompt)
        self.assertIn("urban–rural temperature contrast", prompt)
        self.assertIn("for the next month (2017-04)", prompt)
        self.assertIn("historical UHI data (2016-04 to 2017-03)", prompt)
        self.assertIn("[Built Surface Area:] 149272 m²/km²", prompt)
        self.assertIn("[City Compactness:] 0.141", prompt)
        self.assertIn("[Population Density:] 6594 people/km²", prompt)
        self.assertIn("[Solar Radiation:] 9537774", prompt)
        self.assertIn("[Snow Thickness:] " + ", ".join(["0.00"] * 12) + " cm", prompt)
        self.assertIn("J m⁻² d⁻¹", prompt)
        self.assertNotIn("kJ", prompt)
        self.assertIn(
            "Reflect short-term factors of UHI, consisting of average observations "
            "of the last 12 months (12 data points).",
            prompt,
        )
        self.assertIn(
            "[Area:] 751.39 km²\n[Built Height:] 6.35 m\n"
            "[Built Surface Area:] 149272 m²/km²",
            prompt,
        )
        self.assertIn(
            "[Average Cloud Cover:] " + ", ".join(["0.83"] * 12)
            + "\n[Solar Radiation:]",
            prompt,
        )
        self.assertTrue(
            prompt.endswith(
                "[Result:] Predicted UHI intensity for the next month: 0.664 °C\n"
            )
        )

    def test_sft_and_ppo_serializations_are_identical(self):
        row = _sample_row()
        ppo_example = format_prompt_PPO(row)
        self.assertEqual(
            format_prompt(row),
            ppo_example["prompt"] + ppo_example["completion"],
        )

    def test_batched_inference_uses_the_same_target_free_prompt(self):
        row = _sample_row()
        inference_prompt = create_prompt(_batched_sample(row), index=0)
        self.assertEqual(inference_prompt, format_prompt_PPO(row)["prompt"])
        self.assertTrue(inference_prompt.endswith("next month:"))
        self.assertFalse(inference_prompt.endswith(" "))
        self.assertTrue(format_prompt_PPO(row)["completion"].startswith(" "))

    def test_result_only_inference_changes_only_the_output_cue(self):
        batched = _batched_sample(_sample_row())
        standard = create_prompt(batched, index=0)
        result_only = create_result_only_prompt(batched, index=0)
        standard_suffix = (
            "\n\nPrediction\n\n[Result:] "
            "Predicted UHI intensity for the next month:"
        )

        self.assertTrue(standard.endswith(standard_suffix))
        self.assertEqual(
            result_only,
            standard[: -len(standard_suffix)] + "\n\nPrediction\n\n[Result:]",
        )
        self.assertNotIn("Predicted UHI intensity", result_only)

    def test_task_only_inference_removes_the_prediction_section(self):
        batched = _batched_sample(_sample_row())
        standard = create_prompt(batched, index=0)
        task_only = create_task_only_prompt(batched, index=0)
        standard_suffix = (
            "\n\nPrediction\n\n[Result:] "
            "Predicted UHI intensity for the next month:"
        )

        self.assertEqual(task_only, standard[: -len(standard_suffix)])
        self.assertTrue(task_only.endswith("predict the UHI intensity for the next month."))
        self.assertNotIn("\n\nPrediction", task_only)
        self.assertNotIn("[Result:]", task_only)

    def test_task_only_ppo_matches_task_only_inference(self):
        row = _sample_row()
        ppo_example = format_task_only_prompt_PPO(row)
        inference_prompt = create_task_only_prompt(_batched_sample(row), index=0)

        self.assertEqual(ppo_example["prompt"], inference_prompt)
        self.assertTrue(
            ppo_example["prompt"].endswith(
                "predict the UHI intensity for the next month."
            )
        )
        self.assertNotIn("\n\nPrediction", ppo_example["prompt"])
        self.assertNotIn("[Result:]", ppo_example["prompt"])
        self.assertEqual(ppo_example["completion"], " 0.664 °C\n")

    def test_numeric_ablation_is_an_unlabeled_147_value_list(self):
        row = _sample_row()
        serialized = format_numeric_plain_prompt_PPO(row)
        prompt = serialized["prompt"]
        pieces = prompt.strip()[1:-1].split(", ")

        self.assertTrue(prompt.startswith("["))
        self.assertTrue(prompt.endswith("]\n"))
        self.assertEqual(len(pieces), 147)
        self.assertEqual(pieces[:2], ["2017", "4"])
        self.assertEqual(pieces[2:5], ["0.325", "0.172", "-0.141"])
        self.assertEqual(
            pieces[14:27],
            [
                "751.39", "6.35", "149272", "26.61", "120.626",
                "27.784", "0.141", "6594", "27.73", "0.74",
                "2.63", "26.64", "1.15",
            ],
        )
        self.assertEqual(pieces[27], "0.83")
        self.assertEqual(pieces[39], "9537774")
        self.assertEqual(pieces[-1], "1.70")
        self.assertEqual(serialized["completion"], "0.664\n")
        self.assertNotIn("UHI", prompt)
        self.assertNotIn(":", prompt)

    def test_numeric_sft_ppo_and_batched_inference_are_identical(self):
        row = _sample_row()
        ppo_example = format_numeric_plain_prompt_PPO(row)
        self.assertEqual(
            format_numeric_plain_prompt(row),
            ppo_example["prompt"] + ppo_example["completion"],
        )
        self.assertEqual(
            create_numeric_plain_prompt(_batched_sample(row), index=0),
            ppo_example["prompt"],
        )


if __name__ == "__main__":
    unittest.main()
