import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figure_data import load_sample_errors_dir  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import build_comparison_tables as tables  # noqa: E402


class MainEvaluationWorkflowTest(unittest.TestCase):
    def test_ppo_protocol_is_consistent_across_workflows(self):
        main = (ROOT / "scripts" / "run_ppo.sh").read_text()
        self.assertIn('PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"', main)
        self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"', main)
        self.assertIn('PPO_MAX_NEW_TOKENS="${PPO_MAX_NEW_TOKENS:-24}"', main)
        self.assertIn('PPO_INPUT_FORMAT="${PPO_INPUT_FORMAT:-prompt_task_only}"', main)

        component = (ROOT / "scripts" / "run_component_ablations.sh").read_text()
        regional = (ROOT / "scripts" / "run_leave_one_region_out.sh").read_text()
        sensitivity = (ROOT / "scripts" / "run_ppo_penalty_sensitivity.sh").read_text()
        for workflow in (component, regional, sensitivity):
            self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"', workflow)
            self.assertIn("prompt_task_only", workflow)
        self.assertIn('PPO_MAX_NEW_TOKENS="${PPO_MAX_NEW_TOKENS:-24}"', sensitivity)

    def test_main_model_runs_before_every_baseline_group(self):
        script = (ROOT / "scripts" / "run_main_evaluation.sh").read_text()
        main = script.index("run_distilled_ensemble_inference.sh")
        tabular = script.index("run_global_tabular_baselines")
        timeseries = script.index("run_global_timeseries_baselines")
        table = script.index("build_comparison_tables.py")
        self.assertLess(main, tabular)
        self.assertLess(tabular, timeseries)
        self.assertLess(timeseries, table)

    def test_main_model_protocol_is_frozen(self):
        script = (ROOT / "scripts" / "run_main_evaluation.sh").read_text()
        for setting in (
            'DISTILLED_EPOCHS="0 1"',
            "DISTILLED_INPUT_FORMAT=prompt",
            "DISTILLED_EXPECTED_CITIES=100",
            "DISTILLED_EXPECTED_SAMPLES_PER_CITY=36",
            "DISTILLED_INVALID_POLICY=error",
        ):
            self.assertIn(setting, script)
        self.assertIn("MAIN_DISTILLED_ADAPTER_ROOT", script)
        self.assertIn("scripts/run_distilled_ensemble_inference.sh", script)

    def test_final_ensemble_uses_explanation_adapters(self):
        helper = (ROOT / "scripts" / "run_distilled_ensemble_inference.sh").read_text()
        self.assertIn('--adapter_path explanation', helper)
        self.assertIn('DISTILLED_ADAPTER_ROOT', helper)
        self.assertIn('DISTILLED_EPOCHS="${DISTILLED_EPOCHS:-0 1}"', helper)
        self.assertIn('scripts/average_predictions.py', helper)
        self.assertNotIn('--adapter_path PPO', helper)

    def test_validation_rejects_non_finite_predictions(self):
        samples = pd.DataFrame(
            {
                "method": ["UHI-E3LLM"],
                "city_idx": [1],
                "target_year": [2018],
                "target_month": [1],
                "true_uhi": [0.1],
                "predicted_uhi": [float("nan")],
                "absolute_error": [float("nan")],
                "squared_error": [float("nan")],
            }
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            tables.validate_collection(
                samples,
                expected_cities=1,
                expected_samples_per_city=1,
                methods=["UHI-E3LLM"],
            )

    def test_baselines_have_no_public_shell_entry(self):
        for filename in (
            "run_baselines.sh",
            "run_timeseries_baselines.sh",
            "run_timeseries_region.sh",
        ):
            self.assertFalse((ROOT / "scripts" / filename).exists())
        helper = ROOT / "scripts" / "_baseline_workflows.bash"
        self.assertTrue(helper.exists())
        self.assertNotEqual(helper.suffix, ".sh")

    def test_internal_workflow_contains_all_seven_baselines(self):
        helper = (ROOT / "scripts" / "_baseline_workflows.bash").read_text()
        for token in (
            "tabular_baselines.py",
            "xgb lgb rf mlp",
            "baseline/armd",
            "baseline/filternet",
            "baseline/st_mtm",
        ):
            self.assertIn(token, helper)

    def test_figure_entry_uses_complete_main_results_and_case_outputs(self):
        script = (ROOT / "scripts" / "run_figures.sh").read_text()
        self.assertNotIn("--allow-missing", script)
        self.assertIn('case_study/inference_train', script)
        self.assertIn('case_study/inference_valid', script)
        self.assertNotIn('--lime-dir', script)
        self.assertIn("--expected-cities 100", script)
        self.assertIn("--expected-samples-per-city 36", script)
        self.assertIn("method_city_year_metrics_2004_2020.csv", script)
        readme = (ROOT / "README.md").read_text()
        self.assertIn(
            "EFFICIENCY_RUN_MAIN_EVALUATION=0 bash scripts/run_efficiency_measurement.sh",
            readme,
        )

    def test_case_study_producer_matches_figure_paths_and_period(self):
        workflow = (ROOT / "scripts" / "run_case_study_outputs.sh").read_text()
        lime = (ROOT / "src" / "Lime_explanation.py").read_text()
        self.assertIn('case_study/inference_${output_name}', workflow)
        self.assertIn('output_name="train"', workflow)
        self.assertIn('output_name="valid"', workflow)
        self.assertIn('--save_path "${OUTPUT_ROOT}/lime_explanation"', workflow)
        self.assertIn("dataset.filter(target_in_period", lime)
        self.assertIn("len(dataset) != 36", lime)

    def test_result_producers_match_plot_consumers(self):
        figure_data = (ROOT / "scripts" / "figures" / "figure_data.py").read_text()
        main = (ROOT / "scripts" / "run_main_evaluation.sh").read_text()
        self.assertIn('main_ensemble', main)
        self.assertIn('main_ensemble', figure_data)
        for path in ("baselines/tabular", "baselines/armd", "baselines/filternet", "baselines/st_mtm"):
            self.assertIn(path, figure_data)

        fig7_uncertainty = (
            ROOT / "scripts" / "figures" / "plot_fig7_uncertainty.py"
        ).read_text()
        self.assertIn('"uncertainty" / "bootstrap"', fig7_uncertainty)
        self.assertIn('"uncertainty" / "calibration"', fig7_uncertainty)
        self.assertIn('calibration_errors.csv', fig7_uncertainty)
        self.assertIn('100 calibration cities x 36 samples', fig7_uncertainty)
        self.assertNotIn('fig7b_q90_lollipop', fig7_uncertainty)
        fig7_lambda = (ROOT / "scripts" / "figures" / "plot_fig7_lambda.py").read_text()
        self.assertIn("ppo_lambda_summary.csv", fig7_lambda)

    def test_figure_scripts_emit_only_final_revised_panels(self):
        fig3 = (ROOT / "scripts" / "figures" / "plot_fig3_performance.py").read_text()
        fig5 = (ROOT / "scripts" / "figures" / "plot_fig5_case_study.py").read_text()
        fig7 = (ROOT / "scripts" / "figures" / "plot_fig7_uncertainty.py").read_text()
        for stem in ("fig3a_all_results", "fig3b_all_results"):
            self.assertIn(stem, fig3)
        self.assertIn("fig5a_case_city_predictions", fig5)
        self.assertNotIn("fig5c_case_city_lime", fig5)
        self.assertNotIn("fig5b_", fig5)
        for stem in ("fig7a_bootstrap_uncertainty", "fig7b_calibration_error_ecdf"):
            self.assertIn(stem, fig7)
        self.assertNotIn("fig7a_bootstrap_metric_uncertainty", fig7)
        for obsolete in (
            "fig3a_all_results_with_ml_zoom",
            "fig3c_global_city_mae",
            "fig3d_us_city_mae",
            "fig3e_china_city_mae",
        ):
            self.assertNotIn(obsolete, fig3)

    def test_each_regional_main_model_precedes_regional_baselines(self):
        script = (ROOT / "scripts" / "run_leave_one_region_out.sh").read_text()
        loop_start = script.index('for region in "${REGIONS[@]}"; do')
        sft = script.index("run_sft.sh", loop_start)
        fusion = script.index("run_lora_fusion.sh", sft)
        ppo = script.index("run_ppo.sh", fusion)
        distillation = script.index("run_explanation_distillation.sh", ppo)
        ensemble = script.index("run_distilled_ensemble_inference.sh", ppo)
        baselines = script.index("run_all_regional_baselines", ensemble)
        self.assertIn('PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-4}"', script)
        self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"', script)
        self.assertIn("PPO_INPUT_FORMAT=prompt_task_only", script)
        self.assertLess(sft, fusion)
        self.assertLess(fusion, ppo)
        self.assertLess(ppo, distillation)
        self.assertLess(distillation, ensemble)
        self.assertLess(ppo, ensemble)
        self.assertLess(ensemble, baselines)
        self.assertNotIn("--reward_baseline 0.17", script)
        self.assertIn("build_comparison_tables.py regional", script)

    def test_timeseries_period_parser_accepts_compact_dates(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            pd.DataFrame(
                {
                    "city_id": [1, 1],
                    "target_year_month": [201801, 201802],
                    "true": [0.1, 0.2],
                    "pred": [0.2, 0.1],
                }
            ).to_csv(root / "city_1.csv", index=False)
            result = load_sample_errors_dir(root, "ARMD")
            self.assertEqual(result["target_year"].tolist(), [2018, 2018])
            self.assertEqual(result["target_month"].tolist(), [1, 2])

    def test_main_table_is_emitted_with_proposed_method_first(self):
        rows = []
        for method_index, method in enumerate(tables.METHOD_ORDER):
            offset = method_index / 100.0
            for city_id in range(1, 101):
                for sample_index in range(36):
                    year = 2018 + sample_index // 12
                    month = sample_index % 12 + 1
                    true = city_id / 100.0 + sample_index / 1000.0
                    pred = true + offset
                    rows.append(
                        {
                            "method": method,
                            "city_idx": city_id,
                            "target_year": year,
                            "target_month": month,
                            "true_uhi": true,
                            "predicted_uhi": pred,
                            "absolute_error": abs(offset),
                            "squared_error": offset**2,
                        }
                    )
        samples = pd.DataFrame(rows)

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / "data"
            output_root = root / "outputs"
            metrics_dir = data_dir / "city_metrics"
            metrics_dir.mkdir(parents=True)
            areas = [125, 250, 600, 1000, 2000] * 20
            pd.DataFrame({"id": range(1, 101), "Area": areas}).to_csv(
                metrics_dir / "city_metrics_test.csv", index=False
            )

            with patch.object(tables, "collect", return_value=samples):
                tables.build_main(data_dir, output_root)

            table = pd.read_csv(
                output_root / "tables" / "table1_main_performance.csv"
            )
            self.assertEqual(table.iloc[0]["Method"], "UHI-E3LLM")
            self.assertEqual(table["Method"].tolist(), tables.METHOD_ORDER)
            self.assertEqual(
                table.columns.tolist(),
                ["Method", "Average", *tables.AREA_LABELS],
            )
            self.assertRegex(
                table.iloc[0]["Average"],
                r"^\d+\.\d{4}/\d+\.\d{4}/\d+\.\d{4}$",
            )

    def test_regional_table_is_emitted_with_proposed_method_first(self):
        parts = []
        for method_index, method in enumerate(tables.METHOD_ORDER):
            for sample_index in range(36):
                parts.append(
                    {
                        "method": method,
                        "city_idx": 1,
                        "target_year": 2018 + sample_index // 12,
                        "target_month": sample_index % 12 + 1,
                        "true_uhi": sample_index / 100.0,
                        "predicted_uhi": sample_index / 100.0 + method_index / 100.0,
                        "absolute_error": method_index / 100.0,
                        "squared_error": (method_index / 100.0) ** 2,
                    }
                )
        region_samples = pd.DataFrame(parts)

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fold_dir = root / "data" / "regional_folds"
            fold_dir.mkdir(parents=True)
            for region in tables.REGIONS:
                pd.DataFrame({"id": [1]}).to_csv(
                    fold_dir / f"fold_{region}_test.csv", index=False
                )

            with patch.object(tables, "load_region", return_value=region_samples):
                tables.build_regional(root / "data", root / "outputs")

            result = pd.read_csv(
                root
                / "outputs"
                / "tables"
                / "table2_leave_one_region_out.csv"
            )
            self.assertEqual(result.iloc[0]["Method"], "UHI-E3LLM")
            self.assertEqual(result["Method"].tolist(), tables.METHOD_ORDER)
            self.assertEqual(
                result.columns.tolist(), ["Method", *tables.REGION_TABLE_ORDER]
            )

    def test_ablation_table_is_built_from_aligned_predictions(self):
        def fake_variant(_root, variant):
            offset = tables.ABLATION_ORDER.index(variant) / 100.0
            rows = []
            for city_id in range(1, 101):
                for sample_index in range(36):
                    true = city_id / 100.0 + sample_index / 1000.0
                    rows.append(
                        {
                            "method": variant,
                            "city_idx": city_id,
                            "target_year": 2018 + sample_index // 12,
                            "target_month": sample_index % 12 + 1,
                            "true_uhi": true,
                            "predicted_uhi": true + offset,
                            "absolute_error": abs(offset),
                            "squared_error": offset**2,
                        }
                    )
            return pd.DataFrame(rows)

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metrics_dir = root / "data" / "city_metrics"
            metrics_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "id": range(1, 101),
                    "Area": [125, 250, 600, 1000, 2000] * 20,
                }
            ).to_csv(metrics_dir / "city_metrics_test.csv", index=False)

            with patch.object(
                tables, "load_ablation_variant", side_effect=fake_variant
            ):
                tables.build_ablation(root / "data", root / "outputs")

            table = pd.read_csv(
                root / "outputs" / "tables" / "table3_ablation.csv"
            )
            self.assertEqual(
                table["Method"].tolist(), tables.ABLATION_ORDER
            )
            self.assertEqual(
                table.columns.tolist(),
                ["Method", "Average", *tables.AREA_LABELS],
            )

    def test_public_table_entry_exposes_all_three_tables(self):
        entry = (ROOT / "scripts" / "run_tables.sh").read_text()
        for target in ("table1", "table2", "table3", "all"):
            self.assertIn(target, entry)
        self.assertIn("build_comparison_tables.py", entry)


if __name__ == "__main__":
    unittest.main()
