from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from figure_data import validate_collection  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from prepare_full_period_figure_data import METHODS, validate_existing  # noqa: E402


class FigureWorkflowTest(unittest.TestCase):
    def test_full_period_contract_accepts_exact_2004_2020_keys(self):
        rows = []
        for year in range(2004, 2021):
            for month in range(1, 13):
                rows.append(
                    {
                        "method": "UHI-E3LLM",
                        "city_idx": 1,
                        "target_year": year,
                        "target_month": month,
                        "true_uhi": 0.1,
                        "predicted_uhi": 0.2,
                        "absolute_error": 0.1,
                        "squared_error": 0.01,
                    }
                )
        validate_collection(
            pd.DataFrame(rows),
            expected_cities=1,
            expected_samples_per_city=204,
            methods=["UHI-E3LLM"],
        )

    def test_main_figure_entry_excludes_deferred_panels(self):
        entry = (ROOT / "scripts" / "run_figures.sh").read_text()
        self.assertNotIn("plot_fig5bd_explanations.py", entry)
        self.assertNotIn("assemble_fig5.py", entry)
        self.assertNotIn("plot_fig6_ablations.py", entry)
        self.assertNotIn("plot_fig6_regional_3d.py", entry)
        self.assertNotIn("export_fixed_html_figures.py", entry)
        self.assertIn("fig5c_lime_importance.csv", entry)

    def test_fig4c_source_is_generated_from_raw_data(self):
        entry = (ROOT / "scripts" / "run_figures.sh").read_text()
        plot_script = (
            ROOT / "scripts" / "figures" / "plot_fig4_case_study.py"
        ).read_text()
        generated = "fig4c_factor_evolution_5cities.csv"
        self.assertIn("build_fig4c_source.py", entry)
        self.assertIn("FIG4C_RAW_DATA_DIR", entry)
        self.assertIn('${DATA_DIR}/raw', entry)
        self.assertIn(f'FIGURE_DATA_DIR}}/{generated}', entry)
        self.assertIn("outputs/figures/data", plot_script)

    def test_qgis_panel_has_no_repository_script(self):
        self.assertFalse((ROOT / "scripts" / "figures" / "plot_fig4a_qgis.py").exists())

    def test_manual_html_and_qgis_panels_are_not_released(self):
        self.assertFalse((ROOT / "scripts" / "export_fixed_html_figures.py").exists())
        self.assertFalse(
            (ROOT / "scripts" / "figures" / "export_fig3_html_pdf.mjs").exists()
        )
        self.assertFalse(
            (ROOT / "scripts" / "figures" / "plot_fig6_regional_3d.py").exists()
        )
        self.assertFalse((ROOT / "scripts" / "build_regional_figure_data.py").exists())
        self.assertFalse(
            (ROOT / "scripts" / "figures" / "plot_fig6_ablations.py").exists()
        )
        self.assertFalse(
            (ROOT / "data" / "case_study" / "Selected_5Cities.geojson").exists()
        )
        documentation = " ".join((ROOT / "README.md").read_text().split())
        self.assertIn("manually composed using HTML", documentation)
        self.assertIn("Fig. 2(d--f)", documentation)
        self.assertIn("manually prepared in QGIS", documentation)

    def test_fig2h_uses_embedded_manual_wos_counts(self):
        plot = (
            ROOT / "scripts" / "figures" / "plot_fig2h_factor_evidence.py"
        ).read_text()
        entry = (ROOT / "scripts" / "run_figures.sh").read_text()
        self.assertIn("MANUAL_WOS_COUNTS", plot)
        self.assertIn("manual factor-specific Web of", plot)
        self.assertNotIn("--evidence", entry)
        self.assertNotIn("figure_sources", entry)
        self.assertFalse((ROOT / "figure_sources").exists())

    def test_case_study_plot_has_no_error_filter_or_lime_jitter(self):
        case_plot = (ROOT / "scripts" / "figures" / "plot_fig5_case_study.py").read_text()
        lime_plot = (ROOT / "scripts" / "figures" / "plot_fig5c_lime.py").read_text()
        self.assertNotIn('rows["absolute_error"] <= 0.3', case_plot)
        self.assertNotIn("rng.uniform", lime_plot)
        self.assertNotIn("visual_y_offset", lime_plot)

    def test_full_period_data_is_prepared_inside_figure_entry(self):
        figure_entry = (ROOT / "scripts" / "run_figures.sh").read_text()
        producer = (ROOT / "scripts" / "prepare_full_period_figure_data.py").read_text()
        self.assertFalse((ROOT / "scripts" / "run_full_period_evaluation.sh").exists())
        self.assertIn("prepare_full_period_figure_data.py", figure_entry)
        self.assertIn('"DISTILLED_MODE": "test_full"', producer)
        self.assertIn("run_distilled_ensemble_inference.sh", producer)
        self.assertIn("tabular_full_period_inference.py", producer)
        for model_dir in ("baseline/armd", "baseline/filternet", "baseline/st_mtm"):
            self.assertIn(model_dir, producer)
        self.assertIn("build_full_period_figure_data.py", producer)

    def test_full_period_figure_data_contract_accepts_complete_table(self):
        rows = []
        for method in sorted(METHODS):
            for city_idx in range(1, 101):
                for year in range(2004, 2021):
                    rows.append(
                        {
                            "method": method,
                            "city_idx": city_idx,
                            "target_year": year,
                            "samples": 12,
                            "mae": 0.1,
                            "mse": 0.01,
                            "rmse": 0.1,
                        }
                    )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "method_city_year_metrics_2004_2020.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            self.assertTrue(validate_existing(source))


if __name__ == "__main__":
    unittest.main()
