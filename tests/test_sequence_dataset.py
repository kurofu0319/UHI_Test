import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baseline"))

from sequence_dataset import (
    DYNAMIC_FIELDS,
    FEATURE_INDEX,
    FEATURE_NAMES,
    read_city_samples,
    read_city_series,
    save_city_series,
    samples_to_series,
)
from utils import format_prompt
from common import load_data


def _legacy_samples(city_id=9):
    rows = []
    for target in range(12, 216):
        year, month = divmod(target, 12)
        row = {
            "year_info": [2003 + year],
            "month_info": [month + 1],
            "location_info": [[120.125, 27.875]],
            "area_info": [751.39],
            "pop_info": [65.94],
            "built_surface_info": [1492.72],
            "built_height_info": [6.35],
            "urbanization_level_info": [26.61],
            "urban_land_use_info": [[27.73, 0.74, 2.63, 26.64, 1.15]],
            "compactness_info": [0.14107],
            "city_idx": [city_id],
        }
        history_indices = range(target - 12, target)
        for field_index, field in enumerate(DYNAMIC_FIELDS):
            row[field] = [[
                field_index + index / 1000 for index in history_indices
            ]]
        row["x"] = [[index / 1000 for index in history_indices]]
        row["y"] = [target / 1000]
        rows.append(row)
    return rows


class SequenceDatasetTest(unittest.TestCase):
    def test_tensor_round_trip_preserves_sample_prompts_and_split_windows(self):
        city_id = 9
        old_rows = _legacy_samples(city_id)
        year_month, values = samples_to_series(old_rows, city_id)
        self.assertEqual(values.shape, (216, 24))
        self.assertEqual(tuple(FEATURE_NAMES).index("Amod2"), 23)
        self.assertTrue(np.isnan(values[-1, 13:23]).all())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_city_series(root / "train" / f"UHI_city_{city_id}", year_month[:180], values[:180])
            save_city_series(root / "valid" / f"UHI_city_{city_id}", year_month[180:], values[180:])

            stored = load_from_disk(str(root / "train" / f"UHI_city_{city_id}"))
            self.assertEqual(len(stored), 1)
            self.assertEqual(set(stored.column_names), {"year_month", "values"})
            self.assertNotIn("x", stored.column_names)
            self.assertNotIn("y", stored.column_names)
            self.assertEqual(
                read_city_series(root / "train" / f"UHI_city_{city_id}")[1].shape,
                (180, 24),
            )

            reconstructed = list(read_city_samples(root, city_id, "all"))
            self.assertEqual(len(reconstructed), 204)
            for old, new in zip(old_rows, reconstructed):
                self.assertEqual(format_prompt(old), format_prompt(new))
                for source in DYNAMIC_FIELDS:
                    np.testing.assert_allclose(old[source], new[source], rtol=0, atol=1e-12)
                np.testing.assert_allclose(old["x"], new["x"], rtol=0, atol=1e-8)
                np.testing.assert_allclose(old["y"], new["y"], rtol=0, atol=1e-12)

            train_rows = read_city_samples(root, city_id, "train")
            valid_rows = read_city_samples(root, city_id, "valid")
            self.assertEqual((len(train_rows), len(valid_rows)), (168, 36))
            self.assertEqual(train_rows[-1]["year_info"], [2017])
            self.assertEqual(train_rows[-1]["month_info"], [12])
            self.assertEqual(valid_rows[0]["year_info"], [2018])
            self.assertEqual(valid_rows[0]["month_info"], [1])
            np.testing.assert_allclose(
                train_rows[-1]["y"], valid_rows[0]["x"][0][-1:], rtol=0, atol=1e-8
            )

    def test_tabular_baseline_loader_reads_tensor_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "city_metrics").mkdir()
            pd.DataFrame({"id": [9]}).to_csv(
                root / "city_metrics" / "city_metrics_train.csv", index=False
            )
            pd.DataFrame({"id": [10]}).to_csv(
                root / "city_metrics" / "city_metrics_test.csv", index=False
            )

            for city_id in (9, 10):
                year_month, values = samples_to_series(
                    _legacy_samples(city_id), city_id
                )
                save_city_series(
                    root / "train" / f"UHI_city_{city_id}",
                    year_month[:180], values[:180],
                )
                save_city_series(
                    root / "valid" / f"UHI_city_{city_id}",
                    year_month[180:], values[180:],
                )

            train_x, train_y, train_cities = load_data(root, mode="train")
            test_x, test_y, test_cities = load_data(root, mode="test")
            full_x, full_y, full_cities = load_data(root, mode="test_full")

            self.assertEqual(train_x.shape, (168, 147))
            self.assertEqual(test_x.shape, (36, 147))
            self.assertEqual(full_x.shape, (204, 147))
            self.assertEqual((len(train_y), len(test_y), len(full_y)), (168, 36, 204))
            self.assertEqual(set(train_cities), {9})
            self.assertEqual(set(test_cities), {10})
            self.assertEqual(set(full_cities), {10})
            np.testing.assert_array_equal(train_x[[0, -1], -2:], [[2004, 1], [2017, 12]])
            np.testing.assert_array_equal(test_x[[0, -1], -2:], [[2018, 1], [2020, 12]])


if __name__ == "__main__":
    unittest.main()
