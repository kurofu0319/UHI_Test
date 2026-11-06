import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sample_modeling_cities import (  # noqa: E402
    AREA_LABELS,
    assign_strata,
    calibration_manifest,
    largest_remainder_quotas,
    stratified_sample,
)


def synthetic_candidates() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    sizes = {"Very small": 40, "Small": 60, "Medium": 20, "Large": 12, "Very large": 8}
    ranges = {
        "Very small": (100, 150),
        "Small": (150, 450),
        "Medium": (450, 800),
        "Large": (800, 1500),
        "Very large": (1500, 20000),
    }
    rows = []
    city_id = 0
    for label, count in sizes.items():
        low, high = ranges[label]
        for _ in range(count):
            rows.append({"id": city_id, "Area": float(rng.uniform(low, high - 1))})
            city_id += 1
    return pd.DataFrame(rows)


class CitySamplingTest(unittest.TestCase):
    def test_frozen_manifests_match_original_experiment_counts(self):
        metrics_dir = ROOT / "data" / "city_metrics"
        expected = {
            "modeling_split_600.csv": [192, 277, 54, 43, 34],
            "city_metrics_train.csv": [159, 229, 44, 39, 29],
            "city_metrics_test.csv": [33, 48, 10, 4, 5],
        }
        for filename, counts in expected.items():
            frame = pd.read_csv(metrics_dir / filename)
            observed = assign_strata(frame).value_counts().reindex(AREA_LABELS)
            self.assertEqual(observed.tolist(), counts)

    def test_largest_remainder_quotas_sum_and_feasibility(self):
        sizes = pd.Series([336, 462, 105, 61, 56], index=AREA_LABELS)
        quotas = largest_remainder_quotas(sizes, 600)
        self.assertEqual(int(quotas.sum()), 600)
        self.assertTrue((quotas <= sizes).all())
        self.assertEqual(quotas.tolist(), [197, 272, 62, 36, 33])
        train = largest_remainder_quotas(quotas, 500)
        self.assertEqual(int(train.sum()), 500)
        self.assertEqual(train.tolist(), [164, 227, 52, 30, 27])

    def test_stratified_sample_is_deterministic_and_disjoint(self):
        frame = synthetic_candidates()
        frame["stratum"] = assign_strata(frame)
        sizes = frame["stratum"].value_counts().reindex(AREA_LABELS)
        quotas = largest_remainder_quotas(sizes, 70)
        first = stratified_sample(frame, quotas, np.random.default_rng(123))
        second = stratified_sample(frame, quotas, np.random.default_rng(123))
        self.assertEqual(first["id"].tolist(), second["id"].tolist())
        self.assertEqual(len(first), 70)
        self.assertFalse(first["id"].duplicated().any())
        per_stratum = first["stratum"].value_counts().reindex(AREA_LABELS)
        self.assertEqual(per_stratum.tolist(), quotas.tolist())

        train_quotas = largest_remainder_quotas(quotas, 50)
        train = stratified_sample(first, train_quotas, np.random.default_rng(123))
        test = first[~first["id"].isin(train["id"])]
        self.assertEqual(len(train), 50)
        self.assertEqual(len(test), 20)
        self.assertTrue(set(train["id"]).isdisjoint(set(test["id"])))
        for label in AREA_LABELS:
            self.assertGreater((test["stratum"] == label).sum(), 0)

    def test_out_of_range_area_is_rejected(self):
        frame = pd.DataFrame({"id": [0, 1], "Area": [120.0, 99.0]})
        with self.assertRaisesRegex(ValueError, "outside the area strata"):
            assign_strata(frame)

    def test_very_large_category_has_no_upper_bound(self):
        frame = pd.DataFrame({"id": [0, 1], "Area": [1500.0, 12000.0]})
        observed = assign_strata(frame)
        self.assertEqual(observed.astype(str).tolist(), ["Very large", "Very large"])

    def test_calibration_manifest_uses_city_ids_and_coordinates(self):
        frame = pd.DataFrame(
            {
                "id": [9, 3],
                "centroid_x": [120.0, -70.0],
                "centroid_y": [30.0, -20.0],
            }
        )
        observed = calibration_manifest(frame)
        self.assertEqual(observed["id"].tolist(), [1, 2])
        self.assertEqual(observed["city_id"].tolist(), [3, 9])
        self.assertEqual(observed["longitude"].tolist(), [-70.0, 120.0])


if __name__ == "__main__":
    unittest.main()
