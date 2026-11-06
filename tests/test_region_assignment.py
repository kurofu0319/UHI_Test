import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assign_city_regions import (  # noqa: E402
    EXPECTED_1020_COUNTS,
    assign_regions,
    classify_region,
    validate_expected_counts,
)
from validate_regional_fold import REGION_NAMES, validate_fold  # noqa: E402


class RegionAssignmentTest(unittest.TestCase):
    def test_representative_coordinates(self):
        cases = [
            ((-100.0, 40.0), "North America"),
            ((-70.0, -20.0), "Latin America"),
            ((20.0, 5.0), "Africa"),
            ((15.0, 50.0), "Europe"),
            ((120.0, 30.0), "East Asia & Oceania"),
            ((70.0, 30.0), "West Asia"),
        ]
        for (longitude, latitude), expected in cases:
            with self.subTest(longitude=longitude, latitude=latitude):
                self.assertEqual(classify_region(longitude, latitude), expected)

    def test_boundary_precedence_matches_frozen_rules(self):
        self.assertEqual(classify_region(-30.0, 10.0), "Africa")
        self.assertEqual(classify_region(-30.0001, 10.0), "North America")
        self.assertEqual(classify_region(40.0, 35.0), "West Asia")
        self.assertEqual(classify_region(93.0, 35.0), "East Asia & Oceania")

    def test_all_candidate_counts_and_frozen_600_assignments(self):
        metrics_dir = ROOT / "data" / "city_metrics"
        all_cities = assign_regions(pd.read_csv(metrics_dir / "city_metrics_all.csv"))
        self.assertEqual(len(all_cities), 1020)
        validate_expected_counts(all_cities, EXPECTED_1020_COUNTS)

        frozen = pd.read_csv(metrics_dir / "region_assignments_600.csv")
        regenerated = assign_regions(frozen.drop(columns="region"))
        pd.testing.assert_series_equal(
            regenerated["region"], frozen["region"], check_names=False
        )

    def test_invalid_coordinates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "longitude outside"):
            classify_region(181.0, 0.0)
        with self.assertRaisesRegex(ValueError, "latitude outside"):
            classify_region(0.0, 91.0)

    def test_frozen_leave_one_region_out_folds_are_exact_complements(self):
        fold_root = ROOT / "data" / "regional_folds"
        assignments = ROOT / "data" / "city_metrics" / "region_assignments_600.csv"
        observed_test_cities = 0
        for region in REGION_NAMES:
            train_count, test_count = validate_fold(
                region,
                fold_root / f"fold_{region}_train.csv",
                fold_root / f"fold_{region}_test.csv",
                assignments,
            )
            self.assertEqual(train_count + test_count, 600)
            observed_test_cities += test_count
        self.assertEqual(observed_test_cities, 600)


if __name__ == "__main__":
    unittest.main()
