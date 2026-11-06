import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_city_manifests import REGION_FILENAMES, write_regional_folds  # noqa: E402
from validate_regional_fold import validate_fold  # noqa: E402


class PreprocessingManifestTest(unittest.TestCase):
    def test_external_regional_metadata_source_is_not_required(self):
        source = ROOT / "data" / "city_metrics" / "region_assignments_source.csv"
        script = (ROOT / "scripts" / "build_city_manifests.py").read_text()
        self.assertFalse(source.exists())
        self.assertNotIn("region_assignments_source", script)
        self.assertNotIn("--city-metadata", script)

    def test_regional_folds_are_exact_complements(self):
        regions = list(REGION_FILENAMES)
        assignments = pd.DataFrame(
            {
                "id": range(600),
                "region": [regions[index % len(regions)] for index in range(600)],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments_path = root / "assignments.csv"
            assignments.to_csv(assignments_path, index=False)
            write_regional_folds(assignments, root / "folds")
            self.assertEqual(len(list((root / "folds").glob("*.csv"))), 12)
            for display_name, filename in REGION_FILENAMES.items():
                train_count, test_count = validate_fold(
                    filename,
                    root / "folds" / f"fold_{filename}_train.csv",
                    root / "folds" / f"fold_{filename}_test.csv",
                    assignments_path,
                )
                self.assertEqual(train_count + test_count, 600)
                self.assertEqual(test_count, 100, display_name)


if __name__ == "__main__":
    unittest.main()
