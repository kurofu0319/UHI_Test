import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lora_similarity_threshold import (  # noqa: E402
    OUTPUT_FILES,
    prepare_output_dir,
    validate_thresholds,
)


class LoRASimilarityThresholdTest(unittest.TestCase):
    def test_thresholds_must_be_finite_unique_and_increasing(self):
        self.assertEqual(
            validate_thresholds([0.4, 0.5, 0.6, 0.7, 0.8]),
            [0.4, 0.5, 0.6, 0.7, 0.8],
        )
        for invalid in ([0.5, 0.4], [0.5, 0.5], [float("nan")], [1.1], []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_thresholds(list(invalid))

    def test_existing_outputs_require_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "similarity"
            output_dir.mkdir()
            (output_dir / OUTPUT_FILES[0]).write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_dir(output_dir, overwrite=False)
            prepare_output_dir(output_dir, overwrite=True)

    def test_shell_uses_explicit_epoch_checkpoints(self):
        script = (ROOT / "scripts" / "run_lora_similarity_threshold.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SIMILARITY_EPOCHS="${SIMILARITY_EPOCHS:-0 1 2 3 4}"', script)
        self.assertIn('checkpoint-epoch${epoch_index}', script)
        self.assertNotIn("find \"${ADAPTER_ROOT}\"", script)


if __name__ == "__main__":
    unittest.main()
