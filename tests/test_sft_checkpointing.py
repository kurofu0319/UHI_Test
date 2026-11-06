import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SFT_trainer import (  # noqa: E402
    EpochCheckpointCallback,
    _resolve_resume_checkpoint,
)


class SFTCheckpointingTest(unittest.TestCase):
    def test_epoch_checkpoints_use_zero_based_stable_names(self):
        callback = EpochCheckpointCallback()
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(output_dir=output_dir)
            for epoch_index in range(5):
                global_step = (epoch_index + 1) * 5250
                os.mkdir(os.path.join(output_dir, f"checkpoint-{global_step}"))
                state = SimpleNamespace(
                    epoch=float(epoch_index + 1), global_step=global_step
                )
                callback.on_save(args, state, None)

            expected = [f"checkpoint-epoch{i}" for i in range(5)]
            self.assertEqual(sorted(os.listdir(output_dir)), expected)
            self.assertEqual(
                _resolve_resume_checkpoint("latest", output_dir),
                os.path.join(output_dir, "checkpoint-epoch4"),
            )
            self.assertEqual(
                _resolve_resume_checkpoint("checkpoint-epoch2", output_dir),
                os.path.join(output_dir, "checkpoint-epoch2"),
            )


if __name__ == "__main__":
    unittest.main()
