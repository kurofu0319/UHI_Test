import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PPOTrainer import (  # noqa: E402
    _checkpoint_directories,
    _prepare_checkpoint_directories,
    _save_latest_training_state,
    _save_ppo_checkpoint_locally,
)


class PPOCheckpointingTest(unittest.TestCase):
    def test_training_state_atomically_replaces_previous_save(self):
        class Accelerator:
            is_main_process = True

            def __init__(self):
                self.save_count = 0

            @staticmethod
            def wait_for_everyone():
                return None

            def save_state(self, directory):
                self.save_count += 1
                Path(directory, "optimizer.bin").write_text(
                    str(self.save_count), encoding="utf-8"
                )

        with tempfile.TemporaryDirectory() as temporary_dir:
            state_dir = Path(temporary_dir) / "training_state_latest"
            accelerator = Accelerator()
            _save_latest_training_state(accelerator, state_dir, {"step": 1})
            _save_latest_training_state(accelerator, state_dir, {"step": 2})

            self.assertEqual(
                (state_dir / "optimizer.bin").read_text(encoding="utf-8"), "2"
            )
            self.assertEqual(
                (state_dir / "ppo_training_state.json").read_text(
                    encoding="utf-8"
                ),
                '{\n  "step": 2\n}\n',
            )
            self.assertFalse(
                (Path(temporary_dir) / ".training_state_latest.tmp").exists()
            )

    def test_local_save_does_not_call_trainer_hub_save(self):
        class Saveable:
            def __init__(self, filename):
                self.filename = filename

            def save_pretrained(self, directory):
                Path(directory, self.filename).write_text("saved", encoding="utf-8")

        class Accelerator:
            @staticmethod
            def unwrap_model(model):
                return model

        class Config:
            @staticmethod
            def to_dict():
                return {"batch_size": 1}

        class Trainer:
            accelerator = Accelerator()
            model = Saveable("model.bin")
            tokenizer = Saveable("tokenizer.json")
            config = Config()

            @staticmethod
            def save_pretrained(_directory):
                raise AssertionError("hub-aware trainer save must not be called")

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "checkpoint"
            _save_ppo_checkpoint_locally(Trainer(), checkpoint)
            self.assertTrue((checkpoint / "model.bin").is_file())
            self.assertTrue((checkpoint / "tokenizer.json").is_file())
            self.assertEqual(
                (checkpoint / "config.json").read_text(encoding="utf-8"),
                '{\n  "batch_size": 1\n}\n',
            )

    def test_multi_epoch_paths_are_zero_based_epoch_directories(self):
        root = Path("output")
        self.assertEqual(
            _checkpoint_directories(root, train_epochs=2, start_epoch=0),
            [root / "epoch0", root / "epoch1"],
        )
        self.assertEqual(
            _checkpoint_directories(root, train_epochs=2, start_epoch=1),
            [root / "epoch1", root / "epoch2"],
        )

    def test_single_epoch_ablation_saves_exactly_to_requested_path(self):
        root = Path("lambda_5")
        self.assertEqual(
            _checkpoint_directories(root, train_epochs=1, start_epoch=0), [root]
        )

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "epoch0"
            checkpoint.mkdir()
            marker = checkpoint / "marker"
            marker.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                _prepare_checkpoint_directories([checkpoint], overwrite=False)

            _prepare_checkpoint_directories([checkpoint], overwrite=True)
            self.assertFalse(checkpoint.exists())

    def test_resume_source_cannot_be_an_overwrite_target(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "epoch0"
            checkpoint.mkdir()
            with self.assertRaises(ValueError):
                _prepare_checkpoint_directories(
                    [checkpoint], overwrite=True, resume_dir=checkpoint
                )


if __name__ == "__main__":
    unittest.main()
