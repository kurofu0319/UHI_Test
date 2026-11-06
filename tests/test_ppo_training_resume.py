import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PPOTrainer import (  # noqa: E402
    _load_training_resume_metadata,
    _load_local_ppo_weights,
    _validate_resume_run_config,
)


def _args():
    return SimpleNamespace(
        batch_size=1,
        mini_batch_size=1,
        ppo_epochs=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-6,
        seed=0,
        max_length=1300,
        max_new_tokens=24,
        input_format="prompt_task_only",
        reward_baseline=0.15,
        penalty_lambda=10.0,
        valid_min=-5.0,
        valid_max=5.0,
        max_decimal_places=3,
    )


class PPOTrainingResumeTest(unittest.TestCase):
    def test_local_weight_loader_restores_lora_and_value_head(self):
        class Pretrained(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict(
                    {"default": torch.nn.Linear(2, 1, bias=False)}
                )

        class Model:
            def __init__(self):
                self.pretrained_model = Pretrained()
                self.value_head_state = None

            def post_init(self, state):
                self.value_head_state = state

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_file(
                {"lora_A.weight": torch.tensor([[2.0, 3.0]])},
                checkpoint / "adapter_model.safetensors",
            )
            torch.save(
                {"v_head.summary.weight": torch.tensor([[4.0, 5.0]])},
                checkpoint / "pytorch_model.bin",
            )
            model = Model()
            loaded = _load_local_ppo_weights(model, checkpoint)

            self.assertEqual(loaded, 1)
            self.assertTrue(
                torch.equal(
                    model.pretrained_model.lora_A["default"].weight,
                    torch.tensor([[2.0, 3.0]]),
                )
            )
            self.assertIn("v_head.summary.weight", model.value_head_state)

    def test_metadata_requires_matching_complete_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            state = root / "state"
            checkpoint.mkdir()
            state.mkdir()
            (state / "model.safetensors").touch()
            (state / "optimizer.bin").touch()
            for rank in range(2):
                (state / f"random_states_{rank}.pkl").touch()
            (state / "ppo_training_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "adapter_checkpoint": str(checkpoint),
                        "epoch": 0,
                        "ppo_steps_completed_in_epoch": 8400,
                    }
                ),
                encoding="utf-8",
            )

            resolved, metadata = _load_training_resume_metadata(
                state, checkpoint, 2
            )
            self.assertEqual(resolved, state.resolve())
            self.assertEqual(metadata["ppo_steps_completed_in_epoch"], 8400)

    def test_run_config_rejects_training_protocol_change(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            config = {
                "per_process_batch_size": 1,
                "per_process_mini_batch_size": 1,
                "distributed_processes": 4,
                "ppo_epochs_per_batch": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 3e-6,
                "seed": 0,
                "max_prompt_tokens": 1300,
                "max_new_tokens": 24,
                "input_format": "prompt_task_only",
                "reward": {
                    "y_bar": 0.15,
                    "lambda": 10.0,
                    "valid_min": -5.0,
                    "valid_max": 5.0,
                    "max_decimal_places": 3,
                },
            }
            (checkpoint / "ppo_run_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            _validate_resume_run_config(checkpoint, _args(), 4)

            changed = _args()
            changed.max_new_tokens = 32
            with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                _validate_resume_run_config(checkpoint, changed, 4)


if __name__ == "__main__":
    unittest.main()
