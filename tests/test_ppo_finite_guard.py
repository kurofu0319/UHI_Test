import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PPOTrainer import (  # noqa: E402
    FiniteSafePPOTrainer,
    _all_ranks_tensors_finite,
)


class _Model:
    def train(self):
        return self


class _Optimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.steps = 0
        self.zero_grad_calls = 0

    def step(self):
        self.steps += 1

    def zero_grad(self):
        self.zero_grad_calls += 1
        self.parameter.grad = None


class _Accelerator:
    is_main_process = False
    sync_gradients = True

    def __init__(self, parameter, gradient_value):
        self.parameter = parameter
        self.gradient_value = gradient_value
        self.backward_calls = 0

    def backward(self, _loss):
        self.backward_calls += 1
        self.parameter.grad = torch.tensor(self.gradient_value)


def _trainer(loss_value, gradient_value=1.0):
    trainer = object.__new__(FiniteSafePPOTrainer)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer.model = _Model()
    trainer.model_params = [parameter]
    trainer.optimizer = _Optimizer(parameter)
    trainer.accelerator = _Accelerator(parameter, gradient_value)
    trainer.config = SimpleNamespace(max_grad_norm=None)
    trainer.non_finite_loss_skips = 0
    trainer.non_finite_gradient_skips = 0

    def loss(*_args):
        loss_p = torch.tensor(loss_value, requires_grad=True)
        loss_v = torch.tensor(0.0, requires_grad=True)
        return loss_p, loss_v, {"loss/policy": loss_p.detach()}

    trainer.loss = loss
    return trainer


class PPOFiniteGuardTest(unittest.TestCase):
    def test_tensor_finiteness_check(self):
        self.assertTrue(_all_ranks_tensors_finite([torch.tensor([1.0, 2.0])]))
        self.assertFalse(_all_ranks_tensors_finite([torch.tensor([float("nan")])]))

    def test_finite_update_is_unchanged(self):
        trainer = _trainer(1.0, gradient_value=0.5)
        stats = trainer.train_minibatch(*([torch.tensor(0.0)] * 8))

        self.assertEqual(trainer.accelerator.backward_calls, 1)
        self.assertEqual(trainer.optimizer.steps, 1)
        self.assertEqual(trainer.optimizer.zero_grad_calls, 1)
        self.assertEqual(stats["safety/skipped_non_finite_loss"].item(), 0.0)
        self.assertEqual(stats["safety/skipped_non_finite_gradient"].item(), 0.0)

    def test_non_finite_loss_skips_backward_and_optimizer(self):
        trainer = _trainer(float("nan"))
        stats = trainer.train_minibatch(*([torch.tensor(0.0)] * 8))

        self.assertEqual(trainer.accelerator.backward_calls, 0)
        self.assertEqual(trainer.optimizer.steps, 0)
        self.assertEqual(trainer.non_finite_loss_skips, 1)
        self.assertEqual(stats["loss/policy"].item(), 0.0)
        self.assertEqual(stats["safety/skipped_non_finite_loss"].item(), 1.0)

    def test_non_finite_gradient_skips_optimizer(self):
        trainer = _trainer(1.0, gradient_value=float("inf"))
        stats = trainer.train_minibatch(*([torch.tensor(0.0)] * 8))

        self.assertEqual(trainer.accelerator.backward_calls, 1)
        self.assertEqual(trainer.optimizer.steps, 0)
        self.assertEqual(trainer.non_finite_gradient_skips, 1)
        self.assertEqual(
            stats["safety/skipped_non_finite_gradient"].item(), 1.0
        )


if __name__ == "__main__":
    unittest.main()
