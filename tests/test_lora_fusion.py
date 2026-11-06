import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_fusion import (  # noqa: E402
    AdapterDeltaState,
    delta_cosine_similarity,
    exponential_mae_weights,
    greedy_select_adapters,
)
from greedy_adapter_fusion import (  # noqa: E402
    add_rank_concatenation_fusion,
    save_named_adapter,
)


class LoRAFusionTest(unittest.TestCase):
    def test_low_rank_cosine_matches_materialized_delta(self):
        a1 = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
        b1 = torch.tensor([[1.0, 0.0], [2.0, 1.0], [-1.0, 0.5]])
        a2 = torch.tensor([[0.5, 1.0], [-2.0, 0.25]])
        b2 = torch.tensor([[0.0, 1.0], [1.0, -1.0], [0.5, 2.0]])
        state1 = AdapterDeltaState({"q_proj": (a1, b1)}, scale=2.0)
        state2 = AdapterDeltaState({"q_proj": (a2, b2)}, scale=0.5)

        actual, modules = delta_cosine_similarity(state1, state2)
        delta1 = state1.scale * (b1 @ a1)
        delta2 = state2.scale * (b2 @ a2)
        expected = torch.nn.functional.cosine_similarity(
            delta1.flatten(), delta2.flatten(), dim=0
        ).item()
        self.assertEqual(modules, 1)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_exponential_mae_weights_are_normalized(self):
        weights = exponential_mae_weights([0.20, 0.10], beta=10.0)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(weights[1] / weights[0], math.e, places=6)

    def test_greedy_selection_applies_similarity_and_mae_rules(self):
        individual = {"epoch0": 0.20, "epoch1": 0.18, "epoch2": 0.17}
        similarities = {
            ("epoch0", "epoch1"): 0.90,
            ("epoch0", "epoch2"): 0.50,
            ("epoch1", "epoch2"): 0.85,
        }
        evaluated = []

        def evaluate(names, weights):
            evaluated.append((list(names), list(weights)))
            return {("epoch2", "epoch1"): 0.16}[tuple(names)]

        result = greedy_select_adapters(
            individual,
            similarities,
            similarity_threshold=0.6,
            beta=10.0,
            evaluate_fusion=evaluate,
        )
        self.assertEqual(result.selected, ["epoch2", "epoch1"])
        self.assertAlmostEqual(result.validation_mae, 0.16)
        self.assertEqual(len(evaluated), 1)
        self.assertEqual(result.steps[-1]["reason"], "cosine_similarity_below_threshold")

    def test_rank_concatenation_is_exact_and_reloadable(self):
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        peft_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, peft_config, adapter_name="epoch4")
        model.add_adapter("epoch5", peft_config)
        torch.manual_seed(7)
        for module in model.modules():
            if hasattr(module, "lora_A"):
                for name in ("epoch4", "epoch5"):
                    if name in module.lora_A:
                        module.lora_A[name].weight.data.normal_()
                        module.lora_B[name].weight.data.normal_()

        source_deltas = {}
        for module_name, module in model.named_modules():
            if not hasattr(module, "lora_A") or "epoch4" not in module.lora_A:
                continue
            source_deltas[module_name] = (
                0.25 * module.get_delta_weight("epoch4")
                + 0.75 * module.get_delta_weight("epoch5")
            )

        add_rank_concatenation_fusion(
            model,
            ["epoch4", "epoch5"],
            [0.25, 0.75],
            "fusion",
        )
        fused_deltas = {}
        for module_name, module in model.named_modules():
            if not hasattr(module, "get_delta_weight") or "fusion" not in getattr(
                module, "lora_A", {}
            ):
                continue
            fused_deltas[module_name] = module.get_delta_weight("fusion").clone()
            self.assertEqual(
                module.lora_A["fusion"].weight.shape[0],
                module.lora_A["epoch4"].weight.shape[0]
                + module.lora_A["epoch5"].weight.shape[0],
            )
            self.assertTrue(
                torch.allclose(
                    module.get_delta_weight("fusion"),
                    source_deltas[module_name],
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
        self.assertTrue(fused_deltas)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "adapter"
            save_named_adapter(model, "fusion", output)
            self.assertTrue((output / "adapter_config.json").is_file())
            self.assertTrue((output / "adapter_model.safetensors").is_file())

            reloaded = PeftModel.from_pretrained(
                LlamaForCausalLM(model.base_model.model.config),
                output,
                adapter_name="fusion",
            )
            reloaded_modules = dict(reloaded.named_modules())
            for module_name, expected in fused_deltas.items():
                actual = reloaded_modules[module_name].get_delta_weight("fusion")
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
                )

    def test_single_selected_adapter_is_exact(self):
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        peft_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, peft_config, adapter_name="epoch4")
        torch.manual_seed(9)
        source_deltas = {}
        for module_name, module in model.named_modules():
            if not hasattr(module, "lora_A") or "epoch4" not in module.lora_A:
                continue
            module.lora_A["epoch4"].weight.data.normal_()
            module.lora_B["epoch4"].weight.data.normal_()
            source_deltas[module_name] = module.get_delta_weight("epoch4").clone()

        add_rank_concatenation_fusion(
            model,
            ["epoch4"],
            [1.0],
            "fusion",
        )
        for module_name, expected in source_deltas.items():
            actual = dict(model.named_modules())[module_name].get_delta_weight("fusion")
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_rank_concatenation_preserves_fp32_adapter_precision_on_bf16_base(self):
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ).to(torch.bfloat16)
        peft_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, peft_config, adapter_name="epoch4")
        model.add_adapter("epoch5", peft_config)
        torch.manual_seed(11)
        for module_name, module in model.named_modules():
            if not hasattr(module, "lora_A") or "epoch4" not in module.lora_A:
                continue
            for name in ("epoch4", "epoch5"):
                module.lora_A[name].float()
                module.lora_B[name].float()
                module.lora_A[name].weight.data.normal_()
                module.lora_B[name].weight.data.normal_()

        add_rank_concatenation_fusion(
            model,
            ["epoch4", "epoch5"],
            [0.25, 0.75],
            "fusion",
        )
        fused_modules = 0
        for module in model.modules():
            if not hasattr(module, "lora_A") or "fusion" not in module.lora_A:
                continue
            fused_modules += 1
            self.assertEqual(module.lora_A["fusion"].weight.dtype, torch.float32)
            self.assertEqual(module.lora_B["fusion"].weight.dtype, torch.float32)
        self.assertGreater(fused_modules, 0)


if __name__ == "__main__":
    unittest.main()
