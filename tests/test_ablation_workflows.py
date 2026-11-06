import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AblationWorkflowTest(unittest.TestCase):
    def test_module_ablation_retains_all_non_removed_stages(self):
        script = (ROOT / "scripts" / "run_module_ablations.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("SFT_INPUT_FORMAT=numeric_plain", script)
        self.assertIn("bash scripts/run_sft.sh", script)
        self.assertIn("FUSION_INPUT_FORMAT=numeric_plain", script)
        self.assertIn("PPO_INPUT_FORMAT=numeric_plain", script)
        self.assertIn("PPO_EPOCHS=2", script)
        self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"', script)
        self.assertIn('DISTILLED_EPOCHS="0 1"', script)
        self.assertIn("numeric_fusion_epochs", script)
        self.assertIn("explanation_samples_numeric_list", script)
        self.assertIn("without_generalizability", script)
        self.assertIn("without_explainability", script)
        self.assertIn("run_distilled_ensemble_inference.sh", script)
        without_explainability = script.split("# 3. w/o Explainability Enhancer", 1)[1]
        self.assertIn("run_ensemble_inference.sh", without_explainability)
        self.assertNotIn("run_distilled_ensemble_inference.sh", without_explainability)
        self.assertIn("checkpoint-epoch${SINGLE_SFT_EPOCH}", script)
        self.assertNotIn("sort -V | tail -n 1", script)

    def test_component_ablation_uses_final_policy_without_ensemble(self):
        script = (ROOT / "scripts" / "run_component_ablations.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("PPO_EPOCHS=2", script)
        self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"', script)
        self.assertIn("PPO_INPUT_FORMAT=prompt_task_only", script)
        self.assertIn('DISTILLED_EPOCHS="0 1"', script)
        self.assertIn('DISTILLED_EPOCHS="1"', script)
        self.assertIn("checkpoint-epoch${SINGLE_SFT_EPOCH}", script)
        self.assertNotIn("sort -V | tail -n 1", script)
        no_ensemble = script.split("# 3. w/o ensemble inference", 1)[1]
        self.assertIn('${MAIN_PPO_ROOT}/epoch1', no_ensemble)
        self.assertNotIn('${MAIN_PPO_ROOT}/epoch0', no_ensemble)
        self.assertIn("run_explanation_variant", no_ensemble)
        self.assertIn("run_distilled_ensemble_inference.sh", no_ensemble)


if __name__ == "__main__":
    unittest.main()
