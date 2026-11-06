import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
SMOKE_MODEL = ROOT / "model" / "smoke-tiny-llama"
SMOKE_CITY_IDS = (413, 661, 222, 677, 864)
SECTION_HEADINGS = (
    "Geographical Context:",
    "Structural Contextual Influence:",
    "Dynamic Environmental Modulation:",
    "Cross-Factor Synergy:",
    "Prediction Synthesis:",
    "Planning-Oriented Recommendations:",
)


def _scalar(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return int(value)


class _DeepSeekFixtureHandler(BaseHTTPRequestHandler):
    requests_seen = 0
    result_pattern = re.compile(
        r"\[Result:\]\s*Predicted UHI intensity for the next month:\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    )

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length))
        prompt = payload["messages"][0]["content"]
        match = self.result_pattern.search(prompt)
        if match is None:
            self.send_error(400, "Missing supplied result")
            return
        prediction = match.group(1)
        explanation = "\n\n".join(
            f"{heading} Local end-to-end fixture explanation."
            for heading in SECTION_HEADINGS
        )
        response = {
            "id": "fixture-chat-completion",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "[Result:] Predicted UHI intensity for the next month: "
                            f"{prediction} °C\n\n{explanation}"
                        ),
                        "reasoning_content": "Local fixture reasoning.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        type(self).requests_seen += 1
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        return


class ExplanationWorkflowContractTest(unittest.TestCase):
    def test_default_workflow_distills_both_ppo_policies(self):
        workflow = (ROOT / "scripts" / "run_explanation_distillation.sh").read_text()
        self.assertIn('EXPLANATION_POLICY_EPOCHS="${EXPLANATION_POLICY_EPOCHS:-0 1}"', workflow)
        self.assertIn('EXPLANATION_DISTILLED_ADAPTER_ROOT', workflow)
        self.assertIn('EXPLANATION_POLICY_ADAPTER_DIR="${PPO_POLICY_ADAPTER_ROOT}/epoch${policy_epoch}"', workflow)
        self.assertIn("use_flat_release_samples", workflow)
        self.assertIn("Using ${flat_sample_count} released flat explanation samples", workflow)


@unittest.skipUnless(
    os.environ.get("RUN_EXPLANATION_E2E") == "1"
    and SMOKE_MODEL.is_dir()
    and all(
        (ROOT / "data" / "train" / f"UHI_city_{city_id}").is_dir()
        for city_id in SMOKE_CITY_IDS
    ),
    "Set RUN_EXPLANATION_E2E=1 when local smoke artifacts are available.",
)
class ExplanationWorkflowEndToEndTest(unittest.TestCase):
    def test_generation_epoch1_materialization_and_distillation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".smoke") as temporary_dir:
            temporary = Path(temporary_dir)
            data_dir = temporary / "data"
            for city_id in SMOKE_CITY_IDS:
                source = load_from_disk(
                    str(ROOT / "data" / "train" / f"UHI_city_{city_id}")
                )
                first = min(
                    source,
                    key=lambda sample: (
                        _scalar(sample["year_info"]),
                        _scalar(sample["month_info"]),
                    ),
                )
                destination = data_dir / "train" / f"UHI_city_{city_id}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                Dataset.from_list([first]).save_to_disk(str(destination))
            city_metrics = data_dir / "city_metrics" / "city_metrics_train.csv"
            city_metrics.parent.mkdir(parents=True, exist_ok=True)
            city_metrics.write_text(
                "id\n" + "".join(f"{city_id}\n" for city_id in SMOKE_CITY_IDS),
                encoding="utf-8",
            )

            epoch1_dir = temporary / "ppo" / "epoch1"
            base = AutoModelForCausalLM.from_pretrained(SMOKE_MODEL)
            policy = get_peft_model(
                base,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            policy.save_pretrained(epoch1_dir)

            _DeepSeekFixtureHandler.requests_seen = 0
            server = ThreadingHTTPServer(("127.0.0.1", 0), _DeepSeekFixtureHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            try:
                environment = os.environ.copy()
                for proxy_name in (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                ):
                    environment.pop(proxy_name, None)
                smoke_gpu = os.environ.get("EXPLANATION_E2E_GPU", "")
                environment.update(
                    {
                        "PYTHON": sys.executable,
                        "DATA_DIR": str(data_dir),
                        "OUTPUT_ROOT": str(temporary / "outputs"),
                        "LOG_DIR": str(temporary / "logs"),
                        "GPU_ID": smoke_gpu or "0",
                        "BASE_MODEL": "smoke-tiny-llama",
                        "EXPLANATION_POLICY_ADAPTER_DIR": str(epoch1_dir),
                        "EXPLANATION_POLICY_EPOCH": "1",
                        "MERGED_MODEL": "smoke-ppo-epoch1-materialized",
                        "MERGED_MODEL_DIR": str(temporary / "merged"),
                        "EXPLANATION_DATA_ROOT": str(temporary / "explanations"),
                        "EXPLANATION_ADAPTER": "smoke-explanation",
                        "EXPLANATION_ADAPTER_DIR": str(temporary / "adapter"),
                        "EXPLANATION_EXPECTED_TRAIN_CITIES": "5",
                        "EXPLANATION_EXPECTED_TRAIN_SAMPLES_PER_CITY": "1",
                        "EXPLANATION_SAMPLE_COUNT": "5",
                        "EXPLANATION_SAMPLING_SEED": "42",
                        "EXPLANATION_FIRST_TARGET": "2004-01",
                        "EXPLANATION_LAST_TARGET": "2004-01",
                        "EXPLANATION_CONCURRENCY": "2",
                        "EXPLANATION_API_TIMEOUT": "10",
                        "EXPLANATION_API_MAX_ATTEMPTS": "1",
                        "NO_PROXY": "127.0.0.1,localhost",
                        "no_proxy": "127.0.0.1,localhost",
                        "DISTILL_DEVICE": "cuda:0" if smoke_gpu else "cpu",
                        "DISTILL_TORCH_DTYPE": "float32",
                        "DISTILL_MAX_LENGTH": "1200",
                        "DISTILL_MAX_STEPS": "1",
                        "DISTILL_NUM_WORKERS": "1",
                        "OPENAI_API_KEY": "fixture-key",
                        "OPENAI_BASE_URL": (
                            f"http://127.0.0.1:{server.server_port}/v1"
                        ),
                    }
                )
                completed = subprocess.run(
                    ["bash", "scripts/run_explanation_distillation.sh"],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    check=False,
                )
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertEqual(_DeepSeekFixtureHandler.requests_seen, 5)
            self.assertEqual(
                len(list((temporary / "explanations" / "content").rglob("*.txt"))),
                5,
            )
            self.assertTrue((temporary / "merged" / "config.json").is_file())
            self.assertTrue((temporary / "adapter" / "adapter_config.json").is_file())
            config = json.loads(
                (temporary / "adapter" / "distillation_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["training_samples"], 5)


if __name__ == "__main__":
    unittest.main()
