import csv
import json
import os
import sys
from pathlib import Path


ARGS = sys.argv[1:]
ROOT = Path.cwd()


def value(flag, default=None):
    return ARGS[ARGS.index(flag) + 1] if flag in ARGS else default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def adapter(path):
    path = Path(path)
    write_json(path / "adapter_config.json", {})
    (path / "adapter_model.safetensors").write_bytes(b"test fixture")


def measurement(path, kind, method, stage, amount):
    metadata = {}
    if kind == "inference":
        metadata = {
            "batch_size": 36,
            "warmup_calls": 5,
            "measured_calls": 5,
            "timing_protocol": "independent_repetitions_mean",
            "max_new_tokens": 19,
            "max_length": 6000,
            "temperature": 0.0,
            "input_format": "prompt",
        }
    write_json(path, {
        "kind": kind, "method": method, "stage": stage,
        "value": amount,
        "unit": "seconds" if kind == "training" else "ms_per_sample",
        "hardware": "test fixture",
        "note": "model execution only" if kind == "inference" else "training loop",
        "metadata": metadata,
    })


def rows(offset):
    result = []
    for city in range(501, 601):
        for index in range(36):
            year, month = 2018 + index // 12, index % 12 + 1
            target = 0.2 + index / 1000.0
            result.append({
                "city_idx": city, "target_year": year, "target_month": month,
                "target_year_month": f"{year:04d}-{month:02d}",
                "true_uhi": target, "predicted_uhi": target + offset,
            })
    return result


if ARGS == ["--version"]:
    print("test python")
    sys.exit(0)
if ARGS[0] == "-c":
    if "import json, sys" in ARGS[1]:
        os.execv(sys.executable, [sys.executable, *ARGS])
    versions = "4.46.3 0.11.2"
    if "peft" in ARGS[1]:
        versions += " 0.19.1"
    if "openai" in ARGS[1]:
        versions += " 1.0"
    print(versions)
    sys.exit(0)

with Path(os.environ["SMOKE_COMMANDS"]).open("a") as handle:
    handle.write(json.dumps(ARGS) + "\n")

script = Path(ARGS[0]).name
if script in {
    "average_predictions.py", "derive_ppo_reward_baseline.py",
    "figure_data.py", "build_comparison_tables.py", "build_efficiency_measurements.py",
}:
    relative = Path("scripts/figures/figure_data.py") if script == "figure_data.py" else Path("scripts") / script
    os.execv(sys.executable, [
        sys.executable, str(Path(os.environ["SMOKE_REAL_ROOT"]) / relative), *ARGS[1:],
    ])
elif script == "SFT_trainer.py":
    output = Path(value("--output_dir"))
    for epoch in range(int(value("--train_epochs"))):
        adapter(output / f"checkpoint-epoch{epoch}")
    adapter(output)
    measurement(value("--efficiency_record"), "training", "UHI-E3LLM", "SFT",
                float(os.environ.get("SMOKE_TRAINING_SECONDS", "10")))
elif script == "greedy_adapter_fusion.py":
    start, stop = ARGS.index("--adapter_paths") + 1, ARGS.index("--dataset_path")
    assert all((Path(path) / "adapter_model.safetensors").is_file() for path in ARGS[start:stop])
    adapter(value("--output_adapter_dir"))
    write_json(Path(value("--merged_model_dir")) / "config.json", {})
    write_json(Path(value("--report_dir")) / "selection.json", {"epochs": [0, 1]})
elif script == "PPOTrainer.py":
    assert (ROOT / "model" / value("--model") / "config.json").is_file()
    for epoch in range(int(value("--train_epochs"))):
        adapter(Path(value("--save_path")) / f"epoch{epoch}")
    measurement(value("--efficiency_record"), "training", "UHI-E3LLM", "PPO",
                float(os.environ.get("SMOKE_TRAINING_SECONDS", "20")))
elif script == "merge_lora.py":
    assert (Path(value("--adapter_dir")) / "adapter_config.json").is_file()
    assert (ROOT / "model" / value("--model") / "config.json").is_file()
    write_json(Path(value("--output_dir")) / "config.json", {})
elif script == "explanation_distillation.py":
    assert (Path(value("--model_path")) / "config.json").is_file()
    assert len(list(Path(value("--data_path")).glob("sample_*.txt"))) == 900
    adapter(value("--output_dir"))
    epoch = value("--efficiency_member_epoch")
    measurement(value("--efficiency_record"), "training", "UHI-E3LLM",
                f"Distillation_epoch{epoch}", 5.0)
elif script == "Inference.py":
    model = value("--model")
    assert (ROOT / "model" / model / "config.json").is_file()
    if value("--mode") == "train":
        write_csv(Path(value("--save_path")) / model / f"_{value('--checkpoint')}_inference.csv", [
            {"city_idx": city, "samples": 168, "mae": 0.2, "mse": 0.04, "rmse": 0.2}
            for city in range(1, 501)
        ])
    else:
        assert (Path(value("--adapter_dir")) / "adapter_config.json").is_file()
        epoch = int(value("--checkpoint").rsplit("epoch", 1)[1])
        predictions = rows(0.1 * (epoch + 1))
        for city in range(501, 601):
            write_csv(
                Path(value("--save_path")) / model / "explanation" / "city_results"
                / f"city_{city}_predictions.csv",
                [row for row in predictions if row["city_idx"] == city],
            )
        if "--efficiency_record" in ARGS:
            measurement(value("--efficiency_record"), "inference", "UHI-E3LLM",
                        f"inference_epoch{epoch}", 100.0 + 40.0 * epoch)
elif script in {"smoke_tabular", "smoke_timeseries"}:
    output = Path(value("--output-root"))
    raw = Path(value("--raw-dir"))
    methods = (
        [("xgb", "XGBoost"), ("rf", "RandomForest"), ("lgb", "LightGBM"), ("mlp", "MLP")]
        if script == "smoke_tabular" else
        [("armd", "ARMD"), ("filternet", "FilterNet"), ("st_mtm", "ST-MTM")]
    )
    for index, (name, method) in enumerate(methods):
        data = rows(0.3 + index * 0.1)
        if script == "smoke_tabular":
            write_csv(output / "baselines/tabular/predictions" / f"{name}_test_predictions.csv", data)
        else:
            for city in range(501, 601):
                write_csv(output / "baselines" / name / "sample_errors" / f"city_{city}.csv",
                          [row for row in data if row["city_idx"] == city])
        stages = ("pretrain", "finetune") if method == "ST-MTM" else ("train",)
        for stage in stages:
            measurement(raw / f"training_{name}_{stage}.json", "training", method, stage, 3.0)
        measurement(raw / f"inference_{name}.json", "inference", method, "inference", 1.0)
else:
    raise RuntimeError(f"Unexpected smoke command: {ARGS}")
