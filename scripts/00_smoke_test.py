#!/usr/bin/env python3
"""Verify Qwen3.5 model loading, text tokenization, and hidden-state extraction."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

from experiment_common import (
    REPO_ROOT,
    ensure_run_dirs,
    load_processor_and_model,
    read_config,
    read_jsonl,
    resolve_repo_path,
    response_mean_activations,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = read_config(args.config)
    run_dir = ensure_run_dirs(config)
    rows = read_jsonl(resolve_repo_path(config["paths"]["dataset_all"]))[: args.limit]

    processor, model = load_processor_and_model(config, args.device)
    records = []
    first_array = None
    for row in rows:
        array, meta = response_mean_activations(
            processor,
            model,
            row["user_prompt"],
            row["response"],
            device=args.device,
            max_input_tokens=int(config["model"]["max_input_tokens"]),
        )
        first_array = array if first_array is None else first_array
        records.append({"id": row["id"], "combo": row["combo"], **meta})

    output = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_type": type(model).__name__,
        "processor_type": type(processor).__name__,
        "records": records,
        "first_activation_shape": list(first_array.shape) if first_array is not None else None,
    }
    write_json(run_dir / "analysis" / "smoke_test.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

