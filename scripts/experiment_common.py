#!/usr/bin/env python3
"""Shared helpers for the pre-training vector discovery experiment."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else REPO_ROOT / "config" / "experiment_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_run_dirs(config: dict[str, Any]) -> Path:
    run_dir = Path(config["paths"]["run_dir"])
    for name in [
        "activations",
        "analysis",
        "figures",
        "logs",
        "probes",
        "responses",
        "scores",
        "vectors",
    ]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = float(np.linalg.norm(vector))
    if denom < eps:
        return np.zeros_like(vector)
    return vector / denom


def projection(values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    unit = normalized(direction.astype(np.float64))
    return values.astype(np.float64) @ unit


def orthogonalize(vector: np.ndarray, against: np.ndarray) -> np.ndarray:
    against_unit = normalized(against.astype(np.float64))
    cleaned = vector.astype(np.float64) - float(vector.astype(np.float64) @ against_unit) * against_unit
    return normalized(cleaned).astype(np.float32)


def shard_rows(rows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def text_messages(user_prompt: str, response: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


def load_processor_and_model(config: dict[str, Any], device: str = "cuda:0"):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = config["model"]["local_path"]
    dtype = torch_dtype(config["model"]["dtype"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return processor, model


def tokenize_chat(processor, messages: list[dict[str, Any]], add_generation_prompt: bool) -> dict[str, torch.Tensor]:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def get_hidden_states(outputs) -> tuple[torch.Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return tuple(hidden_states)
    language_model_output = getattr(outputs, "language_model_output", None)
    if language_model_output is not None and getattr(language_model_output, "hidden_states", None) is not None:
        return tuple(language_model_output.hidden_states)
    raise RuntimeError(f"Could not find hidden_states in output type {type(outputs)}")


def move_inputs(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def response_mean_activations(
    processor,
    model,
    user_prompt: str,
    response: str,
    device: str = "cuda:0",
    max_input_tokens: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    prefix_inputs = tokenize_chat(processor, text_messages(user_prompt), add_generation_prompt=True)
    full_inputs = tokenize_chat(processor, text_messages(user_prompt, response), add_generation_prompt=False)
    prefix_len = int(prefix_inputs["input_ids"].shape[-1])
    full_len = int(full_inputs["input_ids"].shape[-1])
    if full_len > max_input_tokens:
        raise ValueError(f"Tokenized transcript length {full_len} exceeds max_input_tokens={max_input_tokens}")
    response_start = min(prefix_len, full_len - 1)
    model_inputs = move_inputs(full_inputs, device)
    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = get_hidden_states(outputs)
    pooled = []
    for state in hidden_states:
        span = state[:, response_start:full_len, :]
        pooled.append(span.mean(dim=1).squeeze(0).float().cpu().numpy())
    array = np.stack(pooled).astype(np.float16)
    meta = {
        "prefix_tokens": prefix_len,
        "full_tokens": full_len,
        "response_tokens": max(1, full_len - response_start),
        "hidden_state_count": len(hidden_states),
        "hidden_size": int(array.shape[-1]),
    }
    return array, meta

