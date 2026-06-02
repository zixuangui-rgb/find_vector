#!/usr/bin/env python3
"""Shared helpers for the pre-training vector discovery experiment."""

from __future__ import annotations

import csv
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_config(path: str | Path | None = None) -> dict[str, Any]:
    requested = path or os.environ.get("FIND_VECTOR_CONFIG")
    config_path = Path(requested) if requested else REPO_ROOT / "config" / "experiment_config.yaml"
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


def tokenize_chat(
    processor,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    template_kwargs = chat_template_kwargs or {}
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
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


def locate_subsequence(sequence: list[int], subsequence: list[int], start: int = 0) -> tuple[int, int] | None:
    if not subsequence:
        return None
    stop = len(sequence) - len(subsequence) + 1
    for index in range(max(0, start), max(0, stop)):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index, index + len(subsequence)
    return None


def response_pooled_activations(
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
    full_token_ids = full_inputs["input_ids"][0].tolist()
    response_token_ids = processor.tokenizer.encode(response, add_special_tokens=False)
    span = locate_subsequence(full_token_ids, response_token_ids, start=max(0, prefix_len - 2))
    if span is None:
        response_start = min(prefix_len, full_len - 1)
        response_end = full_len
        span_method = "prefix_fallback"
    else:
        response_start, response_end = span
        span_method = "exact_response_subsequence"
    model_inputs = move_inputs(full_inputs, device)
    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = get_hidden_states(outputs)
    pooled_mean = []
    pooled_first = []
    pooled_last = []
    for state in hidden_states:
        response_state = state[:, response_start:response_end, :]
        pooled_mean.append(response_state.mean(dim=1).squeeze(0).float().cpu().numpy())
        pooled_first.append(response_state[:, 0, :].squeeze(0).float().cpu().numpy())
        pooled_last.append(response_state[:, -1, :].squeeze(0).float().cpu().numpy())
    arrays = {
        "response_mean": np.stack(pooled_mean).astype(np.float16),
        "response_first": np.stack(pooled_first).astype(np.float16),
        "response_last": np.stack(pooled_last).astype(np.float16),
    }
    meta = {
        "prefix_tokens": prefix_len,
        "full_tokens": full_len,
        "response_start": response_start,
        "response_end": response_end,
        "response_tokens": max(1, response_end - response_start),
        "response_span_method": span_method,
        "hidden_state_count": len(hidden_states),
        "hidden_size": int(arrays["response_mean"].shape[-1]),
    }
    return arrays, meta


def response_mean_activations(
    processor,
    model,
    user_prompt: str,
    response: str,
    device: str = "cuda:0",
    max_input_tokens: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    arrays, meta = response_pooled_activations(
        processor,
        model,
        user_prompt,
        response,
        device=device,
        max_input_tokens=max_input_tokens,
    )
    return arrays["response_mean"], meta


def row_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if messages:
        return messages
    return text_messages(row["user_prompt"])


def nested_attr(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        current = getattr(current, part)
    return current


def decoder_layers(model) -> tuple[str, torch.nn.ModuleList]:
    candidates = [
        "model.language_model.layers",
        "language_model.model.layers",
        "model.layers",
        "language_model.layers",
    ]
    for path in candidates:
        try:
            layers = nested_attr(model, path)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return path, layers
    fallback = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList) and len(module) > 4
    ]
    if not fallback:
        raise RuntimeError("Could not locate decoder layer ModuleList")
    return max(fallback, key=lambda pair: len(pair[1]))


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


@contextmanager
def residual_interventions(model, interventions: list[dict[str, Any]]) -> Iterator[None]:
    """Apply residual additions or erasures at hidden-state layer indices."""
    if not interventions:
        yield
        return
    _, layers = decoder_layers(model)
    handles = []
    for intervention in interventions:
        layer_index = int(intervention["layer_index"])
        if layer_index < 1 or layer_index > len(layers):
            raise ValueError(f"layer_index={layer_index} must be in [1, {len(layers)}]")
        mode = intervention["mode"]
        vector = torch.as_tensor(intervention["vector"], dtype=torch.float32)
        vector = vector / vector.norm().clamp_min(1e-8)
        magnitude = float(intervention.get("magnitude", 1.0))
        token_position = intervention.get("token_position", "last")

        def hook(_module, _inputs, output, *, mode=mode, vector=vector, magnitude=magnitude, token_position=token_position):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            unit = vector.to(device=hidden.device, dtype=hidden.dtype)
            if token_position == "last":
                target = hidden[:, -1:, :]
            elif token_position == "all":
                target = hidden
            else:
                raise ValueError(f"Unsupported token_position: {token_position}")
            if mode == "add":
                updated = target + magnitude * unit.view(1, 1, -1)
            elif mode == "erase":
                projection_values = (target * unit.view(1, 1, -1)).sum(dim=-1, keepdim=True)
                updated = target - projection_values * unit.view(1, 1, -1)
            else:
                raise ValueError(f"Unsupported intervention mode: {mode}")
            hidden = hidden.clone()
            if token_position == "last":
                hidden[:, -1:, :] = updated
            else:
                hidden = updated
            return _replace_hidden(output, hidden)

        handles.append(layers[layer_index - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate_response(
    processor,
    model,
    messages: list[dict[str, Any]],
    interventions: list[dict[str, Any]] | None = None,
    device: str = "cuda:0",
    max_new_tokens: int = 128,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    template_kwargs = {"enable_thinking": False}
    if chat_template_kwargs:
        template_kwargs.update(chat_template_kwargs)
    inputs = tokenize_chat(
        processor,
        messages,
        add_generation_prompt=True,
        chat_template_kwargs=template_kwargs,
    )
    input_tokens = int(inputs["input_ids"].shape[-1])
    model_inputs = move_inputs(inputs, device)
    with residual_interventions(model, interventions or []):
        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    generated_ids = output_ids[0, input_tokens:]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, {"input_tokens": input_tokens, "generated_tokens": int(generated_ids.shape[-1])}
