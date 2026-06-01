#!/usr/bin/env python3
"""Generate frozen steering, erasure, random-control, and side-effect responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
from tqdm import tqdm

from experiment_common import (
    REPO_ROOT,
    ensure_run_dirs,
    generate_response,
    load_processor_and_model,
    read_config,
    read_jsonl,
    row_messages,
    seed_everything,
)


def shard_for(value: str, num_shards: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def add_intervention(component: dict, candidates: dict, vectors) -> dict:
    layer = int(component["layer_index"])
    direction = component["direction"]
    sign = 1.0 if component["sign"] == "pos" else -1.0
    scale = float(candidates["layer_scales"][str(layer)][direction]["projection_std"])
    return {
        "mode": "add",
        "layer_index": layer,
        "vector": vectors[direction][layer],
        "magnitude": sign * float(component["alpha_std"]) * scale,
    }


def random_vector(shape, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=shape).astype(np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def condition_specs(config: dict, frozen: dict, candidates: dict, vectors) -> list[dict]:
    seed = int(config["seed"])
    selected = frozen["selected"]
    specs = [{"condition_id": "base", "kind": "base", "interventions": []}]
    for name in ["emotion_add", "emotion_subtract", "belief_add", "belief_subtract"]:
        component = selected[name]
        specs.append({"condition_id": name, "kind": "target_addition", "interventions": [add_intervention(component, candidates, vectors)], **component})
    composed = selected["emotion_add_belief_subtract"]
    specs.append(
        {
            "condition_id": "emotion_add_belief_subtract",
            "kind": "composed_addition",
            "interventions": [add_intervention(component, candidates, vectors) for component in composed["components"]],
            **composed,
        }
    )
    for name in ["erase_belief", "erase_emotion"]:
        erase = selected[name]
        specs.append(
            {
                "condition_id": name,
                "kind": "target_erasure",
                "interventions": [
                    {
                        "mode": "erase",
                        "layer_index": erase["layer_index"],
                        "vector": vectors[erase["direction"]][erase["layer_index"]],
                    }
                ],
                **erase,
            }
        )
    for match_name in ["emotion_add", "belief_subtract"]:
        component = selected[match_name]
        target = add_intervention(component, candidates, vectors)
        layer = int(component["layer_index"])
        vector = random_vector(vectors[component["direction"]][layer].shape, seed + 1000 + layer)
        specs.append(
            {
                "condition_id": f"random_add_match_{match_name}",
                "kind": "random_addition",
                "match_target": match_name,
                "layer_index": layer,
                "random_seed": seed + 1000 + layer,
                "interventions": [{**target, "vector": vector}],
            }
        )
    erase = selected["erase_belief"]
    layer = int(erase["layer_index"])
    specs.append(
        {
            "condition_id": "random_erase_match_belief",
            "kind": "random_erasure",
            "match_target": "erase_belief",
            "layer_index": layer,
            "random_seed": seed + 2000 + layer,
            "interventions": [
                {
                    "mode": "erase",
                    "layer_index": layer,
                    "vector": random_vector(vectors[erase["direction"]][layer].shape, seed + 2000 + layer),
                }
            ],
        }
    )
    return specs


def serializable_spec(spec: dict) -> dict:
    return {key: value for key, value in spec.items() if key != "interventions"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    config = read_config(args.config)
    seed_everything(int(config["seed"]) + args.shard_index)
    run_dir = ensure_run_dirs(config)
    prompts = [
        row
        for row in read_jsonl(REPO_ROOT / "data" / "generation_prompts_v1.jsonl")
        if row["evaluation_split"] != "calibration"
    ]
    candidates = json.loads((run_dir / "analysis" / "candidate_layers.json").read_text(encoding="utf-8"))
    frozen = json.loads((run_dir / "analysis" / "frozen_intervention_configs.json").read_text(encoding="utf-8"))
    vectors = np.load(run_dir / "vectors" / "diffmean_vectors_by_layer.npz")
    specs = condition_specs(config, frozen, candidates, vectors)

    output = run_dir / "responses" / f"frozen_tests_shard{args.shard_index:02d}.jsonl"
    completed = set()
    if output.exists():
        completed = {row["generation_id"] for row in read_jsonl(output)}
    jobs = []
    for prompt in prompts:
        for spec in specs:
            generation_id = f"{prompt['id']}::{spec['condition_id']}"
            if generation_id not in completed and shard_for(generation_id, args.num_shards) == args.shard_index:
                jobs.append((prompt, spec, generation_id))

    processor, model = load_processor_and_model(config, args.device)
    started = time.time()
    with output.open("a", encoding="utf-8") as f:
        for prompt, spec, generation_id in tqdm(jobs, desc=f"frozen shard {args.shard_index}"):
            item_started = time.time()
            response, token_meta = generate_response(
                processor,
                model,
                row_messages(prompt),
                interventions=spec["interventions"],
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                "generation_id": generation_id,
                "prompt_id": prompt["id"],
                "evaluation_split": prompt["evaluation_split"],
                "prompt_type": prompt["prompt_type"],
                "category": prompt["category"],
                "user_prompt": prompt["user_prompt"],
                "claim": prompt.get("claim"),
                "expected_answer": prompt.get("expected_answer"),
                "response": response,
                **serializable_spec(spec),
                **token_meta,
                "seconds": time.time() - item_started,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print(json.dumps({"shard": args.shard_index, "jobs_completed": len(jobs), "seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
