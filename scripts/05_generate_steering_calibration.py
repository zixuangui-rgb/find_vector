#!/usr/bin/env python3
"""Generate calibration responses under residual-stream steering interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

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


def condition_specs(config: dict, candidates: dict, vectors) -> list[dict]:
    specs = [{"condition_id": "base", "interventions": [], "kind": "base"}]
    alphas = [float(value) for value in config["steering"]["alphas_std"]]
    seed = int(config["seed"])
    for layer in candidates["selected_layers"]:
        scales = candidates["layer_scales"][str(layer)]
        for direction in ["emotion", "belief", "emotion_orth", "belief_orth"]:
            for sign_name, sign in [("pos", 1.0), ("neg", -1.0)]:
                for alpha in alphas:
                    magnitude = sign * alpha * float(scales[direction]["projection_std"])
                    specs.append(
                        {
                            "condition_id": f"add_{direction}_L{layer}_{sign_name}_a{alpha:g}",
                            "kind": "target_addition",
                            "direction": direction,
                            "layer_index": layer,
                            "sign": sign_name,
                            "alpha_std": alpha,
                            "magnitude": magnitude,
                            "interventions": [
                                {
                                    "mode": "add",
                                    "layer_index": layer,
                                    "vector": vectors[direction][layer],
                                    "magnitude": magnitude,
                                }
                            ],
                        }
                    )
        rng = np.random.default_rng(seed + layer)
        random_vector = rng.normal(size=vectors["emotion"][layer].shape).astype(np.float32)
        random_vector /= max(float(np.linalg.norm(random_vector)), 1e-8)
        scale = float(np.mean([scales["emotion"]["projection_std"], scales["belief"]["projection_std"]]))
        for sign_name, sign in [("pos", 1.0), ("neg", -1.0)]:
            for alpha in [1.0, 2.0]:
                magnitude = sign * alpha * scale
                specs.append(
                    {
                        "condition_id": f"add_random_L{layer}_{sign_name}_a{alpha:g}",
                        "kind": "random_addition",
                        "direction": "random",
                        "layer_index": layer,
                        "sign": sign_name,
                        "alpha_std": alpha,
                        "magnitude": magnitude,
                        "random_seed": seed + layer,
                        "interventions": [
                            {
                                "mode": "add",
                                "layer_index": layer,
                                "vector": random_vector,
                                "magnitude": magnitude,
                            }
                        ],
                    }
                )
    return specs


def serializable_spec(spec: dict) -> dict:
    return {key: value for key, value in spec.items() if key != "interventions"}


def shard_for(value: str, num_shards: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    config = read_config(args.config)
    seed_everything(int(config["seed"]) + args.shard_index)
    run_dir = ensure_run_dirs(config)
    prompts = [
        row
        for row in read_jsonl(REPO_ROOT / "data" / "generation_prompts_v1.jsonl")
        if row["evaluation_split"] == "calibration"
    ]
    candidates = json.loads((run_dir / "analysis" / "candidate_layers.json").read_text(encoding="utf-8"))
    vectors = np.load(run_dir / "vectors" / "diffmean_vectors_by_layer.npz")
    specs = condition_specs(config, candidates, vectors)

    output = run_dir / "responses" / f"steering_calibration_shard{args.shard_index:02d}.jsonl"
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
        for prompt, spec, generation_id in tqdm(jobs, desc=f"calibration shard {args.shard_index}"):
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
