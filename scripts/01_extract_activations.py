#!/usr/bin/env python3
"""Extract response-pooled hidden states for one data shard."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiment_common import (
    REPO_ROOT,
    ensure_run_dirs,
    load_processor_and_model,
    read_config,
    read_jsonl,
    resolve_repo_path,
    response_pooled_activations,
    seed_everything,
    shard_rows,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = read_config(args.config)
    seed_everything(int(config["seed"]) + args.shard_index)
    run_dir = ensure_run_dirs(config)
    source = resolve_repo_path(config["paths"][f"dataset_{args.split}"])
    rows = shard_rows(read_jsonl(source), args.shard_index, args.num_shards)
    processor, model = load_processor_and_model(config, args.device)

    arrays_by_pooling = {
        "response_mean": [],
        "response_first": [],
        "response_last": [],
    }
    metadata = []
    started = time.time()
    for row in tqdm(rows, desc=f"extract {args.split} shard {args.shard_index}"):
        pooled_arrays, token_meta = response_pooled_activations(
            processor,
            model,
            row["user_prompt"],
            row["response"],
            device=args.device,
            max_input_tokens=int(config["model"]["max_input_tokens"]),
        )
        for pooling, array in pooled_arrays.items():
            arrays_by_pooling[pooling].append(array)
        metadata.append(
            {
                "id": row["id"],
                "scenario_id": row["scenario_id"],
                "split": row["split"],
                "category": row["category"],
                "combo": row["combo"],
                "emotion_validation": int(row["emotion_validation"]),
                "belief_endorsement": int(row["belief_endorsement"]),
                "user_prompt": row["user_prompt"],
                "response": row["response"],
                **token_meta,
            }
        )

    stacked = {
        pooling: np.stack(arrays).astype(np.float16)
        for pooling, arrays in arrays_by_pooling.items()
    }
    activations = stacked["response_mean"]
    out_base = run_dir / "activations" / f"residual_{args.split}_shard{args.shard_index:02d}"
    np.savez_compressed(out_base.with_suffix(".npz"), activations=activations, **stacked)
    write_json(out_base.with_suffix(".json"), metadata)
    write_json(
        run_dir / "logs" / f"extract_{args.split}_shard{args.shard_index:02d}.json",
        {
            "split": args.split,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "rows": len(rows),
            "shape": list(activations.shape),
            "poolings": sorted(stacked),
            "seconds": time.time() - started,
        },
    )
    print(json.dumps({"rows": len(rows), "shape": list(activations.shape)}, indent=2))


if __name__ == "__main__":
    main()
