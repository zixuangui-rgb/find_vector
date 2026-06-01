#!/usr/bin/env python3
"""Extract held-out naturalized 2x2 activations for probe generalization checks."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from tqdm import tqdm

from experiment_common import (
    REPO_ROOT,
    ensure_run_dirs,
    load_processor_and_model,
    read_config,
    read_jsonl,
    response_pooled_activations,
    seed_everything,
    shard_rows,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = read_config(args.config)
    seed_everything(int(config["seed"]) + args.shard_index)
    run_dir = ensure_run_dirs(config)
    rows = shard_rows(read_jsonl(REPO_ROOT / "data" / "2x2_items_natural_generalization_v1.jsonl"), args.shard_index, args.num_shards)
    processor, model = load_processor_and_model(config, args.device)

    arrays_by_pooling = {"response_mean": [], "response_first": [], "response_last": []}
    metadata = []
    started = time.time()
    for row in tqdm(rows, desc=f"generalization shard {args.shard_index}"):
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
        metadata.append({**row, **token_meta})

    stacked = {pooling: np.stack(arrays).astype(np.float16) for pooling, arrays in arrays_by_pooling.items()}
    out_base = run_dir / "activations" / f"residual_natural_generalization_shard{args.shard_index:02d}"
    np.savez_compressed(out_base.with_suffix(".npz"), activations=stacked["response_mean"], **stacked)
    write_json(out_base.with_suffix(".json"), metadata)
    write_json(
        run_dir / "logs" / f"extract_natural_generalization_shard{args.shard_index:02d}.json",
        {"rows": len(rows), "shape": list(stacked["response_mean"].shape), "seconds": time.time() - started},
    )
    print(json.dumps({"rows": len(rows), "shape": list(stacked["response_mean"].shape)}, indent=2))


if __name__ == "__main__":
    main()
