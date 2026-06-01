#!/usr/bin/env python3
"""Merge activation shards for train/dev/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment_common import REPO_ROOT, ensure_run_dirs, read_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    config = read_config(args.config)
    run_dir = ensure_run_dirs(config)
    arrays_by_pooling = {
        "response_mean": [],
        "response_first": [],
        "response_last": [],
    }
    metadata = []
    for shard_index in range(args.num_shards):
        base = run_dir / "activations" / f"residual_{args.split}_shard{shard_index:02d}"
        shard = np.load(base.with_suffix(".npz"))
        for pooling in arrays_by_pooling:
            arrays_by_pooling[pooling].append(shard[pooling])
        metadata.extend(json.loads(base.with_suffix(".json").read_text(encoding="utf-8")))

    merged_by_pooling = {
        pooling: np.concatenate(arrays, axis=0)
        for pooling, arrays in arrays_by_pooling.items()
    }
    merged = merged_by_pooling["response_mean"]
    out_base = run_dir / "activations" / f"residual_{args.split}"
    np.savez_compressed(out_base.with_suffix(".npz"), activations=merged, **merged_by_pooling)
    write_json(out_base.with_suffix(".json"), metadata)
    write_json(
        run_dir / "analysis" / f"activation_shapes_{args.split}.json",
        {"split": args.split, "rows": len(metadata), "shape": list(merged.shape), "poolings": sorted(merged_by_pooling)},
    )
    print(json.dumps({"split": args.split, "rows": len(metadata), "shape": list(merged.shape)}, indent=2))


if __name__ == "__main__":
    main()
