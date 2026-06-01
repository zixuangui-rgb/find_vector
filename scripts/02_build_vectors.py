#!/usr/bin/env python3
"""Build DiffMean, conditional, and orthogonalized vectors for every layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_common import (
    REPO_ROOT,
    ensure_run_dirs,
    normalized,
    orthogonalize,
    projection,
    read_config,
    write_csv,
    write_json,
)


def artifact_name(stem: str, pooling: str, suffix: str) -> str:
    return f"{stem}{suffix}" if pooling == "response_mean" else f"{stem}_{pooling}{suffix}"


def load_split(run_dir: Path, split: str, pooling: str) -> tuple[np.ndarray, list[dict]]:
    activations = np.load(run_dir / "activations" / f"residual_{split}.npz")[pooling].astype(np.float32)
    metadata = json.loads((run_dir / "activations" / f"residual_{split}.json").read_text(encoding="utf-8"))
    return activations, metadata


def labels(metadata: list[dict], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in metadata], dtype=np.int64)


def mean_difference(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return values[target == 1].mean(axis=0) - values[target == 0].mean(axis=0)


def masked_difference(values: np.ndarray, metadata: list[dict], positive_combo: str, negative_combo: str) -> np.ndarray:
    positive = np.asarray([row["combo"] == positive_combo for row in metadata])
    negative = np.asarray([row["combo"] == negative_combo for row in metadata])
    return values[positive].mean(axis=0) - values[negative].mean(axis=0)


def auc_for_direction(values: np.ndarray, target: np.ndarray, direction: np.ndarray) -> float:
    scores = projection(values, direction)
    auc = float(roc_auc_score(target, scores))
    return max(auc, 1.0 - auc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--pooling", choices=["response_mean", "response_first", "response_last"], default="response_mean")
    args = parser.parse_args()

    config = read_config(args.config)
    run_dir = ensure_run_dirs(config)
    train_x, train_meta = load_split(run_dir, "train", args.pooling)
    dev_x, dev_meta = load_split(run_dir, "dev", args.pooling)
    test_x, test_meta = load_split(run_dir, "test", args.pooling)
    emotion_train = labels(train_meta, "emotion_validation")
    belief_train = labels(train_meta, "belief_endorsement")
    emotion_dev = labels(dev_meta, "emotion_validation")
    belief_dev = labels(dev_meta, "belief_endorsement")
    emotion_test = labels(test_meta, "emotion_validation")
    belief_test = labels(test_meta, "belief_endorsement")

    layer_count = int(train_x.shape[1])
    vectors: dict[str, list[np.ndarray]] = {
        "emotion": [],
        "belief": [],
        "emotion_given_low_belief": [],
        "emotion_given_high_belief": [],
        "belief_given_low_emotion": [],
        "belief_given_high_emotion": [],
        "emotion_orth": [],
        "belief_orth": [],
    }
    rows = []
    for layer in range(layer_count):
        x = train_x[:, layer, :]
        v_emotion = normalized(mean_difference(x, emotion_train)).astype(np.float32)
        v_belief = normalized(mean_difference(x, belief_train)).astype(np.float32)
        conditional = {
            "emotion_given_low_belief": normalized(masked_difference(x, train_meta, "high_emotion_low_belief", "low_emotion_low_belief")).astype(np.float32),
            "emotion_given_high_belief": normalized(masked_difference(x, train_meta, "high_emotion_high_belief", "low_emotion_high_belief")).astype(np.float32),
            "belief_given_low_emotion": normalized(masked_difference(x, train_meta, "low_emotion_high_belief", "low_emotion_low_belief")).astype(np.float32),
            "belief_given_high_emotion": normalized(masked_difference(x, train_meta, "high_emotion_high_belief", "high_emotion_low_belief")).astype(np.float32),
        }
        v_emotion_orth = orthogonalize(v_emotion, v_belief)
        v_belief_orth = orthogonalize(v_belief, v_emotion)
        vectors["emotion"].append(v_emotion)
        vectors["belief"].append(v_belief)
        vectors["emotion_orth"].append(v_emotion_orth)
        vectors["belief_orth"].append(v_belief_orth)
        for key, value in conditional.items():
            vectors[key].append(value)

        rows.append(
            {
                "layer_index": layer,
                "is_embedding_state": int(layer == 0),
                "cos_emotion_belief": float(v_emotion @ v_belief),
                "cos_emotion_conditions": float(conditional["emotion_given_low_belief"] @ conditional["emotion_given_high_belief"]),
                "cos_belief_conditions": float(conditional["belief_given_low_emotion"] @ conditional["belief_given_high_emotion"]),
                "dev_auc_emotion_diffmean": auc_for_direction(dev_x[:, layer, :], emotion_dev, v_emotion),
                "dev_auc_belief_diffmean": auc_for_direction(dev_x[:, layer, :], belief_dev, v_belief),
                "test_auc_emotion_diffmean": auc_for_direction(test_x[:, layer, :], emotion_test, v_emotion),
                "test_auc_belief_diffmean": auc_for_direction(test_x[:, layer, :], belief_test, v_belief),
                "test_auc_emotion_orth": auc_for_direction(test_x[:, layer, :], emotion_test, v_emotion_orth),
                "test_auc_belief_orth": auc_for_direction(test_x[:, layer, :], belief_test, v_belief_orth),
            }
        )

    np.savez_compressed(
        run_dir / "vectors" / artifact_name("diffmean_vectors_by_layer", args.pooling, ".npz"),
        **{key: np.stack(value) for key, value in vectors.items()},
    )
    write_csv(run_dir / "analysis" / artifact_name("vector_geometry_by_layer", args.pooling, ".csv"), rows)
    best_emotion = max(rows[1:], key=lambda row: row["test_auc_emotion_diffmean"])
    best_belief = max(rows[1:], key=lambda row: row["test_auc_belief_diffmean"])
    summary = {"pooling": args.pooling, "best_emotion_layer": best_emotion, "best_belief_layer": best_belief, "layer_count_including_embedding": layer_count}
    write_json(run_dir / "analysis" / artifact_name("vector_build_summary", args.pooling, ".json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
