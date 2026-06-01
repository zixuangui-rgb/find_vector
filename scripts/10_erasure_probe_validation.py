#!/usr/bin/env python3
"""Representation-level erasure tests for separability of the two directions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from experiment_common import ensure_run_dirs, normalized, read_config, write_csv


def load_split(run_dir: Path, split: str) -> tuple[np.ndarray, list[dict]]:
    x = np.load(run_dir / "activations" / f"residual_{split}.npz")["response_mean"].astype(np.float32)
    meta = json.loads((run_dir / "activations" / f"residual_{split}.json").read_text(encoding="utf-8"))
    return x, meta


def target(meta: list[dict], concept: str) -> np.ndarray:
    return np.asarray([int(row[concept]) for row in meta], dtype=np.int64)


def metrics(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probs = model.predict_proba(x)[:, 1]
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, probs)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }


def erase_direction(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    unit = normalized(direction.astype(np.float64)).astype(np.float32)
    return x - (x @ unit)[:, None] * unit[None, :]


def inlp_erase(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, iterations: int, seed: int) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    z_train = (train_x - mean) / std
    z_test = (test_x - mean) / std
    for index in range(iterations):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed + index)
        clf.fit(z_train, train_y)
        direction = normalized(clf.coef_.ravel().astype(np.float64)).astype(np.float32)
        z_train = erase_direction(z_train, direction)
        z_test = erase_direction(z_test, direction)
    return (z_test * std + mean).astype(np.float32)


def main() -> None:
    config = read_config()
    run_dir = ensure_run_dirs(config)
    train_x, train_meta = load_split(run_dir, "train")
    test_x, test_meta = load_split(run_dir, "test")
    vectors = np.load(run_dir / "vectors" / "diffmean_vectors_by_layer.npz")
    layer_count = int(test_x.shape[1])
    y = {
        "emotion_validation": target(test_meta, "emotion_validation"),
        "belief_endorsement": target(test_meta, "belief_endorsement"),
    }
    y_train = {
        "emotion_validation": target(train_meta, "emotion_validation"),
        "belief_endorsement": target(train_meta, "belief_endorsement"),
    }
    rows = []
    raw_metrics: dict[tuple[int, str], dict[str, float]] = {}
    for layer in range(1, layer_count):
        probes = {
            "emotion_validation": joblib.load(run_dir / "probes" / "response_mean" / f"emotion_validation_layer{layer:02d}.joblib"),
            "belief_endorsement": joblib.load(run_dir / "probes" / "response_mean" / f"belief_endorsement_layer{layer:02d}.joblib"),
        }
        for concept, model in probes.items():
            result = metrics(model, test_x[:, layer, :], y[concept])
            raw_metrics[(layer, concept)] = result
            rows.append({"layer_index": layer, "erasure_method": "none", "erased_direction": "none", "target_probe": concept, **result, "delta_auc_vs_raw": 0.0})

        erased_sets = {
            "single_emotion": erase_direction(test_x[:, layer, :], vectors["emotion"][layer]),
            "single_belief": erase_direction(test_x[:, layer, :], vectors["belief"][layer]),
            "single_emotion_orth": erase_direction(test_x[:, layer, :], vectors["emotion_orth"][layer]),
            "single_belief_orth": erase_direction(test_x[:, layer, :], vectors["belief_orth"][layer]),
            "inlp_emotion_5": inlp_erase(train_x[:, layer, :], y_train["emotion_validation"], test_x[:, layer, :], 5, int(config["seed"])),
            "inlp_belief_5": inlp_erase(train_x[:, layer, :], y_train["belief_endorsement"], test_x[:, layer, :], 5, int(config["seed"]) + 100),
        }
        for erased_name, erased_x in erased_sets.items():
            for concept, model in probes.items():
                result = metrics(model, erased_x, y[concept])
                rows.append(
                    {
                        "layer_index": layer,
                        "erasure_method": "inlp" if erased_name.startswith("inlp") else "single_projection",
                        "erased_direction": erased_name,
                        "target_probe": concept,
                        **result,
                        "delta_auc_vs_raw": result["auc"] - raw_metrics[(layer, concept)]["auc"],
                    }
                )
    write_csv(run_dir / "analysis" / "erasure_probe_metrics.csv", rows)
    print(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
