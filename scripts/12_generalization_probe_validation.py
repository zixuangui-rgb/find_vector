#!/usr/bin/env python3
"""Evaluate v1 probes on the held-out naturalized 2x2 set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from experiment_common import ensure_run_dirs, projection, read_config, write_csv


def target(meta: list[dict], concept: str) -> np.ndarray:
    return np.asarray([int(row[concept]) for row in meta], dtype=np.int64)


def metrics_from_probs(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, probs)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }


def main() -> None:
    config = read_config()
    run_dir = ensure_run_dirs(config)
    x = np.load(run_dir / "activations" / "residual_natural_generalization.npz")["response_mean"].astype(np.float32)
    meta = json.loads((run_dir / "activations" / "residual_natural_generalization.json").read_text(encoding="utf-8"))
    vectors = np.load(run_dir / "vectors" / "diffmean_vectors_by_layer.npz")
    layer_count = int(x.shape[1])
    rows = []
    for concept in ["emotion_validation", "belief_endorsement"]:
        y = target(meta, concept)
        for layer in range(1, layer_count):
            probe = joblib.load(run_dir / "probes" / "response_mean" / f"{concept}_layer{layer:02d}.joblib")
            probs = probe.predict_proba(x[:, layer, :])[:, 1]
            result = metrics_from_probs(y, probs)
            rows.append({"concept": concept, "layer_index": layer, "method": "linear_probe", **result})
            direction = "emotion" if concept == "emotion_validation" else "belief"
            scores = projection(x[:, layer, :], vectors[direction][layer])
            auc = float(roc_auc_score(y, scores))
            rows.append({"concept": concept, "layer_index": layer, "method": "diffmean_projection", "auc": max(auc, 1.0 - auc)})
    write_csv(run_dir / "analysis" / "generalization_probe_metrics.csv", rows)
    print(json.dumps({"rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
