#!/usr/bin/env python3
"""Train per-layer linear probes and evaluate cross-condition generalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_common import REPO_ROOT, ensure_run_dirs, read_config, write_csv, write_json


def artifact_name(stem: str, pooling: str, suffix: str) -> str:
    return f"{stem}{suffix}" if pooling == "response_mean" else f"{stem}_{pooling}{suffix}"


def load_split(run_dir: Path, split: str, pooling: str) -> tuple[np.ndarray, list[dict]]:
    activations = np.load(run_dir / "activations" / f"residual_{split}.npz")[pooling].astype(np.float32)
    metadata = json.loads((run_dir / "activations" / f"residual_{split}.json").read_text(encoding="utf-8"))
    return activations, metadata


def target(metadata: list[dict], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in metadata], dtype=np.int64)


def metrics(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probs = model.predict_proba(x)[:, 1]
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, probs)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }


def evaluate_conditions(model, x: np.ndarray, metadata: list[dict], concept: str) -> list[dict]:
    rows = []
    condition_field = "belief_endorsement" if concept == "emotion_validation" else "emotion_validation"
    y = target(metadata, concept)
    for condition in [0, 1]:
        mask = np.asarray([int(row[condition_field]) == condition for row in metadata])
        result = metrics(model, x[mask], y[mask])
        rows.append({"concept": concept, "condition_field": condition_field, "condition_value": condition, "n": int(mask.sum()), **result})
    return rows


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
    layer_count = int(train_x.shape[1])
    c_values = [float(value) for value in config["probe"]["c_values"]]
    max_iter = int(config["probe"]["max_iter"])

    metric_rows = []
    cross_rows = []
    best_by_concept = {}
    for concept in ["emotion_validation", "belief_endorsement"]:
        y_train = target(train_meta, concept)
        y_dev = target(dev_meta, concept)
        y_test = target(test_meta, concept)
        best_row = None
        for layer in range(1, layer_count):
            estimator = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=int(config["seed"]))),
                ]
            )
            search = GridSearchCV(estimator, {"clf__C": c_values}, scoring="roc_auc", cv=5, n_jobs=-1)
            search.fit(train_x[:, layer, :], y_train)
            dev_metrics = metrics(search.best_estimator_, dev_x[:, layer, :], y_dev)
            test_metrics = metrics(search.best_estimator_, test_x[:, layer, :], y_test)
            row = {
                "concept": concept,
                "layer_index": layer,
                "best_c": float(search.best_params_["clf__C"]),
                "dev_auc": dev_metrics["auc"],
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_f1": dev_metrics["f1"],
                "test_auc": test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
            }
            metric_rows.append(row)
            probe_dir = run_dir / "probes" / args.pooling
            probe_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(search.best_estimator_, probe_dir / f"{concept}_layer{layer:02d}.joblib")
            for cross in evaluate_conditions(search.best_estimator_, test_x[:, layer, :], test_meta, concept):
                cross_rows.append({"layer_index": layer, **cross})
            if best_row is None or row["dev_auc"] > best_row["dev_auc"]:
                best_row = row
        best_by_concept[concept] = best_row

    write_csv(run_dir / "analysis" / artifact_name("probe_metrics_by_layer", args.pooling, ".csv"), metric_rows)
    write_csv(run_dir / "analysis" / artifact_name("probe_cross_condition_metrics", args.pooling, ".csv"), cross_rows)
    write_json(run_dir / "analysis" / artifact_name("probe_selection_summary", args.pooling, ".json"), {"pooling": args.pooling, **best_by_concept})
    print(json.dumps(best_by_concept, indent=2))


if __name__ == "__main__":
    main()
