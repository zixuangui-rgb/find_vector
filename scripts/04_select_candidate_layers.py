#!/usr/bin/env python3
"""Select residual-stream layers for causal steering calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiment_common import REPO_ROOT, ensure_run_dirs, projection, read_config, write_csv, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    config = read_config()
    run_dir = ensure_run_dirs(config)
    geometry_rows = read_csv(run_dir / "analysis" / "vector_geometry_by_layer.csv")
    probe_rows = read_csv(run_dir / "analysis" / "probe_metrics_by_layer.csv")
    cross_rows = read_csv(run_dir / "analysis" / "probe_cross_condition_metrics.csv")

    probes: dict[tuple[int, str], dict[str, str]] = {}
    for row in probe_rows:
        probes[(int(row["layer_index"]), row["concept"])] = row
    cross: dict[tuple[int, str], list[float]] = {}
    for row in cross_rows:
        cross.setdefault((int(row["layer_index"]), row["concept"]), []).append(float(row["auc"]))

    ranked = []
    for geometry in geometry_rows:
        layer = int(geometry["layer_index"])
        if layer == 0:
            continue
        emotion_probe = probes[(layer, "emotion_validation")]
        belief_probe = probes[(layer, "belief_endorsement")]
        emotion_dev = float(emotion_probe["dev_auc"])
        belief_dev = float(belief_probe["dev_auc"])
        cross_floor = min(
            min(cross[(layer, "emotion_validation")]),
            min(cross[(layer, "belief_endorsement")]),
        )
        cosine = abs(float(geometry["cos_emotion_belief"]))
        condition_stability = mean(
            [
                abs(float(geometry["cos_emotion_conditions"])),
                abs(float(geometry["cos_belief_conditions"])),
            ]
        )
        joint_score = (
            0.35 * mean([emotion_dev, belief_dev])
            + 0.25 * cross_floor
            + 0.20 * (1.0 - cosine)
            + 0.20 * condition_stability
        )
        ranked.append(
            {
                "layer_index": layer,
                "joint_score": joint_score,
                "emotion_dev_auc": emotion_dev,
                "belief_dev_auc": belief_dev,
                "cross_condition_auc_floor": cross_floor,
                "abs_cos_emotion_belief": cosine,
                "condition_stability": condition_stability,
            }
        )
    ranked.sort(key=lambda row: row["joint_score"], reverse=True)

    calibration_layers = int(config["steering"]["calibration_layers"])
    best_emotion = max(ranked, key=lambda row: row["emotion_dev_auc"])["layer_index"]
    best_belief = max(ranked, key=lambda row: row["belief_dev_auc"])["layer_index"]
    best_low_cos = max(
        ranked,
        key=lambda row: mean([row["emotion_dev_auc"], row["belief_dev_auc"]]) - row["abs_cos_emotion_belief"],
    )["layer_index"]
    layer_count = max(row["layer_index"] for row in ranked) + 1
    thirds = [
        [row for row in ranked if 1 <= row["layer_index"] <= max(2, layer_count // 4)],
        [row for row in ranked if max(3, layer_count // 4 + 1) <= row["layer_index"] <= max(4, layer_count // 2)],
        [row for row in ranked if row["layer_index"] > max(4, layer_count // 2)],
    ]
    stratified = [group[0]["layer_index"] for group in thirds if group]
    preferred = [ranked[0]["layer_index"], best_emotion, best_belief, best_low_cos, *stratified]
    preferred.extend(row["layer_index"] for row in ranked)
    selected = []
    for layer in preferred:
        if layer not in selected:
            selected.append(layer)
        if len(selected) == calibration_layers:
            break

    train = np.load(run_dir / "activations" / "residual_train.npz")["response_mean"].astype(np.float32)
    vector_file = run_dir / "vectors" / "diffmean_vectors_by_layer.npz"
    vectors = np.load(vector_file)
    layer_scales = {}
    for layer in selected:
        layer_scales[str(layer)] = {}
        for direction in ["emotion", "belief", "emotion_orth", "belief_orth"]:
            values = projection(train[:, layer, :], vectors[direction][layer])
            layer_scales[str(layer)][direction] = {
                "projection_std": float(values.std(ddof=1)),
                "projection_mean": float(values.mean()),
            }

    output = {
        "selected_layers": selected,
        "best_emotion_layer": best_emotion,
        "best_belief_layer": best_belief,
        "best_low_cosine_layer": best_low_cos,
        "vector_file": str(vector_file),
        "layer_scales": layer_scales,
        "ranked_layers": ranked,
    }
    write_json(run_dir / "analysis" / "candidate_layers.json", output)
    write_csv(run_dir / "analysis" / "candidate_layers.csv", ranked)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
