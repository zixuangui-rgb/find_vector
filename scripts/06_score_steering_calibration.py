#!/usr/bin/env python3
"""Merge calibration shards and produce rule-scored condition effects."""

from __future__ import annotations

import json
from collections import defaultdict

from experiment_common import ensure_run_dirs, read_config, read_jsonl, write_csv, write_jsonl
from scoring_common import score_response


SCORE_FIELDS = [
    "emotion_validation_rule",
    "belief_endorsement_rule",
    "epistemic_calibration_rule",
    "harshness_rule",
    "response_chars",
]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    run_dir = ensure_run_dirs(read_config())
    rows_by_id = {}
    for path in sorted((run_dir / "responses").glob("steering_calibration_shard*.jsonl")):
        for row in read_jsonl(path):
            rows_by_id[row["generation_id"]] = row
    rows = sorted(rows_by_id.values(), key=lambda row: row["generation_id"])
    write_jsonl(run_dir / "responses" / "steering_calibration_merged.jsonl", rows)

    scored = []
    for row in rows:
        scored.append({**row, **score_response(row)})
    write_jsonl(run_dir / "scores" / "steering_calibration_rule_scores.jsonl", scored)

    baselines = {row["prompt_id"]: row for row in scored if row["condition_id"] == "base"}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        if row["prompt_id"] not in baselines:
            continue
        groups[row["condition_id"]].append(row)

    effect_rows = []
    for condition_id, values in groups.items():
        first = values[0]
        output = {
            "condition_id": condition_id,
            "kind": first["kind"],
            "direction": first.get("direction"),
            "layer_index": first.get("layer_index"),
            "sign": first.get("sign"),
            "alpha_std": first.get("alpha_std"),
            "token_position": first.get("token_position"),
            "n": len(values),
        }
        for field in SCORE_FIELDS:
            field_values = [float(row[field]) for row in values]
            deltas = [float(row[field]) - float(baselines[row["prompt_id"]][field]) for row in values]
            output[f"mean_{field}"] = mean(field_values)
            output[f"delta_{field}"] = mean(deltas)
        effect_rows.append(output)
    effect_rows.sort(key=lambda row: row["condition_id"])
    write_csv(run_dir / "analysis" / "steering_calibration_rule_effects.csv", effect_rows)
    print(json.dumps({"responses": len(rows), "conditions": len(effect_rows)}, indent=2))


if __name__ == "__main__":
    main()
