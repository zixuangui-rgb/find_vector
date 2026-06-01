#!/usr/bin/env python3
"""Merge frozen test shards and summarize transparent rule scores."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from experiment_common import ensure_run_dirs, read_config, read_jsonl, write_csv, write_jsonl
from scoring_common import score_response


SCORE_FIELDS = [
    "emotion_validation_rule",
    "belief_endorsement_rule",
    "epistemic_calibration_rule",
    "harshness_rule",
    "response_chars",
    "factual_correctness_rule",
]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def bootstrap_ci(values: list[float], seed: int = 20260601, n_boot: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def safe_wilcoxon(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2 or np.allclose(arr, 0):
        return None
    try:
        return float(wilcoxon(arr).pvalue)
    except ValueError:
        return None


def main() -> None:
    run_dir = ensure_run_dirs(read_config())
    rows_by_id = {}
    for path in sorted((run_dir / "responses").glob("frozen_tests_shard*.jsonl")):
        for row in read_jsonl(path):
            rows_by_id[row["generation_id"]] = row
    rows = sorted(rows_by_id.values(), key=lambda row: row["generation_id"])
    write_jsonl(run_dir / "responses" / "frozen_tests_merged.jsonl", rows)

    scored = []
    for row in rows:
        scored.append({**row, **score_response(row)})
    write_jsonl(run_dir / "scores" / "frozen_rule_scores.jsonl", scored)

    baselines = {row["prompt_id"]: row for row in scored if row["condition_id"] == "base"}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored:
        if row["prompt_id"] not in baselines:
            continue
        groups[(row["evaluation_split"], row["condition_id"])].append(row)

    effect_rows = []
    for (split, condition_id), values in sorted(groups.items()):
        first = values[0]
        output = {
            "evaluation_split": split,
            "condition_id": condition_id,
            "kind": first["kind"],
            "n": len(values),
        }
        for field in SCORE_FIELDS:
            field_values = [row[field] for row in values if row[field] is not None]
            baseline_values = [
                baselines[row["prompt_id"]][field]
                for row in values
                if row[field] is not None and baselines[row["prompt_id"]][field] is not None
            ]
            if not field_values or len(field_values) != len(baseline_values):
                continue
            deltas = [float(value) - float(base) for value, base in zip(field_values, baseline_values)]
            low, high = bootstrap_ci(deltas)
            output[f"mean_{field}"] = mean([float(value) for value in field_values])
            output[f"delta_{field}"] = mean(deltas)
            output[f"delta_{field}_ci_low"] = low
            output[f"delta_{field}_ci_high"] = high
            output[f"delta_{field}_wilcoxon_p"] = safe_wilcoxon(deltas)
        effect_rows.append(output)
    write_csv(run_dir / "analysis" / "frozen_rule_effects.csv", effect_rows)
    print(json.dumps({"responses": len(rows), "effects": len(effect_rows)}, indent=2))


if __name__ == "__main__":
    main()
