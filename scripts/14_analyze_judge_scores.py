#!/usr/bin/env python3
"""Summarize rubric-judge scores for frozen tests."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from experiment_common import ensure_run_dirs, read_config, read_jsonl, write_csv, write_jsonl


FIELDS = [
    "emotion_validation",
    "belief_endorsement",
    "epistemic_calibration",
    "gentle_correction",
    "naturalness",
    "harshness",
    "factual_correctness",
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


def numeric(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    run_dir = ensure_run_dirs(read_config())
    rows_by_id = {}
    for path in sorted((run_dir / "scores").glob("frozen_judge_scores_shard*.jsonl")):
        for row in read_jsonl(path):
            rows_by_id[row["generation_id"]] = row
    rows = sorted(rows_by_id.values(), key=lambda row: row["generation_id"])
    write_jsonl(run_dir / "scores" / "frozen_judge_scores_merged.jsonl", rows)

    baselines = {row["prompt_id"]: row for row in rows if row["condition_id"] == "base" and not row.get("parse_error")}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("parse_error") or row["prompt_id"] not in baselines:
            continue
        groups[(row["evaluation_split"], row["condition_id"])].append(row)

    effect_rows = []
    for (split, condition_id), values in sorted(groups.items()):
        output = {"evaluation_split": split, "condition_id": condition_id, "n": len(values)}
        for field in FIELDS:
            paired = []
            current_values = []
            for row in values:
                current = numeric(row.get(field))
                base = numeric(baselines[row["prompt_id"]].get(field))
                if current is None or base is None:
                    continue
                current_values.append(current)
                paired.append(current - base)
            if not paired:
                continue
            low, high = bootstrap_ci(paired)
            output[f"mean_{field}"] = mean(current_values)
            output[f"delta_{field}"] = mean(paired)
            output[f"delta_{field}_ci_low"] = low
            output[f"delta_{field}_ci_high"] = high
            output[f"delta_{field}_wilcoxon_p"] = safe_wilcoxon(paired)
        effect_rows.append(output)
    write_csv(run_dir / "analysis" / "frozen_judge_effects.csv", effect_rows)
    parse_errors = sum(1 for row in rows if row.get("parse_error"))
    print(json.dumps({"judge_rows": len(rows), "parse_errors": parse_errors, "effects": len(effect_rows)}, indent=2))


if __name__ == "__main__":
    main()
