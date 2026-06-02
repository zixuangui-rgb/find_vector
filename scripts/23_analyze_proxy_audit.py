#!/usr/bin/env python3
"""Merge and summarize blinded pairwise proxy-audit scores."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np

from experiment_common import ensure_run_dirs, read_config, read_jsonl, write_csv, write_json, write_jsonl


FIELDS = [
    "emotion_validation",
    "belief_endorsement",
    "epistemic_calibration",
    "naturalness",
    "harshness",
    "factual_correctness",
]


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def condition_delta(row: dict[str, Any], field: str) -> float | None:
    raw = numeric(row.get(f"a_minus_b_{field}"))
    if raw is None:
        return None
    return raw if row.get("condition_is_a") else -raw


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def bootstrap_ci(values: list[float], seed: int = 20260602, n_boot: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("parse_error"):
            continue
        groups[tuple(row.get(field) for field in group_fields)].append(row)

    summaries = []
    for key, values in sorted(groups.items(), key=lambda pair: tuple(str(part) for part in pair[0])):
        output = {field: value for field, value in zip(group_fields, key)}
        output["n"] = len(values)
        for field in FIELDS:
            deltas = [condition_delta(row, field) for row in values]
            deltas = [value for value in deltas if value is not None]
            if not deltas:
                continue
            low, high = bootstrap_ci(deltas)
            output[f"condition_vs_base_{field}"] = mean(deltas)
            output[f"condition_vs_base_{field}_ci_low"] = low
            output[f"condition_vs_base_{field}_ci_high"] = high
        summaries.append(output)
    return summaries


def main() -> None:
    config = read_config()
    run_dir = ensure_run_dirs(config)
    rows_by_id = {}
    for path in sorted((run_dir / "scores").glob("proxy_pairwise_audit_shard*.jsonl")):
        for row in read_jsonl(path):
            rows_by_id[row["audit_id"]] = row
    rows = sorted(rows_by_id.values(), key=lambda row: row["audit_id"])
    write_jsonl(run_dir / "scores" / "proxy_pairwise_audit_merged.jsonl", rows)

    effects = summarize(rows, ["condition_id", "evaluation_split"])
    by_group = summarize(rows, ["condition_id", "benchmark_group"])
    by_condition = summarize(rows, ["condition_id"])
    write_csv(run_dir / "analysis" / "proxy_pairwise_audit_effects.csv", effects)
    write_csv(run_dir / "analysis" / "proxy_pairwise_audit_by_benchmark.csv", by_group)
    write_csv(run_dir / "analysis" / "proxy_pairwise_audit_by_condition.csv", by_condition)

    parse_errors = sum(1 for row in rows if row.get("parse_error"))
    summary = {
        "audit_rows": len(rows),
        "parse_errors": parse_errors,
        "parse_error_rate": parse_errors / max(1, len(rows)),
        "conditions": sorted({row.get("condition_id") for row in rows}),
    }
    write_json(run_dir / "analysis" / "proxy_pairwise_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
