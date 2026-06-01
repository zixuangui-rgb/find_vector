#!/usr/bin/env python3
"""Select steering settings on calibration prompts and freeze them for final tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiment_common import ensure_run_dirs, read_config, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def value(row: dict[str, str], field: str) -> float:
    return float(row[field])


def as_addition(row: dict[str, str], name: str) -> dict:
    return {
        "name": name,
        "mode": "add",
        "direction": row["direction"],
        "layer_index": int(row["layer_index"]),
        "sign": row["sign"],
        "alpha_std": float(row["alpha_std"]),
    }


def select(rows: list[dict[str, str]], directions: set[str], sign: str, objective) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row["kind"] == "target_addition" and row["direction"] in directions and row["sign"] == sign
    ]
    if not candidates:
        raise RuntimeError(f"No candidates for directions={directions}, sign={sign}")
    return max(candidates, key=objective)


def main() -> None:
    run_dir = ensure_run_dirs(read_config())
    rows = read_csv(run_dir / "analysis" / "steering_calibration_rule_effects.csv")
    emotion = select(
        rows,
        {"emotion", "emotion_orth"},
        "pos",
        lambda row: value(row, "delta_emotion_validation_rule")
        - 1.25 * max(0.0, value(row, "delta_belief_endorsement_rule"))
        - 0.05 * max(0.0, value(row, "delta_harshness_rule")),
    )
    belief_pos = select(
        rows,
        {"belief", "belief_orth"},
        "pos",
        lambda row: value(row, "delta_belief_endorsement_rule")
        - 0.20 * max(0.0, -value(row, "delta_emotion_validation_rule")),
    )
    belief_neg = select(
        rows,
        {"belief", "belief_orth"},
        "neg",
        lambda row: -value(row, "delta_belief_endorsement_rule")
        - 0.75 * max(0.0, -value(row, "delta_emotion_validation_rule"))
        - 0.10 * max(0.0, value(row, "delta_harshness_rule")),
    )
    emotion_neg = select(
        rows,
        {"emotion", "emotion_orth"},
        "neg",
        lambda row: -value(row, "delta_emotion_validation_rule"),
    )
    selections = {
        "emotion_add": as_addition(emotion, "emotion_add"),
        "emotion_subtract": as_addition(emotion_neg, "emotion_subtract"),
        "belief_add": as_addition(belief_pos, "belief_add"),
        "belief_subtract": as_addition(belief_neg, "belief_subtract"),
    }
    selections["emotion_add_belief_subtract"] = {
        "name": "emotion_add_belief_subtract",
        "mode": "composed_add",
        "components": [selections["emotion_add"], selections["belief_subtract"]],
    }
    selections["erase_belief"] = {
        "name": "erase_belief",
        "mode": "erase",
        "direction": selections["belief_subtract"]["direction"],
        "layer_index": selections["belief_subtract"]["layer_index"],
    }
    selections["erase_emotion"] = {
        "name": "erase_emotion",
        "mode": "erase",
        "direction": selections["emotion_add"]["direction"],
        "layer_index": selections["emotion_add"]["layer_index"],
    }
    output = {
        "selection_source": "calibration prompts with transparent lexical heuristic; frozen before final evaluation",
        "selected": selections,
        "calibration_effect_rows": {
            "emotion_add": emotion,
            "emotion_subtract": emotion_neg,
            "belief_add": belief_pos,
            "belief_subtract": belief_neg,
        },
    }
    write_json(run_dir / "analysis" / "frozen_intervention_configs.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
