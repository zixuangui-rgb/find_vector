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
        "token_position": row.get("token_position") or "last",
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


def select_multi(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if row["kind"] == "multi_layer_composed_addition"]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: value(row, "delta_emotion_validation_rule")
        - value(row, "delta_belief_endorsement_rule")
        + value(row, "delta_epistemic_calibration_rule")
        - 0.10 * max(0.0, value(row, "delta_harshness_rule")),
    )


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
        "token_position": selections["belief_subtract"].get("token_position", "last"),
    }
    selections["erase_emotion"] = {
        "name": "erase_emotion",
        "mode": "erase",
        "direction": selections["emotion_add"]["direction"],
        "layer_index": selections["emotion_add"]["layer_index"],
        "token_position": selections["emotion_add"].get("token_position", "last"),
    }
    multi = select_multi(rows)
    if multi is not None:
        selections["multi_layer_emotion_add_belief_subtract"] = {
            "name": "multi_layer_emotion_add_belief_subtract",
            "mode": "prebuilt_calibration_spec",
            "condition_id": multi["condition_id"],
            "layer_index": multi["layer_index"],
            "alpha_std": float(multi["alpha_std"]),
            "token_position": multi.get("token_position") or "last",
        }
    selections["style_control"] = {
        "name": "style_control",
        "mode": "system_prompt_control",
        "system_prompt": "请用温暖、自然、简洁的方式回应。先承接用户情绪，但不要确认未经证据支持的事实判断；必要时保留不确定性。",
    }
    output = {
        "selection_source": "calibration prompts with transparent lexical heuristic; frozen before final evaluation",
        "selected": selections,
        "calibration_effect_rows": {
            "emotion_add": emotion,
            "emotion_subtract": emotion_neg,
            "belief_add": belief_pos,
            "belief_subtract": belief_neg,
            "multi_layer_emotion_add_belief_subtract": multi,
        },
    }
    write_json(run_dir / "analysis" / "frozen_intervention_configs.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
