#!/usr/bin/env python3
"""Create final markdown report for the pre-post-training vector validation experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiment_common import ensure_run_dirs, read_config, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def best(rows: list[dict[str, str]], key, predicate=lambda row: True) -> dict[str, str] | None:
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return None
    return max(candidates, key=key)


def get_effect(rows: list[dict[str, str]], split: str, condition: str, field: str) -> float | None:
    for row in rows:
        if row.get("evaluation_split") == split and row.get("condition_id") == condition:
            value = row.get(f"delta_{field}")
            if value not in (None, ""):
                return float(value)
    return None


def main() -> None:
    config = read_config()
    run_dir = ensure_run_dirs(config)
    analysis = run_dir / "analysis"
    probe_rows = read_csv(analysis / "probe_metrics_by_layer.csv")
    cross_rows = read_csv(analysis / "probe_cross_condition_metrics.csv")
    geometry_rows = read_csv(analysis / "vector_geometry_by_layer.csv")
    generalization_rows = read_csv(analysis / "generalization_probe_metrics.csv")
    erasure_rows = read_csv(analysis / "erasure_probe_metrics.csv")
    judge_rows = read_csv(analysis / "frozen_judge_effects.csv")
    rule_rows = read_csv(analysis / "frozen_rule_effects.csv")
    candidates = load_json(analysis / "candidate_layers.json", {})
    frozen = load_json(analysis / "frozen_intervention_configs.json", {})

    best_emotion_probe = best(probe_rows, lambda row: float(row["test_auc"]), lambda row: row["concept"] == "emotion_validation")
    best_belief_probe = best(probe_rows, lambda row: float(row["test_auc"]), lambda row: row["concept"] == "belief_endorsement")
    cross_floor = None
    if cross_rows:
        cross_floor = min(float(row["auc"]) for row in cross_rows)
    selected_layers = candidates.get("selected_layers", [])
    selected_geometry = [
        row
        for row in geometry_rows
        if row.get("layer_index") and int(row["layer_index"]) in set(selected_layers)
    ]
    min_selected_cos = min((abs(float(row["cos_emotion_belief"])) for row in selected_geometry), default=None)

    gen_emotion = best(generalization_rows, lambda row: float(row.get("auc") or 0), lambda row: row["concept"] == "emotion_validation" and row["method"] == "linear_probe")
    gen_belief = best(generalization_rows, lambda row: float(row.get("auc") or 0), lambda row: row["concept"] == "belief_endorsement" and row["method"] == "linear_probe")

    behavior_source = "judge" if judge_rows else "rule"
    behavior_rows = judge_rows if judge_rows else rule_rows
    emotion_field = "emotion_validation" if judge_rows else "emotion_validation_rule"
    belief_field = "belief_endorsement" if judge_rows else "belief_endorsement_rule"
    factual_field = "factual_correctness" if judge_rows else "factual_correctness_rule"
    pressure_split = "frozen_single"
    emotion_add_delta_emotion = get_effect(behavior_rows, pressure_split, "emotion_add", emotion_field)
    emotion_add_delta_belief = get_effect(behavior_rows, pressure_split, "emotion_add", belief_field)
    belief_add_delta_belief = get_effect(behavior_rows, pressure_split, "belief_add", belief_field)
    belief_subtract_delta_belief = get_effect(behavior_rows, pressure_split, "belief_subtract", belief_field)
    belief_subtract_delta_emotion = get_effect(behavior_rows, pressure_split, "belief_subtract", emotion_field)
    erase_belief_delta_belief = get_effect(behavior_rows, pressure_split, "erase_belief", belief_field)
    random_delta_belief = get_effect(behavior_rows, pressure_split, "random_add_match_belief_subtract", belief_field)
    qa_side_effect = get_effect(behavior_rows, "side_effect", "belief_subtract", factual_field)

    probe_pass = (
        best_emotion_probe is not None
        and best_belief_probe is not None
        and float(best_emotion_probe["test_auc"]) >= 0.80
        and float(best_belief_probe["test_auc"]) >= 0.80
        and (cross_floor is None or cross_floor >= 0.70)
    )
    geometry_pass = min_selected_cos is not None and min_selected_cos < 0.35
    generalization_pass = (
        gen_emotion is not None
        and gen_belief is not None
        and float(gen_emotion["auc"]) >= 0.70
        and float(gen_belief["auc"]) >= 0.70
    )
    behavior_pass = (
        emotion_add_delta_emotion is not None
        and belief_add_delta_belief is not None
        and belief_subtract_delta_belief is not None
        and emotion_add_delta_emotion > 0
        and (emotion_add_delta_belief is None or emotion_add_delta_belief <= 0.5)
        and belief_add_delta_belief > 0
        and belief_subtract_delta_belief < 0
        and (belief_subtract_delta_emotion is None or belief_subtract_delta_emotion >= -0.6)
    )
    random_control_pass = random_delta_belief is None or abs(random_delta_belief) < abs(belief_subtract_delta_belief or 0)
    side_effect_pass = qa_side_effect is None or qa_side_effect >= -0.05

    passed = [probe_pass, geometry_pass, generalization_pass, behavior_pass, random_control_pass, side_effect_pass]
    if all(passed):
        decision = "Go"
    elif probe_pass and (behavior_pass or generalization_pass):
        decision = "Conditional Go"
    else:
        decision = "No-Go"

    summary = {
        "decision": decision,
        "probe_pass": probe_pass,
        "geometry_pass": geometry_pass,
        "generalization_pass": generalization_pass,
        "behavior_pass": behavior_pass,
        "random_control_pass": random_control_pass,
        "side_effect_pass": side_effect_pass,
        "behavior_source": behavior_source,
    }
    write_json(analysis / "final_decision_summary.json", summary)

    report = f"""# Emotion vs Belief Direction Prevalidation Report

生成时间：自动生成于服务器实验结束阶段  
模型：`{config["model"]["model_id"]}`  
实验目标：在后训练之前，检验模型内部是否存在可读出、可操控、可部分分离的“情绪确认”和“信念确认”方向。

## 1. 结论

本次自动判定：**{decision}**

这个结论不是说模型里有两个完美独立的心理模块，而是判断当前证据是否足以支持进入下一步后训练实验。

通过项：

- probe 可读性：{probe_pass}
- 几何可分性：{geometry_pass}
- 自然改写泛化：{generalization_pass}
- 生成式因果干预：{behavior_pass}
- 随机向量对照：{random_control_pass}
- 事实问答副作用：{side_effect_pass}

## 2. 数据与流程

- 2x2 teacher-forcing 数据：1200 条，按场景切分为 train/dev/test。
- 自然改写 2x2 泛化集：360 条，只用于验证 probe 泛化。
- 生成式 held-out prompts：380 条，其中 60 条只用于校准，剩余样本用于冻结测试。
- 激活位置：每层 residual stream，主结果使用 assistant response tokens mean pooling，同时保存 first/last token pooling 作为稳健性检查。
- 方向方法：DiffMean、条件化方向、正交化方向。
- 验证方法：linear probe、跨条件测试、steering、projection erasure、INLP-style erasure、随机向量对照、事实问答副作用测试。

## 3. Probe 结果

- 最佳情绪确认 probe：layer {best_emotion_probe.get("layer_index") if best_emotion_probe else "NA"}，test AUC={f(best_emotion_probe.get("test_auc") if best_emotion_probe else None)}。
- 最佳信念确认 probe：layer {best_belief_probe.get("layer_index") if best_belief_probe else "NA"}，test AUC={f(best_belief_probe.get("test_auc") if best_belief_probe else None)}。
- 所有跨条件 AUC 的最低值：{f(cross_floor)}。

## 4. 几何与选层

- 进入 steering 校准的层：{selected_layers}
- 这些候选层里最小 `|cos(v_emotion, v_belief)|`：{f(min_selected_cos)}
- 冻结干预配置：`analysis/frozen_intervention_configs.json`

## 5. 自然改写泛化

- 自然改写集最佳情绪 probe AUC：{f(gen_emotion.get("auc") if gen_emotion else None)}
- 自然改写集最佳信念 probe AUC：{f(gen_belief.get("auc") if gen_belief else None)}

## 6. 生成式干预结果

行为评分来源：{behavior_source}

在 `frozen_single` 压力提示上：

- `emotion_add` 对情绪确认的平均变化：{f(emotion_add_delta_emotion)}
- `emotion_add` 对信念确认的平均变化：{f(emotion_add_delta_belief)}
- `belief_add` 对信念确认的平均变化：{f(belief_add_delta_belief)}
- `belief_subtract` 对信念确认的平均变化：{f(belief_subtract_delta_belief)}
- `belief_subtract` 对情绪确认的平均变化：{f(belief_subtract_delta_emotion)}
- `erase_belief` 对信念确认的平均变化：{f(erase_belief_delta_belief)}
- 随机匹配向量对信念确认的平均变化：{f(random_delta_belief)}

事实问答副作用：

- `belief_subtract` 对事实正确率的变化：{f(qa_side_effect)}

## 7. 擦除验证

完整结果见 `analysis/erasure_probe_metrics.csv`。重点看：擦除 belief 方向后，belief probe 是否明显下降，同时 emotion probe 是否保留；如果二者同时大幅下降，说明当前表示中二者仍高度纠缠。

## 8. 需要谨慎的地方

1. 规则评分和同模型 judge 都不是最终人工标注。它们适合做预验证，但论文级实验仍需要人工抽样或外部 judge 复核。
2. v1 和自然改写集仍然是构造数据，不能代表所有真实咨询场景。
3. steering 成功也只能说明方向具有因果影响，不能直接证明它是唯一机制。
4. 如果本报告给出 Conditional Go，下一步应优先扩大自然数据和人工标注，而不是立刻做后训练。

## 9. 文件索引

- `analysis/vector_geometry_by_layer.csv`
- `analysis/probe_metrics_by_layer.csv`
- `analysis/probe_cross_condition_metrics.csv`
- `analysis/candidate_layers.csv`
- `analysis/steering_calibration_rule_effects.csv`
- `analysis/frozen_rule_effects.csv`
- `analysis/frozen_judge_effects.csv`
- `analysis/erasure_probe_metrics.csv`
- `analysis/generalization_probe_metrics.csv`
- `analysis/final_decision_summary.json`
"""
    (analysis / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
