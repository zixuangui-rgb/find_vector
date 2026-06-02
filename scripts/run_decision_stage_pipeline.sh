#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/find_vector_workspace/find_vector}"
PYTHON="${PYTHON:-/data/find_vector_env/bin/python}"
CONFIG="${CONFIG:-config/decision_stage_config.yaml}"
export FIND_VECTOR_CONFIG="$CONFIG"

RUN_DIR="/data/find_vector_runs/qwen35_9b_emotion_belief_decision_stage_v1"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/full_pipeline.log"
}

run_four_gpu() {
  local label="$1"
  shift
  log "starting $label"
  pids=()
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" "$@" --shard-index "$gpu" --num-shards 4 \
      > "$LOG_DIR/${label}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  log "completed $label"
}

assert_no_thinking_pollution() {
  local label="$1"
  if grep -R -I -i -E "Thinking Process|Here's a thinking process" "$RUN_DIR/responses" "$RUN_DIR/scores" >/dev/null 2>&1; then
    log "ERROR: thinking-process pollution detected after $label"
    grep -R -I -i -E "Thinking Process|Here's a thinking process" "$RUN_DIR/responses" "$RUN_DIR/scores" | head -20 \
      >> "$LOG_DIR/full_pipeline.log" || true
    exit 2
  fi
  log "thinking-pollution check passed after $label"
}

log "decision-stage pipeline start"
log "using config=$CONFIG"

log "generating deterministic decision-stage datasets and benchmark prompts"
"$PYTHON" scripts/20_generate_decision_stage_data.py >> "$LOG_DIR/full_pipeline.log" 2>&1

if [[ ! -f "$RUN_DIR/analysis/smoke_test.json" ]]; then
  log "running smoke test"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/00_smoke_test.py --config "$CONFIG" --limit 4 \
    > "$LOG_DIR/smoke_test.log" 2>&1
fi

for split in train dev test; do
  if [[ ! -f "$RUN_DIR/activations/residual_${split}.npz" ]]; then
    log "starting teacher-forcing extraction split=$split"
    pids=()
    for gpu in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/01_extract_activations.py \
        --config "$CONFIG" --split "$split" --shard-index "$gpu" --num-shards 4 \
        > "$LOG_DIR/extract_${split}_gpu${gpu}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    "$PYTHON" scripts/01b_merge_activations.py --config "$CONFIG" --split "$split" --num-shards 4 \
      >> "$LOG_DIR/full_pipeline.log" 2>&1
    log "completed teacher-forcing extraction split=$split"
  fi
done

for pooling in response_mean response_first response_last; do
  if [[ "$pooling" == "response_mean" ]]; then
    vector_summary="$RUN_DIR/analysis/vector_build_summary.json"
    probe_summary="$RUN_DIR/analysis/probe_selection_summary.json"
  else
    vector_summary="$RUN_DIR/analysis/vector_build_summary_${pooling}.json"
    probe_summary="$RUN_DIR/analysis/probe_selection_summary_${pooling}.json"
  fi
  if [[ ! -f "$vector_summary" ]]; then
    log "building vectors pooling=$pooling"
    "$PYTHON" scripts/02_build_vectors.py --config "$CONFIG" --pooling "$pooling" >> "$LOG_DIR/full_pipeline.log" 2>&1
  fi
  if [[ ! -f "$probe_summary" ]]; then
    log "training probes pooling=$pooling"
    "$PYTHON" scripts/03_probe_validation.py --config "$CONFIG" --pooling "$pooling" >> "$LOG_DIR/full_pipeline.log" 2>&1
  fi
done

if [[ ! -f "$RUN_DIR/analysis/candidate_layers.json" ]]; then
  log "selecting candidate layers"
  "$PYTHON" scripts/04_select_candidate_layers.py >> "$LOG_DIR/full_pipeline.log" 2>&1
fi

if [[ ! -f "$RUN_DIR/analysis/steering_calibration_rule_effects.csv" ]]; then
  run_four_gpu steering_calibration "$PYTHON" scripts/05_generate_steering_calibration.py --config "$CONFIG" --max-new-tokens 128
  "$PYTHON" scripts/06_score_steering_calibration.py --config "$CONFIG" >> "$LOG_DIR/full_pipeline.log" 2>&1
  assert_no_thinking_pollution "steering_calibration"
fi

if [[ ! -f "$RUN_DIR/analysis/frozen_intervention_configs.json" ]]; then
  log "selecting frozen configs"
  "$PYTHON" scripts/07_select_frozen_configs.py >> "$LOG_DIR/full_pipeline.log" 2>&1
fi

if [[ ! -f "$RUN_DIR/analysis/frozen_rule_effects.csv" ]]; then
  run_four_gpu frozen_tests "$PYTHON" scripts/08_generate_frozen_tests.py --config "$CONFIG" --max-new-tokens 160
  "$PYTHON" scripts/09_score_frozen_rules.py >> "$LOG_DIR/full_pipeline.log" 2>&1
  assert_no_thinking_pollution "frozen_tests"
fi

if [[ ! -f "$RUN_DIR/analysis/erasure_probe_metrics.csv" ]]; then
  log "running representation erasure validation"
  "$PYTHON" scripts/10_erasure_probe_validation.py >> "$LOG_DIR/full_pipeline.log" 2>&1
fi

if [[ ! -f "$RUN_DIR/activations/residual_natural_generalization.npz" ]]; then
  run_four_gpu natural_generalization "$PYTHON" scripts/11_extract_generalization_activations.py --config "$CONFIG"
  "$PYTHON" scripts/11b_merge_generalization_activations.py --config "$CONFIG" >> "$LOG_DIR/full_pipeline.log" 2>&1
fi

if [[ ! -f "$RUN_DIR/analysis/generalization_probe_metrics.csv" ]]; then
  log "running naturalized probe generalization"
  "$PYTHON" scripts/12_generalization_probe_validation.py >> "$LOG_DIR/full_pipeline.log" 2>&1
fi

if [[ ! -f "$RUN_DIR/analysis/frozen_judge_effects.csv" ]]; then
  run_four_gpu frozen_judge "$PYTHON" scripts/13_judge_frozen_outputs.py --config "$CONFIG" --max-new-tokens 180
  "$PYTHON" scripts/14_analyze_judge_scores.py >> "$LOG_DIR/full_pipeline.log" 2>&1
  assert_no_thinking_pollution "frozen_judge"
fi

if [[ ! -f "$RUN_DIR/analysis/proxy_pairwise_audit_summary.json" ]]; then
  run_four_gpu proxy_pairwise_audit "$PYTHON" scripts/22_pairwise_proxy_audit.py --config "$CONFIG" --max-new-tokens 160 --max-pairs-per-condition 180
  "$PYTHON" scripts/23_analyze_proxy_audit.py >> "$LOG_DIR/full_pipeline.log" 2>&1
  assert_no_thinking_pollution "proxy_pairwise_audit"
fi

log "writing decision-stage final report"
"$PYTHON" scripts/21_decision_stage_report.py >> "$LOG_DIR/full_pipeline.log" 2>&1
log "decision-stage pipeline complete"
