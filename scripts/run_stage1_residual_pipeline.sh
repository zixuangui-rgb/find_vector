#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/find_vector_workspace/find_vector}"
PYTHON="${PYTHON:-/data/find_vector_env/bin/python}"
RUN_DIR="/data/find_vector_runs/qwen35_9b_emotion_belief_prevalidation_v1"
mkdir -p "$RUN_DIR/logs"
cd "$REPO_DIR"

for split in train dev test; do
  echo "[$(date '+%F %T')] starting extraction split=$split" | tee -a "$RUN_DIR/logs/stage1_orchestrator.log"
  pids=()
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/01_extract_activations.py \
      --split "$split" --shard-index "$gpu" --num-shards 4 \
      > "$RUN_DIR/logs/extract_${split}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$PYTHON" scripts/01b_merge_activations.py --split "$split" --num-shards 4 \
    >> "$RUN_DIR/logs/stage1_orchestrator.log" 2>&1
  echo "[$(date '+%F %T')] completed extraction split=$split" | tee -a "$RUN_DIR/logs/stage1_orchestrator.log"
done

"$PYTHON" scripts/02_build_vectors.py >> "$RUN_DIR/logs/stage1_orchestrator.log" 2>&1
"$PYTHON" scripts/03_probe_validation.py >> "$RUN_DIR/logs/stage1_orchestrator.log" 2>&1
echo "[$(date '+%F %T')] stage1 residual pipeline complete" | tee -a "$RUN_DIR/logs/stage1_orchestrator.log"

