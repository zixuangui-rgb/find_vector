#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/find_vector_workspace/find_vector}"
RUN_DIR="${RUN_DIR:-/data/find_vector_runs/qwen35_9b_emotion_belief_prevalidation_v1}"
MODEL_DIR="${MODEL_DIR:-/data/models/Qwen3.5-9B}"
LOG_DIR="${LOG_DIR:-/data/find_vector_runs/setup_logs}"
DOWNLOAD_SCREEN="${DOWNLOAD_SCREEN:-find_vector_model_download}"
PIPELINE_SCREEN="${PIPELINE_SCREEN:-find_vector_full_pipeline}"
LOG_FILE="$LOG_DIR/auto_start_full_pipeline.log"

mkdir -p "$LOG_DIR" "$RUN_DIR/logs"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

model_is_complete() {
  grep -q "DOWNLOAD_COMPLETE" "$LOG_DIR/model_download.log" 2>/dev/null &&
    find "$MODEL_DIR" -maxdepth 1 -name 'model.safetensors-*.safetensors' | grep -q .
}

log "auto-start watcher begins"

while screen -ls | grep -q "$DOWNLOAD_SCREEN"; do
  size=$(du -sh "$MODEL_DIR" 2>/dev/null | awk '{print $1}' || true)
  log "waiting for model download; model_dir_size=${size:-unknown}"
  sleep 60
done

log "model download screen exited"
tail -n 80 "$LOG_DIR/model_download.log" >> "$LOG_FILE" 2>&1 || true

if ! model_is_complete; then
  log "download did not complete cleanly; not launching pipeline"
  exit 1
fi

cd "$REPO_DIR"
git pull --ff-only >> "$LOG_FILE" 2>&1 || true

log "running smoke test before full launch"
CUDA_VISIBLE_DEVICES=0 /data/find_vector_env/bin/python scripts/00_smoke_test.py --limit 4 \
  > "$RUN_DIR/logs/smoke_test_auto_start.log" 2>&1

log "launching full prevalidation pipeline in screen=$PIPELINE_SCREEN"
screen -S "$PIPELINE_SCREEN" -X quit >/dev/null 2>&1 || true
screen -dmS "$PIPELINE_SCREEN" bash -lc "cd '$REPO_DIR' && bash scripts/run_full_prevalidation_pipeline.sh"
log "full prevalidation pipeline launched"
