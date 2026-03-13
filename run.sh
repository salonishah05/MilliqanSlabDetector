#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV="$PROJECT_DIR/env/bin/activate"

PIPELINE="$PROJECT_DIR/scripts/anomalyDetector.py"

STATE_DIR="$PROJECT_DIR/state"
STATE_FILE="$STATE_DIR/last_run.txt"

LOG_DIR="$PROJECT_DIR/logs"
OUTPUT_DIR="$PROJECT_DIR/outputs"

MODEL="$PROJECT_DIR/models/cond_autoencoder.keras"

BASE_URL="http://cms2.physics.ucsb.edu/mqslab"
RUN_DIR_TEMPLATE="${BASE_URL}/Run%d/"
PULSES_FILE_TEMPLATE="MQSlabRun%d_mqspulses.ant"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$STATE_DIR"
LOGFILE="$LOG_DIR/run_$(date +%Y-%m-%d_%H%M%S).log"

log() {
  echo "$(date '+%F %T') | $*" | tee -a "$LOGFILE"
}


[[ -f "$VENV" ]] || { log "ERROR: venv activate not found at $VENV"; exit 1; }
[[ -f "$PIPELINE" ]] || { log "ERROR: pipeline not found at $PIPELINE"; exit 1; }
[[ -f "$MODEL" ]] || { log "ERROR: model not found at $MODEL"; exit 1; }

LAST_RUN=0
if [[ -f "$STATE_FILE" ]]; then
  LAST_RUN="$(cat "$STATE_FILE" || echo 0)"
fi
log "Last run processed: $LAST_RUN"

INDEX_HTML="$(curl -fsSL "$BASE_URL/" || true)"
if [[ -z "$INDEX_HTML" ]]; then
  log "ERROR: Could not fetch index: $BASE_URL/"
  exit 1
fi

LATEST_RUN="$(
  echo "$INDEX_HTML" \
    | grep -oE 'Run[0-9]{3,6}/' \
    | grep -oE '[0-9]{3,6}' \
    | sort -n \
    | tail -1
)"

if [[ -z "${LATEST_RUN:-}" ]]; then
  log "ERROR: Could not find at $BASE_URL/"
  exit 1
fi

log "Latest available run: $LATEST_RUN"

if (( LATEST_RUN <= LAST_RUN )); then
  log "Nothing new to process"
  exit 0
fi

RUN_DIR="$(printf "$RUN_DIR_TEMPLATE" "$LATEST_RUN")"
PULSES_FILE="$(printf "$PULSES_FILE_TEMPLATE" "$LATEST_RUN")"
RUN_URL="${RUN_DIR}${PULSES_FILE}"

log "URL: $RUN_URL"


METRICS_OUT="$OUTPUT_DIR/run${LATEST_RUN}/metrics.parquet"
PULSES_OUT="$OUTPUT_DIR/run${LATEST_RUN}/pulses.parquet"

source "$VENV"

python3 "$PIPELINE" \
  --url "$RUN_URL" \
  --outputFile "$METRICS_OUT" \
  --pulse_output "$PULSES_OUT" \
  --model_path "$MODEL" \
  2>&1 | tee -a "$LOGFILE"

echo "$LATEST_RUN" > "$STATE_FILE"
log "Run complete (finished run $LATEST_RUN)"