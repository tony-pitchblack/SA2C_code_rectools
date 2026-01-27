#!/usr/bin/env bash
set -euo pipefail

if [ ! -f ".env" ]; then
  echo "Missing .env in $PWD." >&2
  echo "Create .env with:" >&2
  echo "  MLFLOW_HOST=0.0.0.0" >&2
  echo "  MLFLOW_PORT=5000" >&2
  exit 1
fi

set -a
source ".env"
set +a

if [ -z "${MLFLOW_HOST:-}" ] || [ -z "${MLFLOW_PORT:-}" ]; then
  echo ".env must define MLFLOW_HOST and MLFLOW_PORT." >&2
  exit 1
fi

PORT_CLEAN="$(printf '%s' "$MLFLOW_PORT" | tr -d "\"'")"
HOST_CLEAN="$(printf '%s' "$MLFLOW_HOST" | tr -d "\"'")"

DISPLAY_HOST="$HOST_CLEAN"
if [ -z "$DISPLAY_HOST" ] || [ "$DISPLAY_HOST" = "0.0.0.0" ] || [ "$DISPLAY_HOST" = "::" ]; then
  DISPLAY_HOST="localhost"
fi

SESSION="mlflow-${PORT_CLEAN}"
ROOT="$PWD/mlflow"
mkdir -p "$ROOT/artifacts"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "http://${DISPLAY_HOST}:${PORT_CLEAN}"
  echo "tmux attach -t ${SESSION}"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -ic 'cd \"$PWD\"; exec mlflow server --host \"$HOST_CLEAN\" --port \"$PORT_CLEAN\" --backend-store-uri \"sqlite:///$ROOT/mlflow.db\" --default-artifact-root \"file://$ROOT/artifacts\" --serve-artifacts --workers 1'"
echo "http://${DISPLAY_HOST}:${PORT_CLEAN}"
echo "tmux attach -t ${SESSION}"
