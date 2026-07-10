#!/usr/bin/env bash
# Open-loop eval for GWP0.5.
# Usage:
#   CHECKPOINT=<path> NORM_STATS=<path> DATA_PATH=<path> ./run_inference_openloop.sh server
#   CHECKPOINT=<path> NORM_STATS=<path> DATA_PATH=<path> ./run_inference_openloop.sh client

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT="${CHECKPOINT:-}"
NORM_STATS="${NORM_STATS:-}"
DATA_PATH="${DATA_PATH:-}"
DATA_IDX="${DATA_IDX:-1}"
REPLAN_STEPS="${REPLAN_STEPS:-30}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-11444}"
DEVICE="${DEVICE:-cuda:1}"
BASE_MODEL="${BASE_MODEL:-}"
FIXED_T5_PATH="${FIXED_T5_PATH:-}"

require_path() {
    local name="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "${name} must be set. Example: ${name}=<path> $0 server" >&2
        exit 1
    fi
}

require_path CHECKPOINT "${CHECKPOINT}"
require_path NORM_STATS "${NORM_STATS}"
require_path DATA_PATH "${DATA_PATH}"

ARGS=(
    --checkpoint "${CHECKPOINT}"
    --norm-stats "${NORM_STATS}"
    --data-path "${DATA_PATH}"
    --data-idx "${DATA_IDX}"
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
)

if [[ -n "${BASE_MODEL}" ]]; then
    ARGS+=(--base-model "${BASE_MODEL}")
fi
if [[ -n "${FIXED_T5_PATH}" ]]; then
    ARGS+=(--fixed-t5-path "${FIXED_T5_PATH}")
fi

case "${1:-}" in
    server)
        require_path BASE_MODEL "${BASE_MODEL}"
        shift
        python "${SCRIPT_DIR}/inference_openloop.py" --server "${ARGS[@]}" "$@"
        ;;
    client)
        shift
        python "${SCRIPT_DIR}/inference_openloop.py" "${ARGS[@]}" --replan-steps "${REPLAN_STEPS}" "$@"
        ;;
    *)
        echo "Usage: CHECKPOINT=<path> NORM_STATS=<path> DATA_PATH=<path> $0 server|client" >&2
        exit 1
        ;;
esac
