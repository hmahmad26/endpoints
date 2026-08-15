#!/usr/bin/env bash
# Kimi K2.6 native INT4 (moonshotai/Kimi-K2.6) — SGLang server + agentic benchmark client.
#
# Host: smci355-ccs-aus-n08-17 (8× MI35x, ROCm)
#
# Verified recipe (client-only, no probe):
#   1. ./examples/10_Agentic_Inference/kimi_l26_int4.sh start-server
#   2. ./examples/10_Agentic_Inference/kimi_l26_int4.sh wait-for-endpoint
#   3. Holdout (50 traj, ~103 min):
#        ./examples/10_Agentic_Inference/kimi_l26_int4.sh tmux-holdout rep2
#      Full v4 (990 traj, ~2 days):
#        ./examples/10_Agentic_Inference/kimi_l26_int4.sh tmux-v4-full
#   4. Monitor: tail -f logs/<report_dir>/run.log
#   5. Attach:  tmux attach -t <session>
#
# Other commands: download | stop-server | logs-server | probe | status
# Legacy (uses probe): benchmark-repeat | tmux-repeat | tmux-v4-repeat

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --- edit these if paths/host change ---
export HOSTNAME="${HOSTNAME:-smci355-ccs-aus-n08-17}"
export MODEL_PATH="${MODEL_PATH:-/data/workloads-inference/models/Kimi-K2.6-INT4}"
export HF_MODEL_ID="${HF_MODEL_ID:-moonshotai/Kimi-K2.6}"
export DATASET_PATH="${DATASET_PATH:-/home/hmahmad/endpoints/agentic_combined_v4_holdout_50.jsonl}"
export ENDPOINT_URL="${ENDPOINT_URL:-http://${HOSTNAME}:8000}"
export CONTAINER_NAME="${CONTAINER_NAME:-sglang-kimi-k26-int4}"
export SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang-rocm:v0.5.14-rocm700-mi35x-20260626}"
export BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-examples/10_Agentic_Inference/kimi_agentic_benchmark_holdout50_int4.yaml}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.6}"
export TP_SIZE="${TP_SIZE:-8}"
export SERVE_PORT="${SERVE_PORT:-8000}"
export BENCHMARK_CONFIG_V4_FULL="${BENCHMARK_CONFIG_V4_FULL:-examples/10_Agentic_Inference/kimi_agentic_benchmark_v4_full.yaml}"
export DATASET_PATH_V4="${DATASET_PATH_V4:-/home/hmahmad/endpoints/agentic_combined_v4.jsonl}"
export TMUX_SESSION_V4="${TMUX_SESSION_V4:-kimi-int4-v4-full}"
export REPORT_DIR_V4="${REPORT_DIR_V4:-logs/kimi_agentic_v4_int4}"
export TMUX_SESSION_HOLDOUT="${TMUX_SESSION_HOLDOUT:-kimi-int4-holdout-baseline}"
export MASTER_LOG_V4="${MASTER_LOG_V4:-logs/kimi_agentic_v4_int4_master.log}"
export REPORT_DIR_PREFIX_V4="${REPORT_DIR_PREFIX_V4:-logs/kimi_agentic_v4_int4_rep}"
export TMUX_SESSION="${TMUX_SESSION:-kimi-int4-holdout5x}"
export MASTER_LOG="${MASTER_LOG:-logs/kimi_agentic_holdout_50_int4_5x_master.log}"
export REPORT_DIR_PREFIX="${REPORT_DIR_PREFIX:-logs/kimi_agentic_holdout_50_int4_rep}"

uv() {
  if [[ -f "${HOME}/.local/bin/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  fi
  command uv "$@"
}

require_model() {
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model not found: ${MODEL_PATH}" >&2
    echo "Run: $0 download" >&2
    exit 1
  fi
  local shard_count
  shard_count="$(find "${MODEL_PATH}" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)"
  if [[ "${shard_count}" -lt 64 ]]; then
    echo "Warning: expected 64 safetensors shards, found ${shard_count}." >&2
    echo "Download may still be in progress. Check: du -sh ${MODEL_PATH}" >&2
  fi
}

require_dataset() {
  if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "Dataset not found: ${DATASET_PATH}" >&2
    exit 1
  fi
}

cmd_download() {
  mkdir -p "${MODEL_PATH}"
  uv run hf download "${HF_MODEL_ID}" --local-dir "${MODEL_PATH}"
}

cmd_start_server() {
  require_model
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Removing existing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}"
  fi
  # Kimi on ROCm: SGLANG_ROCM_FUSED_DECODE_MLA=0 avoids ForwardMetadata crash
  # (sglang issues #20691 / #19824). INT4 uses compressed-tensors — do NOT use
  # MXFP4-only flags (--enforce-shared-experts-fusion, etc.).
  docker run -d \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --shm-size=16g \
    --network host \
    --ipc host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
    -v "${MODEL_PATH}:/model:ro" \
    "${SGLANG_IMAGE}" \
    bash -c 'SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 \
      sglang serve \
        --model-path /model \
        --served-model-name kimi-k2.6 \
        --tp 8 \
        --trust-remote-code \
        --reasoning-parser kimi_k2 \
        --tool-call-parser kimi_k2 \
        --decode-attention-backend aiter \
        --prefill-attention-backend aiter \
        --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static 0.75 \
        --cuda-graph-backend-decode disabled \
        --disable-radix-cache \
        --watchdog-timeout 3600 \
        --host 0.0.0.0 \
        --port 8000'
  echo "Started ${CONTAINER_NAME}. Tail logs with: $0 logs-server"
}

cmd_logs_server() {
  docker logs -f "${CONTAINER_NAME}"
}

cmd_stop_server() {
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
  echo "Stopped ${CONTAINER_NAME} (if it was running)."
}

cmd_probe() {
  require_dataset
  uv run inference-endpoint probe \
    --endpoints "${ENDPOINT_URL}" \
    --model "${SERVED_MODEL_NAME}"
}

require_server() {
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container ${CONTAINER_NAME} is not running." >&2
    echo "Start it first: $0 start-server" >&2
    exit 1
  fi
  echo "Container ${CONTAINER_NAME} is running."

  local health_url="http://127.0.0.1:${SERVE_PORT}/v1/models"
  if ! curl -sf --max-time 30 "${health_url}" >/dev/null; then
    echo "Endpoint not reachable at ${health_url}" >&2
    echo "Check server logs: $0 logs-server" >&2
    exit 1
  fi
  echo "Endpoint health check OK: ${health_url}"

  local attempt max_attempts=1
  for attempt in $(seq 1 "${max_attempts}"); do
    echo "Probe attempt ${attempt}/${max_attempts}..."
    if cmd_probe; then
      return 0
    fi
    if [[ "${attempt}" -lt "${max_attempts}" ]]; then
      echo "Probe failed; retrying in 60s..." >&2
      sleep 60
    fi
  done

  echo "WARNING: probe failed after ${max_attempts} attempts, but health check passed." >&2
  echo "Proceeding with benchmark (server is reachable)." >&2
}

wait_for_endpoint() {
  local health_url="http://127.0.0.1:${SERVE_PORT}/v1/models"
  local max_attempts="${1:-40}"
  local attempt
  for attempt in $(seq 1 "${max_attempts}"); do
    if curl -sf --max-time 10 "${health_url}" >/dev/null 2>&1; then
      echo "Endpoint ready: ${health_url}"
      return 0
    fi
    echo "Waiting for endpoint (${attempt}/${max_attempts})..."
    sleep 15
  done
  echo "Endpoint not ready at ${health_url}" >&2
  exit 1
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed." >&2
    exit 1
  fi
}

# Verified client launcher: uv run benchmark, tee run.log, keep shell open after exit.
tmux_run_client() {
  local session="$1"
  local report_dir="$2"
  local config="$3"

  require_tmux
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Tmux session already exists: ${session}" >&2
    echo "Attach: tmux attach -t ${session}" >&2
    exit 1
  fi

  mkdir -p "${report_dir}"
  tmux new-session -d -s "${session}" -c "${REPO_ROOT}" \
    "cd ${REPO_ROOT} && uv run inference-endpoint benchmark from-config --config ${config} 2>&1 | tee ${report_dir}/run.log; echo benchmark finished exit=\$?; exec bash"

  echo "Started tmux session: ${session}"
  echo "  config:     ${config}"
  echo "  report_dir: ${report_dir}"
  echo "  attach:     tmux attach -t ${session}"
  echo "  monitor:    tail -f ${report_dir}/run.log"
}

cmd_benchmark() {
  require_dataset
  uv sync --extra dev --extra test
  uv run inference-endpoint benchmark from-config \
    --config "${BENCHMARK_CONFIG}"
}

cmd_benchmark_repeat() {
  local n="${1:-5}"
  if ! [[ "${n}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid repeat count: ${n} (expected positive integer)" >&2
    exit 1
  fi

  require_dataset
  require_server
  uv sync --extra dev --extra test

  mkdir -p logs
  echo "=== benchmark-repeat: ${n} runs ===" | tee "${MASTER_LOG}"
  echo "Started at $(date -Is)" | tee -a "${MASTER_LOG}"
  echo "Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"

  local i report_dir tmp_config
  for i in $(seq 1 "${n}"); do
    report_dir="${REPORT_DIR_PREFIX}${i}"
    tmp_config="/tmp/kimi_holdout50_rep${i}.yaml"
    mkdir -p "${report_dir}"
    sed "s|^report_dir:.*|report_dir: ${report_dir}|" \
      "${BENCHMARK_CONFIG}" > "${tmp_config}"

    echo "" | tee -a "${MASTER_LOG}"
    echo "=== Run ${i}/${n} -> ${report_dir} ===" | tee -a "${MASTER_LOG}"
    echo "Run ${i} started at $(date -Is)" | tee -a "${MASTER_LOG}"

    uv run inference-endpoint benchmark from-config \
      --config "${tmp_config}" \
      2>&1 | tee -a "${report_dir}/run.log" "${MASTER_LOG}"

    echo "Run ${i} finished at $(date -Is)" | tee -a "${MASTER_LOG}"
  done

  echo "" | tee -a "${MASTER_LOG}"
  echo "=== All ${n} runs complete at $(date -Is) ===" | tee -a "${MASTER_LOG}"
}

cmd_benchmark_v4() {
  export BENCHMARK_CONFIG="${BENCHMARK_CONFIG_V4_FULL}"
  export DATASET_PATH="${DATASET_PATH_V4}"
  cmd_benchmark
}

cmd_benchmark_v4_repeat() {
  local n="${1:-1}"
  if ! [[ "${n}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid repeat count: ${n} (expected positive integer)" >&2
    exit 1
  fi

  BENCHMARK_CONFIG="${BENCHMARK_CONFIG_V4_FULL}" \
  DATASET_PATH="${DATASET_PATH_V4}" \
  require_dataset
  require_server
  uv sync --extra dev --extra test

  mkdir -p logs
  echo "=== benchmark-v4-repeat: ${n} runs ===" | tee "${MASTER_LOG_V4}"
  echo "Started at $(date -Is)" | tee -a "${MASTER_LOG_V4}"
  echo "Master log: ${MASTER_LOG_V4}" | tee -a "${MASTER_LOG_V4}"

  local i report_dir tmp_config
  for i in $(seq 1 "${n}"); do
    report_dir="${REPORT_DIR_PREFIX_V4}${i}"
    tmp_config="/tmp/kimi_v4_full_rep${i}.yaml"
    mkdir -p "${report_dir}"
    sed "s|^report_dir:.*|report_dir: ${report_dir}|" \
      "${BENCHMARK_CONFIG_V4_FULL}" > "${tmp_config}"

    echo "" | tee -a "${MASTER_LOG_V4}"
    echo "=== Run ${i}/${n} -> ${report_dir} ===" | tee -a "${MASTER_LOG_V4}"
    echo "Run ${i} started at $(date -Is)" | tee -a "${MASTER_LOG_V4}"

    uv run inference-endpoint benchmark from-config \
      --config "${tmp_config}" \
      2>&1 | tee -a "${report_dir}/run.log" "${MASTER_LOG_V4}"

    echo "Run ${i} finished at $(date -Is)" | tee -a "${MASTER_LOG_V4}"
  done

  echo "" | tee -a "${MASTER_LOG_V4}"
  echo "=== All ${n} runs complete at $(date -Is) ===" | tee -a "${MASTER_LOG_V4}"
}

cmd_benchmark_v4_full() {
  cmd_benchmark_v4
}

cmd_tmux_holdout() {
  local suffix="${1:-rep1}"
  local report_dir="${REPORT_DIR_PREFIX}${suffix}"
  local tmp_config="/tmp/kimi_holdout_${suffix}.yaml"

  sed "s|^report_dir:.*|report_dir: ${report_dir}|" \
    "${BENCHMARK_CONFIG}" > "${tmp_config}"
  tmux_run_client "${TMUX_SESSION_HOLDOUT}" "${report_dir}" "${tmp_config}"
}

cmd_tmux_v4_full() {
  tmux_run_client "${TMUX_SESSION_V4}" "${REPORT_DIR_V4}" "${BENCHMARK_CONFIG_V4_FULL}"
}

cmd_tmux_v4_repeat() {
  local n="${1:-1}"
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed." >&2
    exit 1
  fi
  if tmux has-session -t "${TMUX_SESSION_V4}" 2>/dev/null; then
    echo "Tmux session already exists: ${TMUX_SESSION_V4}" >&2
    echo "Attach: tmux attach -t ${TMUX_SESSION_V4}" >&2
    echo "Or kill first: tmux kill-session -t ${TMUX_SESSION_V4}" >&2
    exit 1
  fi

  tmux new-session -d -s "${TMUX_SESSION_V4}" -c "${REPO_ROOT}" \
    "./examples/10_Agentic_Inference/kimi_l26_int4.sh benchmark-v4-repeat ${n}"

  echo "Started detached tmux session: ${TMUX_SESSION_V4}"
  echo "  runs:     ${n} (990 trajectories each, ~30-40 hr per run)"
  echo "  attach:   tmux attach -t ${TMUX_SESSION_V4}"
  echo "  monitor:  tail -f ${MASTER_LOG_V4}"
}

cmd_tmux_repeat() {
  local n="${1:-5}"
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed." >&2
    exit 1
  fi
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "Tmux session already exists: ${TMUX_SESSION}" >&2
    echo "Attach: tmux attach -t ${TMUX_SESSION}" >&2
    echo "Or kill first: tmux kill-session -t ${TMUX_SESSION}" >&2
    exit 1
  fi

  tmux new-session -d -s "${TMUX_SESSION}" -c "${REPO_ROOT}" \
    "./examples/10_Agentic_Inference/kimi_l26_int4.sh benchmark-repeat ${n}"

  echo "Started detached tmux session: ${TMUX_SESSION}"
  echo "  attach:  tmux attach -t ${TMUX_SESSION}"
  echo "  monitor: tail -f ${MASTER_LOG}"
}

cmd_status() {
  echo "HOSTNAME=${HOSTNAME}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "DATASET_PATH=${DATASET_PATH}"
  echo "ENDPOINT_URL=${ENDPOINT_URL}"
  echo "BENCHMARK_CONFIG=${BENCHMARK_CONFIG}"
  if [[ -d "${MODEL_PATH}" ]]; then
    du -sh "${MODEL_PATH}"
    find "${MODEL_PATH}" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l | xargs echo "safetensors shards:"
  else
    echo "Model directory missing."
  fi
  docker ps -a --filter "name=${CONTAINER_NAME}" --format 'table {{.Names}}\t{{.Status}}'
}

usage() {
  sed -n '2,16p' "$0" | tr -d '#'
  echo
  echo "Commands:"
  echo "  Verified: start-server | wait-for-endpoint | tmux-holdout [suffix] | tmux-v4-full"
  echo "  Legacy:   benchmark-repeat [N] | tmux-repeat [N] | tmux-v4-repeat [N] | probe | benchmark | benchmark-v4"
  echo "  Other:    download | logs-server | stop-server | status"
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    download) cmd_download ;;
    start-server) cmd_start_server ;;
    logs-server) cmd_logs_server ;;
    stop-server) cmd_stop_server ;;
    wait-for-endpoint) wait_for_endpoint "${2:-40}" ;;
    probe) cmd_probe ;;
    benchmark) cmd_benchmark ;;
    benchmark-v4) cmd_benchmark_v4 ;;
    benchmark-v4-full) cmd_benchmark_v4 ;;
    tmux-holdout) cmd_tmux_holdout "${2:-rep1}" ;;
    tmux-v4-full) cmd_tmux_v4_full ;;
    tmux-v4-repeat) cmd_tmux_v4_repeat "${2:-1}" ;;
    benchmark-repeat) cmd_benchmark_repeat "${2:-5}" ;;
    tmux-repeat) cmd_tmux_repeat "${2:-5}" ;;
    status) cmd_status ;;
    -h | --help | help | "") usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
