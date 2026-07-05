#!/usr/bin/env bash
# =============================================================================
# Kimi K2.6 INT4 — server + client commands (benchmark settings live in YAML files)
# =============================================================================
#
# Benchmark configs (edit paths there on a new server):
#   examples/10_Agentic_Inference/kimi_agentic_benchmark_holdout50_int4.yaml  (holdout)
#   examples/10_Agentic_Inference/kimi_agentic_benchmark_v4_full.yaml         (full v4)
#
# Set paths below, copy-paste each STEP, or run subcommands:
#   ./kimi_l26_int4_runbook.sh start-server
#   ./kimi_l26_int4_runbook.sh holdout rep1
#   ./kimi_l26_int4_runbook.sh v4-full
#
# ---------------------------------------------------------------------------
# PATHS — edit these
# ---------------------------------------------------------------------------
# export REPO_ROOT=/home/you/endpoints
# export MODEL_PATH=/data/workloads-inference/models/Kimi-K2.6-INT4
# export CONTAINER_NAME=sglang-kimi-k26-int4
# export SGLANG_IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm700-mi35x-20260626
#
# ---------------------------------------------------------------------------
# STEP 0 — preflight
# ---------------------------------------------------------------------------
# cd $REPO_ROOT
# rocm-smi --showuse
# ls $MODEL_PATH/*.safetensors | wc -l
#
# ---------------------------------------------------------------------------
# STEP 1 — download model (~555 GB, once)
# ---------------------------------------------------------------------------
# cd $REPO_ROOT
# mkdir -p $MODEL_PATH
# uv run hf download moonshotai/Kimi-K2.6 --local-dir $MODEL_PATH
#
# ---------------------------------------------------------------------------
# STEP 2 — start SGLang container server (8 GPU, port 8000)
# ---------------------------------------------------------------------------
# docker rm -f $CONTAINER_NAME 2>/dev/null || true
#
# docker run -d \
#   --name $CONTAINER_NAME \
#   --privileged \
#   --cap-add=CAP_SYS_ADMIN \
#   --security-opt seccomp=unconfined \
#   --shm-size=16g \
#   --network host \
#   --ipc host \
#   --device=/dev/kfd \
#   --device=/dev/dri \
#   --device=/dev/mem \
#   -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
#   -v $MODEL_PATH:/model:ro \
#   $SGLANG_IMAGE \
#   bash -c 'SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 \
#     sglang serve \
#       --model-path /model \
#       --served-model-name kimi-k2.6 \
#       --tp 8 \
#       --trust-remote-code \
#       --reasoning-parser kimi_k2 \
#       --tool-call-parser kimi_k2 \
#       --decode-attention-backend aiter \
#       --prefill-attention-backend aiter \
#       --kv-cache-dtype fp8_e4m3 \
#       --mem-fraction-static 0.75 \
#       --cuda-graph-backend-decode disabled \
#       --disable-radix-cache \
#       --host 0.0.0.0 \
#       --port 8000'
#
# docker logs -f $CONTAINER_NAME
#
# ---------------------------------------------------------------------------
# STEP 3 — wait for server (curl only, no probe)
# ---------------------------------------------------------------------------
# for i in $(seq 1 40); do
#   curl -sf --max-time 10 http://127.0.0.1:8000/v1/models && echo ready && break
#   echo "waiting $i/40..."; sleep 15
# done
#
# ---------------------------------------------------------------------------
# STEP 4a — HOLDOUT client (50 traj, ~103 min) in tmux
# ---------------------------------------------------------------------------
# cd $REPO_ROOT
# REP=rep1
# REPORT_DIR=logs/kimi_agentic_holdout_50_int4_${REP}
# mkdir -p $REPORT_DIR
# sed "s|^report_dir:.*|report_dir: ${REPORT_DIR}|" \
#   examples/10_Agentic_Inference/kimi_agentic_benchmark_holdout50_int4.yaml \
#   > /tmp/kimi_holdout_${REP}.yaml
# tmux new-session -d -s kimi-int4-holdout-baseline \
#   "cd $REPO_ROOT && uv run inference-endpoint benchmark from-config \
#    --config /tmp/kimi_holdout_${REP}.yaml 2>&1 | tee ${REPORT_DIR}/run.log; exec bash"
# tail -f ${REPORT_DIR}/run.log
#
# ---------------------------------------------------------------------------
# STEP 4b — FULL V4 client (990 traj, ~2 days) in tmux
# ---------------------------------------------------------------------------
# cd $REPO_ROOT
# REPORT_DIR=logs/kimi_agentic_v4_int4
# mkdir -p $REPORT_DIR
# sed "s|^report_dir:.*|report_dir: ${REPORT_DIR}|" \
#   examples/10_Agentic_Inference/kimi_agentic_benchmark_v4_full.yaml \
#   > /tmp/kimi_v4_full.yaml
# tmux new-session -d -s kimi-int4-v4-full \
#   "cd $REPO_ROOT && uv run inference-endpoint benchmark from-config \
#    --config /tmp/kimi_v4_full.yaml 2>&1 | tee ${REPORT_DIR}/run.log; exec bash"
# tail -f ${REPORT_DIR}/run.log
#
# ---------------------------------------------------------------------------
# MONITOR / STOP
# ---------------------------------------------------------------------------
# docker ps --filter name=$CONTAINER_NAME
# curl -sf http://127.0.0.1:8000/v1/models
# tmux ls
# tail -f logs/kimi_agentic_holdout_50_int4_rep1/run.log
# tail -f logs/kimi_agentic_v4_int4/run.log
# docker rm -f $CONTAINER_NAME
#
# Do NOT run probe before benchmark. Keep server alive for the full client run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/data/workloads-inference/models/Kimi-K2.6-INT4}"
HF_MODEL_ID="${HF_MODEL_ID:-moonshotai/Kimi-K2.6}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-kimi-k26-int4}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang-rocm:v0.5.14-rocm700-mi35x-20260626}"
HOLDOUT_YAML="${SCRIPT_DIR}/kimi_agentic_benchmark_holdout50_int4.yaml"
V4_YAML="${SCRIPT_DIR}/kimi_agentic_benchmark_v4_full.yaml"
HOLDOUT_REPORT_PREFIX="${HOLDOUT_REPORT_PREFIX:-logs/kimi_agentic_holdout_50_int4_rep}"
V4_REPORT_DIR="${V4_REPORT_DIR:-logs/kimi_agentic_v4_int4}"
TMUX_HOLDOUT="${TMUX_HOLDOUT:-kimi-int4-holdout-baseline}"
TMUX_V4="${TMUX_V4:-kimi-int4-v4-full}"
SERVE_PORT="${SERVE_PORT:-8000}"

cd "${REPO_ROOT}"

patch_report_dir() {
  local src="$1"
  local dst="$2"
  local report_dir="$3"
  sed "s|^report_dir:.*|report_dir: ${report_dir}|" "${src}" > "${dst}"
}

case "${1:-help}" in
  download)
    mkdir -p "${MODEL_PATH}"
    uv run hf download "${HF_MODEL_ID}" --local-dir "${MODEL_PATH}"
    ;;

  start-server)
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
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
          --host 0.0.0.0 \
          --port 8000'
    echo "Started ${CONTAINER_NAME}.  docker logs -f ${CONTAINER_NAME}"
    ;;

  stop-server)
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    echo "Stopped ${CONTAINER_NAME}"
    ;;

  wait-endpoint)
    for i in $(seq 1 "${2:-40}"); do
      if curl -sf --max-time 10 "http://127.0.0.1:${SERVE_PORT}/v1/models" >/dev/null 2>&1; then
        echo "Endpoint ready"
        exit 0
      fi
      echo "waiting ${i}/${2:-40}..."
      sleep 15
    done
    echo "Endpoint not ready" >&2
    exit 1
    ;;

  logs-server)
    docker logs -f "${CONTAINER_NAME}"
    ;;

  holdout)
    REP="${2:-rep1}"
    REPORT_DIR="${HOLDOUT_REPORT_PREFIX}${REP}"
    CONFIG="/tmp/kimi_holdout_${REP}.yaml"
    mkdir -p "${REPORT_DIR}"
    patch_report_dir "${HOLDOUT_YAML}" "${CONFIG}" "${REPORT_DIR}"
    tmux new-session -d -s "${TMUX_HOLDOUT}" \
      "cd ${REPO_ROOT} && uv run inference-endpoint benchmark from-config \
       --config ${CONFIG} 2>&1 | tee ${REPORT_DIR}/run.log; exec bash"
    echo "Holdout started: tmux attach -t ${TMUX_HOLDOUT}"
    echo "Monitor: tail -f ${REPORT_DIR}/run.log"
    ;;

  v4-full)
    REPORT_DIR="${V4_REPORT_DIR}"
    CONFIG="/tmp/kimi_v4_full.yaml"
    mkdir -p "${REPORT_DIR}"
    patch_report_dir "${V4_YAML}" "${CONFIG}" "${REPORT_DIR}"
    tmux new-session -d -s "${TMUX_V4}" \
      "cd ${REPO_ROOT} && uv run inference-endpoint benchmark from-config \
       --config ${CONFIG} 2>&1 | tee ${REPORT_DIR}/run.log; exec bash"
    echo "V4-full started: tmux attach -t ${TMUX_V4}"
    echo "Monitor: tail -f ${REPORT_DIR}/run.log"
    ;;

  monitor)
    case "${2:-}" in
      holdout) tail -f "${HOLDOUT_REPORT_PREFIX}${3:-rep1}/run.log" ;;
      v4)      tail -f "${V4_REPORT_DIR}/run.log" ;;
      *)       echo "Usage: $0 monitor holdout [rep] | monitor v4" >&2; exit 1 ;;
    esac
    ;;

  help | *)
    echo "Usage: $0 {download|start-server|stop-server|wait-endpoint|logs-server|holdout|v4-full|monitor}"
    echo "Benchmark YAML: ${HOLDOUT_YAML}  and  ${V4_YAML}"
    echo "Copy-paste commands: see comment block at top of this file."
    ;;
esac
