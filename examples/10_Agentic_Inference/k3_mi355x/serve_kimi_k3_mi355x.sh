#!/usr/bin/env bash
# Serve Kimi K3 for the agentic inference benchmark on a single 8x MI355X node.
#
# Matches the official SGLang cookbook cell:
#   hw=mi355x, pdMode=unified, strategy=balanced, quant=mxfp4, spec=none, hicache=off
#   https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3#hw=mi355x&pdMode=unified&strategy=balanced&quant=mxfp4&mmTransport=auto&spec=none&hicache=off
#
# Local additions (not in the cookbook command): --served-model-name for the
# client YAMLs in this directory, and a pinned HF revision mounted from the
# shared cache so weights are not re-downloaded.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Cookbook image for MI350X / MI355X ROCm.
IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260910}"

# The HF snapshot tree is symlinks into ../../blobs, so the whole repo directory
# must be mounted; mounting only the snapshot breaks every weight file.
REPO_DIR="${REPO_DIR:-/data/workloads-inference/hf_hub_cache/models--moonshotai--Kimi-K3}"
REVISION="${REVISION:-9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
MODEL_PATH="/models/kimi-k3-repo/snapshots/${REVISION}"

CONTAINER="${CONTAINER:-kimi-k3-server}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"

# Cookbook Balanced cell flags (defaults match the published command).
ATTN_BACKEND="${ATTN_BACKEND:-triton}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
DTYPE="${DTYPE:-bfloat16}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-256}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_mi355x_serve}"

mkdir -p "${LOG_DIR}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: image ${IMAGE} not found. Pull it with:" >&2
  echo "  docker pull ${IMAGE}" >&2
  exit 1
fi

if [[ ! -d "${REPO_DIR}/snapshots/${REVISION}" ]]; then
  echo "ERROR: checkpoint ${REPO_DIR}/snapshots/${REVISION} not found." >&2
  exit 1
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --network=host --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 128g \
  -v "${REPO_DIR}:/models/kimi-k3-repo:ro" \
  -v "${LOG_DIR}:/logs" \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_AITER_K3_OPT=1 \
  -e AITER_FLYDSL_FORCE=1 \
  -e AITER_SITUV2_A8W4=1 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --served-model-name kimi-k3 \
    --tp-size "${TP_SIZE}" \
    --attention-backend "${ATTN_BACKEND}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --dtype "${DTYPE}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS}" \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --host 0.0.0.0 \
    --port "${PORT}"

echo "Started ${CONTAINER} on port ${PORT} (image ${IMAGE}, TP${TP_SIZE})."
echo "Cookbook: MI355X Unified / Balanced / MXFP4 / Non-Spec"
echo "Logs:   docker logs -f ${CONTAINER}"
echo "Ready:  until curl -sf http://localhost:${PORT}/health; do sleep 20; done"
