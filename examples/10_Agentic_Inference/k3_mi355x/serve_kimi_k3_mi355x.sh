#!/usr/bin/env bash
# Serve Kimi K3 for the agentic inference benchmark on a single 8x MI355X node.
#
# Single-node analogue of the GB200 "Unified / Balanced / Non-Spec" topology in
# ../KIMI_K3_RECIPE.md: TP8 instead of TP16/DCP16, ROCm image instead of the
# ARM64 CUDA one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# The stock rocm/sgl-dev:rocm720-mi35x-k3-20260802 image cannot run K3: two of
# its kernels fail to compile for gfx950. Build Dockerfile.hipfallback first.
IMAGE="${IMAGE:-kimi-k3-rocm:hipfallback}"

# The HF snapshot tree is symlinks into ../../blobs, so the whole repo directory
# must be mounted; mounting only the snapshot breaks every weight file.
REPO_DIR="${REPO_DIR:-/data/workloads-inference/hf_hub_cache/models--moonshotai--Kimi-K3}"
REVISION="${REVISION:-9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
MODEL_PATH="/models/kimi-k3-repo/snapshots/${REVISION}"

CONTAINER="${CONTAINER:-kimi-k3-server}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"

# DCP requires either cutedsl_mla with a DCP-patched FlashInfer or the
# tokenspeed_mla backend; neither ships in the ROCm image, so this runs plain
# TP8. The aiter backend silently multiplies mem_fraction_static by 0.85 when
# context_len > 8192, capping the budget below the 249.3 GB/GPU weight
# footprint, so it can never load K3 regardless of the value passed.
ATTN_BACKEND="${ATTN_BACKEND:-triton}"
# The ~16.9 GB/GPU the allocator leaves free at 0.94 is the runtime working set
# (activations, chunked-prefill buffers, CUDA-graph private pools), not waste.
# Raising this to 0.97 loads fine and nearly doubles the KV pool, but the
# scheduler then OOMs on a routine allocation about 90 seconds into real agentic
# traffic and SIGQUITs the server. Do not raise it to buy KV capacity.
MEM_FRACTION="${MEM_FRACTION:-0.94}"

# Halves the SSM state: without it the mamba pool caps max_running_requests
# at 14, which is below any useful concurrency.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
# Assumes ~150k average total request length, same as the GB200 recipe.
MAMBA_RATIO="${MAMBA_RATIO:-0.54}"

# patch_warp.py in Dockerfile.hipfallback repairs the tiny_gemm warp mask that
# blocked decode graph capture on gfx950, so graphs can be captured on the
# patched image. Accepted values: full, breakable, tc_piecewise, disabled.
CUDA_GRAPH_DECODE="${CUDA_GRAPH_DECODE:-disabled}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_mi355x_serve}"

mkdir -p "${LOG_DIR}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: image ${IMAGE} not found. Build it with Dockerfile.hipfallback." >&2
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
  -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --served-model-name kimi-k3 \
    --tp-size "${TP_SIZE}" \
    --attention-backend "${ATTN_BACKEND}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS}" \
    --cuda-graph-backend-decode "${CUDA_GRAPH_DECODE}" \
    --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}" \
    --mamba-full-memory-ratio "${MAMBA_RATIO}" \
    --mamba-radix-cache-strategy extra_buffer \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --host 0.0.0.0 \
    --port "${PORT}"

echo "Started ${CONTAINER} on port ${PORT} (image ${IMAGE}, TP${TP_SIZE})."
echo "Logs:   docker logs -f ${CONTAINER}"
echo "Ready:  until curl -sf http://localhost:${PORT}/health; do sleep 20; done"
