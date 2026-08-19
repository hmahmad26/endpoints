#!/usr/bin/env bash
# =============================================================================
# Kimi K3 agentic benchmark on 8x MI355X — server + client commands
# (benchmark settings live in the YAML files next to this script)
# =============================================================================
#
# Configs:
#   kimi_k3_agentic_smoke.yaml      short validation run
#   kimi_k3_agentic_benchmark.yaml  full run: mean OSL + inline accuracy
#
# One full run produces both mean OSL and inline accuracy; no separate
# inference pass is needed for either. SWE-bench Verified is a standalone
# mini-swe-agent evaluation against the same endpoint.
#
# Subcommands:
#   ./kimi_k3_mi355x_runbook.sh preflight
#   ./kimi_k3_mi355x_runbook.sh start-server
#   ./kimi_k3_mi355x_runbook.sh wait-endpoint
#   ./kimi_k3_mi355x_runbook.sh check-endpoint
#   ./kimi_k3_mi355x_runbook.sh smoke
#   ./kimi_k3_mi355x_runbook.sh full
#   ./kimi_k3_mi355x_runbook.sh swebench
#   ./kimi_k3_mi355x_runbook.sh monitor smoke|full
#   ./kimi_k3_mi355x_runbook.sh stop-server
#
# Do NOT run `inference-endpoint probe` before the benchmark. Keep the server
# alive for the whole client run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

CONTAINER="${CONTAINER:-kimi-k3-server}"
PORT="${PORT:-30000}"
SMOKE_YAML="${SCRIPT_DIR}/kimi_k3_agentic_smoke.yaml"
# Runnable full run. kimi_k3_agentic_benchmark.yaml keeps the submission-shaped
# config (official dataset path, 613 trajectories) but cannot run while the
# published artifact is corrupt.
FULL_YAML="${FULL_YAML:-${SCRIPT_DIR}/kimi_k3_agentic_full_clean266.yaml}"
SMOKE_REPORT_DIR="${SMOKE_REPORT_DIR:-logs/kimi_k3_mi355x_smoke}"
FULL_REPORT_DIR="${FULL_REPORT_DIR:-logs/kimi_k3_mi355x_full_clean266}"
TMUX_SMOKE="${TMUX_SMOKE:-kimi-k3-smoke}"
TMUX_FULL="${TMUX_FULL:-kimi-k3-full}"

cd "${REPO_ROOT}"

patch_report_dir() {
  sed "s|^report_dir:.*|report_dir: $3|" "$1" > "$2"
}

launch_run() {
  local session="$1" config="$2" report_dir="$3" mode_flag="$4"
  mkdir -p "${report_dir}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session ${session} already exists. Attach or kill it first." >&2
    exit 1
  fi
  tmux new-session -d -s "${session}" \
    "cd ${REPO_ROOT} && uv run inference-endpoint benchmark from-config \
     --config ${config} ${mode_flag} 2>&1 | tee ${report_dir}/run.log; exec bash"
  echo "Started: tmux attach -t ${session}"
  echo "Monitor: tail -f ${report_dir}/run.log"
}

case "${1:-help}" in
  preflight)
    echo "--- GPUs ---"
    rocm-smi --showuse 2>/dev/null | head -20 || echo "rocm-smi unavailable"
    echo "--- port ${PORT} ---"
    ss -ltn "sport = :${PORT}" 2>/dev/null | tail -n +2 | grep -q . \
      && echo "OCCUPIED" || echo "free"
    echo "--- image ---"
    docker image inspect kimi-k3-rocm:hipfallback >/dev/null 2>&1 \
      && echo "kimi-k3-rocm:hipfallback present" || echo "MISSING: build Dockerfile.hipfallback"
    ;;

  start-server)
    "${SCRIPT_DIR}/serve_kimi_k3_mi355x.sh"
    ;;

  stop-server)
    docker rm -f "${CONTAINER}" 2>/dev/null || true
    echo "Stopped ${CONTAINER}"
    ;;

  logs-server)
    docker logs -f "${CONTAINER}"
    ;;

  wait-endpoint)
    for i in $(seq 1 "${2:-60}"); do
      if curl -sf --max-time 10 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Endpoint ready"
        exit 0
      fi
      echo "waiting ${i}/${2:-60}..."
      sleep 20
    done
    echo "Endpoint not ready" >&2
    exit 1
    ;;

  check-endpoint)
    # One real completion before committing hours to a run.
    curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"kimi-k3","messages":[{"role":"user","content":"Reply with the word ready."}],"max_tokens":32,"temperature":1.0,"top_p":1.0}'
    echo
    ;;

  smoke)
    CONFIG="/tmp/kimi_k3_smoke.yaml"
    patch_report_dir "${SMOKE_YAML}" "${CONFIG}" "${SMOKE_REPORT_DIR}"
    launch_run "${TMUX_SMOKE}" "${CONFIG}" "${SMOKE_REPORT_DIR}" ""
    ;;

  full)
    CONFIG="/tmp/kimi_k3_full.yaml"
    patch_report_dir "${FULL_YAML}" "${CONFIG}" "${FULL_REPORT_DIR}"
    launch_run "${TMUX_FULL}" "${CONFIG}" "${FULL_REPORT_DIR}" ""
    ;;

  swebench)
    # Standalone SWE-bench Verified evaluation per ../KIMI_K3_RECIPE.md. This is
    # a separate mini-swe-agent run against the same served endpoint, not the
    # framework's swe_bench_scorer dataset.
    SWE_OUTPUT_DIR="${SWE_OUTPUT_DIR:-${REPO_ROOT}/logs/kimi_k3_mi355x_swebench}"
    mkdir -p "${SWE_OUTPUT_DIR}"
    mini-extra swebench \
      --config examples/10_Agentic_Inference/accuracy/tmp_scripts/kimi_swebench_local.yaml \
      --subset verified \
      --split test \
      --slice 0:200 \
      --workers "${SWE_WORKERS:-100}" \
      --output "${SWE_OUTPUT_DIR}"
    ;;

  monitor)
    case "${2:-}" in
      smoke) tail -f "${SMOKE_REPORT_DIR}/run.log" ;;
      full)  tail -f "${FULL_REPORT_DIR}/run.log" ;;
      *) echo "Usage: $0 monitor smoke|full" >&2; exit 1 ;;
    esac
    ;;

  help | *)
    echo "Usage: $0 {preflight|start-server|stop-server|logs-server|wait-endpoint|check-endpoint|smoke|full|swebench|monitor}"
    echo "Configs: ${SMOKE_YAML}"
    echo "         ${FULL_YAML}"
    ;;
esac
