# Kimi K3 Agentic Inference — 8x MI355X

Single-node AMD port of the client methodology in
[`../KIMI_K3_RECIPE.md`](../KIMI_K3_RECIPE.md), with serving aligned to the
official SGLang cookbook cell:

[MI355X / Unified / Balanced / MXFP4 / Non-Spec](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3#hw=mi355x&pdMode=unified&strategy=balanced&quant=mxfp4&mmTransport=auto&spec=none&hicache=off)

Same checkpoint family and measurement approach as the GB200 recipe. Topology
and container follow the MI355X cookbook (1×8 GPUs) rather than four GB200
nodes.

The K3 recipe is the authority for anything K3-*client*-specific and takes
precedence over [`../README.md`](../README.md), which predates it and covers
Kimi K2.6 and Qwen3.6-35B-A3B. Where the two differ, the client configs here
follow the K3 recipe: no inter-turn delay, and SWE-bench Verified run standalone
rather than through the framework's `swe_bench_scorer` dataset.

## Contents

| File                             | Purpose                                                     |
| -------------------------------- | ----------------------------------------------------------- |
| `serve_kimi_k3_mi355x.sh`        | Launch SGLang per the MI355X Balanced cookbook cell, TP8    |
| `kimi_k3_agentic_benchmark.yaml` | Full run: agentic replay + inline accuracy                  |
| `kimi_k3_agentic_smoke.yaml`     | Short validation run                                        |
| `kimi_k3_mi355x_runbook.sh`      | Preflight, serve, health-check, and tmux-backed run wrapper |

One full run produces inline accuracy. Mean OSL / TPS need a usable fast
`tokenizer_name`; see "Deviations" below.

## Prerequisites

**Checkpoint.** `moonshotai/Kimi-K3` at revision
`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, an MXFP4 `compressed-tensors`
checkpoint of roughly 1.5 TB. The serve script mounts the whole HF repo
directory, not the snapshot subdirectory: the snapshot tree is symlinks into
`../../blobs`, so mounting only the snapshot breaks every weight file.

**Container.** Cookbook image
`lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260910`. Override with `IMAGE=…`
if needed. Pull once:

```bash
docker pull lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260910
```

## Serving topology

Cookbook MI355X Unified / Balanced / MXFP4 / Non-Spec (also listed for MI350X):

| Setting | Value |
| ------- | ----- |
| Image | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260910` |
| Env | `SGLANG_USE_AITER=1`, `SGLANG_AITER_K3_OPT=1`, `AITER_FLYDSL_FORCE=1`, `AITER_SITUV2_A8W4=1` |
| `--tp-size` | `8` |
| `--attention-backend` | `triton` |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--dtype` | `bfloat16` |
| `--mem-fraction-static` | `0.85` |
| `--cuda-graph-max-bs-decode` | `256` |
| `--reasoning-parser` / `--tool-call-parser` | `kimi_k3` |

Local additions in `serve_kimi_k3_mi355x.sh`:

- `--served-model-name kimi-k3` so the client YAMLs match
- Mount of the pinned local HF revision (no Hub re-download)

Compared to the GB200 recipe in `../KIMI_K3_RECIPE.md`: single node (no
`--nnodes` / rendezvous), no `--dcp-size`, no `SGLANG_FORCE_COARSE_WAR_BARRIER`,
and the AMD cookbook env/flags above instead of the ARM64 `lmsysorg/sglang:kimi-k3`
image.

## Running

From the repo root:

```bash
cd examples/10_Agentic_Inference/k3_mi355x

./kimi_k3_mi355x_runbook.sh preflight       # GPUs free, port free, image present
./kimi_k3_mi355x_runbook.sh start-server    # weight load, then graph capture
./kimi_k3_mi355x_runbook.sh wait-endpoint
./kimi_k3_mi355x_runbook.sh check-endpoint  # one real completion

./kimi_k3_mi355x_runbook.sh smoke           # short validation run
./kimi_k3_mi355x_runbook.sh full            # full agentic + inline accuracy
```

Do not run `inference-endpoint probe` before the benchmark, and keep the server
alive for the whole client run.

Standalone SWE-bench Verified, per the K3 recipe, against the same endpoint:

```bash
./kimi_k3_mi355x_runbook.sh swebench
```

That wraps `mini-extra swebench` over
`../accuracy/tmp_scripts/kimi_swebench_local.yaml` at `--slice 0:200`. Point the
config's `api_base` at the served endpoint before running it.

## Deviations from the GB200 K3 client recipe

Results here are characterization data on MI355X; they are not comparable to the
GB200 results table in `../KIMI_K3_RECIPE.md`.

**Concurrency is 14, not 64.** The GB200 client uses 64. The official MI355X
Balanced cell does not set `--mamba-ssm-dtype bfloat16`, so the mamba/KDA state
pool typically admits on the order of ~14 concurrent long requests. The client
YAMLs use `target_concurrency: 14` to match; higher values mostly queue. Raise
client concurrency only after confirming a higher `max_running_requests` on the
live server.

**`tokenizer_name` is omitted, so mean OSL is unavailable.** The GB200 recipe
sets it and gets mean OSL from the same run. No fast tokenizer exists for K3 —
it ships only `tiktoken.model` and a Python `TikTokenTokenizer` — and the
metrics aggregator refuses to start without one, so the run reports `TPS: N/A`.
Setting `tokenizer_name` here crashes the metrics aggregator at startup.

## Dataset

Both configs read `datasets/agentic_combined_v6.jsonl`: 613 conversations
(500 workflow, 113 coding), 40,700 messages, which is the size the submission
rules require `num_trajectories_to_issue` to be a multiple of. Place the file
under `examples/10_Agentic_Inference/datasets/` on each node (it is not always
present in a fresh clone).

Do not use the artifact published at `endpoints.mlcommons-storage.org` if it is
truncated or invalid UTF-8; see prior notes in git history / `make_clean_dataset.py`
if you must salvage that download. Prefer v6 when available.

## Relationship to the older agentic guide

[`../README.md`](../README.md) predates the K3 recipe and describes the Kimi
K2.6 and Qwen3.6-35B-A3B workloads. Two of its requirements are deliberately not
followed here because the K3 recipe supersedes them:

- It requires `inject_tool_delay: true`; the K3 recipe replays with no
  inter-turn delay, and its reported numbers come from that setting.
- It requires a `swe_bench_scorer` accuracy dataset driven by the SWE-bench
  service; the K3 recipe runs SWE-bench Verified standalone through
  `mini-swe-agent` instead.

Its sampling parameters also do not apply: the authoritative sets there cover
K2.6 (`top_p: 0.95`, `chat_template_kwargs.thinking`, `preserve_thinking`) and
Qwen3.6, while the K3 recipe specifies `top_p: 1.0` and no
`chat_template_kwargs`.

## Tuning notes

Prefer changing knobs via the
[SGLang Kimi-K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
playground (for example `--mamba-full-memory-ratio` from the mamba calculator)
rather than inventing MI355X-only forks of the Balanced cell.

After boot, confirm admission from server logs (`max_running_requests` / pool
sizes) before raising `target_concurrency` in the client YAML.

Wall-clock budgeting should use the full run duration, not only the performance
window. Throughput is high while contexts are short and degrades as history
grows and the KV pool fills. Calibrate from the smoke run's `duration_s` in
`accuracy/accuracy_results.json`, not only the `Duration` line in `report.txt`.
