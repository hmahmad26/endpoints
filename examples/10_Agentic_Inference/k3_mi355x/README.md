# Kimi K3 Agentic Inference — 8x MI355X

Single-node AMD port of the GB200 recipe in [`../KIMI_K3_RECIPE.md`](../KIMI_K3_RECIPE.md).

Same checkpoint, same client, same measurement methodology. The serving
topology and container differ because this is one 8-GPU MI355X node rather than
four GB200 nodes.

The K3 recipe is the authority for anything K3-specific and takes precedence
over [`../README.md`](../README.md), which predates it and covers Kimi K2.6 and
Qwen3.6-35B-A3B. Where the two differ, the client configs here follow the K3
recipe: no inter-turn delay, and SWE-bench Verified run standalone rather than
through the framework's `swe_bench_scorer` dataset.

## Contents

| File                             | Purpose                                                     |
| -------------------------------- | ----------------------------------------------------------- |
| `serve_kimi_k3_mi355x.sh`        | Launch SGLang in the patched ROCm container, TP8             |
| `kimi_k3_agentic_benchmark.yaml` | Full run: one pass yields mean OSL and inline accuracy       |
| `kimi_k3_agentic_smoke.yaml`     | Short validation run                                         |
| `kimi_k3_mi355x_runbook.sh`      | Preflight, serve, health-check, and tmux-backed run wrapper  |

One full run produces both mean OSL and inline accuracy; a separate inference
pass for either is unnecessary.

## Prerequisites

**Checkpoint.** `moonshotai/Kimi-K3` at revision
`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, an MXFP4 `compressed-tensors`
checkpoint of roughly 1.5 TB. The serve script mounts the whole HF repo
directory, not the snapshot subdirectory: the snapshot tree is symlinks into
`../../blobs`, so mounting only the snapshot breaks every weight file.

**Container.** `kimi-k3-rocm:hipfallback`, built from
`rocm/sgl-dev:rocm720-mi35x-k3-20260802` with two source patches. The stock
image cannot run K3 — two kernels fail to compile for gfx950:

- `warp::inclusive_sum` passes a 32-bit mask to `__shfl_up_sync`, which is wrong
  for 64-wide AMD wavefronts. Backporting the ROCm guard from sglang `main`
  repairs both the decode CUDA graph (`tiny_gemm`) and K3's attention-residual
  kernel.
- K3's `SituAndMul` activation JIT-builds `situ_and_mul.cuh`, which includes the
  CUDA-only `cuda_fp8.h`. Pointing `forward_hip` at `forward_native` runs the
  activation in PyTorch instead of a fused kernel: correct, but slower.

## Serving topology

| GB200 recipe                                       | Here                            | Reason                                     |
| -------------------------------------------------- | ------------------------------- | ------------------------------------------ |
| `lmsysorg/sglang:kimi-k3` (ARM64)                  | `kimi-k3-rocm:hipfallback`      | CUDA image cannot run on ROCm              |
| 4 nodes x 4 GPUs = 16                              | 1 node x 8 GPUs                 | Available hardware                         |
| `--tp-size 16` plus multinode rendezvous flags     | `--tp-size 8`, no such flags    | Single node needs no rendezvous            |
| `--dcp-size 16`                                    | omitted                         | See "DCP" below                            |
| `--mem-fraction-static 0.85`                       | `0.97`                          | See "Memory" below                         |
| `SGLANG_FORCE_COARSE_WAR_BARRIER=1`                | dropped                         | CUDA-graph workaround with no ROCm analogue |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`        | kept                            | Framework-level, not vendor-specific       |
| `--mamba-full-memory-ratio 0.54`                   | kept                            | Same ~150k average request length          |
| `--mamba-radix-cache-strategy extra_buffer`        | kept                            | Same cache strategy                        |
| `--reasoning-parser` / `--tool-call-parser kimi_k3` | kept                            | Model-specific                             |
| —                                                  | `--attention-backend triton`    | See "Attention backend" below              |
| —                                                  | `--mamba-ssm-dtype bfloat16`    | Doubles `max_running_requests` from 14     |

**DCP.** K3 DCP in SGLang needs either `cutedsl_mla` with a DCP-patched
FlashInfer, or the `tokenspeed_mla` backend. Neither ships in the ROCm image, so
this runs plain TP8.

**Attention backend.** Must be `triton`. With `aiter` and context length above
8192, SGLang silently multiplies `mem_fraction_static` by 0.85, capping the
budget at 244.8 GB — below the 249.3 GB per-GPU weight footprint — so loading
fails for every value passed, including 1.0.

**Memory.** The same weights are split across fewer, larger GPUs here: roughly
188 GB per GPU of 288 GB, versus about 94 GB per GPU on GB200. The higher static
fraction is what leaves a usable KV and mamba pool at all, and it is the first
thing to adjust if concurrency proves unstable.

## Running

From the repo root:

```bash
cd examples/10_Agentic_Inference/k3_mi355x

./kimi_k3_mi355x_runbook.sh preflight       # GPUs free, port free, image present
./kimi_k3_mi355x_runbook.sh start-server    # ~5 min: weight load, then graph capture
./kimi_k3_mi355x_runbook.sh wait-endpoint
./kimi_k3_mi355x_runbook.sh check-endpoint  # one real completion

./kimi_k3_mi355x_runbook.sh smoke           # short validation run
./kimi_k3_mi355x_runbook.sh full            # mean OSL + inline accuracy
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

## Deviations from the K3 recipe

Everything in `../KIMI_K3_RECIPE.md` is matched except the following, all forced
by the hardware. Results produced here are characterization data; they are not
comparable to the recipe's GB200 results table.

**Concurrency is 28, not 64.** After weights there is roughly 21 GB per GPU
left; the mamba state cache caps `max_running_requests` at 14, doubled to 28 by
`--mamba-ssm-dtype bfloat16`. Reaching 64 at ~150k-token requests needs about
twice the aggregate memory, which is what the 16-GPU GB200 config has.

**No DCP**, and `--attention-backend triton` with a higher static memory
fraction, as described under "Serving topology" above.

**`tokenizer_name` is omitted, so mean OSL is unavailable.** The recipe sets it
and gets mean OSL from the same run. No fast tokenizer exists for K3 — it ships
only `tiktoken.model` and a Python `TikTokenTokenizer` — and the metrics
aggregator refuses to start without one, so the run reports `TPS: N/A`.
Conversion via `TikTokenConverter` does not work: K3's pattern uses regex
character-class intersection (`&&[^\p{Han}]`) that the fast tokenizer's regex
engine parses differently, and verification samples tokenize differently.
Setting `tokenizer_name` here crashes the metrics aggregator at startup, before
any request is issued.

## Dataset

Both configs read `datasets/agentic_combined_v6.jsonl`: 613 conversations
(500 workflow, 113 coding), 40,700 messages, which is the size the submission
rules require `num_trajectories_to_issue` to be a multiple of. Datasets are
gitignored, so this file has to be placed on each node separately.

Do not use the artifact published at `endpoints.mlcommons-storage.org`. It is a
truncated prefix of the same data — 268 of the 613 conversations, ending
mid-record, with a binary splice that makes it invalid UTF-8 — so
`pd.read_json(..., lines=True)` cannot open it and the run fails at dataset
load. Every conversation it does contain is present in v6. `make_clean_dataset.py`
drops the unreadable conversations from that artifact if you ever need to work
with it directly; it is not needed when v6 is available.

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
`chat_template_kwargs`. Worth noting that `preserve_thinking` exists to stop the
chat template stripping reasoning tokens from previous turns, which is a
multi-turn correctness property; whether the `kimi_k3` chat template preserves
them by default has not been verified here.

## Tuning notes

`CUDA_GRAPH_DECODE` defaults to `disabled` in the serve script. The warp patch
is intended to make decode graph capture work on gfx950, so `enabled` is worth
measuring — disabling graphs costs significant decode throughput. Override with:

```bash
CUDA_GRAPH_DECODE=enabled ./serve_kimi_k3_mi355x.sh
```

The KV pool is what runs out on long agentic contexts, and the static memory
fraction is the lever. At `0.94` a full run saturated KV at 97-98% while the
mamba pool never rose above 6%, collapsing concurrency from the configured 28
down to 1-2 and cutting generation throughput from 200-500 tok/s to 66 tok/s.

`--mamba-full-memory-ratio 0.54` is not the culprit and should not be lowered:
it sizes the mamba cache at 141 states, which at 5 slots per request is exactly
the 28 concurrent requests we ask for. Reducing it would cut concurrency.

The waste is elsewhere. At `0.94` the pool allocator reported `avail mem=16.88 GB`
per GPU still unallocated, while KV received only 7.16 GB (278,159 tokens).
Raising the static fraction hands that headroom to KV, which is why the default
here is `0.97`. Lower it if weight loading OOMs.

Wall-clock budgeting should use the full run duration, not the performance
window. Beware extrapolating from early progress: throughput is high while
contexts are short and degrades sharply as conversation history accumulates and
the KV pool fills. Those differ substantially: the performance window closes when the first
user finishes its final trajectory, while the remaining trajectories continue to
drain for accuracy. Calibrate from the smoke run's `duration_s` in
`accuracy/accuracy_results.json`, not the `Duration` line in `report.txt`.
