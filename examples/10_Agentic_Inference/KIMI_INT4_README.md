# Kimi K2.6 INT4 — Holdout-50 Agentic Benchmark Run

This documents the **verified holdout run** on 8× MI300X (ROCm): SGLang INT4 server + `inference-endpoint` benchmark client. No probe step before the client.

**Successful run:** `logs/kimi_agentic_holdout_50_int4_rep1/` — ~103 minutes, **70.12%** overall accuracy (1233/1233 turns, 0 failures).

---

## What you need

- 8× MI300X / MI35x with ROCm
- Docker, tmux, curl, [uv](https://docs.astral.sh/uv/)
- ~600 GB disk for model weights
- This repo cloned with datasets:
  - `agentic_combined_v4_holdout_50.jsonl` (50 trajectories)
- Benchmark config: `examples/10_Agentic_Inference/kimi_agentic_benchmark_holdout50_int4.yaml`

---

## Paths (edit for your machine)

```bash
export REPO_ROOT=/home/hmahmad/endpoints
export MODEL_PATH=/data/workloads-inference/models/Kimi-K2.6-INT4
export CONTAINER_NAME=sglang-kimi-k26-int4
export SGLANG_IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm700-mi35x-20260626
```

On a new host, also update paths inside `kimi_agentic_benchmark_holdout50_int4.yaml`:

- `model_params.tokenizer_name` → your `MODEL_PATH`
- `datasets[0].path` → your holdout jsonl path

---

## Step 1 — Download model (once)

```bash
cd /home/hmahmad/endpoints
mkdir -p /data/workloads-inference/models/Kimi-K2.6-INT4
uv run hf download moonshotai/Kimi-K2.6 --local-dir /data/workloads-inference/models/Kimi-K2.6-INT4
```

Expect 64 safetensors shards (~555 GB).

---

## Step 2 — Start SGLang server

Remove any old container, then launch:

```bash
docker rm -f sglang-kimi-k26-int4 2>/dev/null || true

docker run -d \
  --name sglang-kimi-k26-int4 \
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
  -v /data/workloads-inference/models/Kimi-K2.6-INT4:/model:ro \
  lmsysorg/sglang-rocm:v0.5.14-rocm700-mi35x-20260626 \
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
```

Watch until the model is loaded (~10–20 min):

```bash
docker logs -f sglang-kimi-k26-int4
```

---

## Step 3 — Wait for endpoint

Use curl only. **Do not run probe** before the benchmark.

```bash
for i in $(seq 1 40); do
  curl -sf --max-time 10 http://127.0.0.1:8000/v1/models && echo "ready" && break
  echo "waiting $i/40..."
  sleep 15
done
```

---

## Step 4 — Run holdout benchmark in tmux

50 trajectories, concurrency 8, ~103 minutes.

Pick a run label (e.g. `rep1`) so each repeat gets its own log directory:

```bash
cd /home/hmahmad/endpoints

REP=rep1
REPORT_DIR=logs/kimi_agentic_holdout_50_int4_${REP}
mkdir -p ${REPORT_DIR}

sed "s|^report_dir:.*|report_dir: ${REPORT_DIR}|" \
  examples/10_Agentic_Inference/kimi_agentic_benchmark_holdout50_int4.yaml \
  > /tmp/kimi_holdout_${REP}.yaml

tmux new-session -d -s kimi-int4-holdout-baseline \
  "cd /home/hmahmad/endpoints && uv run inference-endpoint benchmark from-config \
   --config /tmp/kimi_holdout_${REP}.yaml 2>&1 | tee ${REPORT_DIR}/run.log; exec bash"
```

The `exec bash` at the end keeps the tmux session open after the benchmark finishes.

---

## Step 5 — Monitor

```bash
tail -f logs/kimi_agentic_holdout_50_int4_rep1/run.log
```

```bash
tmux attach -t kimi-int4-holdout-baseline
```

```bash
docker ps --filter name=sglang-kimi-k26-int4
curl -sf http://127.0.0.1:8000/v1/models
```

---

## Step 6 — Results

After completion, check:

```bash
cat logs/kimi_agentic_holdout_50_int4_rep1/scores.json
cat logs/kimi_agentic_holdout_50_int4_rep1/report.txt
cat logs/kimi_agentic_holdout_50_int4_rep1/results.json
```

Rep1 reference numbers:

| Metric | Value |
|--------|-------|
| Duration | ~103 min |
| Turns | 1233 issued, 1233 successful, 0 failed |
| Overall accuracy | 70.12% |
| Coding | 67.17% |
| Workflow | 93.84% |
| Mean OSL (per turn) | ~320 tokens |

---

## Stop server

```bash
docker rm -f sglang-kimi-k26-int4
```

---

## Notes

- **INT4 server flags:** `SGLANG_ROCM_FUSED_DECODE_MLA=0` avoids a ROCm MLA crash. Use `aiter` for both prefill and decode (not triton decode). Do not use MXFP4-only flags on INT4.
- **Client:** plain `uv run inference-endpoint benchmark from-config` — no `uv sync` extras required, no probe preflight.
- **YAML settings that matter:** `target_concurrency: 8`, `enable_salt: true`, `inject_tool_delay: true`, `eval_method: agentic_inference_inline`, `num_trajectories_to_issue: 50`.
- **Keep the server up** for the whole client run. If the container dies mid-run, remaining turns fail with `ConnectionRefusedError` and accuracy collapses.
