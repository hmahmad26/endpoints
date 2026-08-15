# Kimi K3 Recipe

## SGLang

Use checkpoint `moonshotai/Kimi-K3` at revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`. Use the GB200/ARM64 `lmsysorg/sglang:kimi-k3` image pinned to digest `sha256:c249615b7bd28698d8c4a7682277aa9a8a6e284023791381daa97b0fdab2be2b`.

### Serve

Use the official SGLang [`GB200 / Unified / Balanced / Non-Spec`](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3#hw=gb200&pdMode=unified&strategy=balanced&spec=none&hicache=off) topology: four GB200 nodes (16 GPUs), TP16, DCP16, and no speculative decoding.

```bash
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

sglang serve \
  --trust-remote-code \
  --model-path /models/Kimi-K3 \
  --tp-size 16 \
  --nnodes 4 \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${HEAD_NODE}:20000" \
  --dcp-size 16 \
  --mem-fraction-static 0.85 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mamba-full-memory-ratio 0.54 \
  --mamba-radix-cache-strategy extra_buffer \
  --host 0.0.0.0 \
  --port 30000
```

The `0.54` Mamba ratio assumes an average total request length of 150,000 tokens. On the tested `lmsysorg/sglang:kimi-k3` image, use `SGLANG_FORCE_COARSE_WAR_BARRIER=1` together with `--mamba-radix-cache-strategy extra_buffer`. This fixed the observed low-batch CUDA-graph launch failures while leaving overlap scheduling and decode CUDA graphs enabled.

### Mean OSL and inline accuracy

Save this as `sglang-endpoint.yaml`. One run replays all 613 trajectories at concurrency 64 with no inter-turn delay. It produces both mean OSL and inline accuracy; it is not necessary to run a separate inference pass for either value.

```yaml
name: "kimi-k3-agentic-combined-v6-sglang-gb200-c64-t613"
type: "online"

model_params:
  name: "/models/Kimi-K3"
  tokenizer_name: "${MODEL_DIR}"
  temperature: 1.0
  top_p: 1.0
  max_new_tokens: 8192
  streaming: "on"

datasets:
  - name: agentic_combined
    type: performance
    path: examples/10_Agentic_Inference/datasets/mlperf_agentic_inference_dataset.jsonl
    accuracy_config:
      eval_method: agentic_inference_inline
      num_repeats: 1
    agentic_inference:
      turn_timeout_s: 86400.0
      enable_salt: true
      inject_tool_delay: false
      num_trajectories_to_issue: 613
      stop_issuing_on_first_user_complete: false

settings:
  runtime:
    min_duration_ms: 0
    max_duration_ms: 0
    scheduler_random_seed: 42
    dataloader_random_seed: 42
  load_pattern:
    type: agentic_inference
    target_concurrency: 64
  client:
    num_workers: 4
    warmup_connections: 0
    max_connections: -1
    max_idle_time: 0.5
    worker_initialization_timeout: 600.0

endpoint_config:
  endpoints:
    - "${ENDPOINT_URL}"
  api_type: openai

report_dir: "${REPORT_DIR}"
```

```bash
inference-endpoint benchmark from-config --config sglang-endpoint.yaml
```

### Standalone SWE-bench Verified accuracy

````yaml
agent:
  system_template: |
    You are a helpful assistant that can interact with a computer shell to solve programming tasks.
  instance_template: |
    <pr_description>
    Consider the following PR description:
    {{task}}
    </pr_description>

    <instructions>
    # Task Instructions

    ## Overview

    You're a software engineer interacting continuously with a computer by submitting commands.
    You'll be helping implement necessary changes to meet requirements in the PR description.
    Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
    <IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

    For each response:

    1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
    2. Provide one or more bash tool calls to execute

    ## Important Boundaries

    - MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
    - DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

    ## Recommended Workflow

    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust

    ## Command Execution Rules

    You are operating in an environment where

    1. You issue at least one command
    2. The system executes the command(s) in a subshell
    3. You see the result(s)
    4. You write your next command(s)

    Each response should include:

    1. **Reasoning text** where you explain your analysis and plan
    2. At least one tool call with your command

    **CRITICAL REQUIREMENTS:**

    - Your response SHOULD include reasoning text explaining what you're doing
    - Your response MUST include AT LEAST ONE bash tool call. You can make MULTIPLE tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
    - Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
    - However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

    Example of a CORRECT response:
    <example_response>
    I need to understand the Builder-related code. Let me find relevant files and check the project structure.

    [Makes multiple bash tool calls: {"command": "ls -la"}, {"command": "find src -name '*.java' | grep -i builder"}, {"command": "cat README.md | head -50"}]
    </example_response>

    ## Environment Details

    - You have a full Linux shell environment
    - Always use non-interactive flags (-y, -f) for commands
    - Avoid interactive tools like vi, nano, or any that require user input
    - You can use bash commands or invoke any tool that is available in the environment
    - You can also create new tools or scripts to help you with the task
    - If a tool isn't available, you can also install it

    ## Submission

    When you've completed your work, you MUST submit your changes as a git patch.
    Follow these steps IN ORDER, with SEPARATE commands:

    Step 1: Create the patch file
    Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
    Do NOT commit your changes.

    <IMPORTANT>
    The patch must only contain changes to the specific source files you modified to fix the issue.
    Do not submit file creations or changes to any of the following files:

    - test and reproduction files
    - helper scripts, tests, or tools that you created
    - installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
    - binary or compiled files
    </IMPORTANT>

    Step 2: Verify your patch
    Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

    Step 3: Submit (EXACT command required)
    You MUST use this EXACT command to submit:

    ```bash
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
    ```

    If the command fails (nonzero exit status), it will not submit.

    <CRITICAL>
    - Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
    - If you modify patch.txt after verifying, you SHOULD verify again before submitting.
    - You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
    </CRITICAL>
    </instructions>
  step_limit: 250
  cost_limit: 3.

environment:
  cwd: "/testbed"
  timeout: 3600
  interpreter: ["bash", "-c"]
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"
  environment_class: docker
  pull_timeout: 3600
  container_timeout: 6h

model:
  cost_tracking: "ignore_errors"
  observation_template: |
    {% if output.exception_info -%}
    <exception>{{output.exception_info}}</exception>
    {% endif -%}
    <returncode>{{output.returncode}}</returncode>
    {% if output.output | length < 10000 -%}
    <output>
    {{ output.output -}}
    </output>
    {%- else -%}
    <warning>
    The output of your last command was too long.
    Please try a different command that produces less output.
    If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
    If you're using grep or find and it produced too much output, you can use a more selective search pattern.
    If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
    </warning>
    {%- set elided_chars = output.output | length - 10000 -%}
    <output_head>
    {{ output.output[:5000] }}
    </output_head>
    <elided_chars>
    {{ elided_chars }} characters elided
    </elided_chars>
    <output_tail>
    {{ output.output[-5000:] }}
    </output_tail>
    {%- endif -%}
  format_error_template: |
    Tool call error:

    <error>
    {{error}}
    </error>

    Here is general guidance on how to submit correct toolcalls:

    Every response needs to use the 'bash' tool at least once to execute commands.

    Call the bash tool with your command as the argument:
    - Tool: bash
    - Arguments: {"command": "your_command_here"}

    If you have completed your assignment, please consult the first message about how to
    submit your solution (you will not be able to continue working on this task after that).
  model_name: "moonshotai/Kimi-K3"
  model_kwargs:
    custom_llm_provider: "openai"
    api_key: "test"
    drop_params: true
    temperature: 1.0
    top_p: 1.0
    max_tokens: 8192
    parallel_tool_calls: false
    api_base: "http://127.0.0.1:30003/v1"
    timeout: 3600
````

```bash
mini-extra swebench \
  --config examples/10_Agentic_Inference/accuracy/tmp_scripts/kimi_swebench_local.yaml \
  --subset verified \
  --split test \
  --slice 0:200 \
  --workers 100 \
  --output "${SWE_OUTPUT_DIR}"
```

### Results

#### Inline accuracy

| Repeat | Mean OSL (tokens/turn) | Inline overall | Inline coding | Inline workflow |
| ------ | ---------------------- | -------------- | ------------- | --------------- |
| 1      | 435.90                 | 59.08%         | 52.41%        | 89.93%          |
| 2      | 429.23                 | 58.98%         | 52.18%        | 90.43%          |
| 3      | 431.06                 | 58.66%         | 51.86%        | 90.12%          |

#### Standalone SWE-bench Verified accuracy

| Repeat | Standalone SWE-bench Verified |
| ------ | ----------------------------- |
| 1      | 191 / 200 (95.5%)             |
| 2      | 191 / 200 (95.5%)             |
| 3      | 189 / 200 (94.5%)             |
| 4      | 189 / 200 (94.5%)             |
| 5      | 188 / 200 (94.0%)             |
