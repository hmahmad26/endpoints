#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a Perfetto-compatible Chrome Trace Event JSON from a multi-turn benchmark run.

Reads events.jsonl and the ground-truth dataset to produce a trace viewable
at https://ui.perfetto.dev.

Tracks per concurrency slot:
    - Conversation bars on a dedicated row, hash-colored per conversation
    - Turn slices on a second row, colored by domain:
        coding:   blue (has GT tool calls) / grey (none)
        workflow: green (model intent matches GT) / red (mismatch) / grey (unscorable)
    - TTFT slices (issued → recv_first) nested inside turn slices
    - Tool delay slices (orange) between turns, sized by measured gap
    - A counter track showing in-flight request count over time

Tooltip on each turn slice differs by domain:
    coding:   gt_tool_calls vs. model_tool_calls
    workflow: gt_intent_codes vs. model_intent + intent_match (also tool calls)

Domain is auto-detected from the dataset (workflow if any row carries
intent_codes), or can be forced with --domain.

Usage:
    python generate_trace.py --report-dir logs/kimi_agentic_int4_run_01 \\
        --dataset agentic_combined_v4.jsonl \\
        [--domain auto|coding|workflow] [--out trace.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("generate_trace")

# Conversation bars get distinct colors by cycling through this palette,
# which is the set of Chrome-trace cnames that Perfetto renders reliably
# with visually distinct colors. With ~24 entries and 50 conversations, some
# repeats are expected but adjacent conversations get different colors.
_CONV_PALETTE = [
    "thread_state_running",
    "thread_state_iowait",
    "thread_state_uninterruptible",
    "thread_state_runnable",
    "rail_response",
    "rail_animation",
    "rail_idle",
    "rail_load",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "cq_build_abandoned",
    "cq_build_attempt_running",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
    "generic_work",
    "good",
    "bad",
    "terrible",
    "yellow",
    "olive",
    "startup",
    "heap_dump_stack_frame",
    "vsync_highlight_color",
]

# Blue for turns with tool calls, grey for turns without.
_TOOL_CALL_COLOR = "rail_load"
_NO_TOOL_CALL_COLOR = "generic_work"

_TTFT_COLOR = "thread_state_sleeping"
_DELAY_COLOR = "thread_state_iowait"  # orange — semantically "waiting"

# Reliably-rendered Perfetto cnames for match/mismatch
_MATCH_COLOR = "thread_state_running"          # green
_MISMATCH_COLOR = "thread_state_uninterruptible"  # red

# Minimal exe extraction for tooltip — mirrors score_inline_accuracy.py logic.
_HEREDOC_RE = re.compile(
    r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?[\s\S]*?\n\1\s*$",
    re.MULTILINE,
)
_QUOTED_RE = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")
_STAGE_SEP_RE = re.compile(r"\|\||\||&&|;|\n")
_ENVKV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPERS = {"env", "time", "nice", "sudo", "exec", "command"}
_PATH_LEAF = re.compile(r"[^/]+$")

# Intent code extraction for workflow domain — mirrors score_inline_accuracy.py.
_INTENT_RE = re.compile(r"\bintent:\s*(I\d{3})\b", re.IGNORECASE)
_BARE_INTENT_RE = re.compile(r"\bI(\d{3})\b")


def _extract_intent_code(content: str | None, reasoning: str | None) -> str | None:
    for text in (reasoning or "", content or ""):
        m = _INTENT_RE.search(text)
        if m:
            return m.group(1).upper()
    for text in (reasoning or "", content or ""):
        ms = list(_BARE_INTENT_RE.finditer(text))
        if ms:
            return f"I{ms[-1].group(1)}"
    return None


def _extract_tool_summary(tool_calls: list[dict] | None) -> str:
    """Return a concise summary of tool calls: 'bash(find, grep)' or 'none'."""
    if not tool_calls:
        return "none"
    parts = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = (tc.get("function") or {})
        name = fn.get("name", "?")
        if name == "bash":
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            cmd = (args.get("command") or args.get("cmd") or "") if isinstance(args, dict) else ""
            exes = _extract_bash_exes(cmd)
            parts.append(f"bash({', '.join(exes)})" if exes else "bash(?)")
        else:
            parts.append(name)
    return "; ".join(parts)


def _extract_bash_exes(cmd: str) -> list[str]:
    """Extract raw executable names from a bash command (no canonicalization)."""
    if not cmd:
        return []
    cmd = _HEREDOC_RE.sub(" ", cmd)
    cmd = _QUOTED_RE.sub(" ", cmd)
    exes = []
    for stage in _STAGE_SEP_RE.split(cmd):
        tokens = stage.split()
        i = 0
        while i < len(tokens) and (_ENVKV_RE.match(tokens[i]) or tokens[i] in _WRAPPERS):
            i += 1
        if i < len(tokens):
            leaf = _PATH_LEAF.search(tokens[i])
            if leaf:
                exes.append(leaf.group(0))
    return exes


def _load_gt_dataset(
    dataset_path: Path,
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], float], bool]:
    """Load GT rows keyed by (conversation_id, turn) and delay map.

    Returns (gt_by_key, delays, has_intent_codes). has_intent_codes is True
    if any row carries an `intent_codes` field (signals workflow domain).
    """
    gt_by_key: dict[tuple[str, int], dict] = {}
    delays: dict[tuple[str, int], float] = {}
    has_intent = False
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_type" in row:
                continue
            if "conversation_id" not in row:
                continue
            key = (row["conversation_id"], int(row["turn"]))
            gt_by_key[key] = row
            d = row.get("delay_seconds")
            if d is not None:
                delays[key] = float(d)
            if row.get("intent_codes"):
                has_intent = True
    return gt_by_key, delays, has_intent


def _load_events(
    events_path: Path,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Parse events.jsonl into issued, recv_first, and complete maps keyed by sample_uuid."""
    issued: dict[str, dict] = {}
    recv_first: dict[str, dict] = {}
    completed: dict[str, dict] = {}

    with events_path.open() as f:
        for line in f:
            ev = json.loads(line)
            et = ev.get("event_type", "")
            uuid = ev.get("sample_uuid", "")
            if not uuid:
                continue
            if et == "sample.issued":
                issued[uuid] = ev
            elif et == "sample.recv_first":
                recv_first[uuid] = ev
            elif et == "sample.complete":
                completed[uuid] = ev

    return issued, recv_first, completed


def _join_text_field(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "".join(p for p in value if isinstance(p, str))
    return None


def _merge_tool_calls(tool_calls: list | tuple | None) -> list[dict] | None:
    """Merge streamed tool-call chunks into final OpenAI-style tool call dicts."""
    if not tool_calls:
        return None
    if not isinstance(tool_calls[0], list | tuple):
        return [tc for tc in tool_calls if isinstance(tc, dict)]

    merged: dict[int, dict] = {}
    for chunk in tool_calls:
        if not isinstance(chunk, list | tuple):
            continue
        for partial in chunk:
            if not isinstance(partial, dict):
                continue
            idx = partial.get("index", 0)
            tool_call = merged.setdefault(
                int(idx), {"type": "function", "function": {"arguments": ""}}
            )
            if partial.get("id"):
                tool_call["id"] = partial["id"]
            if partial.get("type"):
                tool_call["type"] = partial["type"]
            fn = partial.get("function") or {}
            if isinstance(fn, dict):
                if fn.get("name"):
                    tool_call["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_call["function"]["arguments"] += fn["arguments"]
    return [merged[i] for i in sorted(merged)]


def _parse_model_output(data: object) -> tuple[str | None, str | None, list[dict] | None]:
    """Extract (content, reasoning, tool_calls) from a COMPLETE event's data field."""
    if isinstance(data, list):
        offset = 1 if data and data[0] == "TextModelOutput" else 0
        content = data[offset + 0] if len(data) > offset else None
        reasoning = data[offset + 1] if len(data) > offset + 1 else None
        tcs = data[offset + 2] if len(data) > offset + 2 else None
        content_str = _join_text_field(content)
        reasoning_str = _join_text_field(reasoning)
        merged_tcs = _merge_tool_calls(tcs) if isinstance(tcs, list | tuple) else None
        return content_str, reasoning_str, merged_tcs
    if isinstance(data, dict):
        return (
            data.get("content"),
            data.get("reasoning_content"),
            _merge_tool_calls(data.get("tool_calls"))
            if isinstance(data.get("tool_calls"), list | tuple)
            else None,
        )
    return None, None, None


def generate_trace(
    report_dir: Path,
    dataset_path: Path,
    out_path: Path | None = None,
    domain: str = "auto",
) -> Path:
    events_path = report_dir / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"missing {events_path}")
    if not dataset_path.exists():
        raise SystemExit(f"missing {dataset_path}")

    if out_path is None:
        out_path = report_dir / "trace.json"

    gt_by_key, delay_map, has_intent = _load_gt_dataset(dataset_path)
    if domain == "auto":
        domain = "workflow" if has_intent else "coding"
    logger.info(
        "loaded %d GT rows, %d delay entries, domain=%s (intent_codes present=%s)",
        len(gt_by_key), len(delay_map), domain, has_intent,
    )

    issued, recv_first, completed = _load_events(events_path)
    logger.info(
        "loaded %d issued, %d recv_first, %d complete events",
        len(issued), len(recv_first), len(completed),
    )

    # Build turn records
    turns: list[dict] = []
    for uuid, iss_ev in issued.items():
        comp_ev = completed.get(uuid)
        if comp_ev is None:
            continue
        rf_ev = recv_first.get(uuid)
        conv_id = iss_ev.get("conversation_id", "")
        turn_num = int(iss_ev.get("turn", 0))

        # GT: client turn is at (conv_id, turn_num), assistant response at turn_num+1
        gt_client = gt_by_key.get((conv_id, turn_num))
        gt_assistant = gt_by_key.get((conv_id, turn_num + 1))

        # Parse model output: content, reasoning, tool_calls
        model_content, model_reasoning, model_tcs = _parse_model_output(comp_ev.get("data"))

        # GT assistant's expected tool calls and intent codes
        gt_tcs = (gt_assistant.get("tool_calls") or []) if gt_assistant else []
        gt_intent_codes = (gt_assistant.get("intent_codes") or []) if gt_assistant else []

        # Model's extracted intent code (workflow domain)
        model_intent = _extract_intent_code(model_content, model_reasoning)

        turns.append({
            "uuid": uuid,
            "conv_id": conv_id,
            "turn": turn_num,
            "issued_ns": iss_ev["timestamp_ns"],
            "recv_first_ns": rf_ev["timestamp_ns"] if rf_ev else None,
            "complete_ns": comp_ev["timestamp_ns"],
            "client_role": gt_client.get("role", "?") if gt_client else "?",
            "gt_tool_calls": gt_tcs,
            "model_tool_calls": model_tcs or [],
            "has_gt_tool_calls": bool(gt_tcs),
            "gt_intent_codes": gt_intent_codes,
            "model_intent": model_intent,
            "turn_domain": (
                "workflow" if gt_intent_codes else "coding"
            ) if domain == "auto" else domain,
        })

    turns.sort(key=lambda t: t["issued_ns"])
    t0_ns = turns[0]["issued_ns"] if turns else 0

    # --- Slot assignment (conversations stick to their slot) ---
    slot_free_at: list[int] = []
    turn_slots: list[int] = []
    conv_to_slot: dict[str, int] = {}

    last_turn_idx_for_conv: dict[str, int] = {}
    for i, t in enumerate(turns):
        last_turn_idx_for_conv[t["conv_id"]] = i

    for i, t in enumerate(turns):
        conv_id = t["conv_id"]
        if conv_id in conv_to_slot:
            slot = conv_to_slot[conv_id]
        else:
            active_slots = set(conv_to_slot.values())
            slot = None
            for s in range(len(slot_free_at)):
                if s not in active_slots and slot_free_at[s] <= t["issued_ns"]:
                    slot = s
                    break
            if slot is None:
                slot = len(slot_free_at)
                slot_free_at.append(0)
            conv_to_slot[conv_id] = slot

        turn_slots.append(slot)
        slot_free_at[slot] = t["complete_ns"]

        if last_turn_idx_for_conv[conv_id] == i:
            del conv_to_slot[conv_id]

    n_slots = len(slot_free_at)
    logger.info("assigned %d turns across %d concurrency slots", len(turns), n_slots)

    # --- Build conversation spans per slot ---
    # slot -> [(conv_id, first_issued_us, last_complete_us, conv_index_in_slot)]
    slot_conv_spans: dict[int, list[dict]] = {s: [] for s in range(n_slots)}
    # Group turns by (slot, conv_id) preserving order
    slot_conv_turns: dict[tuple[int, str], list[dict]] = {}
    for i, t in enumerate(turns):
        key = (turn_slots[i], t["conv_id"])
        slot_conv_turns.setdefault(key, []).append(t)

    for (slot, conv_id), tlist in slot_conv_turns.items():
        first_us = (tlist[0]["issued_ns"] - t0_ns) / 1000.0
        last_us = (tlist[-1]["complete_ns"] - t0_ns) / 1000.0
        slot_conv_spans[slot].append({
            "conv_id": conv_id,
            "first_us": first_us,
            "last_us": last_us,
            "n_turns": len(tlist),
        })

    # Sort spans by start time per slot
    for slot in slot_conv_spans:
        slot_conv_spans[slot].sort(key=lambda s: s["first_us"])

    # --- Generate trace events ---
    # Use TWO tids per slot so Perfetto renders them as separate stacked
    # tracks: the conversation bar gets its own row above the turn row.
    #     tid layout: conv_tid(s) = s*2 + 1, turn_tid(s) = s*2 + 2
    # The counter track sits after all slot tracks.
    trace_events: list[dict] = []
    pid = 1

    def conv_tid(slot: int) -> int:
        return slot * 2 + 1

    def turn_tid(slot: int) -> int:
        return slot * 2 + 2

    trace_events.append({
        "ph": "M", "pid": pid, "tid": 0,
        "name": "process_name",
        "args": {"name": "Multi-Turn Benchmark"},
    })
    for s in range(n_slots):
        trace_events.append({
            "ph": "M", "pid": pid, "tid": conv_tid(s),
            "name": "thread_name",
            "args": {"name": f"Slot {s} — conversations"},
        })
        trace_events.append({
            "ph": "M", "pid": pid, "tid": conv_tid(s),
            "name": "thread_sort_index",
            "args": {"sort_index": s * 2},
        })
        trace_events.append({
            "ph": "M", "pid": pid, "tid": turn_tid(s),
            "name": "thread_name",
            "args": {"name": f"Slot {s} — turns"},
        })
        trace_events.append({
            "ph": "M", "pid": pid, "tid": turn_tid(s),
            "name": "thread_sort_index",
            "args": {"sort_index": s * 2 + 1},
        })
    counter_tid = n_slots * 2 + 1
    trace_events.append({
        "ph": "M", "pid": pid, "tid": counter_tid,
        "name": "thread_name",
        "args": {"name": "In-Flight Requests"},
    })
    trace_events.append({
        "ph": "M", "pid": pid, "tid": counter_tid,
        "name": "thread_sort_index",
        "args": {"sort_index": n_slots * 2 + 10},
    })

    # Conversation-level bars on their own dedicated track per slot.
    # Perfetto auto-colors by hashing the slice name; to avoid hash collisions
    # for names with shared prefixes (e.g. all "sim_000XXX" in workflow), we
    # prepend a unique index to make each name hash to a different color.
    all_conv_ids = sorted({span["conv_id"] for spans in slot_conv_spans.values() for span in spans})
    conv_color_idx = {cid: i for i, cid in enumerate(all_conv_ids)}

    for slot, spans in slot_conv_spans.items():
        tid = conv_tid(slot)
        for span in spans:
            conv_short = span["conv_id"].split("__")[-1] if "__" in span["conv_id"] else span["conv_id"]
            idx = conv_color_idx[span["conv_id"]]
            # Name format: "{idx} {conv_short}" — prefix forces name hashes to differ.
            display_name = f"{idx} {conv_short}"
            trace_events.append({
                "ph": "X",
                "pid": pid,
                "tid": tid,
                "ts": span["first_us"],
                "dur": span["last_us"] - span["first_us"],
                "name": display_name,
                "cat": span["conv_id"],
                "args": {
                    "conversation_id": span["conv_id"],
                    "n_turns": span["n_turns"],
                    "slot": slot,
                },
            })

    # Build a lookup: for each turn, find the next turn of the same conversation.
    # The delay between turn N and turn N+next belongs to (conv, next_turn_num)
    # in the dataset (the wait before the next request is issued).
    # Also identify the first inference per conversation for the [user] label.
    next_turn_for: dict[str, dict] = {}
    first_uuid_for_conv: dict[str, str] = {}
    by_conv_sorted: dict[str, list[dict]] = {}
    for t in turns:
        by_conv_sorted.setdefault(t["conv_id"], []).append(t)
    for conv_id, tlist in by_conv_sorted.items():
        tlist.sort(key=lambda x: x["issued_ns"])
        first_uuid_for_conv[conv_id] = tlist[0]["uuid"]
        for idx in range(len(tlist) - 1):
            next_turn_for[tlist[idx]["uuid"]] = tlist[idx + 1]

    # Turn slices (on the per-slot "turns" track)
    for i, t in enumerate(turns):
        slot = turn_slots[i]
        tid = turn_tid(slot)
        conv_id = t["conv_id"]
        turn_num = t["turn"]

        issued_us = (t["issued_ns"] - t0_ns) / 1000.0
        complete_us = (t["complete_ns"] - t0_ns) / 1000.0
        dur_us = complete_us - issued_us

        # Dataset's delay for THIS turn (wait that occurred before this turn was issued).
        delay_for_this_s = delay_map.get((conv_id, turn_num), 0.0)

        gt_summary = _extract_tool_summary(t["gt_tool_calls"])
        model_summary = _extract_tool_summary(t["model_tool_calls"])

        # Model response shape — kept for tooltip ("what the model actually did")
        model_response_kind = "tools" if t["model_tool_calls"] else "reply"

        # Label kind reflects the GT-expected response shape, so [tool]/[reply]
        # describes what *should* happen at turn_num+1. A model that produces a
        # reply when GT expected tool calls still renders as [tool] with an ✗.
        #   user  — first inference of the conversation (cold start)
        #   tool  — GT response at turn_num+1 contains tool calls
        #   reply — GT response at turn_num+1 has no tool calls
        is_first_inference = t["uuid"] == first_uuid_for_conv.get(conv_id)
        if is_first_inference:
            label_kind = "user"
        else:
            label_kind = "tool" if t["has_gt_tool_calls"] else "reply"

        # Domain-appropriate match (per-turn for mixed datasets when domain=auto)
        turn_domain = t.get("turn_domain", domain)
        if turn_domain == "workflow":
            match_status = (
                "yes" if t["model_intent"] and t["model_intent"] in set(t["gt_intent_codes"])
                else "no" if t["gt_intent_codes"]
                else "n/a"
            )
        else:
            match_status = (
                "yes" if t["has_gt_tool_calls"] and gt_summary == model_summary
                else "no" if t["has_gt_tool_calls"]
                else "n/a"
            )

        # Gate the visible ✓/✗ symbol by domain × label kind. The match is still
        # computed above (and shown in the tooltip) but suppressed in the label
        # for the domain-irrelevant kind: [reply] in coding, [tool] in workflow.
        if turn_domain == "workflow":
            symbol_applies = label_kind in {"reply", "user"}
        else:
            symbol_applies = label_kind in {"tool", "user"}
        visible_status = match_status if symbol_applies else "n/a"

        if visible_status == "yes":
            color = _MATCH_COLOR
            cat = "match"
        elif visible_status == "no":
            color = _MISMATCH_COLOR
            cat = "mismatch"
        else:
            color = _NO_TOOL_CALL_COLOR
            cat = "unscorable"

        # Build tooltip args
        args: dict[str, object] = {
            "conversation_id": conv_id,
            "prompt_turn": turn_num,
            "response_turn": turn_num + 1,
            "label_kind": label_kind,
            "prompt_ends_with_role": t["client_role"],
            "model_response_kind": model_response_kind,
            "duration_ms": round(dur_us / 1000, 1),
            "dataset_delay_before_s": round(delay_for_this_s, 3),
            "sample_uuid": t["uuid"],
            "gt_tool_calls": gt_summary,
            "model_tool_calls": model_summary,
        }
        if turn_domain == "workflow":
            args["gt_intent_codes"] = ",".join(t["gt_intent_codes"]) if t["gt_intent_codes"] else "(none)"
            args["model_intent"] = t["model_intent"] or "(none)"
            args["intent_match"] = match_status
        else:
            args["tool_call_match"] = match_status

        status_suffix = {"yes": "✓ ", "no": "✗ ", "n/a": ""}[visible_status]
        trace_events.append({
            "ph": "X",
            "pid": pid,
            "tid": tid,
            "ts": issued_us,
            "dur": dur_us,
            "name": f"{status_suffix}t{turn_num} [{label_kind}]",
            "cat": cat,
            "cname": color,
            "args": args,
        })

        # TTFT slice (always nested inside the turn slice)
        if t["recv_first_ns"] is not None:
            rf_us = (t["recv_first_ns"] - t0_ns) / 1000.0
            ttft_dur_us = rf_us - issued_us
            trace_events.append({
                "ph": "X",
                "pid": pid,
                "tid": tid,
                "ts": issued_us,
                "dur": ttft_dur_us,
                "name": "TTFT",
                "cat": "ttft",
                "cname": _TTFT_COLOR,
                "args": {
                    "ttft_ms": round(ttft_dur_us / 1000, 1),
                    "conversation_id": conv_id,
                    "turn": turn_num,
                },
            })

        # Tool delay slice — placed between this turn's completion and the
        # NEXT turn of the same conversation. Its width is the actual measured
        # gap; the dataset value is recorded as metadata for comparison.
        next_t = next_turn_for.get(t["uuid"])
        if next_t is not None:
            next_issued_us = (next_t["issued_ns"] - t0_ns) / 1000.0
            gap_us = next_issued_us - complete_us
            next_delay_s = delay_map.get((conv_id, next_t["turn"]), 0.0)
            if gap_us > 0:
                trace_events.append({
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": complete_us,
                    "dur": gap_us,
                    "name": f"delay {next_delay_s:.2f}s",
                    "cat": "tool_delay",
                    "cname": _DELAY_COLOR,
                    "args": {
                        "measured_gap_ms": round(gap_us / 1000, 2),
                        "dataset_delay_s": round(next_delay_s, 3),
                        "conversation_id": conv_id,
                        "between_turn": turn_num,
                        "and_turn": next_t["turn"],
                    },
                })

    # Counter track: in-flight requests over time
    timeline: list[tuple[int, int]] = []
    for t in turns:
        timeline.append((t["issued_ns"], +1))
        timeline.append((t["complete_ns"], -1))
    timeline.sort()

    inflight = 0
    for ts_ns, delta in timeline:
        inflight += delta
        trace_events.append({
            "ph": "C",
            "pid": pid,
            "tid": counter_tid,
            "ts": (ts_ns - t0_ns) / 1000.0,
            "name": "in_flight",
            "args": {"in_flight": inflight},
        })

    # Write trace
    trace = {"traceEvents": trace_events}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(trace, f)

    logger.info(
        "wrote %d trace events to %s (%.1f MB)",
        len(trace_events), out_path, out_path.stat().st_size / 1e6,
    )
    return out_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(
        description="Generate a Perfetto trace from a multi-turn benchmark run."
    )
    p.add_argument(
        "--report-dir", required=True, type=Path,
        help="Benchmark report directory (contains events.jsonl)",
    )
    p.add_argument(
        "--dataset", required=True, type=Path,
        help="Ground-truth dataset JSONL (for roles, tool_calls, delay_seconds)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output trace JSON path (default: <report-dir>/trace.json)",
    )
    p.add_argument(
        "--domain", choices=("auto", "coding", "workflow"), default="auto",
        help="Dataset domain. 'auto' detects workflow if any row has intent_codes, "
             "otherwise coding.",
    )
    args = p.parse_args()
    generate_trace(args.report_dir, args.dataset, args.out, args.domain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
