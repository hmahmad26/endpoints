#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write a loadable subset of the MLPerf Agentic Inference dataset.

The artifact published at endpoints.mlcommons-storage.org (md5
1fcacbcb79bee4258f70c65030ba4ebc) carries a binary splice that is not valid
UTF-8 and is truncated mid-record at EOF. The client reads the file with
``pd.read_json(path, lines=True)``, which decodes the whole file strictly before
any trajectory selection happens, so the corruption blocks every run regardless
of ``num_trajectories_to_issue``.

Any conversation owning a damaged record is dropped whole. Replaying a
conversation with one turn missing would exercise a different conversation than
the one published, which is worse than omitting it.

The output is not a valid substitute for the official dataset in a submission;
it exists so the stack can be exercised while a fixed artifact is pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_damage(lines: list[bytes]) -> tuple[list[dict], set[str]]:
    """Classify each line, returning damage details and the conversations to drop."""
    damage: list[dict] = []
    damaged_ids: set[str] = set()

    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        reason = None
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            reason = f"invalid utf-8 at byte {exc.start} of line"
            text = line.decode("utf-8", "replace")

        if reason is None:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                reason = f"invalid json: {exc.msg} at col {exc.colno}"

        if reason is None:
            continue

        conv_id = _recover_conversation_id(text)
        damage.append({"line": lineno, "conversation_id": conv_id, "reason": reason})
        if conv_id:
            damaged_ids.add(conv_id)

    return damage, damaged_ids


def _recover_conversation_id(text: str) -> str | None:
    """Pull conversation_id out of a record that may not parse as JSON."""
    try:
        value = json.loads(text, strict=False).get("conversation_id")
        return str(value) if value is not None else None
    except json.JSONDecodeError:
        pass

    marker = '"conversation_id":'
    pos = text.find(marker)
    if pos == -1:
        return None
    tail = text[pos + len(marker) :].lstrip()
    if not tail.startswith('"'):
        return None
    end = tail.find('"', 1)
    return tail[1:end] if end != -1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    raw = args.source.read_bytes()
    lines = raw.split(b"\n")
    damage, damaged_ids = find_damage(lines)

    kept_lines: list[bytes] = []
    kept_ids: set[str] = set()
    dropped_records = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            dropped_records += 1
            continue

        conv_id = record.get("conversation_id")
        if conv_id is not None and str(conv_id) in damaged_ids:
            dropped_records += 1
            continue

        if conv_id is not None:
            kept_ids.add(str(conv_id))
        kept_lines.append(line)

    args.output.write_bytes(b"\n".join(kept_lines) + b"\n")

    print(f"source              : {args.source} ({len(raw):,} bytes)")
    print(f"source newline-term : {raw.endswith(b'\\n')}")
    print(f"damaged lines       : {len(damage)}")
    for entry in damage:
        print(
            f"    line {entry['line']:>6}  conv={entry['conversation_id']!r}"
            f"  {entry['reason']}"
        )
    print(f"dropped trajectories: {len(damaged_ids)} {sorted(damaged_ids)}")
    print(f"dropped records     : {dropped_records}")
    print(f"kept records        : {len(kept_lines):,}")
    print(f"kept trajectories   : {len(kept_ids)}")
    print(f"output              : {args.output}")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "source": str(args.source),
                    "source_bytes": len(raw),
                    "damaged_lines": damage,
                    "dropped_conversation_ids": sorted(damaged_ids),
                    "dropped_records": dropped_records,
                    "kept_records": len(kept_lines),
                    "kept_trajectories": len(kept_ids),
                },
                indent=2,
            )
            + "\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
