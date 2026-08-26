#!/usr/bin/env python3
"""Record memory mappings for a process and its descendants."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time


MAP_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")


def parse_literal_heap(text: str) -> dict[str, int]:
    """Sum RSS/PSS/private-dirty fields belonging to literal [heap] maps."""

    result = {
        "heap_rss_kib": 0,
        "heap_pss_kib": 0,
        "heap_private_dirty_kib": 0,
    }
    in_heap = False
    for line in text.splitlines():
        if MAP_HEADER.match(line):
            in_heap = line.endswith("[heap]")
        elif in_heap and line.startswith("Rss:"):
            result["heap_rss_kib"] += int(line.split()[1])
        elif in_heap and line.startswith("Pss:"):
            result["heap_pss_kib"] += int(line.split()[1])
        elif in_heap and line.startswith("Private_Dirty:"):
            result["heap_private_dirty_kib"] += int(line.split()[1])
    return result


def parse_rollup(text: str) -> dict[str, int]:
    """Extract the process-level fields needed by qualification."""

    source_names = {
        "Rss:": "rss_kib",
        "Pss:": "pss_kib",
        "Private_Dirty:": "private_dirty_kib",
    }
    result = {name: 0 for name in source_names.values()}
    for line in text.splitlines():
        key = line.split(maxsplit=1)[0] if line else ""
        if key in source_names:
            result[source_names[key]] = int(line.split()[1])
    return result


def descendants(root_pid: int, proc_root: Path = Path("/proc")) -> list[int]:
    """Return a stable breadth-first snapshot of the descendant tree."""

    found: list[int] = []
    seen: set[int] = set()
    pending = deque([root_pid])
    while pending:
        pid = pending.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        found.append(pid)
        children_path = proc_root / str(pid) / "task" / str(pid) / "children"
        try:
            children = sorted(
                int(value) for value in children_path.read_text().split()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            children = []
        pending.extend(child for child in children if child not in seen)
    return found


def _status_fields(text: str) -> tuple[str, int]:
    name = ""
    parent = -1
    for line in text.splitlines():
        if line.startswith("Name:"):
            name = line.split(maxsplit=1)[1]
        elif line.startswith("PPid:"):
            parent = int(line.split()[1])
    return name, parent


def sample_process(pid: int, proc_root: Path) -> dict[str, object] | None:
    """Sample one live process, returning None if it exits mid-sample."""

    process_root = proc_root / str(pid)
    try:
        status = (process_root / "status").read_text(encoding="utf-8")
        rollup = (process_root / "smaps_rollup").read_text(encoding="utf-8")
        smaps = (process_root / "smaps").read_text(encoding="utf-8")
        cmdline = (process_root / "cmdline").read_bytes().replace(b"\0", b" ")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    name, parent = _status_fields(status)
    return {
        "pid": pid,
        "ppid": parent,
        "name": name,
        "cmdline": cmdline.decode("utf-8", errors="replace").strip(),
        **parse_rollup(rollup),
        **parse_literal_heap(smaps),
    }


def monitor(
    *,
    root_pid: int,
    output: Path,
    interval_seconds: float,
    stop_file: Path | None,
    proc_root: Path,
) -> None:
    """Append samples until the root exits or a stop marker appears."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("x", encoding="utf-8") as handle:
        while (proc_root / str(root_pid)).exists():
            if stop_file is not None and stop_file.exists():
                break
            observed_at = datetime.now(timezone.utc).isoformat()
            elapsed = time.monotonic() - started
            for pid in descendants(root_pid, proc_root):
                sample = sample_process(pid, proc_root)
                if sample is None:
                    continue
                sample.update(
                    {
                        "observed_at": observed_at,
                        "elapsed_s": elapsed,
                        "root_pid": root_pid,
                    }
                )
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            time.sleep(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    monitor(
        root_pid=args.root_pid,
        output=args.output,
        interval_seconds=args.interval_seconds,
        stop_file=args.stop_file,
        proc_root=args.proc_root,
    )


if __name__ == "__main__":
    main()
