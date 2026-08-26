#!/usr/bin/env python3
"""Measure glibc ownership in the real-OMat neighbour-only worker path."""

from __future__ import annotations

import argparse
import ctypes
from importlib import metadata
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mace.data.atom_balanced_sampler import ShardWindowAtomBalancedBatchSampler
from mace.data.neighborhood import get_neighborhood
from mace.data.omat_lmdb_dataset import OMatLMDBDataset
from mace.tools import AtomicNumberTable
from matscipy_neighbour_allocator import classify_slope, fit_slope_gib_per_10k


ATOMIC_NUMBERS = list(range(1, 84)) + [89, 90, 91, 92, 93, 94]
MAP_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")
MALLINFO_FIELDS = (
    "arena",
    "ordblks",
    "smblks",
    "hblks",
    "hblkhd",
    "usmblks",
    "fsmblks",
    "uordblks",
    "fordblks",
    "keepcost",
)


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in MALLINFO_FIELDS]


def read_heap_fields(pid: int) -> dict[str, int]:
    """Read RSS/PSS/private-dirty for the process's literal [heap] maps."""

    result = {
        "heap_rss_kib": 0,
        "heap_pss_kib": 0,
        "heap_private_dirty_kib": 0,
    }
    in_heap = False
    with Path(f"/proc/{pid}/smaps").open("r", encoding="utf-8") as handle:
        for line in handle:
            if MAP_HEADER.match(line):
                in_heap = line.rstrip().endswith("[heap]")
                continue
            if not in_heap:
                continue
            if line.startswith("Rss:"):
                result["heap_rss_kib"] += int(line.split()[1])
            elif line.startswith("Pss:"):
                result["heap_pss_kib"] += int(line.split()[1])
            elif line.startswith("Private_Dirty:"):
                result["heap_private_dirty_kib"] += int(line.split()[1])
    return result


class NeighborDataset(Dataset):
    """Decode a real OMat row and retain only a scalar graph-size token."""

    def __init__(
        self,
        base: OMatLMDBDataset,
        sample_every: int,
        events_dir: Path,
    ) -> None:
        self.base = base
        self.sample_every = sample_every
        self.events_dir = events_dir
        self._items = 0
        self._libc = None

    def __len__(self) -> int:
        return len(self.base)

    def _read_mallinfo2(self) -> dict[str, int]:
        if self._libc is None:
            self._libc = ctypes.CDLL("libc.so.6")
            self._libc.mallinfo2.argtypes = []
            self._libc.mallinfo2.restype = Mallinfo2
        info = self._libc.mallinfo2()
        result = {name: int(getattr(info, name)) for name in MALLINFO_FIELDS}
        result["arena_balance"] = (
            result["arena"] - result["uordblks"] - result["fordblks"]
        )
        return result

    def _record(self) -> None:
        pid = os.getpid()
        event = {
            "pid": pid,
            "items": self._items,
            "mallinfo2": self._read_mallinfo2(),
            "heap": read_heap_fields(pid),
        }
        path = self.events_dir / f"worker-{pid}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")

    def __getitem__(self, index: int) -> int:
        config = self.base.read_configuration(index)
        token = len(config.atomic_numbers)
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=config.positions,
            cutoff=self.base.r_max,
            pbc=config.pbc,
            cell=config.cell,
        )
        token += int(edge_index.shape[1])
        del edge_index, shifts, unit_shifts, cell, config
        self._items += 1
        if self._items % self.sample_every == 0:
            self._record()
        return token


def fit_worker(path: Path, warmup_items: int) -> dict[str, object]:
    """Fit one worker's ownership against 256-structure batch equivalents."""

    with path.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    events = [event for event in events if event["items"] >= warmup_items]
    if len(events) < 3:
        raise RuntimeError(f"fewer than three post-warmup events in {path}")
    sample_every = int(events[1]["items"] - events[0]["items"])
    if sample_every <= 0 or any(
        int(right["items"] - left["items"]) != sample_every
        for left, right in zip(events, events[1:])
    ):
        raise RuntimeError(f"non-uniform worker sampling interval in {path}")
    batch_indices = [int(event["items"]) / sample_every for event in events]
    result: dict[str, object] = {
        "pid": int(events[0]["pid"]),
        "events": len(events),
        "sample_every_items": sample_every,
        "first_items": int(events[0]["items"]),
        "last_items": int(events[-1]["items"]),
    }
    for field in MALLINFO_FIELDS + ("arena_balance",):
        values = [int(event["mallinfo2"][field]) for event in events]
        result[f"{field}_gib_per_10k_batches"] = fit_slope_gib_per_10k(
            batch_indices, values
        )
        result[f"{field}_first_bytes"] = values[0]
        result[f"{field}_last_bytes"] = values[-1]
    for field in ("heap_rss_kib", "heap_pss_kib", "heap_private_dirty_kib"):
        values_bytes = [int(event["heap"][field]) * 1024 for event in events]
        result[f"{field}_gib_per_10k_batches"] = fit_slope_gib_per_10k(
            batch_indices, values_bytes
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/Share/omat-aselmdb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=10_000)
    parser.add_argument("--warmup-batches", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=256)
    return parser


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size != 256 or args.sample_every != 256:
        raise ValueError("this probe requires batch-size=sample-every=256")
    if not 3 <= args.warmup_batches < args.batches:
        raise ValueError("require 3 <= warmup-batches < batches")
    torch.set_default_dtype(torch.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    events_dir = args.output / "events"
    events_dir.mkdir()

    base = OMatLMDBDataset(
        args.root,
        r_max=6.0,
        z_table=AtomicNumberTable(ATOMIC_NUMBERS),
        heads=["Default"],
        head="Default",
        max_open_shards=8,
    )
    dataset = NeighborDataset(base, args.sample_every, events_dir)
    sampler = ShardWindowAtomBalancedBatchSampler(
        base.shard_index,
        batch_size=args.batch_size,
        seed=20260729,
        num_replicas=1,
        rank=0,
        window_shards=8,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        multiprocessing_context="spawn",
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(20260729),
    )
    metadata_payload = {
        "schema_version": 1,
        "root": str(Path(args.root).resolve()),
        "batches": args.batches,
        "warmup_batches": args.warmup_batches,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "sample_every": args.sample_every,
        "seed": 20260729,
        "r_max": 6.0,
        "pid": os.getpid(),
        "python": os.sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matscipy": metadata.version("matscipy"),
    }
    _atomic_json(args.output / "metadata.json", metadata_payload)

    iterator = iter(loader)
    worker_pids = [worker.pid for worker in iterator._workers]
    try:
        for _ in range(args.batches):
            value = next(iterator)
            del value
    finally:
        iterator._shutdown_workers()
        base.close()

    paths = sorted(events_dir.glob("worker-*.jsonl"))
    if len(paths) != args.workers:
        raise RuntimeError(
            f"expected {args.workers} worker event files, found {len(paths)}"
        )
    warmup_items = args.warmup_batches * args.batch_size // args.workers
    workers = [fit_worker(path, warmup_items) for path in paths]
    slope_fields = MALLINFO_FIELDS + (
        "arena_balance",
        "heap_rss_kib",
        "heap_pss_kib",
        "heap_private_dirty_kib",
    )
    mean_slopes = {
        field: float(
            np.mean(
                [worker[f"{field}_gib_per_10k_batches"] for worker in workers]
            )
        )
        for field in slope_fields
    }
    critical = ("uordblks", "arena", "heap_private_dirty_kib")
    summary: dict[str, object] = {
        "worker_pids": worker_pids,
        "workers": workers,
        "mean_slopes_gib_per_10k_batches": mean_slopes,
        "limit_gib_per_10k_batches": 0.02,
        "passed": all(classify_slope(mean_slopes[field]) for field in critical),
    }
    _atomic_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {"metadata": metadata_payload, "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
