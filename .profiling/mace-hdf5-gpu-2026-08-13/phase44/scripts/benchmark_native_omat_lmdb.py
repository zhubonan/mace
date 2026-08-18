#!/usr/bin/env python3
###########################################################################################
# Multiworker CPU benchmark for native official OMat LMDB input
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from mace import data, tools
from mace.tools.torch_geometric.dataloader import DataLoader


class InstrumentedDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item.source_row = torch.tensor([index], dtype=torch.int64)
        item.loader_pid = torch.tensor([os.getpid()], dtype=torch.int64)
        item.lmdb_active_environments = torch.tensor(
            [self.dataset.active_environment_count], dtype=torch.int64
        )
        item.lmdb_open_count = torch.tensor(
            [self.dataset.open_environment_count], dtype=torch.int64
        )
        return item


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches) -> None:
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class LimitedBatchSampler(Sampler[list[int]]):
    def __init__(self, sampler, steps: int) -> None:
        self.sampler = sampler
        self.steps = steps

    def __iter__(self):
        for step, batch in enumerate(self.sampler):
            if step == self.steps:
                break
            yield batch

    def __len__(self) -> int:
        return self.steps


def _process_snapshot() -> dict:
    rss_kib = None
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
    return {
        "rss_kib": rss_kib,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "open_fds": len(list(Path("/proc/self/fd").iterdir())),
    }


def _native_dataset(root, *, r_max, max_open_shards):
    return data.OMatLMDBDataset(
        root,
        r_max=r_max,
        z_table=tools.AtomicNumberTable(list(range(1, 84)) + list(range(89, 95))),
        heads=["Default"],
        head="Default",
        max_open_shards=max_open_shards,
    )


def run_benchmark(
    root: Path,
    *,
    sampling: str,
    workers: int,
    repetition: int,
    samples: int,
    batch_size: int,
    window_shards: int,
    max_open_shards: int,
    r_max: float,
    seed: int,
) -> dict:
    if samples < 1 or samples % batch_size:
        raise ValueError("samples must be a positive multiple of batch_size")
    before = _process_snapshot()
    startup = time.perf_counter()
    dataset = _native_dataset(root, r_max=r_max, max_open_shards=max_open_shards)
    startup_seconds = time.perf_counter() - startup
    steps = samples // batch_size

    if sampling == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, repetition]))
        indices = rng.integers(0, len(dataset), size=samples, dtype=np.int64)
        batches = [
            indices[start : start + batch_size].tolist()
            for start in range(0, samples, batch_size)
        ]
        batch_sampler = FixedBatchSampler(batches)
    elif sampling == "shard_window_atom_balanced":
        balanced = data.ShardWindowAtomBalancedBatchSampler(
            dataset.shard_index,
            batch_size=batch_size,
            seed=seed,
            epoch=repetition,
            window_shards=window_shards,
        )
        batch_sampler = LimitedBatchSampler(balanced, steps)
    else:
        raise ValueError(f"Unsupported sampling policy: {sampling}")

    loader = DataLoader(
        InstrumentedDataset(dataset),
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed + repetition),
    )
    checksum = hashlib.sha256()
    active_counts = []
    open_counts = []
    worker_pids = set()
    shard_ids = []
    start = time.perf_counter()
    iterator = iter(loader)
    iterator_ready_seconds = time.perf_counter() - start
    first_batch_seconds = None
    consumed = 0
    for batch in iterator:
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - start
        rows = batch.source_row.detach().cpu().numpy().astype(np.int64, copy=False)
        checksum.update(rows.tobytes())
        active_counts.extend(
            batch.lmdb_active_environments.detach().cpu().numpy().tolist()
        )
        open_counts.extend(batch.lmdb_open_count.detach().cpu().numpy().tolist())
        worker_pids.update(batch.loader_pid.detach().cpu().numpy().tolist())
        shard_ids.extend(dataset.shard_index.locate(int(row))[0] for row in rows)
        consumed += len(rows)
    elapsed = time.perf_counter() - start
    dataset.close()
    after = _process_snapshot()

    return {
        "sampling": sampling,
        "workers": workers,
        "repetition": repetition,
        "samples": consumed,
        "batch_size": batch_size,
        "steps": steps,
        "startup_seconds": startup_seconds,
        "iterator_ready_seconds": iterator_ready_seconds,
        "first_batch_seconds": first_batch_seconds,
        "elapsed_seconds": elapsed,
        "structures_per_second": consumed / elapsed,
        "indices_checksum": checksum.hexdigest(),
        "worker_pids": sorted(int(pid) for pid in worker_pids),
        "worker_processes": len(worker_pids),
        "max_active_environments": max(active_counts, default=0),
        "max_open_environment_count": max(open_counts, default=0),
        "unique_shards": len(set(shard_ids)),
        "shard_transitions": sum(
            left != right for left, right in zip(shard_ids, shard_ids[1:])
        ),
        "rss_delta_kib": (
            None
            if before["rss_kib"] is None or after["rss_kib"] is None
            else after["rss_kib"] - before["rss_kib"]
        ),
        "fd_delta": after["open_fds"] - before["open_fds"],
        "before": before,
        "after": after,
    }


def validate_benchmark_records(
    records: list[dict], *, expected_samples: int, max_open_shards: int
) -> None:
    groups = {}
    for record in records:
        if record["samples"] != expected_samples:
            raise ValueError("Benchmark sample count mismatch")
        if not math.isfinite(record["structures_per_second"]):
            raise ValueError("Benchmark rates must be finite")
        if record["max_active_environments"] > max_open_shards:
            raise ValueError("Worker environment bound exceeded")
        key = (record["sampling"], record["repetition"])
        groups.setdefault(key, []).append(record)

    policies = {record["sampling"] for record in records}
    for policy in policies:
        repetitions = {
            record["repetition"] for record in records if record["sampling"] == policy
        }
        if len(repetitions) < 3:
            raise ValueError("Benchmark requires at least three repetitions")
    for group in groups.values():
        if {record["workers"] for record in group} != {0, 8, 16}:
            raise ValueError("Benchmark repetitions must contain 0/8/16 workers")
        if len({record["indices_checksum"] for record in group}) != 1:
            raise ValueError("Worker variants used a different index stream")


def benchmark_order(repetition: int) -> list[tuple[str, int]]:
    worker_orders = ((0, 8, 16), (8, 16, 0), (16, 0, 8))
    workers = worker_orders[repetition % len(worker_orders)]
    policies = (
        ("random", "shard_window_atom_balanced")
        if repetition % 2 == 0
        else ("shard_window_atom_balanced", "random")
    )
    return [(policy, worker) for policy in policies for worker in workers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-shards", type=int, default=8)
    parser.add_argument("--max-open-shards", type=int, default=8)
    parser.add_argument("--r-max", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    records = []
    for repetition in range(args.repetitions):
        for sampling, workers in benchmark_order(repetition):
            record = run_benchmark(
                args.root,
                sampling=sampling,
                workers=workers,
                repetition=repetition,
                samples=args.samples,
                batch_size=args.batch_size,
                window_shards=args.window_shards,
                max_open_shards=args.max_open_shards,
                r_max=args.r_max,
                seed=args.seed,
            )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    validate_benchmark_records(
        records,
        expected_samples=args.samples,
        max_open_shards=args.max_open_shards,
    )
    summary = {
        "root": str(args.root.resolve()),
        "samples_per_run": args.samples,
        "batch_size": args.batch_size,
        "window_shards": args.window_shards,
        "max_open_shards": args.max_open_shards,
        "repetitions": args.repetitions,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
