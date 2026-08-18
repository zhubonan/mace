#!/usr/bin/env python3
###########################################################################################
# Exact complete-corpus audit for fixed-structure atom-balanced OMat sampling
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import tempfile
from pathlib import Path

import numpy as np

from mace.data import (
    ShardWindowAtomBalancedBatchSampler,
    build_omat_shard_index,
)


def _relative_imbalance(totals: list[int]) -> float:
    mean = sum(totals) / len(totals)
    return 0.0 if mean == 0 else (max(totals) - min(totals)) / mean


def _p95(values: list[float]) -> float:
    return 0.0 if not values else float(np.quantile(values, 0.95))


def _write_natoms(index, path: Path) -> np.memmap:
    values = np.memmap(path, mode="w+", dtype=np.int64, shape=(len(index),))
    start = 0
    for shard_index, shard in enumerate(index.shards):
        end = start + shard.length
        values[start:end] = index.load_natoms(shard_index)
        start = end
    values.flush()
    return values


def _largest_window_rows(index, *, seed, epoch, window_shards) -> int:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    shard_ids = np.asarray(
        [i for i, shard in enumerate(index.shards) if shard.length],
        dtype=np.int64,
    )
    rng.shuffle(shard_ids)
    return max(
        (
            sum(index.shards[int(i)].length for i in shard_ids[start:end])
            for start in range(0, len(shard_ids), window_shards)
            for end in (start + window_shards,)
        ),
        default=0,
    )


def audit_sampler(
    index,
    *,
    batch_size: int,
    world_size: int,
    window_shards: int,
    seed: int,
    epoch: int,
    temporary_directory: Path,
    random_steps: int = 8192,
) -> dict:
    temporary_directory = Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    bit_file = tempfile.NamedTemporaryFile(
        prefix="omat-coverage-", suffix=".bits", dir=temporary_directory, delete=False
    )
    bit_path = Path(bit_file.name)
    bit_file.close()
    natoms_file = tempfile.NamedTemporaryFile(
        prefix="omat-natoms-", suffix=".i64", dir=temporary_directory, delete=False
    )
    natoms_path = Path(natoms_file.name)
    natoms_file.close()

    try:
        bit_count = math.ceil(len(index) / 8)
        coverage = np.memmap(bit_path, mode="w+", dtype=np.uint8, shape=(bit_count,))
        coverage[:] = 0
        natoms = _write_natoms(index, natoms_path)
        samplers = [
            ShardWindowAtomBalancedBatchSampler(
                index,
                batch_size=batch_size,
                seed=seed,
                epoch=epoch,
                num_replicas=world_size,
                rank=rank,
                window_shards=window_shards,
            )
            for rank in range(world_size)
        ]
        rank_steps = [0] * world_size
        duplicates = 0
        used = 0
        checksum = hashlib.sha256()
        balanced_imbalances = []

        for rank_batches in zip(*(iter(sampler) for sampler in samplers)):
            totals = []
            for rank, batch in enumerate(rank_batches):
                indices = np.asarray(batch, dtype=np.int64)
                if len(np.unique(indices)) != len(indices):
                    duplicates += len(indices) - len(np.unique(indices))
                byte_indices = indices >> 3
                masks = np.left_shift(1, indices & 7).astype(np.uint8)
                duplicates += int(np.count_nonzero(coverage[byte_indices] & masks))
                np.bitwise_or.at(coverage, byte_indices, masks)
                used += len(indices)
                rank_steps[rank] += 1
                totals.append(int(natoms[indices].sum()))
                checksum.update(rank.to_bytes(2, "little"))
                checksum.update(indices.tobytes())
            balanced_imbalances.append(_relative_imbalance(totals))

        coverage.flush()
        popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
            axis=1
        )
        marked = sum(
            int(popcount[np.asarray(coverage[start : start + 1_000_000])].sum())
            for start in range(0, len(coverage), 1_000_000)
        )
        if marked != used - duplicates:
            raise RuntimeError(
                f"Coverage bitset marked {marked} unique rows, expected {used - duplicates}"
            )

        comparisons = min(len(balanced_imbalances), int(random_steps))
        random_imbalances = []
        if comparisons:
            sample_count = comparisons * world_size * batch_size
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 991]))
            random_indices = rng.choice(len(index), size=sample_count, replace=False)
            random_indices = random_indices.reshape(comparisons, world_size, batch_size)
            for step in random_indices:
                random_imbalances.append(
                    _relative_imbalance(
                        [int(natoms[rank_batch].sum()) for rank_batch in step]
                    )
                )

        dropped = len(index) - used
        if duplicates:
            raise RuntimeError(f"Sampler produced {duplicates} duplicate indices")
        if dropped < 0 or dropped >= world_size * batch_size:
            raise RuntimeError(f"Sampler drop bound violated: {dropped}")
        if len(set(rank_steps)) != 1:
            raise RuntimeError(f"Ranks have unequal steps: {rank_steps}")

        return {
            "dataset_id": index.dataset_id,
            "rows": len(index),
            "shards": len(index.shards),
            "batch_size": batch_size,
            "world_size": world_size,
            "window_shards": window_shards,
            "seed": seed,
            "epoch": epoch,
            "used": used,
            "dropped": dropped,
            "duplicates": duplicates,
            "disjoint_rank_samples": duplicates == 0,
            "rank_steps": rank_steps,
            "balanced_p95_rank_imbalance": _p95(balanced_imbalances),
            "random_p95_rank_imbalance": _p95(random_imbalances),
            "imbalance_comparison_steps": comparisons,
            "peak_window_shards": max(s.peak_window_shards for s in samplers),
            "peak_window_rows": max(s.peak_window_rows for s in samplers),
            "largest_window_rows": _largest_window_rows(
                index,
                seed=seed,
                epoch=epoch,
                window_shards=window_shards,
            ),
            "checksum": checksum.hexdigest(),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "coverage_bitset_bytes": bit_count,
            "natoms_memmap_bytes": len(index) * np.dtype(np.int64).itemsize,
        }
    finally:
        bit_path.unlink(missing_ok=True)
        natoms_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--window-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--random-steps", type=int, default=8192)
    parser.add_argument("--temporary-directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = audit_sampler(
        build_omat_shard_index(args.root),
        batch_size=args.batch_size,
        world_size=args.world_size,
        window_shards=args.window_shards,
        seed=args.seed,
        epoch=args.epoch,
        temporary_directory=args.temporary_directory,
        random_steps=args.random_steps,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
