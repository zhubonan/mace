###########################################################################################
# Fixed-structure atom-balanced batching for sharded OMat datasets
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from mace.data.omat_lmdb_dataset import OMatShardIndex


def _build_atom_balanced_batches(
    indices: np.ndarray,
    sizes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[list[int], int]], np.ndarray, np.ndarray]:
    """Pack complete rows into fixed-size batches with similar atom totals."""
    tail_count = len(indices) % batch_size
    row_order = rng.permutation(len(indices))
    tail_positions = row_order[:tail_count]
    complete_positions = row_order[tail_count:]

    complete_indices = indices[complete_positions]
    complete_sizes = sizes[complete_positions]
    size_order = np.argsort(complete_sizes, kind="stable")
    complete_indices = complete_indices[size_order]
    complete_sizes = complete_sizes[size_order]

    batch_count = len(complete_indices) // batch_size
    striped_indices = complete_indices.reshape(batch_size, batch_count).copy()
    striped_sizes = complete_sizes.reshape(batch_size, batch_count).copy()
    striped_indices[1::2] = striped_indices[1::2, ::-1].copy()
    striped_sizes[1::2] = striped_sizes[1::2, ::-1].copy()

    batches = striped_indices.T
    batch_totals = striped_sizes.sum(axis=0)
    complete_batches = [
        (batch.tolist(), int(total_atoms))
        for batch, total_atoms in zip(batches, batch_totals, strict=True)
    ]
    return complete_batches, indices[tail_positions], sizes[tail_positions]


class ShardWindowAtomBalancedBatchSampler(Sampler[list[int]]):
    """Build fixed-size, DDP-aligned batches from bounded shard windows."""

    def __init__(
        self,
        index: OMatShardIndex,
        batch_size: int,
        *,
        seed: int = 0,
        epoch: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        window_shards: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if window_shards < 1:
            raise ValueError("window_shards must be positive")

        full_batches = len(index) // batch_size
        usable_batches = full_batches - full_batches % num_replicas
        rank_batches = usable_batches // num_replicas

        self.index = index
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.window_shards = int(window_shards)
        self._rank_batches = rank_batches
        self.peak_planning_shards = 0
        self.peak_window_shards = 0
        self.peak_window_rows = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._rank_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        shard_ids = np.asarray(
            [
                shard_index
                for shard_index, shard in enumerate(self.index.shards)
                if shard.length
            ],
            dtype=np.int64,
        )
        rng.shuffle(shard_ids)

        row_tail_indices = np.empty(0, dtype=np.int64)
        row_tail_sizes = np.empty(0, dtype=np.int64)
        pack_tail: list[tuple[list[int], int]] = []
        yielded = 0
        self.peak_planning_shards = 0
        self.peak_window_shards = 0
        self.peak_window_rows = 0

        planning_shards = min(2 * self.window_shards, 16)
        for planning_start in range(0, len(shard_ids), planning_shards):
            planning_ids = shard_ids[planning_start : planning_start + planning_shards]
            planning_records = []
            for shard_index_value in planning_ids:
                shard_index = int(shard_index_value)
                shard = self.index.shards[shard_index]
                start = (
                    0
                    if shard_index == 0
                    else self.index.cumulative_ends[shard_index - 1]
                )
                sizes = self.index.load_natoms(shard_index)
                planning_records.append(
                    (
                        np.arange(start, start + shard.length, dtype=np.int64),
                        sizes,
                        int(sizes.sum()),
                        len(sizes),
                    )
                )
            self.peak_planning_shards = max(
                self.peak_planning_shards, len(planning_records)
            )

            group_count = (
                len(planning_records) + self.window_shards - 1
            ) // self.window_shards
            planning_groups: list[list[tuple[np.ndarray, np.ndarray, int, int]]] = [
                [] for _ in range(group_count)
            ]
            if group_count == 1:
                planning_groups[0] = planning_records
            else:
                total_atoms = sum(record[2] for record in planning_records)
                total_rows = sum(record[3] for record in planning_records)
                low_mean = min(record[2] / record[3] for record in planning_records)
                high_mean = max(record[2] / record[3] for record in planning_records)
                target_mean = total_atoms / total_rows
                atom_scale = max(1.0, high_mean - low_mean)
                best_objective = None
                best_mask = 0
                full_mask = (1 << len(planning_records)) - 1
                for mask in range(1, full_mask):
                    left_count = mask.bit_count()
                    if (
                        left_count > self.window_shards
                        or len(planning_records) - left_count > self.window_shards
                    ):
                        continue
                    left_atoms = sum(
                        record[2]
                        for record_index, record in enumerate(planning_records)
                        if mask & (1 << record_index)
                    )
                    left_rows = sum(
                        record[3]
                        for record_index, record in enumerate(planning_records)
                        if mask & (1 << record_index)
                    )
                    right_atoms = total_atoms - left_atoms
                    right_rows = total_rows - left_rows
                    objective = (
                        max(
                            abs(left_atoms / left_rows - target_mean),
                            abs(right_atoms / right_rows - target_mean),
                        )
                        / atom_scale,
                        abs(left_rows - right_rows) / (left_rows + right_rows),
                        abs(2 * left_count - len(planning_records)),
                        mask,
                    )
                    if best_objective is None or objective < best_objective:
                        best_objective = objective
                        best_mask = mask
                planning_groups[0] = [
                    record
                    for record_index, record in enumerate(planning_records)
                    if best_mask & (1 << record_index)
                ]
                planning_groups[1] = [
                    record
                    for record_index, record in enumerate(planning_records)
                    if not best_mask & (1 << record_index)
                ]

            for group_index in rng.permutation(group_count):
                window = planning_groups[int(group_index)]
                window_indices = [record[0] for record in window]
                window_sizes = [record[1] for record in window]
                self.peak_window_shards = max(self.peak_window_shards, len(window))
                active_rows = len(row_tail_indices) + sum(
                    len(indices) for indices in window_indices
                )
                self.peak_window_rows = max(self.peak_window_rows, active_rows)

                indices = np.concatenate([row_tail_indices, *window_indices])
                sizes = np.concatenate([row_tail_sizes, *window_sizes])
                complete_batches, row_tail_indices, row_tail_sizes = (
                    _build_atom_balanced_batches(indices, sizes, self.batch_size, rng)
                )

                scheduling_region = pack_tail + complete_batches
                scheduling_region.sort(key=lambda item: item[1])
                complete_pack_batches = (
                    len(scheduling_region) - len(scheduling_region) % self.num_replicas
                )
                pack_tail = scheduling_region[complete_pack_batches:]
                packs = [
                    scheduling_region[start : start + self.num_replicas]
                    for start in range(0, complete_pack_batches, self.num_replicas)
                ]
                rng.shuffle(packs)
                for pack in packs:
                    rank_assignment = rng.permutation(self.num_replicas)
                    batch, _ = pack[int(rank_assignment[self.rank])]
                    yielded += 1
                    yield batch

        if yielded != self._rank_batches:
            raise RuntimeError(
                f"Sampler yielded {yielded} batches, expected {self._rank_batches}"
            )
