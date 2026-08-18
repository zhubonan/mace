###########################################################################################
# Fixed-structure atom-balanced batching for sharded OMat datasets
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from mace.data.omat_lmdb_dataset import OMatShardIndex


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
        self.peak_window_shards = 0
        self.peak_window_rows = 0

        for window_start in range(0, len(shard_ids), self.window_shards):
            window_ids = shard_ids[window_start : window_start + self.window_shards]
            window_indices = []
            window_sizes = []
            for shard_index_value in window_ids:
                shard_index = int(shard_index_value)
                shard = self.index.shards[shard_index]
                start = (
                    0
                    if shard_index == 0
                    else self.index.cumulative_ends[shard_index - 1]
                )
                sizes = self.index.load_natoms(shard_index)
                window_indices.append(
                    np.arange(start, start + shard.length, dtype=np.int64)
                )
                window_sizes.append(sizes)

            self.peak_window_shards = max(self.peak_window_shards, len(window_indices))
            active_rows = len(row_tail_indices) + sum(
                len(indices) for indices in window_indices
            )
            self.peak_window_rows = max(self.peak_window_rows, active_rows)

            indices = np.concatenate([row_tail_indices, *window_indices])
            sizes = np.concatenate([row_tail_sizes, *window_sizes])
            tie_break = rng.permutation(len(indices))
            order = tie_break[np.argsort(sizes[tie_break], kind="stable")]
            indices = indices[order]
            sizes = sizes[order]

            complete_rows = len(indices) - len(indices) % self.batch_size
            complete_batches = []
            for batch_start in range(0, complete_rows, self.batch_size):
                batch_end = batch_start + self.batch_size
                batch = indices[batch_start:batch_end].tolist()
                total_atoms = int(sizes[batch_start:batch_end].sum())
                complete_batches.append((batch, total_atoms))
            row_tail_indices = indices[complete_rows:]
            row_tail_sizes = sizes[complete_rows:]

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
