###########################################################################################
# Fixed-structure atom-and-edge-balanced batching for sharded OMat datasets
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Sampler

from mace.data.omat_lmdb_dataset import OMatShardIndex


@dataclass(frozen=True)
class _ShardPlanningRecord:
    shard_index: int
    start: int
    rows: int
    total_atoms: int
    total_edges: int


def _build_atom_edge_balanced_batches(
    indices: np.ndarray,
    natoms: np.ndarray,
    nedges: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[
    list[tuple[list[int], int, int]], np.ndarray, np.ndarray, np.ndarray
]:
    """Pack complete rows into fixed-size bins, balancing edges before atoms."""

    if len(indices) != len(natoms) or len(indices) != len(nedges):
        raise ValueError("indices, natoms, and nedges must have equal lengths")
    tail_count = len(indices) % batch_size
    row_order = rng.permutation(len(indices))
    tail_positions = row_order[:tail_count]
    complete_positions = row_order[tail_count:]

    complete_indices = indices[complete_positions]
    complete_atoms = natoms[complete_positions]
    complete_edges = nedges[complete_positions]
    tie_break = rng.permutation(len(complete_indices))
    cost_order = np.lexsort((tie_break, complete_atoms, complete_edges))
    complete_indices = complete_indices[cost_order]
    complete_atoms = complete_atoms[cost_order]
    complete_edges = complete_edges[cost_order]

    batch_count = len(complete_indices) // batch_size
    batch_indices: list[list[int]] = [[] for _ in range(batch_count)]
    atom_totals = np.zeros(batch_count, dtype=np.int64)
    edge_totals = np.zeros(batch_count, dtype=np.int64)
    for quantile in range(batch_size):
        start = quantile * batch_count
        stop = start + batch_count
        quantile_indices = complete_indices[start:stop]
        quantile_atoms = complete_atoms[start:stop]
        quantile_edges = complete_edges[start:stop]

        candidate_ties = rng.permutation(batch_count)
        candidate_order = np.lexsort(
            (candidate_ties, -quantile_atoms, -quantile_edges)
        )
        batch_ties = rng.permutation(batch_count)
        target_order = np.lexsort((batch_ties, atom_totals, edge_totals))
        for candidate, target in zip(candidate_order, target_order, strict=True):
            batch_indices[int(target)].append(int(quantile_indices[candidate]))
            atom_totals[target] += int(quantile_atoms[candidate])
            edge_totals[target] += int(quantile_edges[candidate])

    complete_batches = [
        (batch, int(total_atoms), int(total_edges))
        for batch, total_atoms, total_edges in zip(
            batch_indices, atom_totals, edge_totals, strict=True
        )
    ]
    return (
        complete_batches,
        indices[tail_positions],
        natoms[tail_positions],
        nedges[tail_positions],
    )


class ShardWindowAtomEdgeBalancedBatchSampler(Sampler[list[int]]):
    """Build fixed-size, DDP-aligned batches from bounded two-cost windows."""

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
        if index.cost_profile is None:
            raise ValueError("Edge-balanced sampling requires an associated cost profile")
        if len(index.cost_profile.shards) != len(index.shards):
            raise ValueError("Edge cost profile shard count does not match dataset")
        for source_shard, cost_shard in zip(
            index.shards, index.cost_profile.shards, strict=True
        ):
            if (
                cost_shard.relative_shard_path != source_shard.relative_path
                or cost_shard.rows != source_shard.length
            ):
                raise ValueError("Edge cost profile shard mapping does not match dataset")

        full_batches = len(index) // batch_size
        usable_batches = full_batches - full_batches % num_replicas
        self.index = index
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.window_shards = int(window_shards)
        self._rank_batches = usable_batches // num_replicas
        self.peak_planning_shards = 0
        self.peak_window_shards = 0
        self.peak_window_rows = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._rank_batches

    def contract_payload(self) -> dict[str, object]:
        """Return the rank-independent sampling identity checked before training."""

        profile = self.index.cost_profile
        return {
            "format": "mace-omat-edge-sampler-contract-v1",
            "sampler": "shard_window_atom_edge_balanced",
            "dataset_id": self.index.dataset_id,
            "source_manifest_sha256": self.index.manifest_sha256,
            "profile_id": profile.profile_id,
            "profile_manifest_sha256": profile.manifest_sha256,
            "profile_cutoff_hex": float(profile.cutoff).hex(),
            "dataset_rows": len(self.index),
            "dataset_shards": len(self.index.shards),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "num_replicas": self.num_replicas,
            "window_shards": self.window_shards,
            "rank_batches": self._rank_batches,
        }

    def contract_digest(self) -> str:
        encoded = json.dumps(
            self.contract_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _partition_planning_records(
        records: list[_ShardPlanningRecord], window_shards: int
    ) -> list[list[_ShardPlanningRecord]]:
        group_count = (len(records) + window_shards - 1) // window_shards
        groups: list[list[_ShardPlanningRecord]] = [
            [] for _ in range(group_count)
        ]
        if group_count <= 1:
            if records:
                groups[0] = records
            return groups

        total_atoms = sum(record.total_atoms for record in records)
        total_edges = sum(record.total_edges for record in records)
        total_rows = sum(record.rows for record in records)
        target_atom_mean = total_atoms / total_rows
        target_edge_mean = total_edges / total_rows
        atom_means = [record.total_atoms / record.rows for record in records]
        edge_means = [record.total_edges / record.rows for record in records]
        atom_scale = max(1.0, max(atom_means) - min(atom_means))
        edge_scale = max(1.0, max(edge_means) - min(edge_means))

        best_objective = None
        best_mask = 0
        full_mask = (1 << len(records)) - 1
        for mask in range(1, full_mask):
            left_count = mask.bit_count()
            if (
                left_count > window_shards
                or len(records) - left_count > window_shards
            ):
                continue
            left_rows = sum(
                record.rows
                for record_index, record in enumerate(records)
                if mask & (1 << record_index)
            )
            left_atoms = sum(
                record.total_atoms
                for record_index, record in enumerate(records)
                if mask & (1 << record_index)
            )
            left_edges = sum(
                record.total_edges
                for record_index, record in enumerate(records)
                if mask & (1 << record_index)
            )
            right_rows = total_rows - left_rows
            right_atoms = total_atoms - left_atoms
            right_edges = total_edges - left_edges
            objective = (
                max(
                    abs(left_edges / left_rows - target_edge_mean),
                    abs(right_edges / right_rows - target_edge_mean),
                )
                / edge_scale,
                max(
                    abs(left_atoms / left_rows - target_atom_mean),
                    abs(right_atoms / right_rows - target_atom_mean),
                )
                / atom_scale,
                abs(left_rows - right_rows) / total_rows,
                abs(2 * left_count - len(records)),
                mask,
            )
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_mask = mask
        groups[0] = [
            record
            for record_index, record in enumerate(records)
            if best_mask & (1 << record_index)
        ]
        groups[1] = [
            record
            for record_index, record in enumerate(records)
            if not best_mask & (1 << record_index)
        ]
        return groups

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
        row_tail_atoms = np.empty(0, dtype=np.int64)
        row_tail_edges = np.empty(0, dtype=np.int64)
        pack_tail: list[tuple[list[int], int, int]] = []
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
                source_shard = self.index.shards[shard_index]
                cost_shard = self.index.cost_profile.shards[shard_index]
                start = (
                    0
                    if shard_index == 0
                    else self.index.cumulative_ends[shard_index - 1]
                )
                planning_records.append(
                    _ShardPlanningRecord(
                        shard_index=shard_index,
                        start=start,
                        rows=source_shard.length,
                        total_atoms=cost_shard.total_atoms,
                        total_edges=cost_shard.total_edges,
                    )
                )
            self.peak_planning_shards = max(
                self.peak_planning_shards, len(planning_records)
            )
            planning_groups = self._partition_planning_records(
                planning_records, self.window_shards
            )

            for group_index in rng.permutation(len(planning_groups)):
                window = planning_groups[int(group_index)]
                window_indices = []
                window_atoms = []
                window_edges = []
                for record in window:
                    window_indices.append(
                        np.arange(
                            record.start, record.start + record.rows, dtype=np.int64
                        )
                    )
                    window_atoms.append(self.index.load_natoms(record.shard_index))
                    window_edges.append(self.index.load_nedges(record.shard_index))
                self.peak_window_shards = max(self.peak_window_shards, len(window))
                active_rows = len(row_tail_indices) + sum(
                    len(indices) for indices in window_indices
                )
                self.peak_window_rows = max(self.peak_window_rows, active_rows)

                indices = np.concatenate([row_tail_indices, *window_indices])
                natoms = np.concatenate([row_tail_atoms, *window_atoms])
                nedges = np.concatenate([row_tail_edges, *window_edges])
                (
                    complete_batches,
                    row_tail_indices,
                    row_tail_atoms,
                    row_tail_edges,
                ) = _build_atom_edge_balanced_batches(
                    indices, natoms, nedges, self.batch_size, rng
                )

                scheduling_region = pack_tail + complete_batches
                scheduling_region.sort(key=lambda item: (item[2], item[1]))
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
                    batch, _, _ = pack[int(rank_assignment[self.rank])]
                    yielded += 1
                    yield batch

        if yielded != self._rank_batches:
            raise RuntimeError(
                f"Sampler yielded {yielded} batches, expected {self._rank_batches}"
            )
