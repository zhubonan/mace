###########################################################################################
# Tests for fixed-structure atom-and-edge-balanced OMat batch sampling
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mace.data.atom_balanced_sampler import ShardWindowAtomBalancedBatchSampler
from mace.data.atom_edge_balanced_sampler import (
    ShardWindowAtomEdgeBalancedBatchSampler,
)
from mace.data.omat_cost_profile import OMatCostShard
from mace.data.omat_lmdb_dataset import OMatShard, OMatShardIndex


@dataclass(frozen=True)
class MemoryCostProfile:
    values: tuple[np.ndarray, ...]
    shards: tuple[OMatCostShard, ...]
    profile_id: str = "memory-profile"
    manifest_sha256: str = "a" * 64
    cutoff: float = 5.0

    def load_nedges(self, shard_index: int) -> np.ndarray:
        return self.values[shard_index].copy()


def make_cost_index(
    tmp_path: Path, natoms_by_shard: list[list[int]], nedges_by_shard: list[list[int]]
) -> OMatShardIndex:
    assert len(natoms_by_shard) == len(nedges_by_shard)
    family = tmp_path / "family"
    family.mkdir()
    shards = []
    cost_shards = []
    cumulative_ends = []
    total = 0
    edge_values = []
    for shard_id, (natoms, nedges) in enumerate(
        zip(natoms_by_shard, nedges_by_shard, strict=True)
    ):
        assert len(natoms) == len(nedges)
        shard_path = family / f"db_{shard_id}.aselmdb"
        shard_path.touch()
        metadata_path = family / f"metadata_db_{shard_id}.npz"
        np.savez(metadata_path, natoms=np.asarray(natoms, dtype=np.int64))
        relative_path = Path("family") / shard_path.name
        total += len(natoms)
        cumulative_ends.append(total)
        shards.append(
            OMatShard(
                family="family",
                path=shard_path,
                relative_path=relative_path,
                metadata_path=metadata_path,
                length=len(natoms),
            )
        )
        edge_array = np.asarray(nedges, dtype=np.int64)
        edge_values.append(edge_array)
        cost_shards.append(
            OMatCostShard(
                relative_shard_path=relative_path,
                cost_path=None,
                rows=len(natoms),
                total_atoms=int(np.asarray(natoms).sum()),
                total_edges=int(edge_array.sum()),
            )
        )
    profile = MemoryCostProfile(tuple(edge_values), tuple(cost_shards))
    return OMatShardIndex(
        roots=(tmp_path,),
        dataset_id="test",
        shards=tuple(shards),
        cumulative_ends=tuple(cumulative_ends),
        manifest_path=tmp_path / "omat24-manifest.json",
        manifest_sha256="b" * 64,
        cost_profile=profile,
    )


def flatten(batches):
    return [index for batch in batches for index in batch]


def rank_samplers(
    index,
    *,
    seed=17,
    epoch=0,
    batch_size=4,
    num_replicas=2,
    window_shards=2,
):
    return [
        ShardWindowAtomEdgeBalancedBatchSampler(
            index,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
            window_shards=window_shards,
        )
        for rank in range(num_replicas)
    ]


def global_costs(index):
    atoms = np.concatenate(
        [index.load_natoms(shard) for shard in range(len(index.shards))]
    )
    edges = np.concatenate(
        [index.load_nedges(shard) for shard in range(len(index.shards))]
    )
    return atoms, edges


def batch_totals(rank_batches, values):
    return np.asarray(
        [int(values[batch].sum()) for batches in rank_batches for batch in batches]
    )


def test_edge_sampler_preserves_fixed_batches_and_disjoint_ddp_coverage(tmp_path):
    index = make_cost_index(
        tmp_path,
        [[2, 8, 3, 9, 4], [7, 6, 5, 10], [11, 1, 12]],
        [[4, 70, 7, 90, 8], [50, 42, 30, 100], [120, 2, 150]],
    )
    samplers = rank_samplers(index)
    ranks = [list(sampler) for sampler in samplers]

    assert len(ranks[0]) == len(ranks[1]) == len(samplers[0])
    assert all(len(batch) == 4 for batches in ranks for batch in batches)
    rank_sets = [set(flatten(batches)) for batches in ranks]
    assert rank_sets[0].isdisjoint(rank_sets[1])
    used = set().union(*rank_sets)
    assert sum(len(items) for items in rank_sets) == len(used)
    assert 0 <= len(index) - len(used) < 8


def test_edge_sampler_is_deterministic_and_epoch_changes_order(tmp_path):
    index = make_cost_index(
        tmp_path,
        [list(range(1, 33))],
        [list(range(11, 43))],
    )
    first = ShardWindowAtomEdgeBalancedBatchSampler(index, 4, seed=9)
    second = ShardWindowAtomEdgeBalancedBatchSampler(index, 4, seed=9)

    epoch_zero = list(first)
    assert epoch_zero == list(second)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != epoch_zero
    assert first.contract_digest() == second.contract_digest()


def test_edge_sampler_resume_reconstructs_the_requested_epoch_exactly(tmp_path):
    index = make_cost_index(
        tmp_path,
        [list(range(1, 25)), list(range(25, 49))],
        [list(range(101, 125)), list(range(125, 149))],
    )
    resumed = ShardWindowAtomEdgeBalancedBatchSampler(
        index,
        batch_size=4,
        seed=23,
        epoch=7,
        num_replicas=2,
        rank=1,
        window_shards=1,
    )
    advanced = ShardWindowAtomEdgeBalancedBatchSampler(
        index,
        batch_size=4,
        seed=23,
        epoch=0,
        num_replicas=2,
        rank=1,
        window_shards=1,
    )
    advanced.set_epoch(7)

    assert list(resumed) == list(advanced)
    assert resumed.contract_payload() == advanced.contract_payload()
    assert resumed.contract_digest() == advanced.contract_digest()


def test_edge_sampler_reduces_edge_tail_when_atoms_and_edges_are_decorrelated(
    tmp_path,
):
    natoms = [10] * 64
    nedges = list(range(1, 65))
    index = make_cost_index(tmp_path, [natoms], [nedges])
    atom_batches = [
        list(
            ShardWindowAtomBalancedBatchSampler(
                index, 4, seed=17, num_replicas=2, rank=rank, window_shards=1
            )
        )
        for rank in range(2)
    ]
    edge_batches = [list(sampler) for sampler in rank_samplers(index, window_shards=1)]
    atoms, edges = global_costs(index)
    atom_mode_edge_totals = batch_totals(atom_batches, edges)
    edge_mode_edge_totals = batch_totals(edge_batches, edges)
    edge_mode_atom_totals = batch_totals(edge_batches, atoms)

    assert np.ptp(edge_mode_edge_totals) < np.ptp(atom_mode_edge_totals)
    assert np.quantile(edge_mode_edge_totals, 0.99) < np.quantile(
        atom_mode_edge_totals, 0.99
    )
    assert np.ptp(edge_mode_atom_totals) == 0


def test_edge_sampler_preserves_atom_balance_with_variable_edge_density(tmp_path):
    source = np.random.default_rng(0)
    natoms = np.resize(np.asarray([2, 4, 8, 16, 32, 64, 96], dtype=np.int64), 256)
    source.shuffle(natoms)
    edge_density = source.choice(
        np.asarray([12, 16, 24, 48, 72, 96], dtype=np.int64), size=len(natoms)
    )
    nedges = natoms * edge_density
    index = make_cost_index(tmp_path, [natoms.tolist()], [nedges.tolist()])
    atom_batches = [
        list(
            ShardWindowAtomBalancedBatchSampler(
                index, 8, seed=17, num_replicas=1, rank=0, window_shards=1
            )
        )
    ]
    edge_batches = [
        list(
            ShardWindowAtomEdgeBalancedBatchSampler(
                index, 8, seed=17, num_replicas=1, rank=0, window_shards=1
            )
        )
    ]
    atoms, edges = global_costs(index)
    atom_mode_atom_totals = batch_totals(atom_batches, atoms)
    edge_mode_atom_totals = batch_totals(edge_batches, atoms)
    atom_mode_edge_totals = batch_totals(atom_batches, edges)
    edge_mode_edge_totals = batch_totals(edge_batches, edges)

    assert np.ptp(edge_mode_atom_totals) <= np.ptp(atom_mode_atom_totals)
    assert np.ptp(edge_mode_edge_totals) < np.ptp(atom_mode_edge_totals)


def test_edge_sampler_balances_simultaneous_rank_packs(tmp_path):
    natoms = ([3] * 32) + ([30] * 32)
    nedges = list(range(5, 69))
    index = make_cost_index(
        tmp_path,
        [natoms[:16], natoms[16:48], natoms[48:]],
        [nedges[:16], nedges[16:48], nedges[48:]],
    )
    ranks = [list(sampler) for sampler in rank_samplers(index, num_replicas=4)]
    atoms, edges = global_costs(index)
    simultaneous_edge_spread = []
    simultaneous_atom_spread = []
    for step in range(len(ranks[0])):
        simultaneous_edge_spread.append(
            np.ptp([int(edges[ranks[rank][step]].sum()) for rank in range(4)])
        )
        simultaneous_atom_spread.append(
            np.ptp([int(atoms[ranks[rank][step]].sum()) for rank in range(4)])
        )

    assert max(simultaneous_edge_spread) <= 8
    assert max(simultaneous_atom_spread) <= 54


def test_edge_sampler_keeps_an_irreducibly_dense_structure_visible(tmp_path):
    natoms = [8] * 16
    nedges = [32] * 15 + [1_000_000]
    index = make_cost_index(tmp_path, [natoms], [nedges])
    ranks = [
        list(sampler)
        for sampler in rank_samplers(
            index,
            batch_size=4,
            num_replicas=2,
            window_shards=1,
        )
    ]
    used = flatten([batch for batches in ranks for batch in batches])
    _, edges = global_costs(index)
    totals = [int(edges[batch].sum()) for batches in ranks for batch in batches]

    assert sorted(used) == list(range(len(index)))
    assert used.count(15) == 1
    assert max(totals) >= 1_000_000


def test_edge_sampler_loads_atom_and_edge_arrays_in_bounded_windows(tmp_path):
    index = make_cost_index(
        tmp_path,
        [[value, value + 1] for value in range(1, 10)],
        [[10 * value, 10 * value + 2] for value in range(1, 10)],
    )
    original_atoms = index.load_natoms
    original_edges = index.load_nedges
    atom_calls = []
    edge_calls = []

    def load_atoms(shard_index):
        atom_calls.append(shard_index)
        return original_atoms(shard_index)

    def load_edges(shard_index):
        edge_calls.append(shard_index)
        return original_edges(shard_index)

    object.__setattr__(index, "load_natoms", load_atoms)
    object.__setattr__(index, "load_nedges", load_edges)
    sampler = ShardWindowAtomEdgeBalancedBatchSampler(
        index, batch_size=2, seed=4, window_shards=3
    )

    iterator = iter(sampler)
    next(iterator)
    assert len(atom_calls) <= sampler.window_shards
    assert len(edge_calls) <= sampler.window_shards
    list(iterator)
    assert sorted(atom_calls) == list(range(9))
    assert sorted(edge_calls) == list(range(9))
    assert sampler.peak_planning_shards <= min(2 * sampler.window_shards, 16)
    assert sampler.peak_window_shards <= sampler.window_shards
    assert (
        sampler.peak_window_rows <= 2 * sampler.window_shards + sampler.batch_size - 1
    )


def test_edge_sampler_contract_excludes_rank_but_tracks_epoch_and_profile(tmp_path):
    index = make_cost_index(tmp_path, [[2, 3, 4, 5] * 4], [[4, 8, 12, 16] * 4])
    samplers = rank_samplers(index)

    assert samplers[0].contract_payload()["profile_id"] == "memory-profile"
    assert samplers[0].contract_digest() == samplers[1].contract_digest()
    samplers[0].set_epoch(3)
    assert samplers[0].contract_digest() != samplers[1].contract_digest()


def test_edge_sampler_requires_an_associated_cost_profile(tmp_path):
    index = make_cost_index(tmp_path, [[2, 3]], [[4, 8]])
    object.__setattr__(index, "cost_profile", None)

    with pytest.raises(ValueError, match="cost profile"):
        ShardWindowAtomEdgeBalancedBatchSampler(index, batch_size=2)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("batch_size,num_replicas", [(1, 1), (3, 2), (4, 3)])
def test_edge_sampler_stress_coverage_across_empty_shards(
    tmp_path, seed, batch_size, num_replicas
):
    natoms = [[], [1, 3], list(range(2, 13)), [], list(range(13, 29))]
    nedges = [[], [4, 12], list(range(20, 31)), [], list(range(40, 56))]
    index = make_cost_index(tmp_path, natoms, nedges)
    ranks = [
        list(
            ShardWindowAtomEdgeBalancedBatchSampler(
                index,
                batch_size=batch_size,
                seed=seed,
                epoch=2,
                num_replicas=num_replicas,
                rank=rank,
                window_shards=2,
            )
        )
        for rank in range(num_replicas)
    ]
    rank_sets = [set(flatten(batches)) for batches in ranks]
    used = set().union(*rank_sets)

    assert len({len(batches) for batches in ranks}) == 1
    assert all(len(batch) == batch_size for batches in ranks for batch in batches)
    assert sum(len(items) for items in rank_sets) == len(used)
    assert len(index) - len(used) < batch_size * num_replicas
