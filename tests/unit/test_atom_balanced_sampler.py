###########################################################################################
# Tests for fixed-structure atom-balanced OMat batch sampling
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from pathlib import Path

import numpy as np
import pytest

from mace.data.atom_balanced_sampler import ShardWindowAtomBalancedBatchSampler
from mace.data.omat_lmdb_dataset import OMatShard, OMatShardIndex
from mace.tools.torch_geometric.data import Data
from mace.tools.torch_geometric.dataloader import DataLoader


def make_index(tmp_path: Path, shard_sizes: list[list[int]]) -> OMatShardIndex:
    shards = []
    cumulative_ends = []
    total = 0
    for shard_id, sizes in enumerate(shard_sizes):
        family = tmp_path / "family"
        family.mkdir(exist_ok=True)
        shard_path = family / f"db_{shard_id}.aselmdb"
        shard_path.touch()
        metadata_path = family / f"metadata_db_{shard_id}.npz"
        np.savez(metadata_path, natoms=np.asarray(sizes, dtype=np.int64))
        total += len(sizes)
        cumulative_ends.append(total)
        shards.append(
            OMatShard(
                family="family",
                path=shard_path,
                relative_path=Path("family") / shard_path.name,
                metadata_path=metadata_path,
                length=len(sizes),
            )
        )
    return OMatShardIndex(
        roots=(tmp_path,),
        dataset_id="test",
        shards=tuple(shards),
        cumulative_ends=tuple(cumulative_ends),
    )


def flatten(rank_batches):
    return [item for batch in rank_batches for item in batch]


def make_rank_samplers(index, *, epoch=0, window_shards=2):
    return [
        ShardWindowAtomBalancedBatchSampler(
            index,
            batch_size=4,
            seed=17,
            epoch=epoch,
            num_replicas=2,
            rank=rank,
            window_shards=window_shards,
        )
        for rank in range(2)
    ]


def test_sampler_preserves_fixed_batches_and_ddp_coverage(tmp_path):
    index = make_index(tmp_path, [[2, 8, 3, 9, 4], [7, 6, 5, 10], [11, 1, 12]])
    samplers = make_rank_samplers(index)
    batches = [list(sampler) for sampler in samplers]

    assert len(batches[0]) == len(batches[1]) == len(samplers[0])
    assert all(len(batch) == 4 for rank_batches in batches for batch in rank_batches)
    rank_sets = [set(flatten(rank_batches)) for rank_batches in batches]
    assert rank_sets[0].isdisjoint(rank_sets[1])
    used = len(rank_sets[0] | rank_sets[1])
    assert len(flatten(batches[0])) == len(rank_sets[0])
    assert len(flatten(batches[1])) == len(rank_sets[1])
    assert 0 <= len(index) - used < 2 * 4


def test_sampler_is_deterministic_and_set_epoch_changes_order(tmp_path):
    index = make_index(tmp_path, [list(range(1, 17)), list(range(17, 33))])
    first = ShardWindowAtomBalancedBatchSampler(index, 4, seed=9)
    second = ShardWindowAtomBalancedBatchSampler(index, 4, seed=9)

    epoch_zero = list(first)
    assert epoch_zero == list(second)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != epoch_zero


def test_sampler_carries_incomplete_rows_and_ddp_packs_across_windows(tmp_path):
    index = make_index(tmp_path, [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12]])
    batches = [list(sampler) for sampler in make_rank_samplers(index, window_shards=1)]
    used = set(flatten(batches[0])) | set(flatten(batches[1]))

    assert len(batches[0]) == len(batches[1]) == 1
    assert len(used) == 8
    assert len(index) - len(used) == 3


def test_sampler_reduces_simultaneous_rank_atom_imbalance(tmp_path):
    sizes = np.asarray(([1] * 32) + ([100] * 32), dtype=np.int64)
    index = make_index(
        tmp_path, [sizes[:16].tolist(), sizes[16:48].tolist(), sizes[48:].tolist()]
    )
    rank_batches = [list(sampler) for sampler in make_rank_samplers(index)]
    balanced_totals = [
        [sum(sizes[item] for item in rank_batches[rank][step]) for rank in range(2)]
        for step in range(len(rank_batches[0]))
    ]

    rng = np.random.default_rng(17)
    random_indices = rng.permutation(len(index)).reshape(-1, 4)
    random_totals = [
        [int(sizes[batch].sum()) for batch in random_indices[step : step + 2]]
        for step in range(0, len(random_indices), 2)
    ]

    def p95_relative_imbalance(step_totals):
        values = [
            (max(totals) - min(totals)) / (sum(totals) / len(totals))
            for totals in step_totals
        ]
        return float(np.quantile(values, 0.95))

    assert p95_relative_imbalance(balanced_totals) < p95_relative_imbalance(
        random_totals
    )


def test_sampler_loads_metadata_in_bounded_windows(tmp_path):
    index = make_index(tmp_path, [[value] for value in range(1, 8)])
    original = index.load_natoms
    calls = []

    def recording_load(shard_index):
        calls.append(shard_index)
        return original(shard_index)

    object.__setattr__(index, "load_natoms", recording_load)
    sampler = ShardWindowAtomBalancedBatchSampler(
        index, batch_size=2, seed=4, window_shards=3
    )

    iterator = iter(sampler)
    next(iterator)
    assert len(calls) <= 3
    list(iterator)
    assert sorted(calls) == list(range(7))
    assert sampler.peak_window_shards <= 3
    assert sampler.peak_window_rows <= 3 + sampler.batch_size - 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"num_replicas": 0}, "num_replicas"),
        ({"num_replicas": 2, "rank": 2}, "rank"),
        ({"window_shards": 0}, "window_shards"),
    ],
)
def test_sampler_rejects_invalid_configuration(tmp_path, kwargs, match):
    index = make_index(tmp_path, [[1, 2, 3, 4]])
    arguments = {"batch_size": 2, **kwargs}

    with pytest.raises(ValueError, match=match):
        ShardWindowAtomBalancedBatchSampler(index, **arguments)


def test_sampler_batches_collate_through_pyg_loader(tmp_path):
    index = make_index(tmp_path, [[1, 2, 3, 4]])
    dataset = [Data(value=np.asarray([item])) for item in range(len(index))]
    sampler = ShardWindowAtomBalancedBatchSampler(index, batch_size=2, seed=2)
    loader = DataLoader(dataset, batch_sampler=sampler)

    batches = list(loader)

    assert len(batches) == 2
    assert all(batch.num_graphs == 2 for batch in batches)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("batch_size,num_replicas", [(1, 1), (3, 2), (4, 3)])
def test_sampler_stress_coverage_across_empty_shards(
    tmp_path, seed, batch_size, num_replicas
):
    index = make_index(
        tmp_path,
        [[], [1, 3], list(range(2, 13)), [], list(range(13, 29))],
    )
    rank_batches = [
        list(
            ShardWindowAtomBalancedBatchSampler(
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
    rank_sets = [set(flatten(batches)) for batches in rank_batches]
    used = set().union(*rank_sets)

    assert len({len(batches) for batches in rank_batches}) == 1
    assert all(
        len(batch) == batch_size for batches in rank_batches for batch in batches
    )
    assert sum(len(items) for items in rank_sets) == len(used)
    assert len(index) - len(used) < batch_size * num_replicas
