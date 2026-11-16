"""Global optimizer batches must be selected before rank partitioning."""

import weakref

import pytest
import torch
from torch.utils.data import DistributedSampler

from mace.data.global_random_sampler import GlobalRandomBatchSampler


@pytest.mark.parametrize(
    "n,b,r", [(0, 2, 4), (3, 2, 4), (17, 2, 3), (1024, 128, 4), (1033, 128, 4)]
)
def test_exact_global_groups_and_no_padded_or_repeated_rows(n, b, r):
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(17))
    streams = [
        list(GlobalRandomBatchSampler(n, b, seed=17, num_replicas=r, rank=rank))
        for rank in range(r)
    ]
    assert all(len(stream) == n // (b * r) for stream in streams)
    for step in range(n // (b * r)):
        global_indices = permutation[step * b * r : (step + 1) * b * r]
        for rank in range(r):
            assert streams[rank][step] == global_indices[rank::r].tolist()
    flat = [i for stream in streams for batch in stream for i in batch]
    assert len(flat) == len(set(flat)) == (n // (b * r)) * b * r
    assert set(flat) == set(permutation[: len(flat)].tolist())


@pytest.mark.parametrize("epoch", [0, 1, 9])
@pytest.mark.parametrize("replicas", [1, 3, 4])
def test_complete_updates_match_original_distributed_sampler(epoch, replicas):
    for rank in range(replicas):
        new = GlobalRandomBatchSampler(
            1033, 7, seed=20260729, epoch=epoch, num_replicas=replicas, rank=rank
        )
        original = DistributedSampler(
            range(1033),
            num_replicas=replicas,
            rank=rank,
            seed=20260729,
            shuffle=True,
            drop_last=True,
        )
        original.set_epoch(epoch)
        assert [i for batch in new for i in batch] == list(original)[: len(new) * 7]


def test_epoch_boundary_resume_and_private_rng():
    state = torch.random.get_rng_state().clone()
    sampler = GlobalRandomBatchSampler(101, 4, seed=5, num_replicas=3, rank=1)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(9)
    resumed = GlobalRandomBatchSampler(101, 4, seed=5, epoch=9, num_replicas=3, rank=1)
    assert list(sampler) == list(resumed)
    assert first != list(resumed)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dataset_size": -1}, "dataset_size"),
        ({"batch_size": 0}, "batch_size"),
        ({"num_replicas": 0}, "num_replicas"),
        ({"num_replicas": 2, "rank": 2}, "rank"),
        ({"rank": -1}, "rank"),
        ({"seed": -1}, "seed"),
        ({"epoch": -1}, "epoch"),
        ({"seed": 2**64}, "seed"),
        ({"seed": 2**64 - 1, "epoch": 1}, "seed"),
    ],
)
def test_invalid_settings_fail_before_iteration(kwargs, match):
    params = dict(dataset_size=32, batch_size=2)
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        GlobalRandomBatchSampler(**params)


@pytest.mark.parametrize(
    "name", ["dataset_size", "batch_size", "num_replicas", "rank", "seed", "epoch"]
)
def test_noninteger_settings_are_not_silently_truncated(name):
    params = dict(dataset_size=32, batch_size=2)
    params[name] = 1.5
    with pytest.raises(TypeError, match=name):
        GlobalRandomBatchSampler(**params)


def test_invalid_epoch_does_not_destroy_replay_state():
    sampler = GlobalRandomBatchSampler(32, 2, seed=2**64 - 1)
    before = list(sampler)
    with pytest.raises(ValueError, match="epoch"):
        sampler.set_epoch(-1)
    with pytest.raises(ValueError, match="seed"):
        sampler.set_epoch(1)
    assert list(sampler) == before


def test_one_lazy_compact_permutation_and_only_local_index_lists(monkeypatch):
    calls, list_sizes = [], []
    randperm, tolist = torch.randperm, torch.Tensor.tolist

    def tracked_randperm(n, **kwargs):
        result = randperm(n, **kwargs)
        calls.append((n, result.device.type, result.dtype))
        return result

    def tracked_tolist(tensor):
        list_sizes.append(tensor.numel())
        return tolist(tensor)

    monkeypatch.setattr(torch, "randperm", tracked_randperm)
    monkeypatch.setattr(torch.Tensor, "tolist", tracked_tolist)
    sampler = GlobalRandomBatchSampler(101, 4, num_replicas=3)
    sampler.set_epoch(9)
    iterator = iter(sampler)
    assert calls == []
    assert len(list(iterator)) == 8
    assert calls == [(101, "cpu", torch.int64)]
    assert list_sizes == [4] * 8


def test_undersized_dataset_does_not_allocate_permutation(monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        pytest.fail("zero-step sampler allocated a permutation")

    monkeypatch.setattr(torch, "randperm", unexpected_allocation)
    assert list(GlobalRandomBatchSampler(3, 2, num_replicas=4)) == []


def test_contract_checks_plan_parameters_but_not_rank():
    base = GlobalRandomBatchSampler(101, 4, seed=5, num_replicas=3)
    other_rank = GlobalRandomBatchSampler(101, 4, seed=5, num_replicas=3, rank=2)
    assert base.contract_digest() == other_rank.contract_digest()
    assert len(bytes.fromhex(base.contract_digest())) == 32
    for overrides in [
        dict(dataset_size=102),
        dict(batch_size=5),
        dict(seed=6),
        dict(epoch=1),
        dict(num_replicas=4),
    ]:
        params = dict(dataset_size=101, batch_size=4, seed=5, num_replicas=3)
        params.update(overrides)
        assert (
            GlobalRandomBatchSampler(**params).contract_digest()
            != base.contract_digest()
        )


def test_torch_version_is_part_of_rank_agreement(monkeypatch):
    sampler = GlobalRandomBatchSampler(101, 4)
    digest = sampler.contract_digest()
    monkeypatch.setattr(torch, "__version__", str(torch.__version__) + ".different")
    assert sampler.contract_digest() != digest


def test_closing_partial_iterator_releases_permutation(monkeypatch):
    references = []
    randperm = torch.randperm

    def tracked_randperm(*args, **kwargs):
        permutation = randperm(*args, **kwargs)
        references.append(weakref.ref(permutation))
        return permutation

    monkeypatch.setattr(torch, "randperm", tracked_randperm)
    sampler = GlobalRandomBatchSampler(101, 4)
    iterator = iter(sampler)
    next(iterator)
    assert references[0]() is not None
    iterator.close()
    assert references[0]() is None
