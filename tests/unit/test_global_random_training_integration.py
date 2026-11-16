"""The global-random opt-in must replace the loader's shard-first sampler."""

import pytest
import torch
from torch.utils.data import ConcatDataset, DistributedSampler

from mace import tools
from mace.cli.run_train import _build_train_loader_and_sampler
from mace.data.global_random_sampler import GlobalRandomBatchSampler


def parser_args(*extra):
    return tools.build_default_arg_parser().parse_args(
        ["--name", "global-random", *extra]
    )


def test_parser_opt_in_leaves_random_default():
    parser = tools.build_default_arg_parser()
    assert parser.parse_args(["--name", "baseline"]).batch_sampling == "random"
    args = parser.parse_args(["--name", "new", "--batch_sampling", "global_random"])
    assert args.batch_sampling == "global_random"


def test_single_rank_selects_global_batches_before_loading():
    args = parser_args(
        "--batch_size", "7", "--batch_sampling", "global_random", "--seed", "17"
    )
    dataset = ConcatDataset([range(516), range(517)])
    loader, sampler = _build_train_loader_and_sampler(dataset, args, world_size=1, rank=0)
    assert isinstance(sampler, GlobalRandomBatchSampler)
    assert loader.batch_sampler is sampler
    assert loader.batch_size is None
    permutation = torch.randperm(1033, generator=torch.Generator().manual_seed(17))
    assert list(sampler) == [
        permutation[start : start + 7].tolist() for start in range(0, 1029, 7)
    ]
    assert len(sampler) == 147


def test_distributed_ranks_stride_complete_global_batches():
    args = parser_args(
        "--batch_size", "7",
        "--batch_sampling", "global_random",
        "--seed", "17",
        "--distributed",
    )
    loader, sampler = _build_train_loader_and_sampler(
        range(1033), args, world_size=3, rank=2
    )
    permutation = torch.randperm(1033, generator=torch.Generator().manual_seed(17))
    expected = [
        permutation[start : start + 21][2::3].tolist()
        for start in range(0, 1029, 21)
    ]
    assert list(sampler) == expected
    all_rank_indices = []
    for sampler_rank in range(3):
        _, rank_sampler = _build_train_loader_and_sampler(
            range(1033), args, world_size=3, rank=sampler_rank
        )
        all_rank_indices.extend(
            index for batch in rank_sampler for index in batch
        )
    assert len(all_rank_indices) == len(set(all_rank_indices)) == 1029


def test_default_distributed_sampler_is_unchanged():
    args = parser_args("--batch_size", "7", "--distributed", "--seed", "17")
    loader, sampler = _build_train_loader_and_sampler(
        range(1033), args, world_size=3, rank=1
    )
    assert isinstance(sampler, DistributedSampler)
    assert loader.sampler is sampler
    assert sampler.drop_last is (not args.lbfgs)


def test_incomplete_global_batch_is_reported_before_iteration():
    args = parser_args("--batch_size", "128", "--batch_sampling", "global_random")
    loader, sampler = _build_train_loader_and_sampler(
        range(17), args, world_size=1, rank=0
    )
    assert len(sampler) == 0
    assert len(loader) == 0


@pytest.mark.parametrize(
    "flag,match",
    [
        (("--lbfgs",), "LBFGS"),
        (("--data_aug_magmom", "True"), "magnetic"),
    ],
)
def test_unsupported_training_modes_fail_fast(flag, match):
    args = parser_args("--batch_sampling", "global_random", *flag)
    with pytest.raises(ValueError, match=match):
        _build_train_loader_and_sampler(
            range(1033), args, world_size=1, rank=0
        )
