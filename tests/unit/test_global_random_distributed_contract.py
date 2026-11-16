"""Real CPU ranks must agree before consuming globally selected batches."""

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from mace.data.global_random_sampler import GlobalRandomBatchSampler
from mace.tools.train import _verify_distributed_sampler_contract


def _rank_process(rank, world_size, rendezvous, output, mismatch):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=25),
    )
    try:
        sampler = GlobalRandomBatchSampler(
            1033,
            7,
            seed=18 if mismatch and rank == 1 else 17,
            epoch=9,
            num_replicas=world_size,
            rank=rank,
        )
        try:
            _verify_distributed_sampler_contract(
                sampler, distributed=True, device=torch.device("cpu")
            )
        except RuntimeError as exc:
            result = {"error": str(exc), "batches_consumed": 0}
        else:
            batches = list(sampler)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, batches)
            result = {"batches": batches, "all_ranks": gathered}
        Path(output, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("mismatch", [False, True])
def test_real_ranks_agree_or_fail_before_iteration(tmp_path, mismatch):
    context = mp.spawn(
        _rank_process,
        args=(3, str(tmp_path / "rendezvous"), str(tmp_path), mismatch),
        nprocs=3,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("distributed sampler test exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    results = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(3)
    ]
    if mismatch:
        assert all(
            "Distributed sampler contract mismatch" in r["error"]
            and r["batches_consumed"] == 0
            for r in results
        )
    else:
        p = torch.randperm(1033, generator=torch.Generator().manual_seed(26))
        for rank, result in enumerate(results):
            assert result["all_ranks"] == [r["batches"] for r in results]
            flat = [i for batch in result["batches"] for i in batch]
            assert flat == p[:1029][rank::3].tolist()
        assert len(set(i for r in results for b in r["batches"] for i in b)) == 1029
