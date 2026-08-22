###########################################################################################
# CPU/Gloo qualification for opt-in production training diagnostics
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mace.tools.training_diagnostics import (
    TrainingDiagnostics,
    TrainingDiagnosticsConfig,
)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _diagnostics_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_dir: str,
) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        diagnostics = TrainingDiagnostics(
            TrainingDiagnosticsConfig(
                mode="summary",
                interval=1,
                directory=Path(output_dir),
                start_step=0,
                steps=0,
                backend="nvtx",
            ),
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
            distributed=True,
        )
        diagnostics.record_step(
            epoch=4,
            loader_wait_s=0.001 * (rank + 1),
            step_period_s=0.010 + 0.002 * rank,
            h2d_wall_s=0.0005,
            train_wall_s=0.008,
            batch=SimpleNamespace(
                num_graphs=4,
                positions=torch.zeros(10 + rank, 3),
                edge_index=torch.zeros(2, 30 + 10 * rank, dtype=torch.long),
            ),
            optimizer_enabled=True,
            ema_enabled=False,
            compile_requested=True,
            compiled_active=True,
            compiler_graph_count=1,
        )
        diagnostics.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_distributed_diagnostics_write_per_rank_files_and_ordered_aggregate(tmp_path):
    rendezvous_path = tmp_path / "gloo-rendezvous"
    output_dir = tmp_path / "diagnostics"
    mp.spawn(
        _diagnostics_worker,
        args=(2, str(rendezvous_path), str(output_dir)),
        nprocs=2,
        join=True,
    )

    rank_records = [
        _read_jsonl(output_dir / f"rank-{rank:05d}.jsonl") for rank in range(2)
    ]
    assert all(
        [record["record_type"] for record in records] == ["schema", "step"]
        for records in rank_records
    )
    assert [records[1]["rank"] for records in rank_records] == [0, 1]
    assert [records[1]["optimizer_step"] for records in rank_records] == [0, 0]

    aggregate = _read_jsonl(output_dir / "rank-00000-aggregate.jsonl")
    assert [record["record_type"] for record in aggregate] == ["schema", "aggregate"]
    summary = aggregate[1]
    assert summary["epoch"] == 4
    assert summary["optimizer_step"] == 0
    assert summary["atoms"]["min"] == 10.0
    assert summary["atoms"]["max"] == 11.0
    assert summary["atoms"]["worst_rank"] == 1
    assert summary["edges"]["min"] == 30.0
    assert summary["edges"]["max"] == 40.0
    assert summary["edges"]["worst_rank"] == 1
