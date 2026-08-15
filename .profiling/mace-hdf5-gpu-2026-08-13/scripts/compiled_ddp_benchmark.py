#!/usr/bin/env python3
"""Bounded eager-versus-compiled NCCL training qualification."""

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from e3nn import o3
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage

from mace import data, modules, tools
from mace.cli.convert_e3nn_cueq import run as convert_e3nn_to_cueq
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric


train_tools = importlib.import_module("mace.tools.train")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=16)
    parser.add_argument("--r-max", type=float, default=5.0)
    return parser.parse_args()


def _atomic_numbers(path: Path) -> list[int]:
    numbers = set()
    with h5py.File(path, "r") as handle:
        for batch in handle.values():
            for config in batch.values():
                numbers.update(int(value) for value in config["atomic_numbers"][()])
    return sorted(numbers)


def _build_model(
    atomic_numbers: list[int], r_max: float, device: torch.device
) -> torch.nn.Module:
    torch_geometric.seed_everything(1702)
    with mace_compile.disable_e3nn_codegen():
        model = modules.ScaleShiftMACE(
            r_max=r_max,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=1,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=len(atomic_numbers),
            hidden_irreps=o3.Irreps("32x0e + 32x1o"),
            MLP_irreps=o3.Irreps("32x0e"),
            gate=F.silu,
            atomic_energies=np.zeros(len(atomic_numbers)),
            avg_num_neighbors=8.0,
            atomic_numbers=atomic_numbers,
            correlation=2,
            radial_type="bessel",
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        )
    return convert_e3nn_to_cueq(
        model,
        device=device,
        conv_fusion=False,
    )


def _parameter_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)

    local_rank = int(os.environ["LOCAL_RANK"])
    requested_world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.device_count() < requested_world_size:
        raise RuntimeError(
            f"requested {requested_world_size} ranks but only "
            f"{torch.cuda.device_count()} GPUs are visible"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.set_default_dtype(torch.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_args = {
        "energy": True,
        "forces": True,
        "virials": False,
        "stress": True,
    }
    z_table = tools.AtomicNumberTable(_atomic_numbers(args.dataset))
    source_dataset = data.HDF5Dataset(
        str(args.dataset),
        r_max=args.r_max,
        z_table=z_table,
        heads=["Default"],
        head="Default",
    )
    required_samples = (
        args.batch_size * world_size * (args.warmup_steps + args.timed_steps)
    )
    repeats = math.ceil(required_samples / len(source_dataset))
    repeated_dataset = ConcatDataset([source_dataset] * repeats)
    sampler = DistributedSampler(
        repeated_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=1702,
        drop_last=True,
    )
    loader = torch_geometric.dataloader.DataLoader(
        repeated_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    model = _build_model(z_table.zs, args.r_max, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-3,
        amsgrad=True,
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    loss_fn = modules.WeightedEnergyForcesStressLoss(
        energy_weight=1.0,
        forces_weight=10.0,
        stress_weight=1.0,
    ).to(device)

    iterator = iter(loader)
    compiled_facade = None
    if args.mode == "eager":
        forward_model = DistributedDataParallel(model, device_ids=[local_rank])
    else:
        first_batch = next(iterator).to(device)
        compiled_facade = mace_compile.compile_training_model(
            model,
            first_batch.to_dict(),
            output_args=output_args,
            mode="default",
            dynamic=True,
        )
        compiled_facade = train_tools._synchronize_compiled_training_state(
            compiled_facade,
            device=device,
            distributed=True,
        )
        forward_model = DistributedDataParallel(
            compiled_facade,
            device_ids=[local_rank],
        )
        iterator = itertools.chain([first_batch], iterator)

    losses = []
    step_times_ms = []
    structures = 0
    torch.cuda.reset_peak_memory_stats(device)
    total_steps = args.warmup_steps + args.timed_steps
    for step in range(total_steps):
        batch = next(iterator).to(device)
        batch_dict = batch.to_dict()
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.cuda.nvtx.range(f"{args.mode}_ddp_step_{step}"):
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "eager":
                outputs = forward_model(
                    batch_dict,
                    training=True,
                    compute_force=True,
                    compute_virials=False,
                    compute_stress=True,
                )
            else:
                outputs = forward_model(batch_dict)
            loss = loss_fn(pred=outputs, ref=batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            ema.update()
        if compiled_facade is not None:
            train_tools._synchronize_compiled_training_state(
                forward_model,
                device=device,
                distributed=True,
            )
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(
            [(time.perf_counter() - started) * 1.0e3],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        reported_loss = loss.detach().clone()
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
        reported_loss /= world_size
        if step >= args.warmup_steps:
            step_times_ms.append(float(elapsed.item()))
            losses.append(float(reported_loss.item()))
            structures += args.batch_size * world_size

    checksum = _parameter_checksum(model)
    checksums = [None] * world_size
    dist.all_gather_object(checksums, checksum)
    facade = (
        train_tools._compiled_training_facade(forward_model)
        if args.mode == "compiled"
        else None
    )
    record = {
        "rank": rank,
        "world_size": world_size,
        "mode": args.mode,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(local_rank),
        "dataset": str(args.dataset.resolve()),
        "batch_size_per_rank": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "step_times_ms": step_times_ms,
        "median_step_ms": statistics.median(step_times_ms),
        "structures_per_second": structures / (sum(step_times_ms) / 1.0e3),
        "losses": losses,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "is_compiled": None if facade is None else facade.is_compiled,
        "compilation_error": None if facade is None else facade.compilation_error,
        "parameter_checksum": checksum,
        "rank_checksums": checksums,
    }
    (args.output_dir / f"rank-{rank}.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    if rank == 0:
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
