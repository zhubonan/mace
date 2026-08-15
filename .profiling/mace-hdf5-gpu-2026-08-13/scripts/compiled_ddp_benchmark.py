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
from contextlib import nullcontext
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
    parser.add_argument(
        "--model-preset",
        choices=("small", "production"),
        default="small",
    )
    parser.add_argument(
        "--production-ablation",
        choices=(
            "none",
            "no-pair",
            "residual",
            "reduced-cg",
            "ell1",
            "ell2",
            "correlation2",
            "channels32",
        ),
        default="none",
    )
    parser.add_argument("--r-max", type=float)
    parser.add_argument("--dump-fx-only", action="store_true")
    return parser.parse_args()


def _atomic_numbers(path: Path) -> list[int]:
    numbers = set()
    with h5py.File(path, "r") as handle:
        for batch in handle.values():
            for config in batch.values():
                numbers.update(int(value) for value in config["atomic_numbers"][()])
    return sorted(numbers)


def _build_model(
    atomic_numbers: list[int],
    r_max: float,
    device: torch.device,
    model_preset: str,
    production_ablation: str,
    compile_training: bool = False,
) -> torch.nn.Module:
    torch_geometric.seed_everything(1702)
    if model_preset == "production":
        model_options = {
            "num_bessel": 10,
            "num_polynomial_cutoff": 5,
            "max_ell": 3,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticDensityResidualInteractionBlock"
            ],
            "interaction_cls_first": modules.interaction_classes[
                "RealAgnosticDensityInteractionBlock"
            ],
            "hidden_irreps": o3.Irreps("128x0e + 128x1o"),
            "MLP_irreps": o3.Irreps("16x0e"),
            "avg_num_neighbors": 61.964672446250916,
            "correlation": 3,
            "pair_repulsion": True,
            "distance_transform": "Agnesi",
            "radial_MLP": [64, 64, 64],
            "use_reduced_cg": False,
        }
        if production_ablation == "no-pair":
            model_options["pair_repulsion"] = False
        elif production_ablation == "residual":
            residual = modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ]
            model_options["interaction_cls"] = residual
            model_options["interaction_cls_first"] = residual
        elif production_ablation == "reduced-cg":
            model_options["use_reduced_cg"] = True
        elif production_ablation == "ell1":
            model_options["max_ell"] = 1
        elif production_ablation == "ell2":
            model_options["max_ell"] = 2
        elif production_ablation == "correlation2":
            model_options["correlation"] = 2
        elif production_ablation == "channels32":
            model_options["hidden_irreps"] = o3.Irreps("32x0e + 32x1o")
    else:
        model_options = {
            "num_bessel": 4,
            "num_polynomial_cutoff": 4,
            "max_ell": 1,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "interaction_cls_first": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
            "MLP_irreps": o3.Irreps("32x0e"),
            "avg_num_neighbors": 8.0,
            "correlation": 2,
        }
    with mace_compile.disable_e3nn_codegen():
        model = modules.ScaleShiftMACE(
            r_max=r_max,
            num_interactions=2,
            num_elements=len(atomic_numbers),
            gate=F.silu,
            atomic_energies=np.zeros(len(atomic_numbers)),
            atomic_numbers=atomic_numbers,
            radial_type="bessel",
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            **model_options,
        )
    conversion_context = (
        mace_compile.disable_e3nn_codegen() if compile_training else nullcontext()
    )
    with conversion_context:
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


def _dynamo_counters() -> dict[str, dict[str, int]]:
    return {
        category: dict(values)
        for category, values in torch._dynamo.utils.counters.items()
        if values
    }


def _dump_fx_graph(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    validation_batch: dict[str, torch.Tensor],
    output_args: dict[str, bool],
    output_dir: Path,
) -> None:
    from mace.tools.compile_training import make_training_fx

    fx_model = make_training_fx(
        model,
        batch,
        output_args,
        validation_batch=validation_batch,
    )
    (output_dir / "training_fx.py").write_text(
        fx_model.code,
        encoding="utf-8",
    )
    nodes = []
    for index, node in enumerate(fx_model.graph.nodes):
        value = node.meta.get("val")
        nodes.append(
            {
                "index": index,
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": repr(node.args),
                "kwargs": repr(node.kwargs),
                "users": [user.name for user in node.users],
                "value_shape": (
                    [str(dimension) for dimension in value.shape]
                    if isinstance(value, torch.Tensor)
                    else None
                ),
                "stack_trace": node.meta.get("stack_trace"),
                "nn_module_stack": repr(node.meta.get("nn_module_stack")),
                "source_fn_stack": repr(node.meta.get("source_fn_stack")),
            }
        )
    (output_dir / "training_fx_nodes.json").write_text(
        json.dumps(nodes, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fx_nodes": len(nodes),
                "code": str(output_dir / "training_fx.py"),
                "nodes": str(output_dir / "training_fx_nodes.json"),
            },
            indent=2,
        ),
        flush=True,
    )


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
    r_max = args.r_max
    if r_max is None:
        r_max = 6.0 if args.model_preset == "production" else 5.0
    output_args = {
        "energy": True,
        "forces": True,
        "virials": False,
        "stress": True,
    }
    atomic_numbers = _atomic_numbers(args.dataset)
    if args.model_preset == "production":
        atomic_numbers = list(range(1, 84)) + list(range(89, 95))
    z_table = tools.AtomicNumberTable(atomic_numbers)
    source_dataset = data.HDF5Dataset(
        str(args.dataset),
        r_max=r_max,
        z_table=z_table,
        heads=["Default"],
        head="Default",
    )
    required_samples = args.batch_size * world_size * max(
        2,
        args.warmup_steps + args.timed_steps,
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

    model = _build_model(
        z_table.zs,
        r_max,
        device,
        args.model_preset,
        args.production_ablation,
        compile_training=args.mode == "compiled",
    )
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
        validation_batch = next(iterator).to(device)
        if args.dump_fx_only:
            _dump_fx_graph(
                model,
                first_batch.to_dict(),
                validation_batch.to_dict(),
                output_args,
                args.output_dir,
            )
            return
        compiled_facade = mace_compile.compile_training_model(
            model,
            first_batch.to_dict(),
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=validation_batch.to_dict(),
        )
        compiled_facade.materialize_compiled_backward(first_batch.to_dict())
        compiled_facade = train_tools._synchronize_compiled_training_state(
            compiled_facade,
            device=device,
            distributed=True,
        )
        forward_model = DistributedDataParallel(
            compiled_facade,
            device_ids=[local_rank],
        )
        iterator = itertools.chain([first_batch, validation_batch], iterator)

    losses = []
    step_times_ms = []
    batch_shapes = []
    structures = 0
    counters_after_warmup = None
    torch.cuda.reset_peak_memory_stats(device)
    total_steps = args.warmup_steps + args.timed_steps
    for step in range(total_steps):
        batch = next(iterator).to(device)
        batch_dict = batch.to_dict()
        current_shape = {
            "structures": int(batch.ptr.numel() - 1),
            "atoms": int(batch.positions.shape[0]),
            "edges": int(batch.edge_index.shape[1]),
        }
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
            batch_shapes.append(current_shape)
            structures += args.batch_size * world_size
        elif step + 1 == args.warmup_steps:
            counters_after_warmup = _dynamo_counters()

    checksum = _parameter_checksum(model)
    checksums = [None] * world_size
    dist.all_gather_object(checksums, checksum)
    facade = (
        train_tools._compiled_training_facade(forward_model)
        if args.mode == "compiled"
        else None
    )
    validation_batch_shape = None
    dynamic_shape_signatures = None
    if facade is not None:
        validation_batch_shape = {
            "structures": int(validation_batch.ptr.numel() - 1),
            "atoms": int(validation_batch.positions.shape[0]),
            "edges": int(validation_batch.edge_index.shape[1]),
        }
        dynamic_shape_signatures = facade.dynamic_shape_signatures
    record = {
        "rank": rank,
        "world_size": world_size,
        "mode": args.mode,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(local_rank),
        "dataset": str(args.dataset.resolve()),
        "model_preset": args.model_preset,
        "production_ablation": args.production_ablation,
        "r_max": r_max,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size_per_rank": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "step_times_ms": step_times_ms,
        "median_step_ms": statistics.median(step_times_ms),
        "structures_per_second": structures / (sum(step_times_ms) / 1.0e3),
        "losses": losses,
        "batch_shapes": batch_shapes,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "is_compiled": None if facade is None else facade.is_compiled,
        "compilation_error": None if facade is None else facade.compilation_error,
        "validation_batch_shape": validation_batch_shape,
        "dynamic_shape_signatures": dynamic_shape_signatures,
        "dynamo_counters_after_warmup": counters_after_warmup,
        "dynamo_counters_final": _dynamo_counters(),
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
