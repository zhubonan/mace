#!/usr/bin/env python3
"""Fail-closed verification for Phase-44 compiled training artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

POLICIES = {
    "hdf5_random": ("hdf5", "random"),
    "native_random": ("omat_direct", "random"),
    "native_balanced": ("omat_direct", "shard_window_atom_balanced"),
}


def parse_gpu_telemetry(path: Path) -> dict:
    """Summarize dmon samples where the GPU reports active SM work."""
    path = Path(path)
    dmon_path = path / "dmon.csv"
    power_limit_path = path / "gpu-info.csv"
    _require(dmon_path.is_file(), f"missing GPU telemetry: {dmon_path}")
    _require(power_limit_path.is_file(), f"missing GPU telemetry: {power_limit_path}")
    try:
        power_limit_w = float(power_limit_path.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise ValueError(
            f"invalid GPU power limit telemetry: {power_limit_path}"
        ) from error
    _require(
        math.isfinite(power_limit_w) and power_limit_w > 0,
        "invalid GPU power limit telemetry",
    )

    samples = []
    for line in dmon_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        _require(len(fields) >= 8, f"malformed GPU telemetry row: {line}")
        try:
            power_w = float(fields[3])
            sm_percent = float(fields[6])
            memory_percent = float(fields[7])
        except ValueError as error:
            raise ValueError(f"non-numeric GPU telemetry row: {line}") from error
        _require(
            all(
                math.isfinite(value) for value in (power_w, sm_percent, memory_percent)
            ),
            "non-finite GPU telemetry",
        )
        samples.append((power_w, sm_percent, memory_percent))
    _require(samples, "GPU telemetry contains no samples")
    active = [sample for sample in samples if sample[1] > 0]
    _require(active, "GPU telemetry contains no active SM samples")
    active_power = [sample[0] for sample in active]
    active_sm = [sample[1] for sample in active]
    active_memory = [sample[2] for sample in active]
    mean_power = statistics.fmean(active_power)
    return {
        "samples": len(samples),
        "active_samples": len(active),
        "power_limit_w": power_limit_w,
        "active_power_mean_w": mean_power,
        "active_power_peak_w": max(active_power),
        "active_power_mean_limit_fraction": mean_power / power_limit_w,
        "active_sm_mean_percent": statistics.fmean(active_sm),
        "active_sm_peak_percent": max(active_sm),
        "active_memory_mean_percent": statistics.fmean(active_memory),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_rank_records(path: Path) -> list[dict]:
    files = sorted(path.glob("rank-*.json"))
    _require(bool(files), f"No rank artifacts in {path}")
    return [json.loads(file.read_text(encoding="utf-8")) for file in files]


def verify_training_run(
    path: Path,
    *,
    expected_world_size: int,
    expected_source_revision: str,
) -> dict:
    path = Path(path)
    records = _load_rank_records(path)
    _require(len(records) == expected_world_size, "rank artifact count mismatch")
    _require(
        sorted(record["rank"] for record in records)
        == list(range(expected_world_size)),
        "rank identifiers are incomplete",
    )
    first = records[0]
    policy = next(
        (
            name
            for name, pair in POLICIES.items()
            if pair == (first.get("lmdb_backend"), first.get("batch_sampling"))
        ),
        None,
    )
    _require(policy is not None, "unsupported input policy metadata")

    for record in records:
        _require(record.get("world_size") == expected_world_size, "world size mismatch")
        _require(record.get("mode") == "compiled", "run did not use compilation")
        _require(record.get("backend") == "hybrid", "run did not use hybrid backend")
        _require(record.get("derivative_mode") == "positions", "wrong derivative mode")
        precision = record.get("precision_policy", {})
        _require(precision.get("requested") == "highest", "precision is not IEEE FP32")
        _require(
            precision.get("cuda_matmul_allow_tf32") is False,
            "TF32 was enabled",
        )
        _require(
            precision.get("cudnn_allow_tf32") is False,
            "cuDNN TF32 was enabled",
        )
        _require(record.get("batch_size_per_rank") == 128, "batch size mismatch")
        _require(record.get("num_workers") in {8, 16}, "worker count mismatch")
        _require(record.get("warmup_steps") == 4, "warmup count mismatch")
        timed_steps = record.get("timed_steps")
        _require(timed_steps == 256, "timed step count mismatch")
        for field in (
            "step_times_ms",
            "data_wait_times_ms",
            "step_period_times_ms",
            "losses",
            "batch_shapes",
        ):
            _require(len(record.get(field, [])) == timed_steps, f"incomplete {field}")
        _require(
            all(math.isfinite(float(loss)) for loss in record["losses"]),
            "non-finite loss",
        )
        _require(
            all(shape.get("structures") == 128 for shape in record["batch_shapes"]),
            "variable structure count",
        )
        _require(record.get("is_compiled") is True, "compiled facade fell back")
        _require(record.get("compilation_error") is None, "compilation error recorded")
        _require(
            record.get("dynamo_counters_after_warmup")
            == record.get("dynamo_counters_final"),
            "Dynamo graph counters grew after warmup",
        )
        _require(
            record.get("source_revision") == expected_source_revision,
            "source revision mismatch",
        )
        for field in ("benchmark_sha256", "adapter_sha256"):
            value = record.get(field, "")
            _require(len(value) == 64, f"invalid {field}")
        _require(
            (record.get("lmdb_backend"), record.get("batch_sampling"))
            == POLICIES[policy],
            "rank input policies disagree",
        )

    checksums = [record["parameter_checksum"] for record in records]
    _require(len(set(checksums)) == 1, "rank parameter checksums disagree")
    for record in records:
        _require(
            record.get("rank_checksums") == checksums,
            "reported rank checksum vector disagrees",
        )
    benchmark_hashes = {record["benchmark_sha256"] for record in records}
    adapter_hashes = {record["adapter_sha256"] for record in records}
    _require(len(benchmark_hashes) == 1, "rank benchmark hashes disagree")
    _require(len(adapter_hashes) == 1, "rank adapter hashes disagree")
    rates = [float(record["end_to_end_structures_per_second"]) for record in records]
    _require(all(math.isfinite(rate) and rate > 0 for rate in rates), "invalid rate")
    telemetry = parse_gpu_telemetry(path)
    return {
        "path": str(path.resolve()),
        "policy": policy,
        "workers": first["num_workers"],
        "repetition": first["repetition"],
        "world_size": expected_world_size,
        "rate": min(rates),
        "losses": first["losses"],
        "batch_shapes": first["batch_shapes"],
        "parameter_checksum": first["parameter_checksum"],
        "benchmark_sha256": first["benchmark_sha256"],
        "adapter_sha256": first["adapter_sha256"],
        "gpu_telemetry": telemetry,
    }


def verify_gpu_matrix(
    root: Path,
    *,
    expected_source_revision: str,
    expected_world_size: int,
    expected_repetitions: int,
) -> dict:
    root = Path(root)
    runs = []
    for repetition in range(expected_repetitions):
        for workers in (8, 16):
            for policy in POLICIES:
                path = root / f"r{repetition}-{policy}-w{workers}"
                runs.append(
                    verify_training_run(
                        path,
                        expected_world_size=expected_world_size,
                        expected_source_revision=expected_source_revision,
                    )
                )

    benchmark_hashes = {run["benchmark_sha256"] for run in runs}
    adapter_hashes = {run["adapter_sha256"] for run in runs}
    _require(len(benchmark_hashes) == 1, "matrix benchmark hashes disagree")
    _require(len(adapter_hashes) == 1, "matrix adapter hashes disagree")
    medians = {}
    telemetry_medians = {}
    balanced_ok = True
    for workers in (8, 16):
        for policy in POLICIES:
            rates = [
                run["rate"]
                for run in runs
                if run["workers"] == workers and run["policy"] == policy
            ]
            _require(len(rates) == expected_repetitions, "matrix repetition mismatch")
            medians[f"{policy}_w{workers}"] = statistics.median(rates)
            selected_runs = [
                run
                for run in runs
                if run["workers"] == workers and run["policy"] == policy
            ]
            for field in (
                "active_sm_mean_percent",
                "active_power_mean_w",
                "active_power_mean_limit_fraction",
            ):
                telemetry_medians[f"{policy}_w{workers}_{field}"] = statistics.median(
                    run["gpu_telemetry"][field] for run in selected_runs
                )

        random_rate = medians[f"native_random_w{workers}"]
        balanced_rate = medians[f"native_balanced_w{workers}"]
        balanced_ok &= balanced_rate >= 0.98 * random_rate
        for repetition in range(expected_repetitions):
            hdf5 = next(
                run
                for run in runs
                if run["policy"] == "hdf5_random"
                and run["workers"] == workers
                and run["repetition"] == repetition
            )
            native = next(
                run
                for run in runs
                if run["policy"] == "native_random"
                and run["workers"] == workers
                and run["repetition"] == repetition
            )
            _require(
                hdf5["batch_shapes"] == native["batch_shapes"],
                "input parity batch-shape mismatch",
            )
            _require(
                hdf5["losses"][0] == native["losses"][0],
                "input parity first-loss mismatch",
            )
    _require(balanced_ok, "balanced throughput regressed by more than 2%")
    return {
        "root": str(root),
        "runs": len(runs),
        "source_revision": expected_source_revision,
        "world_size": expected_world_size,
        "repetitions": expected_repetitions,
        "batch_size_per_rank": 128,
        "warmup_steps": 4,
        "timed_steps": 256,
        "matmul_precision": "highest",
        "medians": medians,
        "telemetry_medians": telemetry_medians,
        "benchmark_sha256": next(iter(benchmark_hashes)),
        "adapter_sha256": next(iter(adapter_hashes)),
        "acceptance": {
            "balanced_within_two_percent": balanced_ok,
            "compiled_without_fallback": True,
            "matched_batch_shapes_and_first_loss": True,
            "stable_dynamo_graph": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify_gpu_matrix(
        args.root,
        expected_source_revision=args.source_revision,
        expected_world_size=args.world_size,
        expected_repetitions=args.repetitions,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
