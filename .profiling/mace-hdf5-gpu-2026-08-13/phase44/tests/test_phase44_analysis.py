###########################################################################################
# Tests for Phase-44 complete-corpus and CPU qualification analyzers
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import sys
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import ConcatDataset, Dataset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_balanced_sampler import audit_sampler  # noqa: E402
from benchmark_native_omat_lmdb import (  # noqa: E402
    benchmark_order,
    validate_benchmark_records,
)
from run_training_input_backend import (  # noqa: E402
    PrefixShardIndex,
    RepeatedBalancedDistributedSampler,
)
from verify_phase44 import (  # noqa: E402
    parse_gpu_telemetry,
    verify_gpu_matrix,
    verify_training_run,
)

from mace.data import OMatShard, OMatShardIndex  # noqa: E402


def make_index(tmp_path, shard_sizes):
    shards = []
    cumulative = []
    total = 0
    for shard_id, sizes in enumerate(shard_sizes):
        family = tmp_path / "family"
        family.mkdir(exist_ok=True)
        shard_path = family / f"db_{shard_id}.aselmdb"
        shard_path.touch()
        metadata = family / f"metadata_db_{shard_id}.npz"
        np.savez(metadata, natoms=np.asarray(sizes, dtype=np.int64))
        total += len(sizes)
        cumulative.append(total)
        shards.append(
            OMatShard(
                family="family",
                path=shard_path,
                relative_path=Path("family") / shard_path.name,
                metadata_path=metadata,
                length=len(sizes),
            )
        )
    return OMatShardIndex(
        roots=(tmp_path,),
        dataset_id="fixture",
        shards=tuple(shards),
        cumulative_ends=tuple(cumulative),
    )


def test_sampler_audit_reports_exact_coverage_balance_and_memory(tmp_path):
    index = make_index(
        tmp_path,
        [[1] * 16, [1] * 16, [100] * 16, [100] * 16],
    )

    summary = audit_sampler(
        index,
        batch_size=2,
        world_size=2,
        window_shards=2,
        seed=19,
        epoch=1,
        temporary_directory=tmp_path,
        random_steps=64,
    )

    assert summary["rows"] == len(index)
    assert summary["used"] + summary["dropped"] == len(index)
    assert summary["dropped"] < 2 * 2
    assert len(set(summary["rank_steps"])) == 1
    assert summary["disjoint_rank_samples"] is True
    assert summary["duplicates"] == 0
    assert (
        summary["balanced_p95_rank_imbalance"] <= summary["random_p95_rank_imbalance"]
    )
    assert summary["peak_window_rows"] <= summary["largest_window_rows"] + 1
    assert len(summary["checksum"]) == 64


def valid_benchmark_records():
    records = []
    for sampling in ("random", "shard_window_atom_balanced"):
        for repetition in range(3):
            for workers in (0, 8, 16):
                records.append(
                    {
                        "sampling": sampling,
                        "repetition": repetition,
                        "workers": workers,
                        "samples": 256,
                        "indices_checksum": f"{sampling}-{repetition}",
                        "structures_per_second": 100.0 + workers,
                        "max_active_environments": 2,
                        "max_open_environment_count": 3,
                    }
                )
    return records


def test_cpu_benchmark_validator_accepts_three_matched_repetitions():
    validate_benchmark_records(
        valid_benchmark_records(), expected_samples=256, max_open_shards=4
    )


def test_cpu_benchmark_order_balances_policy_and_cold_worker_position():
    orders = [benchmark_order(repetition) for repetition in range(3)]

    assert [order[0][0] for order in orders] == [
        "random",
        "shard_window_atom_balanced",
        "random",
    ]
    for policy in ("random", "shard_window_atom_balanced"):
        first_workers = []
        for order in orders:
            first_workers.append(
                next(workers for name, workers in order if name == policy)
            )
        assert sorted(first_workers) == [0, 8, 16]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows[1].update(indices_checksum="different"), "index stream"),
        (
            lambda rows: rows[0].update(structures_per_second=float("nan")),
            "finite",
        ),
        (lambda rows: rows[0].update(max_active_environments=5), "environment"),
        (lambda rows: rows.pop(), "repetitions"),
    ],
)
def test_cpu_benchmark_validator_fails_closed(mutation, match):
    records = valid_benchmark_records()
    mutation(records)

    with pytest.raises(ValueError, match=match):
        validate_benchmark_records(records, expected_samples=256, max_open_shards=4)


class IndexDataset(Dataset):
    def __init__(self, index):
        self.shard_index = index

    def __len__(self):
        return len(self.shard_index)

    def __getitem__(self, item):
        return item


def test_prefix_shard_index_limits_rows_and_metadata(tmp_path):
    index = make_index(tmp_path, [[1, 2, 3, 4], [20, 21]])

    limited = PrefixShardIndex(index, 3)

    assert len(limited) == 3
    assert len(limited.shards) == 1
    assert limited.shards[0].length == 3
    assert limited.locate(2) == (0, 2)
    assert limited.load_natoms(0).tolist() == [1, 2, 3]
    with pytest.raises(IndexError):
        limited.locate(3)


def test_repeated_balanced_sampler_preserves_fixed_ddp_batches(tmp_path):
    index = make_index(tmp_path, [[1, 2, 3, 4], [20, 21, 22, 23]])
    source = IndexDataset(index)
    repeated = ConcatDataset([source, source])
    samplers = [
        RepeatedBalancedDistributedSampler(
            repeated,
            num_replicas=2,
            rank=rank,
            batch_size=2,
            seed=17,
            window_shards=2,
        )
        for rank in range(2)
    ]

    rank_indices = [list(sampler) for sampler in samplers]
    rank_batches = [
        [indices[start : start + 2] for start in range(0, len(indices), 2)]
        for indices in rank_indices
    ]

    assert all(len(batch) == 2 for batches in rank_batches for batch in batches)
    assert len(rank_batches[0]) == len(rank_batches[1]) == 4
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert all(
        0 <= item < len(repeated) for indices in rank_indices for item in indices
    )


def test_repeated_balanced_sampler_fills_target_after_per_shard_drop(tmp_path):
    index = make_index(tmp_path, [[1, 2, 3, 4, 5]])
    source = IndexDataset(index)
    repeated = ConcatDataset([source, source])
    sampler = RepeatedBalancedDistributedSampler(
        repeated,
        num_replicas=1,
        rank=0,
        batch_size=2,
        seed=17,
        window_shards=1,
    )

    indices = list(sampler)

    assert len(sampler) == len(indices) == 10
    assert all(0 <= item < len(repeated) for item in indices)


def training_record(*, policy, workers, repetition, rate=1000.0):
    checksum = f"checksum-{policy}-{workers}-{repetition}"
    return {
        "rank": 0,
        "world_size": 1,
        "mode": "compiled",
        "backend": "hybrid",
        "derivative_mode": "positions",
        "precision_policy": {
            "requested": "highest",
            "effective": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "batch_size_per_rank": 128,
        "num_workers": workers,
        "warmup_steps": 4,
        "timed_steps": 256,
        "step_times_ms": [100.0] * 256,
        "data_wait_times_ms": [1.0] * 256,
        "step_period_times_ms": [128.0] * 256,
        "losses": [1.0] * 256,
        "batch_shapes": [{"structures": 128, "atoms": 512, "edges": 2048}] * 256,
        "is_compiled": True,
        "compilation_error": None,
        "dynamo_counters_after_warmup": {"stats": {"unique_graphs": 1}},
        "dynamo_counters_final": {"stats": {"unique_graphs": 1}},
        "parameter_checksum": checksum,
        "rank_checksums": [checksum],
        "lmdb_backend": "hdf5" if policy == "hdf5_random" else "omat_direct",
        "batch_sampling": (
            "shard_window_atom_balanced" if policy == "native_balanced" else "random"
        ),
        "source_revision": "abc123",
        "benchmark_sha256": "b" * 64,
        "adapter_sha256": "a" * 64,
        "repetition": repetition,
        "end_to_end_structures_per_second": rate,
    }


def write_training_run(root, policy, workers, repetition, *, rate=1000.0):
    path = root / f"r{repetition}-{policy}-w{workers}"
    path.mkdir(parents=True)
    import json

    (path / "rank-0.json").write_text(
        json.dumps(
            training_record(
                policy=policy,
                workers=workers,
                repetition=repetition,
                rate=rate,
            )
        ),
        encoding="utf-8",
    )
    (path / "gpu-info.csv").write_text("575.00\n", encoding="utf-8")
    (path / "dmon.csv").write_text(
        "#Date Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk\n"
        "#YYYYMMDD HH:MM:SS Idx W C C % % % % % % MHz MHz\n"
        "20260818 10:00:00 0 8 40 - 0 0 0 0 0 0 405 180\n"
        "20260818 10:00:01 0 300 50 - 60 20 0 0 0 0 14001 2500\n"
        "20260818 10:00:02 0 400 55 - 80 30 0 0 0 0 14001 2500\n",
        encoding="utf-8",
    )
    return path


def test_gpu_telemetry_reports_active_sm_and_power_fraction(tmp_path):
    path = write_training_run(tmp_path, "native_random", 8, 0)

    telemetry = parse_gpu_telemetry(path)

    assert telemetry["samples"] == 3
    assert telemetry["active_samples"] == 2
    assert telemetry["active_sm_mean_percent"] == 70.0
    assert telemetry["active_power_mean_w"] == 350.0
    assert telemetry["active_power_mean_limit_fraction"] == pytest.approx(350 / 575)


def test_training_run_verifier_rejects_missing_gpu_telemetry(tmp_path):
    path = write_training_run(tmp_path, "native_random", 8, 0)
    (path / "dmon.csv").unlink()

    with pytest.raises(ValueError, match="telemetry"):
        verify_training_run(
            path,
            expected_world_size=1,
            expected_source_revision="abc123",
        )


def test_training_run_verifier_fails_on_graph_growth(tmp_path):
    path = write_training_run(tmp_path, "native_balanced", 8, 0)
    import json

    record_path = path / "rank-0.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["dynamo_counters_final"] = {"stats": {"unique_graphs": 2}}
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="graph counters"):
        verify_training_run(
            path,
            expected_world_size=1,
            expected_source_revision="abc123",
        )


def test_training_run_verifier_rejects_cudnn_tf32(tmp_path):
    path = write_training_run(tmp_path, "native_balanced", 8, 0)
    import json

    record_path = path / "rank-0.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["precision_policy"]["cudnn_allow_tf32"] = True
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="cuDNN TF32"):
        verify_training_run(
            path,
            expected_world_size=1,
            expected_source_revision="abc123",
        )


def test_gpu_matrix_verifier_accepts_complete_balanced_repetitions(tmp_path):
    for repetition in range(3):
        for workers in (8, 16):
            write_training_run(tmp_path, "hdf5_random", workers, repetition)
            write_training_run(tmp_path, "native_random", workers, repetition)
            write_training_run(
                tmp_path,
                "native_balanced",
                workers,
                repetition,
                rate=990.0,
            )

    summary = verify_gpu_matrix(
        tmp_path,
        expected_source_revision="abc123",
        expected_world_size=1,
        expected_repetitions=3,
    )

    assert summary["runs"] == 18
    assert summary["source_revision"] == "abc123"
    assert summary["world_size"] == 1
    assert summary["repetitions"] == 3
    assert summary["batch_size_per_rank"] == 128
    assert summary["warmup_steps"] == 4
    assert summary["timed_steps"] == 256
    assert summary["matmul_precision"] == "highest"
    assert summary["acceptance"] == {
        "balanced_within_two_percent": True,
        "compiled_without_fallback": True,
        "matched_batch_shapes_and_first_loss": True,
        "stable_dynamo_graph": True,
    }
    assert "exact_hdf5_native_input_parity" not in summary["acceptance"]


def test_gpu_matrix_verifier_rejects_balanced_throughput_regression(tmp_path):
    for repetition in range(3):
        for workers in (8, 16):
            write_training_run(tmp_path, "hdf5_random", workers, repetition)
            write_training_run(tmp_path, "native_random", workers, repetition)
            write_training_run(
                tmp_path,
                "native_balanced",
                workers,
                repetition,
                rate=900.0,
            )

    with pytest.raises(ValueError, match="more than 2%"):
        verify_gpu_matrix(
            tmp_path,
            expected_source_revision="abc123",
            expected_world_size=1,
            expected_repetitions=3,
        )
