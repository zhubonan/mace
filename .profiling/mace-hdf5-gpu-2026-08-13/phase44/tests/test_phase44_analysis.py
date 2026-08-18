###########################################################################################
# Tests for Phase-44 complete-corpus and CPU qualification analyzers
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_balanced_sampler import audit_sampler  # noqa: E402
from benchmark_native_omat_lmdb import (  # noqa: E402
    benchmark_order,
    validate_benchmark_records,
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
