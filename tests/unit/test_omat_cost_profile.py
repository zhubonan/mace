###########################################################################################
# Tests for exact cutoff-specific OMat edge-cost profiles
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import hashlib
import json
from pathlib import Path
import zlib

import lmdb
import numpy as np
import orjson
import pytest

import mace.data.omat_cost_profile as cost_profile_module
from mace import tools
from mace.data.omat_cost_profile import (
    OMAT_COST_PROFILE_FORMAT,
    OMAT_COST_PROFILE_MANIFEST,
    finalize_cost_profile,
    generate_cost_shard,
)
from mace.data.omat_lmdb_dataset import (
    OMatLMDBDataset,
    build_omat_shard_index,
    write_omat_manifest,
)


def _write_manifest_dataset(root: Path, natoms_by_shard) -> object:
    family = root / "family"
    family.mkdir(parents=True)
    for shard_index, natoms in enumerate(natoms_by_shard):
        shard = family / f"db_{shard_index}.aselmdb"
        if natoms:
            shard.touch()
        else:
            environment = lmdb.open(str(shard), subdir=False, map_size=1 << 20)
            environment.close()
        np.savez(
            family / f"metadata_db_{shard_index}.npz",
            natoms=np.asarray(natoms, dtype=np.int64),
        )
    discovered = build_omat_shard_index(root)
    write_omat_manifest(root, discovered, dataset_id="OMat24-test")
    return build_omat_shard_index(root)


def _profile_payload(index, cutoff: float, nedges_by_shard) -> dict:
    shards = []
    total_atoms = 0
    total_edges = 0
    for shard_index, (shard, nedges) in enumerate(
        zip(index.shards, nedges_by_shard, strict=True)
    ):
        natoms = index.load_natoms(shard_index)
        shard_atoms = int(natoms.sum())
        shard_edges = int(np.asarray(nedges).sum())
        total_atoms += shard_atoms
        total_edges += shard_edges
        shards.append(
            {
                "source_path": shard.relative_path.as_posix(),
                "cost_path": (
                    None if shard.length == 0 else f"shards/{shard_index:08d}.npz"
                ),
                "rows": shard.length,
                "total_atoms": shard_atoms,
                "total_edges": shard_edges,
            }
        )
    return {
        "format": OMAT_COST_PROFILE_FORMAT,
        "schema_version": 1,
        "profile_id": "OMat24-test-r5",
        "source": {
            "dataset_id": index.dataset_id,
            "manifest_sha256": index.manifest_sha256,
            "shards": [
                {
                    "path": shard.relative_path.as_posix(),
                    "rows": shard.length,
                }
                for shard in index.shards
            ],
        },
        "contract": {
            "cutoff_hex": float(cutoff).hex(),
            "directed_edges": True,
            "true_self_interaction": False,
            "pbc_cell_semantics": "mace-get-neighborhood-v1",
            "neighbour_list_implementation": (
                "mace.data.neighborhood.get_neighborhood:matscipy.neighbour_list"
            ),
        },
        "provenance": {"mace_version": "test", "matscipy_version": "test"},
        "total_rows": len(index),
        "total_shards": len(index.shards),
        "total_atoms": total_atoms,
        "total_edges": total_edges,
        "shards": shards,
    }


def _write_profile(root: Path, index, cutoff: float, nedges_by_shard) -> Path:
    profile_root = root / "profile"
    costs_root = profile_root / "shards"
    costs_root.mkdir(parents=True)
    for shard_index, (shard, nedges) in enumerate(
        zip(index.shards, nedges_by_shard, strict=True)
    ):
        if shard.length:
            np.savez(
                costs_root / f"{shard_index:08d}.npz",
                nedges=np.asarray(nedges),
            )
    payload = _profile_payload(index, cutoff, nedges_by_shard)
    manifest = profile_root / OMAT_COST_PROFILE_MANIFEST
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return profile_root


def _write_real_manifest_dataset(root: Path, rows: list[dict]):
    family = root / "family"
    family.mkdir(parents=True)
    shard = family / "db_0.aselmdb"
    environment = lmdb.open(str(shard), subdir=False, map_size=16 << 20)
    with environment.begin(write=True) as transaction:
        for row_index, row in enumerate(rows):
            transaction.put(
                str(row_index + 1).encode("ascii"),
                zlib.compress(orjson.dumps(row)),
            )
    environment.close()
    np.savez(
        family / "metadata_db_0.npz",
        natoms=np.asarray([len(row["numbers"]) for row in rows], dtype=np.int64),
    )
    discovered = build_omat_shard_index(root)
    write_omat_manifest(root, discovered, dataset_id="OMat24-real-test")
    return build_omat_shard_index(root)


def _official_structure(positions: list[list[float]]) -> dict:
    return {
        "numbers": [1] * len(positions),
        "positions": positions,
        "cell": (np.eye(3) * 20.0).tolist(),
        "pbc": [False, False, False],
        "energy": 0.0,
        "forces": np.zeros((len(positions), 3)).tolist(),
        "stress": [0.0] * 6,
    }


def test_cost_profile_validates_identity_contract_and_loads_lazily(
    tmp_path, monkeypatch
):
    index = _write_manifest_dataset(tmp_path / "dataset", [[2, 4], []])
    profile_root = _write_profile(tmp_path, index, 5.0, [[6, 12], []])
    original_load = cost_profile_module.np.load
    monkeypatch.setattr(
        cost_profile_module.np,
        "load",
        lambda *args, **kwargs: pytest.fail("startup decompressed a cost array"),
    )

    associated = index.with_cost_profile(profile_root, cutoff=5.0)

    assert associated.cost_profile.cutoff.hex() == float(5.0).hex()
    assert associated.cost_profile.profile_id == "OMat24-test-r5"
    expected_hash = hashlib.sha256(
        (profile_root / OMAT_COST_PROFILE_MANIFEST).read_bytes()
    ).hexdigest()
    assert associated.cost_profile.manifest_sha256 == expected_hash
    monkeypatch.setattr(cost_profile_module.np, "load", original_load)
    assert associated.load_nedges(0).tolist() == [6, 12]
    assert associated.load_nedges(1).tolist() == []
    assert associated.load_nedges(1).dtype == np.int64


def test_cost_profile_can_be_relocated_without_changing_its_identity(tmp_path):
    index = _write_manifest_dataset(tmp_path / "dataset", [[2, 4], []])
    profile_root = _write_profile(tmp_path / "original", index, 5.0, [[6, 12], []])
    original = index.with_cost_profile(profile_root, cutoff=5.0).cost_profile
    relocated_root = tmp_path / "relocated" / "edge-profile"
    relocated_root.parent.mkdir()
    profile_root.rename(relocated_root)

    relocated_index = index.with_cost_profile(relocated_root, cutoff=5.0)
    relocated = relocated_index.cost_profile

    assert relocated.profile_id == original.profile_id
    assert relocated.manifest_sha256 == original.manifest_sha256
    assert relocated.root == relocated_root.resolve()
    assert relocated.shards[0].cost_path.is_relative_to(relocated.root)
    np.testing.assert_array_equal(relocated_index.load_nedges(0), [6, 12])
    np.testing.assert_array_equal(relocated_index.load_nedges(1), [])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("source_hash", "source manifest"),
        ("dataset_id", "dataset identity"),
        ("cutoff", "cutoff"),
        ("directed", "directed-edge"),
        ("self_interaction", "self-interaction"),
        ("shard_path", "shard ordering"),
        ("rows", "row count"),
    ],
)
def test_cost_profile_rejects_incompatible_manifest(tmp_path, mutation, match):
    index = _write_manifest_dataset(tmp_path / "dataset", [[2, 4]])
    profile_root = _write_profile(tmp_path, index, 5.0, [[6, 12]])
    manifest = profile_root / OMAT_COST_PROFILE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "source_hash":
        payload["source"]["manifest_sha256"] = "0" * 64
    elif mutation == "dataset_id":
        payload["source"]["dataset_id"] = "different"
    elif mutation == "cutoff":
        payload["contract"]["cutoff_hex"] = float(4.5).hex()
    elif mutation == "directed":
        payload["contract"]["directed_edges"] = False
    elif mutation == "self_interaction":
        payload["contract"]["true_self_interaction"] = True
    elif mutation == "shard_path":
        payload["source"]["shards"][0]["path"] = "family/db_other.aselmdb"
    elif mutation == "rows":
        payload["shards"][0]["rows"] = 3
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        index.with_cost_profile(profile_root, cutoff=5.0)


@pytest.mark.parametrize(
    "values,match",
    [
        (np.asarray([[2, 4]], dtype=np.int64), "one-dimensional"),
        (np.asarray([2.0, 4.0]), "integer dtype"),
        (np.asarray([2], dtype=np.int64), "length"),
    ],
)
def test_cost_profile_startup_rejects_corrupt_array_header(tmp_path, values, match):
    index = _write_manifest_dataset(tmp_path / "dataset", [[2, 4]])
    profile_root = _write_profile(tmp_path, index, 5.0, [[6, 12]])
    np.savez(profile_root / "shards" / "00000000.npz", nedges=values)

    with pytest.raises(ValueError, match=match):
        index.with_cost_profile(profile_root, cutoff=5.0)


@pytest.mark.parametrize(
    "values,match",
    [
        (np.asarray([2, -1], dtype=np.int64), "non-negative"),
        (np.asarray([2, 4], dtype=np.int64), "total_edges"),
    ],
)
def test_load_nedges_rejects_corrupt_array_values(tmp_path, values, match):
    index = _write_manifest_dataset(tmp_path / "dataset", [[2, 4]])
    profile_root = _write_profile(tmp_path, index, 5.0, [[6, 12]])
    np.savez(profile_root / "shards" / "00000000.npz", nedges=values)
    associated = index.with_cost_profile(profile_root, cutoff=5.0)

    with pytest.raises(ValueError, match=match):
        associated.load_nedges(0)


def test_cost_profile_requires_a_source_manifest(tmp_path):
    family = tmp_path / "dataset" / "family"
    family.mkdir(parents=True)
    (family / "db_0.aselmdb").touch()
    np.savez(family / "metadata_db_0.npz", natoms=np.asarray([2], dtype=np.int64))
    index = build_omat_shard_index(tmp_path / "dataset")

    with pytest.raises(ValueError, match="source OMat manifest"):
        index.with_cost_profile(tmp_path / "profile", cutoff=5.0)


def test_cost_profile_rejects_missing_or_unsafe_cost_path(tmp_path):
    index = _write_manifest_dataset(tmp_path / "dataset", [[2]])
    profile_root = _write_profile(tmp_path, index, 5.0, [[6]])
    manifest = profile_root / OMAT_COST_PROFILE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["shards"][0]["cost_path"] = "../outside.npz"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="remain below"):
        index.with_cost_profile(profile_root, cutoff=5.0)

    payload["shards"][0]["cost_path"] = "shards/missing.npz"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        index.with_cost_profile(profile_root, cutoff=5.0)


def test_generate_and_finalize_cost_profile_matches_online_graphs(tmp_path):
    dataset_root = tmp_path / "dataset"
    index = _write_real_manifest_dataset(
        dataset_root,
        [
            _official_structure([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            _official_structure(
                [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
        ],
    )
    profile_root = tmp_path / "costs"
    shard_result = generate_cost_shard(index, profile_root, cutoff=1.5, shard_index=0)

    assert shard_result.is_file()
    assert not (profile_root / OMAT_COST_PROFILE_MANIFEST).exists()
    with np.load(shard_result, allow_pickle=False) as result:
        generated = np.asarray(result["nedges"])
    dataset = OMatLMDBDataset(
        dataset_root,
        r_max=1.5,
        z_table=tools.AtomicNumberTable([1]),
        heads=["Default"],
    )
    expected = np.asarray(
        [dataset[row].edge_index.shape[1] for row in range(len(dataset))]
    )
    np.testing.assert_array_equal(generated, expected)
    dataset.close()

    manifest = finalize_cost_profile(index, profile_root, cutoff=1.5)
    associated = index.with_cost_profile(profile_root, cutoff=1.5)

    assert manifest == profile_root / OMAT_COST_PROFILE_MANIFEST
    np.testing.assert_array_equal(associated.load_nedges(0), expected)
    assert associated.cost_profile.shards[0].total_atoms == 5


def test_generate_cost_shard_resumes_only_a_valid_complete_result(tmp_path):
    index = _write_real_manifest_dataset(
        tmp_path / "dataset",
        [_official_structure([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])],
    )
    profile_root = tmp_path / "costs"
    result = generate_cost_shard(index, profile_root, cutoff=1.5, shard_index=0)
    with np.load(result, allow_pickle=False) as payload:
        original_nedges = np.asarray(payload["nedges"]).copy()
    original_mtime = result.stat().st_mtime_ns

    resumed = generate_cost_shard(index, profile_root, cutoff=1.5, shard_index=0)

    assert resumed == result
    assert result.stat().st_mtime_ns == original_mtime
    result.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="existing shard result"):
        generate_cost_shard(index, profile_root, cutoff=1.5, shard_index=0)
    regenerated = generate_cost_shard(
        index, profile_root, cutoff=1.5, shard_index=0, force=True
    )
    with np.load(regenerated, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["nedges"], original_nedges)
    assert not list(result.parent.glob(".*.tmp"))


def test_finalize_rejects_partial_profile_and_records_empty_shards(tmp_path):
    index = _write_manifest_dataset(tmp_path / "dataset", [[2], []])
    profile_root = tmp_path / "costs"
    with pytest.raises(ValueError, match="incomplete"):
        finalize_cost_profile(index, profile_root, cutoff=1.5)

    profile_root.mkdir()
    result_root = profile_root / "shards"
    result_root.mkdir()
    cost_profile_module._write_cost_shard_result(
        index=index,
        profile_root=profile_root,
        cutoff=1.5,
        shard_index=0,
        nedges=np.asarray([4], dtype=np.int64),
    )
    manifest = finalize_cost_profile(index, profile_root, cutoff=1.5)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["shards"][1] == {
        "source_path": "family/db_1.aselmdb",
        "cost_path": None,
        "rows": 0,
        "total_atoms": 0,
        "total_edges": 0,
    }
