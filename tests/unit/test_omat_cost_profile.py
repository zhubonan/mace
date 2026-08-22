###########################################################################################
# Tests for exact cutoff-specific OMat edge-cost profiles
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import hashlib
import json
from pathlib import Path

import lmdb
import numpy as np
import pytest

import mace.data.omat_cost_profile as cost_profile_module
from mace.data.omat_cost_profile import (
    OMAT_COST_PROFILE_FORMAT,
    OMAT_COST_PROFILE_MANIFEST,
)
from mace.data.omat_lmdb_dataset import (
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
