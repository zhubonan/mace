import json
from pathlib import Path

import lmdb
import numpy as np
import pytest

from mace.data.omat_lmdb_dataset import (
    OMAT_MANIFEST_FORMAT,
    OMAT_MANIFEST_NAME,
    build_omat_shard_index,
    write_omat_manifest,
)


def write_metadata_shard(
    root: Path,
    suffix: str,
    natoms,
    *,
    metadata_values=None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard = root / f"db_{suffix}.aselmdb"
    shard.touch()
    values = np.asarray(natoms if metadata_values is None else metadata_values)
    np.savez(root / f"metadata_db_{suffix}.npz", natoms=values)
    return shard


def write_lmdb_without_metadata(root: Path, suffix: str, *, nonempty: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard = root / f"db_{suffix}.aselmdb"
    environment = lmdb.open(str(shard), subdir=False, map_size=1 << 20)
    if nonempty:
        with environment.begin(write=True) as transaction:
            transaction.put(b"1", b"row")
    environment.close()
    return shard


def test_index_discovers_multiple_families_in_relative_lexical_order(tmp_path):
    write_metadata_shard(tmp_path / "family_b", "0", [8, 10])
    write_metadata_shard(tmp_path / "family_a", "1", [4])

    index = build_omat_shard_index(tmp_path)

    assert [shard.relative_path.as_posix() for shard in index.shards] == [
        "family_a/db_1.aselmdb",
        "family_b/db_0.aselmdb",
    ]
    assert index.dataset_id == tmp_path.name
    assert index.roots == (tmp_path.resolve(),)
    assert len(index) == 3
    assert index.locate(0) == (0, 0)
    assert index.locate(1) == (1, 0)
    assert index.locate(2) == (1, 1)
    assert index.locate(-1) == (1, 1)
    assert index.load_natoms(1).tolist() == [8, 10]


def test_index_accepts_one_family_root_and_explicit_family_roots(tmp_path):
    family_a = tmp_path / "family_a"
    family_b = tmp_path / "family_b"
    write_metadata_shard(family_a, "0", [4])
    write_metadata_shard(family_b, "0", [8])

    one_family = build_omat_shard_index(family_a)
    explicit = build_omat_shard_index([family_b, family_a])

    assert one_family.shards[0].family == "family_a"
    assert one_family.shards[0].relative_path == Path("family_a/db_0.aselmdb")
    assert [shard.family for shard in explicit.shards] == ["family_a", "family_b"]


def test_index_rejects_duplicate_roots_and_duplicate_family_names(tmp_path):
    family = tmp_path / "family"
    write_metadata_shard(family, "0", [4])
    alias_parent = tmp_path / "other"
    alias_parent.mkdir()
    alias = alias_parent / "family"
    write_metadata_shard(alias, "0", [8])

    with pytest.raises(ValueError, match="Duplicate root"):
        build_omat_shard_index([family, family])
    with pytest.raises(ValueError, match="Duplicate family"):
        build_omat_shard_index([family, alias])


def test_index_locate_rejects_out_of_range_indices(tmp_path):
    write_metadata_shard(tmp_path / "family", "0", [4])
    index = build_omat_shard_index(tmp_path)

    with pytest.raises(IndexError):
        index.locate(1)
    with pytest.raises(IndexError):
        index.locate(-2)


def test_manifest_round_trip_preserves_dataset_identity_and_counts(tmp_path):
    write_metadata_shard(tmp_path / "family", "0", [4, 8])
    discovered = build_omat_shard_index(tmp_path)

    manifest = write_omat_manifest(tmp_path, discovered, dataset_id="OMat24")
    loaded = build_omat_shard_index(tmp_path, manifest=manifest)

    assert manifest.name == OMAT_MANIFEST_NAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload == {
        "format": OMAT_MANIFEST_FORMAT,
        "dataset_id": "OMat24",
        "total_rows": 2,
        "shards": [
            {
                "family": "family",
                "path": "family/db_0.aselmdb",
                "metadata": "family/metadata_db_0.npz",
                "rows": 2,
            }
        ],
    }
    assert loaded.dataset_id == "OMat24"
    assert loaded.shards == discovered.shards
    assert len(loaded) == 2


@pytest.mark.parametrize("unsafe_path", ["/tmp/db.aselmdb", "../db.aselmdb"])
def test_manifest_rejects_unsafe_shard_paths(tmp_path, unsafe_path):
    manifest = tmp_path / OMAT_MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "format": OMAT_MANIFEST_FORMAT,
                "dataset_id": "OMat24",
                "total_rows": 0,
                "shards": [
                    {
                        "family": "family",
                        "path": unsafe_path,
                        "metadata": None,
                        "rows": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative"):
        build_omat_shard_index(tmp_path, manifest=manifest)


def test_manifest_rejects_missing_shard_and_inconsistent_total(tmp_path):
    manifest = tmp_path / OMAT_MANIFEST_NAME
    payload = {
        "format": OMAT_MANIFEST_FORMAT,
        "dataset_id": "OMat24",
        "total_rows": 1,
        "shards": [
            {
                "family": "family",
                "path": "family/db_0.aselmdb",
                "metadata": "family/metadata_db_0.npz",
                "rows": 1,
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="db_0"):
        build_omat_shard_index(tmp_path, manifest=manifest)

    write_metadata_shard(tmp_path / "family", "0", [4])
    payload["total_rows"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="total_rows"):
        build_omat_shard_index(tmp_path, manifest=manifest)


def test_missing_metadata_is_allowed_only_for_an_empty_lmdb(tmp_path):
    empty = write_lmdb_without_metadata(tmp_path / "family", "0", nonempty=False)
    index = build_omat_shard_index(tmp_path)

    assert len(index) == 0
    assert index.shards[0].path == empty.resolve()
    assert index.shards[0].metadata_path is None
    assert index.load_natoms(0).shape == (0,)

    write_lmdb_without_metadata(tmp_path / "family", "1", nonempty=True)
    with pytest.raises(ValueError, match="Missing metadata for non-empty shard"):
        build_omat_shard_index(tmp_path)


def test_metadata_requires_natoms(tmp_path):
    family = tmp_path / "family"
    family.mkdir()
    (family / "db_0.aselmdb").touch()
    np.savez(family / "metadata_db_0.npz", energy=np.asarray([1.0]))

    with pytest.raises(ValueError, match="Missing natoms"):
        build_omat_shard_index(tmp_path)


@pytest.mark.parametrize(
    "values,match",
    [
        (np.asarray([[4, 8]]), "one-dimensional"),
        (np.asarray([4, 0]), "strictly positive"),
    ],
)
def test_load_natoms_validates_shape_and_values(tmp_path, values, match):
    write_metadata_shard(
        tmp_path / "family", "0", [4, 8], metadata_values=values
    )
    index = build_omat_shard_index(tmp_path)

    with pytest.raises(ValueError, match=match):
        index.load_natoms(0)
