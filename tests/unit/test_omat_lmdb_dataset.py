import json
import multiprocessing
import pickle
import zlib
from pathlib import Path

import lmdb
import numpy as np
import orjson
import pytest
import torch

from mace import data, tools
from mace.data.omat_lmdb_dataset import (
    OMAT_MANIFEST_FORMAT,
    OMAT_MANIFEST_NAME,
    OMatLMDBDataset,
    build_omat_shard_index,
    decode_omat_row,
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


def official_row(
    row: int,
    *,
    key_values=None,
    missing: str | None = None,
) -> dict:
    payload = {
        "numbers": [1],
        "positions": [[float(row), 0.0, 0.0]],
        "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "pbc": [False, False, False],
        "energy": float(row) + 0.25,
        "forces": [[1.0, 2.0, 3.0]],
        "stress": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "key_value_pairs": key_values or {},
    }
    if missing is not None:
        del payload[missing]
    return payload


def write_official_shard(
    root: Path,
    suffix: str,
    *,
    rows: int,
    metadata_rows: int | None = None,
    missing: str | None = None,
    key_values=None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard = root / f"db_{suffix}.aselmdb"
    environment = lmdb.open(str(shard), subdir=False, map_size=16 << 20)
    with environment.begin(write=True) as transaction:
        for row in range(rows):
            value = zlib.compress(
                orjson.dumps(
                    official_row(
                        row,
                        key_values=key_values,
                        missing=missing,
                    )
                )
            )
            transaction.put(str(row + 1).encode("ascii"), value)
        if rows:
            transaction.put(b"nextid", zlib.compress(orjson.dumps(rows + 1)))
    environment.close()
    count = rows if metadata_rows is None else metadata_rows
    np.savez(root / f"metadata_db_{suffix}.npz", natoms=np.ones(count, dtype=np.int64))
    return shard


def make_dataset(root: Path, *, max_open_shards: int = 2, transform=None):
    return OMatLMDBDataset(
        root,
        r_max=1.5,
        z_table=tools.AtomicNumberTable([1]),
        heads=["Default"],
        head="Default",
        max_open_shards=max_open_shards,
        transform=transform,
    )


def inspect_dataset_in_worker(dataset, queue):
    queue.put(
        {
            "active_before": dataset.active_environment_count,
            "opens_before": dataset.open_environment_count,
            "energy": dataset[0].energy.item(),
            "active_after": dataset.active_environment_count,
            "opens_after": dataset.open_environment_count,
        }
    )
    dataset.close()


def mark_transformed(item):
    item.transformed = True
    return item


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
        (np.asarray([4.5]), "integer dtype"),
    ],
)
def test_load_natoms_validates_shape_and_values(tmp_path, values, match):
    write_metadata_shard(tmp_path / "family", "0", [4, 8], metadata_values=values)
    index = build_omat_shard_index(tmp_path)

    with pytest.raises(ValueError, match=match):
        index.load_natoms(0)


def test_load_natoms_normalizes_empty_float_metadata(tmp_path):
    write_metadata_shard(
        tmp_path / "family",
        "0",
        [],
        metadata_values=np.asarray([], dtype=np.float64),
    )
    index = build_omat_shard_index(tmp_path)

    natoms = index.load_natoms(0)

    assert natoms.dtype == np.int64
    assert natoms.shape == (0,)


def test_direct_dataset_decodes_configuration_and_builds_graph(tmp_path):
    write_official_shard(tmp_path / "family", "0", rows=1)
    torch.set_default_dtype(torch.float32)
    dataset = make_dataset(tmp_path)

    configuration = dataset.read_configuration(0)
    item = dataset[0]

    assert configuration.atomic_numbers.tolist() == [1]
    assert configuration.positions.tolist() == [[0.0, 0.0, 0.0]]
    assert set(configuration.properties) == {"energy", "forces", "stress"}
    assert configuration.properties["energy"] == pytest.approx(0.25)
    assert configuration.property_weights == {
        "energy": 1.0,
        "forces": 1.0,
        "stress": 1.0,
    }
    assert torch.equal(item.atomic_numbers, torch.tensor([1]))
    assert torch.equal(item.positions, torch.tensor([[0.0, 0.0, 0.0]]))
    assert item.edge_index.shape == (2, 0)
    assert torch.equal(item.energy, torch.tensor(0.25))
    assert torch.equal(item.forces, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(
        item.stress,
        torch.tensor([[[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]]),
    )
    assert item.head.item() == 0
    dataset.close()


def test_direct_dataset_matches_independent_configuration(tmp_path):
    write_official_shard(tmp_path / "family", "0", rows=1)
    torch.set_default_dtype(torch.float32)
    dataset = make_dataset(tmp_path)
    expected = data.AtomicData.from_config(
        data.Configuration(
            atomic_numbers=np.asarray([1], dtype=np.int64),
            positions=np.asarray([[0.0, 0.0, 0.0]]),
            cell=np.eye(3) * 10.0,
            pbc=(False, False, False),
            properties={
                "energy": 0.25,
                "forces": np.asarray([[1.0, 2.0, 3.0]]),
                "stress": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            },
            property_weights={"energy": 1.0, "forces": 1.0, "stress": 1.0},
            weight=1.0,
            config_type="Default",
            head="Default",
        ),
        z_table=tools.AtomicNumberTable([1]),
        cutoff=1.5,
        heads=["Default"],
    )

    actual = dataset[0]

    expected_keys = set(expected.keys)
    actual_keys = set(actual.keys)
    assert actual_keys == expected_keys
    for key in expected_keys:
        expected_value = expected[key]
        actual_value = actual[key]
        if torch.is_tensor(expected_value):
            assert torch.equal(actual_value, expected_value), key
        else:
            assert actual_value == expected_value
    dataset.close()


def test_direct_dataset_applies_official_weights_head_and_transform(tmp_path):
    weights = {
        "config_weight": 0.5,
        "config_energy_weight": 2.0,
        "config_forces_weight": 3.0,
        "config_stress_weight": 4.0,
        "config_type": "rattled",
    }
    write_official_shard(tmp_path / "family", "0", rows=1, key_values=weights)
    dataset = make_dataset(tmp_path, transform=mark_transformed)

    configuration = dataset.read_configuration(0)
    item = dataset[0]

    assert configuration.weight == 0.5
    assert configuration.config_type == "rattled"
    assert configuration.property_weights == {
        "energy": 2.0,
        "forces": 3.0,
        "stress": 4.0,
    }
    assert item.weight.item() == 0.5
    assert item.energy_weight.item() == 2.0
    assert item.forces_weight.item() == 3.0
    assert item.stress_weight.item() == 4.0
    assert item.transformed is True
    dataset.close()


@pytest.mark.parametrize(
    "missing", ["numbers", "positions", "cell", "pbc", "energy", "forces", "stress"]
)
def test_decode_omat_row_requires_active_efs_fields(missing):
    value = zlib.compress(orjson.dumps(official_row(0, missing=missing)))

    with pytest.raises(KeyError, match=missing):
        decode_omat_row(value, head="Default")


def test_direct_dataset_reports_missing_numeric_row_with_context(tmp_path):
    write_official_shard(tmp_path / "family", "7", rows=1, metadata_rows=2)
    dataset = make_dataset(tmp_path)

    with pytest.raises(KeyError, match=r"family.*db_7.*local row 1.*key 2"):
        dataset.read_configuration(1)
    dataset.close()


def test_direct_dataset_wraps_decode_failure_with_row_context(tmp_path):
    write_official_shard(tmp_path / "family", "7", rows=1, missing="forces")
    dataset = make_dataset(tmp_path)

    with pytest.raises(
        RuntimeError,
        match=r"family.*db_7.*local row 0.*key 1.*forces",
    ):
        dataset[0]
    dataset.close()


def test_dataset_opens_worker_local_handles_lazily_and_bounds_lru(tmp_path):
    for shard in range(3):
        write_official_shard(tmp_path / "family", str(shard), rows=1)
    dataset = make_dataset(tmp_path, max_open_shards=2)

    assert dataset.open_environment_count == 0
    assert dataset.active_environment_count == 0
    pickle.dumps(dataset)
    assert dataset.open_environment_count == 0
    dataset[0]
    dataset[0]
    assert dataset.open_environment_count == 1
    dataset[1]
    assert dataset.open_environment_count == 2
    assert dataset.active_environment_count == 2
    dataset[2]
    assert dataset.open_environment_count == 3
    assert dataset.active_environment_count == 2

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.open_environment_count == 0
    assert restored.active_environment_count == 0
    assert restored[0].energy.item() == pytest.approx(0.25)
    restored.close()
    restored.close()
    dataset.close()
    dataset.close()


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_dataset_resets_handle_ownership_in_real_worker(tmp_path, start_method):
    write_official_shard(tmp_path / "family", "0", rows=1)
    dataset = make_dataset(tmp_path)
    dataset[0]
    assert dataset.active_environment_count == 1
    context = multiprocessing.get_context(start_method)
    queue = context.Queue()
    process = context.Process(target=inspect_dataset_in_worker, args=(dataset, queue))
    process.start()
    result = queue.get(timeout=20)
    process.join(timeout=20)

    assert process.exitcode == 0
    assert result == {
        "active_before": 0,
        "opens_before": 0,
        "energy": pytest.approx(0.25),
        "active_after": 1,
        "opens_after": 1,
    }
    dataset.close()
