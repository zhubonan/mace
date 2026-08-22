"""Exact cutoff-specific edge-cost profiles for native OMat LMDB data."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np
from numpy.lib import format as npy_format

from mace.data.neighborhood import get_neighborhood
from mace.data.omat_lmdb_dataset import OMatShardIndex, decode_omat_row

OMAT_COST_PROFILE_FORMAT = "mace-omat-edge-cost-profile-v1"
OMAT_COST_PROFILE_MANIFEST = "omat-edge-cost-profile.json"
OMAT_COST_PROFILE_SCHEMA_VERSION = 1
OMAT_COST_SHARD_FORMAT = "mace-omat-edge-cost-shard-v1"
OMAT_PBC_CELL_SEMANTICS = "mace-get-neighborhood-v1"
OMAT_NEIGHBOUR_IMPLEMENTATION = (
    "mace.data.neighborhood.get_neighborhood:matscipy.neighbour_list"
)


@dataclass(frozen=True)
class OMatCostShard:
    """One edge-cost array aligned with an official source shard."""

    relative_shard_path: Path
    cost_path: Path | None
    rows: int
    total_atoms: int
    total_edges: int


@dataclass(frozen=True)
class OMatCostProfile:
    """Validated immutable edge-cost profile with lazy array loading."""

    root: Path
    manifest_path: Path
    profile_id: str
    manifest_sha256: str
    cutoff: float
    shards: tuple[OMatCostShard, ...]

    def load_nedges(self, shard_index: int) -> np.ndarray:
        shard = self.shards[shard_index]
        if shard.cost_path is None:
            return np.empty(0, dtype=np.int64)
        with np.load(shard.cost_path, allow_pickle=False) as payload:
            if "nedges" not in payload:
                raise ValueError(f"Missing nedges in {shard.cost_path}")
            raw_nedges = payload["nedges"]
        _validate_cost_array(raw_nedges, shard)
        return np.asarray(raw_nedges, dtype=np.int64)


def _require_nonnegative_int(value, *, field: str, manifest: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Invalid {field} in {manifest}")
    return value


def _safe_profile_path(root: Path, value: str, *, field: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Profile {field} path must remain below its root: {value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Profile {field} path must remain below its root: {value}"
        ) from exc
    return resolved


def _cost_array_header(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(path) as archive:
        try:
            stream = archive.open("nedges.npy")
        except KeyError as exc:
            raise ValueError(f"Missing nedges in {path}") from exc
        with stream:
            version = npy_format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = npy_format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _, dtype = npy_format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported nedges array format {version}: {path}")
    return shape, dtype


def _validate_cost_header(path: Path, rows: int) -> None:
    shape, dtype = _cost_array_header(path)
    if len(shape) != 1:
        raise ValueError(f"nedges array must be one-dimensional: {path}")
    if int(shape[0]) != rows:
        raise ValueError(
            f"nedges length {shape[0]} does not match shard length {rows}: {path}"
        )
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"nedges array must have integer dtype: {path}")


def _validate_cost_array(raw_nedges: np.ndarray, shard: OMatCostShard) -> None:
    path = shard.cost_path
    if raw_nedges.ndim != 1:
        raise ValueError(f"nedges array must be one-dimensional: {path}")
    if not np.issubdtype(raw_nedges.dtype, np.integer):
        raise ValueError(f"nedges array must have integer dtype: {path}")
    if len(raw_nedges) != shard.rows:
        raise ValueError(
            f"nedges length {len(raw_nedges)} does not match shard length "
            f"{shard.rows}: {path}"
        )
    nedges = np.asarray(raw_nedges, dtype=np.int64)
    if np.any(nedges < 0):
        raise ValueError(f"nedges array must be non-negative: {path}")
    actual_total = int(nedges.sum())
    if actual_total != shard.total_edges:
        raise ValueError(
            f"nedges total_edges {actual_total} does not match manifest total_edges "
            f"{shard.total_edges}: {path}"
        )


def _validate_source_contract(
    payload: dict, index: OMatShardIndex, manifest: Path
) -> None:
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Invalid source dataset contract in {manifest}")
    if source.get("dataset_id") != index.dataset_id:
        raise ValueError(
            f"Cost profile dataset identity does not match source: {manifest}"
        )
    if source.get("manifest_sha256") != index.manifest_sha256:
        raise ValueError(f"Cost profile source manifest SHA-256 mismatch: {manifest}")
    source_shards = source.get("shards")
    if not isinstance(source_shards, list) or len(source_shards) != len(index.shards):
        raise ValueError(
            f"Cost profile source shard ordering does not match: {manifest}"
        )
    expected = [
        {"path": shard.relative_path.as_posix(), "rows": shard.length}
        for shard in index.shards
    ]
    if source_shards != expected:
        if [
            entry.get("path") for entry in source_shards if isinstance(entry, dict)
        ] != [entry["path"] for entry in expected]:
            raise ValueError(
                f"Cost profile source shard ordering does not match: {manifest}"
            )
        raise ValueError(
            f"Cost profile source shard row count does not match: {manifest}"
        )


def _validate_neighbour_contract(payload: dict, cutoff: float, manifest: Path) -> None:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Invalid neighbour contract in {manifest}")
    if contract.get("cutoff_hex") != float(cutoff).hex():
        raise ValueError(
            f"Cost profile cutoff does not match training cutoff: {manifest}"
        )
    if contract.get("directed_edges") is not True:
        raise ValueError(f"Cost profile must use directed-edge semantics: {manifest}")
    if contract.get("true_self_interaction") is not False:
        raise ValueError(
            f"Cost profile self-interaction semantics do not match: {manifest}"
        )
    if contract.get("pbc_cell_semantics") != OMAT_PBC_CELL_SEMANTICS:
        raise ValueError(f"Cost profile PBC/cell semantics do not match: {manifest}")
    if contract.get("neighbour_list_implementation") != OMAT_NEIGHBOUR_IMPLEMENTATION:
        raise ValueError(
            f"Cost profile neighbour-list implementation does not match: {manifest}"
        )


def load_omat_cost_profile(
    root: Path | str, *, index: OMatShardIndex, cutoff: float
) -> OMatCostProfile:
    """Validate a complete profile without decompressing its edge arrays."""

    if index.manifest_path is None or index.manifest_sha256 is None:
        raise ValueError("Edge-cost profiles require a source OMat manifest")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Edge-cost profile cutoff must be finite and positive")
    profile_root = Path(root).resolve()
    if not profile_root.is_dir():
        raise FileNotFoundError(profile_root)
    manifest = profile_root / OMAT_COST_PROFILE_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifest_bytes = manifest.read_bytes()
    payload = json.loads(manifest_bytes)
    if payload.get("format") != OMAT_COST_PROFILE_FORMAT:
        raise ValueError(f"Unsupported OMat cost-profile format in {manifest}")
    if payload.get("schema_version") != OMAT_COST_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported OMat cost-profile schema in {manifest}")
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError(f"Invalid profile_id in {manifest}")

    _validate_source_contract(payload, index, manifest)
    _validate_neighbour_contract(payload, cutoff, manifest)

    entries = payload.get("shards")
    if not isinstance(entries, list) or len(entries) != len(index.shards):
        raise ValueError(f"Cost profile shard count does not match source: {manifest}")
    shards = []
    seen_paths = set()
    for shard_index, (entry, source_shard) in enumerate(
        zip(entries, index.shards, strict=True)
    ):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid cost-profile shard entry in {manifest}")
        if entry.get("source_path") != source_shard.relative_path.as_posix():
            raise ValueError(f"Cost profile shard ordering does not match: {manifest}")
        rows = _require_nonnegative_int(
            entry.get("rows"), field="shard row count", manifest=manifest
        )
        if rows != source_shard.length:
            raise ValueError(
                f"Cost profile shard row count does not match source: {manifest}"
            )
        total_atoms = _require_nonnegative_int(
            entry.get("total_atoms"), field="total_atoms", manifest=manifest
        )
        total_edges = _require_nonnegative_int(
            entry.get("total_edges"), field="total_edges", manifest=manifest
        )
        path_value = entry.get("cost_path")
        if rows == 0:
            if path_value is not None or total_atoms != 0 or total_edges != 0:
                raise ValueError(
                    f"Empty cost-profile shard must have zero costs: {manifest}"
                )
            cost_path = None
        else:
            if not isinstance(path_value, str):
                raise ValueError(f"Missing cost path for non-empty shard: {manifest}")
            cost_path = _safe_profile_path(profile_root, path_value, field="cost")
            if cost_path in seen_paths:
                raise ValueError(f"Duplicate cost path in {manifest}: {cost_path}")
            if not cost_path.is_file():
                raise FileNotFoundError(cost_path)
            seen_paths.add(cost_path)
            _validate_cost_header(cost_path, rows)
        shards.append(
            OMatCostShard(
                relative_shard_path=source_shard.relative_path,
                cost_path=cost_path,
                rows=rows,
                total_atoms=total_atoms,
                total_edges=total_edges,
            )
        )

    declared_rows = _require_nonnegative_int(
        payload.get("total_rows"), field="total_rows", manifest=manifest
    )
    declared_shards = _require_nonnegative_int(
        payload.get("total_shards"), field="total_shards", manifest=manifest
    )
    declared_atoms = _require_nonnegative_int(
        payload.get("total_atoms"), field="total_atoms", manifest=manifest
    )
    declared_edges = _require_nonnegative_int(
        payload.get("total_edges"), field="total_edges", manifest=manifest
    )
    if declared_rows != sum(shard.rows for shard in shards):
        raise ValueError(
            f"Cost profile total_rows does not match shard sum: {manifest}"
        )
    if declared_shards != len(shards):
        raise ValueError(f"Cost profile total_shards does not match: {manifest}")
    if declared_atoms != sum(shard.total_atoms for shard in shards):
        raise ValueError(
            f"Cost profile total_atoms does not match shard sum: {manifest}"
        )
    if declared_edges != sum(shard.total_edges for shard in shards):
        raise ValueError(
            f"Cost profile total_edges does not match shard sum: {manifest}"
        )

    return OMatCostProfile(
        root=profile_root,
        manifest_path=manifest,
        profile_id=profile_id,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        cutoff=float(cutoff),
        shards=tuple(shards),
    )


def _require_generation_source(index: OMatShardIndex, cutoff: float) -> None:
    if index.manifest_path is None or index.manifest_sha256 is None:
        raise ValueError("Edge-cost generation requires a source OMat manifest")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Edge-cost profile cutoff must be finite and positive")


def _cost_shard_result_path(profile_root: Path, shard_index: int) -> Path:
    return profile_root / "shards" / f"{shard_index:08d}.npz"


def _scalar(payload, name: str, path: Path):
    if name not in payload:
        raise ValueError(f"Missing {name} in edge-cost shard result {path}")
    value = payload[name]
    if value.ndim != 0:
        raise ValueError(f"Invalid {name} in edge-cost shard result {path}")
    return value.item()


def _load_cost_shard_result(
    *,
    index: OMatShardIndex,
    profile_root: Path,
    cutoff: float,
    shard_index: int,
) -> tuple[np.ndarray, int, int]:
    path = _cost_shard_result_path(profile_root, shard_index)
    source_shard = index.shards[shard_index]
    try:
        with np.load(path, allow_pickle=False) as payload:
            if _scalar(payload, "format", path) != OMAT_COST_SHARD_FORMAT:
                raise ValueError(f"Unsupported edge-cost shard format in {path}")
            if _scalar(payload, "schema_version", path) != 1:
                raise ValueError(f"Unsupported edge-cost shard schema in {path}")
            if (
                _scalar(payload, "source_manifest_sha256", path)
                != index.manifest_sha256
            ):
                raise ValueError(f"Source manifest mismatch in {path}")
            if (
                _scalar(payload, "source_path", path)
                != source_shard.relative_path.as_posix()
            ):
                raise ValueError(f"Source shard mismatch in {path}")
            if _scalar(payload, "rows", path) != source_shard.length:
                raise ValueError(f"Source row count mismatch in {path}")
            if _scalar(payload, "cutoff_hex", path) != float(cutoff).hex():
                raise ValueError(f"Cutoff mismatch in {path}")
            if bool(_scalar(payload, "directed_edges", path)) is not True:
                raise ValueError(f"Directed-edge semantics mismatch in {path}")
            if bool(_scalar(payload, "true_self_interaction", path)) is not False:
                raise ValueError(f"Self-interaction semantics mismatch in {path}")
            if _scalar(payload, "pbc_cell_semantics", path) != OMAT_PBC_CELL_SEMANTICS:
                raise ValueError(f"PBC/cell semantics mismatch in {path}")
            if (
                _scalar(payload, "neighbour_list_implementation", path)
                != OMAT_NEIGHBOUR_IMPLEMENTATION
            ):
                raise ValueError(f"Neighbour implementation mismatch in {path}")
            if "nedges" not in payload:
                raise ValueError(f"Missing nedges in {path}")
            raw_nedges = payload["nedges"]
            total_atoms = int(_scalar(payload, "total_atoms", path))
            total_edges = int(_scalar(payload, "total_edges", path))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid existing shard result {path}: {exc}") from exc

    cost_shard = OMatCostShard(
        relative_shard_path=source_shard.relative_path,
        cost_path=path,
        rows=source_shard.length,
        total_atoms=total_atoms,
        total_edges=total_edges,
    )
    _validate_cost_array(raw_nedges, cost_shard)
    expected_atoms = int(index.load_natoms(shard_index).sum())
    if total_atoms != expected_atoms:
        raise ValueError(
            f"Invalid existing shard result {path}: total_atoms {total_atoms} "
            f"does not match source {expected_atoms}"
        )
    return np.asarray(raw_nedges, dtype=np.int64), total_atoms, total_edges


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_cost_shard_result(
    *,
    index: OMatShardIndex,
    profile_root: Path | str,
    cutoff: float,
    shard_index: int,
    nedges: np.ndarray,
) -> Path:
    """Atomically write one independently verifiable shard result."""

    _require_generation_source(index, cutoff)
    source_shard = index.shards[shard_index]
    path = _cost_shard_result_path(Path(profile_root).resolve(), shard_index)
    nedges = np.asarray(nedges)
    total_atoms = int(index.load_natoms(shard_index).sum())
    cost_shard = OMatCostShard(
        relative_shard_path=source_shard.relative_path,
        cost_path=path,
        rows=source_shard.length,
        total_atoms=total_atoms,
        total_edges=int(nedges.sum()),
    )
    _validate_cost_array(nedges, cost_shard)
    _atomic_npz(
        path,
        format=np.asarray(OMAT_COST_SHARD_FORMAT),
        schema_version=np.asarray(1, dtype=np.int64),
        source_manifest_sha256=np.asarray(index.manifest_sha256),
        source_path=np.asarray(source_shard.relative_path.as_posix()),
        rows=np.asarray(source_shard.length, dtype=np.int64),
        cutoff_hex=np.asarray(float(cutoff).hex()),
        directed_edges=np.asarray(True),
        true_self_interaction=np.asarray(False),
        pbc_cell_semantics=np.asarray(OMAT_PBC_CELL_SEMANTICS),
        neighbour_list_implementation=np.asarray(OMAT_NEIGHBOUR_IMPLEMENTATION),
        total_atoms=np.asarray(total_atoms, dtype=np.int64),
        total_edges=np.asarray(cost_shard.total_edges, dtype=np.int64),
        nedges=np.asarray(nedges, dtype=np.int64),
    )
    return path


def generate_cost_shard(
    index: OMatShardIndex,
    profile_root: Path | str,
    cutoff: float,
    shard_index: int,
    *,
    force: bool = False,
) -> Path | None:
    """Generate one resumable exact edge-count result without retaining graphs."""

    _require_generation_source(index, cutoff)
    if shard_index < 0 or shard_index >= len(index.shards):
        raise IndexError(shard_index)
    source_shard = index.shards[shard_index]
    if source_shard.length == 0:
        return None
    root = Path(profile_root).resolve()
    path = _cost_shard_result_path(root, shard_index)
    if path.exists() and not force:
        _load_cost_shard_result(
            index=index,
            profile_root=root,
            cutoff=cutoff,
            shard_index=shard_index,
        )
        return path

    natoms = index.load_natoms(shard_index)
    nedges = np.empty(source_shard.length, dtype=np.int64)
    environment = lmdb.open(
        str(source_shard.path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1024,
    )
    try:
        with environment.begin(write=False) as transaction:
            for local_index in range(source_shard.length):
                key = str(local_index + 1).encode("ascii")
                value = transaction.get(key)
                if value is None:
                    raise KeyError(
                        f"Missing OMat row in {source_shard.relative_path}, key "
                        f"{key.decode('ascii')}"
                    )
                configuration = decode_omat_row(value)
                if len(configuration.atomic_numbers) != int(natoms[local_index]):
                    raise ValueError(
                        f"Atomic count mismatch in {source_shard.relative_path}, "
                        f"row {local_index}"
                    )
                edge_index, _, _, _ = get_neighborhood(
                    configuration.positions,
                    cutoff=float(cutoff),
                    pbc=configuration.pbc,
                    cell=configuration.cell,
                    true_self_interaction=False,
                )
                nedges[local_index] = edge_index.shape[1]
    finally:
        environment.close()

    result = _write_cost_shard_result(
        index=index,
        profile_root=root,
        cutoff=cutoff,
        shard_index=shard_index,
        nedges=nedges,
    )
    _load_cost_shard_result(
        index=index,
        profile_root=root,
        cutoff=cutoff,
        shard_index=shard_index,
    )
    return result


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_cost_profile(
    index: OMatShardIndex,
    profile_root: Path | str,
    cutoff: float,
) -> Path:
    """Publish a profile manifest only after every shard result validates."""

    _require_generation_source(index, cutoff)
    root = Path(profile_root).resolve()
    shards = []
    for shard_index, source_shard in enumerate(index.shards):
        if source_shard.length == 0:
            total_atoms = 0
            total_edges = 0
            cost_path = None
        else:
            path = _cost_shard_result_path(root, shard_index)
            if not path.is_file():
                raise ValueError(
                    f"Edge-cost profile is incomplete; missing shard result {path}"
                )
            _, total_atoms, total_edges = _load_cost_shard_result(
                index=index,
                profile_root=root,
                cutoff=cutoff,
                shard_index=shard_index,
            )
            cost_path = path.relative_to(root).as_posix()
        shards.append(
            {
                "source_path": source_shard.relative_path.as_posix(),
                "cost_path": cost_path,
                "rows": source_shard.length,
                "total_atoms": total_atoms,
                "total_edges": total_edges,
            }
        )

    contract = {
        "cutoff_hex": float(cutoff).hex(),
        "directed_edges": True,
        "true_self_interaction": False,
        "pbc_cell_semantics": OMAT_PBC_CELL_SEMANTICS,
        "neighbour_list_implementation": OMAT_NEIGHBOUR_IMPLEMENTATION,
    }
    profile_identity = hashlib.sha256(
        json.dumps(
            {
                "source_manifest_sha256": index.manifest_sha256,
                "contract": contract,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        "format": OMAT_COST_PROFILE_FORMAT,
        "schema_version": OMAT_COST_PROFILE_SCHEMA_VERSION,
        "profile_id": f"{index.dataset_id}-{profile_identity}",
        "source": {
            "dataset_id": index.dataset_id,
            "manifest_sha256": index.manifest_sha256,
            "shards": [
                {"path": shard.relative_path.as_posix(), "rows": shard.length}
                for shard in index.shards
            ],
        },
        "contract": contract,
        "provenance": {
            "mace_version": _package_version("mace-torch"),
            "matscipy_version": _package_version("matscipy"),
        },
        "total_rows": len(index),
        "total_shards": len(index.shards),
        "total_atoms": sum(entry["total_atoms"] for entry in shards),
        "total_edges": sum(entry["total_edges"] for entry in shards),
        "shards": shards,
    }
    manifest = root / OMAT_COST_PROFILE_MANIFEST
    _atomic_json(manifest, payload)
    load_omat_cost_profile(root, index=index, cutoff=cutoff)
    return manifest
