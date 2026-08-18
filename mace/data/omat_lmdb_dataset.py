"""Lazy indexing utilities for official sharded OMat ASE-LMDB data."""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lmdb
import numpy as np


OMAT_MANIFEST_NAME = "omat24-manifest.json"
OMAT_MANIFEST_FORMAT = "mace-omat-aselmdb-manifest-v1"


@dataclass(frozen=True)
class OMatShard:
    """One official OMat LMDB shard and its size metadata."""

    family: str
    path: Path
    relative_path: Path
    metadata_path: Path | None
    length: int


@dataclass(frozen=True)
class OMatShardIndex:
    """Immutable cumulative index over one or more OMat archive families."""

    roots: tuple[Path, ...]
    dataset_id: str
    shards: tuple[OMatShard, ...]
    cumulative_ends: tuple[int, ...]

    def __len__(self) -> int:
        return self.cumulative_ends[-1] if self.cumulative_ends else 0

    def locate(self, index: int) -> tuple[int, int]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative_ends, index)
        start = 0 if shard_index == 0 else self.cumulative_ends[shard_index - 1]
        return shard_index, index - start

    def load_natoms(self, shard_index: int) -> np.ndarray:
        shard = self.shards[shard_index]
        if shard.metadata_path is None:
            return np.empty(0, dtype=np.int64)
        with np.load(shard.metadata_path, allow_pickle=False) as metadata:
            if "natoms" not in metadata:
                raise ValueError(f"Missing natoms in {shard.metadata_path}")
            natoms = np.asarray(metadata["natoms"], dtype=np.int64)
        if natoms.ndim != 1:
            raise ValueError(
                f"natoms metadata must be one-dimensional: {shard.metadata_path}"
            )
        if len(natoms) != shard.length:
            raise ValueError(
                f"natoms length {len(natoms)} does not match shard length "
                f"{shard.length}: {shard.metadata_path}"
            )
        if np.any(natoms < 1):
            raise ValueError(
                f"natoms metadata must be strictly positive: {shard.metadata_path}"
            )
        return natoms


def _normalize_roots(
    roots: Path | str | Sequence[Path | str],
) -> tuple[Path, ...]:
    if isinstance(roots, (str, Path)):
        normalized = (Path(roots).resolve(),)
    else:
        normalized = tuple(Path(root).resolve() for root in roots)
    if not normalized:
        raise ValueError("At least one OMat LMDB root is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate root in OMat LMDB roots")
    for root in normalized:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if len(normalized) > 1:
        names = [root.name for root in normalized]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate family name in OMat LMDB roots")
    return normalized


def _metadata_path(shard_path: Path) -> Path:
    suffix = shard_path.name.removeprefix("db_").removesuffix(".aselmdb")
    return shard_path.with_name(f"metadata_db_{suffix}.npz")


def _metadata_length(shard_path: Path) -> tuple[Path | None, int]:
    metadata_path = _metadata_path(shard_path)
    if metadata_path.is_file():
        with np.load(metadata_path, allow_pickle=False) as metadata:
            if "natoms" not in metadata:
                raise ValueError(f"Missing natoms in {metadata_path}")
            natoms = metadata["natoms"]
            if natoms.ndim == 0:
                raise ValueError(
                    f"natoms metadata must be one-dimensional: {metadata_path}"
                )
            return metadata_path.resolve(), int(len(natoms))

    environment = lmdb.open(
        str(shard_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with environment.begin(write=False) as transaction:
            entries = int(transaction.stat()["entries"])
    finally:
        environment.close()
    if entries:
        raise ValueError(f"Missing metadata for non-empty shard {shard_path}")
    return None, 0


def _safe_manifest_path(root: Path, value: str, *, field: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest {field} path must be relative: {value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest {field} path must remain below its root: {value}"
        ) from exc
    return resolved


def _cumulative(shards: Sequence[OMatShard]) -> tuple[int, ...]:
    total = 0
    cumulative_ends = []
    for shard in shards:
        total += shard.length
        cumulative_ends.append(total)
    return tuple(cumulative_ends)


def _build_from_manifest(root: Path, manifest: Path) -> OMatShardIndex:
    with manifest.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format") != OMAT_MANIFEST_FORMAT:
        raise ValueError(f"Unsupported OMat manifest format in {manifest}")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"Invalid dataset_id in {manifest}")
    entries = payload.get("shards")
    if not isinstance(entries, list):
        raise ValueError(f"Invalid shards list in {manifest}")

    shards = []
    seen_paths = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid shard entry in {manifest}")
        family = entry.get("family")
        rows = entry.get("rows")
        if not isinstance(family, str) or not family:
            raise ValueError(f"Invalid shard family in {manifest}")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise ValueError(f"Invalid shard row count in {manifest}")
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            raise ValueError(f"Invalid shard path in {manifest}")
        shard_path = _safe_manifest_path(root, path_value, field="shard")
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        if shard_path in seen_paths:
            raise ValueError(f"Duplicate shard path in {manifest}: {shard_path}")
        seen_paths.add(shard_path)

        metadata_value = entry.get("metadata")
        if metadata_value is None:
            metadata_path = None
            if rows:
                raise ValueError(
                    f"Missing metadata for non-empty shard {shard_path}"
                )
        elif isinstance(metadata_value, str):
            metadata_path = _safe_manifest_path(
                root, metadata_value, field="metadata"
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
        else:
            raise ValueError(f"Invalid metadata path in {manifest}")
        shards.append(
            OMatShard(
                family=family,
                path=shard_path,
                relative_path=Path(path_value),
                metadata_path=metadata_path,
                length=rows,
            )
        )

    total_rows = payload.get("total_rows")
    actual_rows = sum(shard.length for shard in shards)
    if total_rows != actual_rows:
        raise ValueError(
            f"Manifest total_rows {total_rows} does not match shard sum "
            f"{actual_rows}: {manifest}"
        )
    return OMatShardIndex(
        roots=(root,),
        dataset_id=dataset_id,
        shards=tuple(shards),
        cumulative_ends=_cumulative(shards),
    )


def build_omat_shard_index(
    roots: Path | str | Sequence[Path | str],
    *,
    manifest: Path | str | None = None,
) -> OMatShardIndex:
    """Build a lazy cumulative index without opening non-empty LMDB shards."""

    normalized = _normalize_roots(roots)
    if manifest is None and len(normalized) == 1:
        candidate = normalized[0] / OMAT_MANIFEST_NAME
        if candidate.is_file():
            manifest = candidate
    if manifest is not None:
        if len(normalized) != 1:
            raise ValueError("A manifest requires exactly one dataset root")
        manifest_path = Path(manifest).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        return _build_from_manifest(normalized[0], manifest_path)

    discovered: list[tuple[str, Path, Path]] = []
    for root in normalized:
        paths = sorted(root.rglob("db_*.aselmdb"), key=lambda path: str(path))
        direct_family_root = bool(paths) and all(path.parent == root for path in paths)
        for path in paths:
            resolved = path.resolve()
            if direct_family_root:
                family = root.name
                relative = Path(family) / path.name
            else:
                within_root = resolved.relative_to(root)
                family = within_root.parts[0]
                relative = within_root
            discovered.append((family, relative, resolved))
    if not discovered:
        raise FileNotFoundError(
            "No db_*.aselmdb shards found below "
            + ", ".join(str(root) for root in normalized)
        )
    discovered.sort(key=lambda item: item[1].as_posix())
    resolved_paths = [item[2] for item in discovered]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Duplicate resolved shard path in OMat LMDB roots")

    shards = []
    for family, relative, path in discovered:
        metadata_path, length = _metadata_length(path)
        shards.append(
            OMatShard(
                family=family,
                path=path,
                relative_path=relative,
                metadata_path=metadata_path,
                length=length,
            )
        )
    dataset_id = normalized[0].name if len(normalized) == 1 else "omat"
    return OMatShardIndex(
        roots=normalized,
        dataset_id=dataset_id,
        shards=tuple(shards),
        cumulative_ends=_cumulative(shards),
    )


def write_omat_manifest(
    root: Path | str,
    index: OMatShardIndex,
    *,
    dataset_id: str,
) -> Path:
    """Atomically write the compact manifest consumed by the native reader."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    entries = []
    for shard in index.shards:
        try:
            relative_path = shard.path.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"Shard is outside manifest root: {shard.path}") from exc
        if shard.metadata_path is None:
            metadata = None
        else:
            try:
                metadata = shard.metadata_path.relative_to(root_path).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Metadata is outside manifest root: {shard.metadata_path}"
                ) from exc
        entries.append(
            {
                "family": shard.family,
                "path": relative_path.as_posix(),
                "metadata": metadata,
                "rows": shard.length,
            }
        )
    payload = {
        "format": OMAT_MANIFEST_FORMAT,
        "dataset_id": dataset_id,
        "total_rows": len(index),
        "shards": entries,
    }
    manifest = root_path / OMAT_MANIFEST_NAME
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest
