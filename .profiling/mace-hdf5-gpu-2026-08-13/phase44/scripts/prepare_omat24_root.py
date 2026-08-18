#!/usr/bin/env python3
###########################################################################################
# Audit and relocate the extracted official OMat24 ASE-LMDB corpus
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from mace.data import (
    OMAT_MANIFEST_NAME,
    build_omat_shard_index,
    write_omat_manifest,
)


@dataclass(frozen=True)
class PreparationResult:
    source: Path
    target: Path
    compatibility_link: Path
    families: int
    shards: int
    rows: int
    logical_bytes: int
    same_filesystem: bool
    applied: bool


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _device_id(path: Path) -> int:
    candidate = _absolute(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"No existing ancestor for {path}")
        candidate = parent
    return int(candidate.stat().st_dev)


def _logical_bytes(index) -> int:
    paths = set()
    for shard in index.shards:
        paths.add(shard.path)
        if shard.metadata_path is not None:
            paths.add(shard.metadata_path)
    return sum(path.stat().st_size for path in paths)


def _is_compatible_link(link: Path, target: Path) -> bool:
    return link.is_symlink() and link.resolve(strict=False) == target.resolve(
        strict=False
    )


def prepare_root(
    source: Path,
    target: Path,
    compatibility_link: Path,
    *,
    expected_rows: int,
    expected_shards: int,
    apply: bool,
) -> PreparationResult:
    source = _absolute(Path(source))
    target = _absolute(Path(target))
    compatibility_link = _absolute(Path(compatibility_link))

    if source.is_symlink():
        raise ValueError(f"Source must not be a symlink: {source}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == target:
        raise ValueError("Source and target must differ")
    try:
        target.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError(f"Target must not be below source: {target}")
    if target.exists() or target.is_symlink():
        raise ValueError(f"Relocation target already exists: {target}")
    if (source / OMAT_MANIFEST_NAME).exists():
        raise ValueError(
            f"Source already contains a manifest that would be overwritten: {source}"
        )

    link_is_source = compatibility_link == source
    compatible_link_exists = _is_compatible_link(compatibility_link, target)
    if (
        not link_is_source
        and os.path.lexists(compatibility_link)
        and not compatible_link_exists
    ):
        raise ValueError(
            f"Existing compatibility path is incompatible: {compatibility_link}"
        )

    source_device = _device_id(source)
    target_device = _device_id(target.parent)
    same_filesystem = source_device == target_device
    if not same_filesystem:
        raise ValueError("Source and target must be on the same filesystem")

    index = build_omat_shard_index(source)
    rows = len(index)
    shards = len(index.shards)
    families = len({shard.family for shard in index.shards})
    logical_bytes = _logical_bytes(index)
    if rows != expected_rows:
        raise ValueError(
            f"Unexpected row count: expected {expected_rows}, found {rows}"
        )
    if shards != expected_shards:
        raise ValueError(
            f"Unexpected shard count: expected {expected_shards}, found {shards}"
        )

    result = PreparationResult(
        source=source,
        target=target,
        compatibility_link=compatibility_link,
        families=families,
        shards=shards,
        rows=rows,
        logical_bytes=logical_bytes,
        same_filesystem=same_filesystem,
        applied=False,
    )
    if not apply:
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_created = False
    link_created = False
    source.rename(target)
    try:
        moved_index = build_omat_shard_index(target)
        manifest = write_omat_manifest(target, moved_index, dataset_id="OMat24")
        manifest_created = True
        reopened = build_omat_shard_index(target, manifest=manifest)
        if len(reopened) != rows or len(reopened.shards) != shards:
            raise RuntimeError("Reopened OMat manifest does not match audited counts")

        if not compatible_link_exists:
            relative_target = os.path.relpath(target, start=compatibility_link.parent)
            compatibility_link.symlink_to(relative_target, target_is_directory=True)
            link_created = True
    except Exception:
        if link_created and compatibility_link.is_symlink():
            compatibility_link.unlink()
        manifest = target / OMAT_MANIFEST_NAME
        if manifest_created and manifest.is_file():
            manifest.unlink()
        target.rename(source)
        raise

    return PreparationResult(**{**asdict(result), "applied": True})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--compatibility-link", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = prepare_root(
        args.source,
        args.target,
        args.compatibility_link,
        expected_rows=args.expected_rows,
        expected_shards=args.expected_shards,
        apply=args.apply,
    )
    payload = asdict(result)
    for field in ("source", "target", "compatibility_link"):
        payload[field] = str(payload[field])
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
