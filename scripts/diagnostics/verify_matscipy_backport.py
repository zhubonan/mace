#!/usr/bin/env python3
"""Verify the exact matscipy 1.2.0 neighbour-list leak backport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_TOOLS_SHA256 = (
    "a37451489ce42c12ce4b734d9168732b2031e2892c45868682d9ac5925150b75"
)
UPSTREAM_COMMIT = "db167a0925a687ea4ebe5785d007cc5a4cc6241f"
EXPECTED_INSERTION = "    free(dims);"
RESIZE_CALL = (
    "    retval = PyArray_Resize((PyArrayObject *) py_arr, "
    "&newshape, 1, NPY_CORDER);"
)
IGNORED_PATHS = {"c/tools.c.pristine", "pristine-files.sha256"}


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries(manifest_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"invalid pristine manifest row: {line!r}")
        if relative in entries:
            raise ValueError(f"duplicate pristine manifest path: {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError("pristine manifest is empty")
    return entries


def compare_manifest(manifest_path: Path, source_root: Path) -> list[str]:
    """Return changed paths and reject missing or newly added source files."""

    expected = _manifest_entries(manifest_path)
    observed = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(source_root).parts
        and path.relative_to(source_root).as_posix() not in IGNORED_PATHS
    }
    if observed != set(expected):
        raise ValueError(
            "unexpected source paths: "
            f"missing={sorted(set(expected) - observed)}, "
            f"added={sorted(observed - set(expected))}"
        )
    return sorted(
        relative
        for relative, expected_hash in expected.items()
        if sha256(source_root / relative) != expected_hash
    )


def _expected_patched_tools(pristine_text: str) -> str:
    if pristine_text.count(RESIZE_CALL) != 1:
        raise ValueError("pristine c/tools.c lacks one exact PyArray_Resize call")
    return pristine_text.replace(
        RESIZE_CALL,
        f"{RESIZE_CALL}\n{EXPECTED_INSERTION}",
        1,
    )


def verify_source_tree(source_root: Path) -> dict[str, object]:
    """Require the official tools hash and only the upstream one-line delta."""

    source_root = source_root.resolve()
    pristine = source_root / "c/tools.c.pristine"
    patched = source_root / "c/tools.c"
    manifest = source_root / "pristine-files.sha256"
    for path in (pristine, patched, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    pristine_hash = sha256(pristine)
    if pristine_hash != EXPECTED_TOOLS_SHA256:
        raise ValueError(
            "pristine c/tools.c SHA-256 mismatch: "
            f"expected {EXPECTED_TOOLS_SHA256}, got {pristine_hash}"
        )
    pristine_text = pristine.read_text(encoding="utf-8")
    patched_text = patched.read_text(encoding="utf-8")
    if patched_text != _expected_patched_tools(pristine_text):
        raise ValueError("c/tools.c does not contain the exact upstream patch")
    changed = compare_manifest(manifest, source_root)
    if changed != ["c/tools.c"]:
        raise ValueError(f"unexpected source delta: {changed}")
    return {
        "schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_root": str(source_root),
        "pristine_tools_sha256": pristine_hash,
        "patched_tools_sha256": sha256(patched),
        "manifest_sha256": sha256(manifest),
        "changed_files": changed,
        "inserted_line": EXPECTED_INSERTION,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically write a provenance report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = verify_source_tree(args.source_root)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
