#!/usr/bin/env python3
"""Measure retained glibc allocations from repeated matscipy neighbour lists."""

from __future__ import annotations

import argparse
import ctypes
from importlib import metadata
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from ase.build import bulk
from matscipy import _matscipy
from matscipy.neighbours import neighbour_list


GIB = float(1024**3)
DEFAULT_LIMIT = 0.02
MALLINFO_FIELDS = (
    "arena",
    "ordblks",
    "smblks",
    "hblks",
    "hblkhd",
    "usmblks",
    "fsmblks",
    "uordblks",
    "fordblks",
    "keepcost",
)


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in MALLINFO_FIELDS]


def fit_slope_gib_per_10k(
    batch_indices: Sequence[int], bytes_used: Sequence[int]
) -> float:
    """Fit bytes versus batch index and normalize to GiB per 10k batches."""

    if len(batch_indices) != len(bytes_used) or len(batch_indices) < 3:
        raise ValueError("slope fit requires at least three paired samples")
    slope = np.polyfit(
        np.asarray(batch_indices, dtype=np.float64),
        np.asarray(bytes_used, dtype=np.float64),
        1,
    )[0]
    return float(slope * 10_000.0 / GIB)


def classify_slope(slope: float, *, limit: float = DEFAULT_LIMIT) -> bool:
    """Return whether a finite allocator slope is strictly below the limit."""

    return bool(np.isfinite(slope) and slope < limit)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON document without exposing a partial destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_mallinfo2() -> dict[str, int]:
    """Read current-process glibc allocator ownership."""

    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.argtypes = []
    libc.mallinfo2.restype = Mallinfo2
    value = libc.mallinfo2()
    return {name: int(getattr(value, name)) for name in MALLINFO_FIELDS}


def run_synthetic_probe(
    *, batches: int, warmup_batches: int, calls_per_batch: int = 256
) -> dict[str, object]:
    """Run a production-normalized neighbour-list allocator probe."""

    if not 3 <= warmup_batches < batches:
        raise ValueError("require 3 <= warmup_batches < batches")
    if calls_per_batch != 256:
        raise ValueError("the production-normalized probe requires 256 calls/batch")

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    batch_indices: list[int] = []
    allocator_rows: list[dict[str, int]] = []
    checksum = 0
    for batch in range(1, batches + 1):
        for _ in range(calls_per_batch):
            senders, receivers, shifts = neighbour_list("ijS", atoms, 6.0)
            checksum += len(senders) + len(receivers) + int(np.abs(shifts).sum())
        if batch >= warmup_batches:
            batch_indices.append(batch)
            allocator_rows.append(read_mallinfo2())

    slopes = {
        name: fit_slope_gib_per_10k(
            batch_indices, [row[name] for row in allocator_rows]
        )
        for name in ("arena", "uordblks", "fordblks", "keepcost")
    }
    return {
        "schema_version": 1,
        "matscipy": metadata.version("matscipy"),
        "matscipy_file": str(Path(_matscipy.__file__).resolve()),
        "numpy": np.__version__,
        "batches": batches,
        "warmup_batches": warmup_batches,
        "calls_per_batch": calls_per_batch,
        "samples": len(batch_indices),
        "checksum": checksum,
        "slopes_gib_per_10k_batches": slopes,
        "limit_gib_per_10k_batches": DEFAULT_LIMIT,
        "passed": classify_slope(slopes["uordblks"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=10_000)
    parser.add_argument("--warmup-batches", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_synthetic_probe(
        batches=args.batches, warmup_batches=args.warmup_batches
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
