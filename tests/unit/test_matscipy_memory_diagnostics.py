from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_allocator_slope_uses_batch_normalization() -> None:
    module = _load(
        "matscipy_neighbour_allocator",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )
    batches = [1_000, 2_000, 3_000, 4_000]
    bytes_used = [int(value * (1024**3) / 10_000) for value in batches]

    assert module.fit_slope_gib_per_10k(batches, bytes_used) == pytest.approx(
        1.0
    )


def test_allocator_slope_rejects_unpaired_or_too_short_samples() -> None:
    module = _load(
        "matscipy_neighbour_allocator_errors",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )

    with pytest.raises(ValueError, match="three paired samples"):
        module.fit_slope_gib_per_10k([1, 2], [3, 4])
    with pytest.raises(ValueError, match="three paired samples"):
        module.fit_slope_gib_per_10k([1, 2, 3], [4, 5])


def test_allocator_threshold_is_strict() -> None:
    module = _load(
        "matscipy_neighbour_allocator_threshold",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )

    assert module.classify_slope(0.019999, limit=0.02) is True
    assert module.classify_slope(0.02, limit=0.02) is False
    assert module.classify_slope(float("nan"), limit=0.02) is False


def test_synthetic_probe_records_the_loaded_matscipy_binary() -> None:
    module = _load(
        "matscipy_neighbour_allocator_metadata",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )

    result = module.run_synthetic_probe(batches=5, warmup_batches=3)

    assert result["matscipy"] == "1.2.0"
    assert Path(result["matscipy_file"]).is_file()
    assert result["calls_per_batch"] == 256


def test_atomic_json_replaces_without_leaving_temporary_file(tmp_path: Path) -> None:
    module = _load(
        "matscipy_neighbour_allocator_json",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )
    output = tmp_path / "probe.json"

    module.atomic_json(output, {"passed": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}
    assert not output.with_suffix(".json.tmp").exists()


def test_process_tree_parser_sums_only_literal_heap() -> None:
    module = _load(
        "monitor_process_tree_memory",
        "scripts/diagnostics/monitor_process_tree_memory.py",
    )
    smaps = """1000-2000 rw-p 00000000 00:00 0 [heap]
Rss:                 128 kB
Pss:                 120 kB
Private_Dirty:       112 kB
2000-3000 rw-p 00000000 00:00 0
Rss:                 256 kB
Pss:                 240 kB
Private_Dirty:       224 kB
"""

    assert module.parse_literal_heap(smaps) == {
        "heap_rss_kib": 128,
        "heap_pss_kib": 120,
        "heap_private_dirty_kib": 112,
    }


def test_descendants_follow_children_recursively_without_duplicates(
    tmp_path: Path,
) -> None:
    module = _load(
        "monitor_process_tree_memory_descendants",
        "scripts/diagnostics/monitor_process_tree_memory.py",
    )
    for pid, children in ((10, "11 12\n"), (11, "13\n"), (12, "13\n"), (13, "")):
        directory = tmp_path / str(pid) / "task" / str(pid)
        directory.mkdir(parents=True)
        (directory / "children").write_text(children, encoding="utf-8")

    assert module.descendants(10, tmp_path) == [10, 11, 12, 13]


def test_rollup_parser_extracts_required_kib_fields() -> None:
    module = _load(
        "monitor_process_tree_memory_rollup",
        "scripts/diagnostics/monitor_process_tree_memory.py",
    )
    rollup = """Rss:                 500 kB
Pss:                 450 kB
Private_Dirty:       400 kB
Shared_Clean:         50 kB
"""

    assert module.parse_rollup(rollup) == {
        "rss_kib": 500,
        "pss_kib": 450,
        "private_dirty_kib": 400,
    }


def test_real_omat_worker_fit_normalizes_256_items_as_one_batch(
    tmp_path: Path,
) -> None:
    _load(
        "matscipy_neighbour_allocator",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )
    module = _load(
        "omat_mallinfo_probe",
        "scripts/diagnostics/omat_mallinfo_probe.py",
    )
    events = tmp_path / "worker-123.jsonl"
    rows = []
    for items in (256, 512, 768, 1_024):
        rows.append(
            {
                "pid": 123,
                "items": items,
                "mallinfo2": {
                    field: items * 1_024 for field in module.MALLINFO_FIELDS
                }
                | {"arena_balance": 0},
                "heap": {
                    "heap_rss_kib": items,
                    "heap_pss_kib": items,
                    "heap_private_dirty_kib": items,
                },
            }
        )
    events.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    fitted = module.fit_worker(events, warmup_items=256)

    expected = 1_024 * 256 * 10_000 / (1024**3)
    assert fitted["uordblks_gib_per_10k_batches"] == pytest.approx(expected)
    assert fitted["heap_private_dirty_kib_gib_per_10k_batches"] == pytest.approx(
        expected
    )
