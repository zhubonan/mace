
# Matscipy Neighbour-List Memory-Leak Backport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify an exact matscipy 1.2.0 memory-leak backport, then relaunch the paired one-layer OMat24 jobs from fresh initialization without linear DataLoader-worker host-memory growth.

**Architecture:** Keep the MACE source change, matscipy build, dependency runtime, OMat diagnostics, and production launch as separately verifiable layers. A committed ASE-based allocator probe and process-tree monitor establish stock RED/patched GREEN behavior; an official-source build applies only upstream `free(dims)` plus local-version metadata; an immutable dependency-only runtime is then admitted by exact version/path/wheel-hash checks before a 10,000-batch real-OMat allocator run, a 10,000-update single-GPU qualification, and paired two-rank production relaunch.

**Tech Stack:** Python 3.12.11, uv, Meson/meson-python, C/NumPy 2.5.1, matscipy 1.2.0, ASE 3.29.0, PyTorch 2.11.0+cu130, e3nn 0.4.4, CuEquivariance 0.11.0, pytest, Slurm, Linux `/proc`, glibc `mallinfo2()`.

## Global Constraints

- Apply only upstream commit `db167a0925a687ea4ebe5785d007cc5a4cc6241f`'s `free(dims)` source change. Derive wheel version `1.2.0+omatfix1` from an isolated build-only Git tag so sdist package files do not need a second edit.
- Retrieve `https://files.pythonhosted.org/packages/3e/62/851f14fef0d47d5546cc3bb315a96d73e3cee9588adb069c09dc5463a64a/matscipy-1.2.0.tar.gz` afresh; require size `10368600` and SHA-256 `1d0691be7e541357ec7a3363e1ed2d7289e5ef8942fbc94064567124d1fb7444` before extraction.
- Require pristine `matscipy-1.2.0/c/tools.c` SHA-256 `a37451489ce42c12ce4b734d9168732b2031e2892c45868682d9ac5925150b75` before patching.
- Never use either incomplete `/tmp` Git clone or the partial `/tmp/matscipy-1.2.0-plan-inspect.tar.gz` from investigation.
- Preserve `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110` unchanged as the stock runtime and rollback artifact.
- The patched runtime contains dependencies only; `mace-torch` and `omat-downsample` distributions remain absent and their immutable sources are supplied through `PYTHONPATH`.
- Use `uv`; run `uv pip`, GPU, and Slurm commands outside the sandbox.
- GPU MACE execution must retain CuEquivariance, compiled training, Float32, batch 256 per rank, and four spawned DataLoader workers.
- Normalize allocator slopes as GiB per 10,000 batches, with 256 neighbour calls/structures per batch; every patched `uordblks`, arena, and literal-heap slope must be `<0.02 GiB/10k batches`.
- Single-GPU qualification must complete at least 10,000 optimizer updates on real OMat24 before production use.
- Paired production jobs use two ranks/two job-wide GPUs each, eight CPUs per rank, at most 46,000 MB host memory per rank (92,000 MB/node), batch 256 per rank, four workers per rank, fresh initialization, and isolated run/compiler-cache roots.
- Preserve all prior diagnostics, failed jobs, run roots, and checkpoints; never reuse a checkpoint from a superseded or failed run.

---

### Task 1: Commit deterministic allocator and process-memory diagnostics

**Files:**
- Create: `scripts/diagnostics/matscipy_neighbour_allocator.py`
- Create: `scripts/diagnostics/omat_mallinfo_probe.py`
- Create: `scripts/diagnostics/monitor_process_tree_memory.py`
- Create: `tests/unit/test_matscipy_memory_diagnostics.py`

**Interfaces:**
- Produces: `fit_slope_gib_per_10k(x: Sequence[int], bytes_used: Sequence[int]) -> float`.
- Produces: `run_synthetic_probe(*, batches: int, warmup_batches: int, calls_per_batch: int = 256) -> dict[str, object]`.
- Produces: allocator JSON with `schema_version`, exact package/runtime metadata, sample counts, `uordblks_gib_per_10k_batches`, and `passed`.
- Produces: process-tree JSONL rows containing timestamp, PID/PPID, command, RSS, PSS, private-dirty, and literal `[heap]` private-dirty.

- [ ] **Step 1: Write failing pure-function and CLI-contract tests.**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allocator_slope_uses_batch_normalization() -> None:
    module = _load(
        "matscipy_neighbour_allocator",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )
    x = [1000, 2000, 3000, 4000]
    y = [int(value * (1024**3) / 10_000) for value in x]
    assert module.fit_slope_gib_per_10k(x, y) == pytest.approx(1.0)


def test_threshold_is_strict_and_serializable() -> None:
    module = _load(
        "matscipy_neighbour_allocator",
        "scripts/diagnostics/matscipy_neighbour_allocator.py",
    )
    assert module.classify_slope(0.019999, limit=0.02) is True
    assert module.classify_slope(0.02, limit=0.02) is False


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
```

- [ ] **Step 2: Run the tests and verify RED.**

Run:

```bash
/home/bonan/work/train-downsample/.venv/bin/python -m pytest tests/unit/test_matscipy_memory_diagnostics.py -q
```

Expected: collection/import failure because all three diagnostic modules are absent.

- [ ] **Step 3: Implement the synthetic probe and shared slope contract.**

`scripts/diagnostics/matscipy_neighbour_allocator.py` must contain these core contracts:

```python
#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from importlib import metadata
from typing import Sequence

import numpy as np
from ase.build import bulk
from matscipy.neighbours import neighbour_list


GIB = float(1024**3)
DEFAULT_LIMIT = 0.02


class Mallinfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena", "ordblks", "smblks", "hblks", "hblkhd",
            "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost",
        )
    ]


def fit_slope_gib_per_10k(
    batch_indices: Sequence[int], bytes_used: Sequence[int]
) -> float:
    if len(batch_indices) != len(bytes_used) or len(batch_indices) < 3:
        raise ValueError("slope fit requires at least three paired samples")
    slope = np.polyfit(
        np.asarray(batch_indices, dtype=np.float64),
        np.asarray(bytes_used, dtype=np.float64),
        1,
    )[0]
    return float(slope * 10_000.0 / GIB)


def classify_slope(slope: float, *, limit: float = DEFAULT_LIMIT) -> bool:
    return bool(np.isfinite(slope) and slope < limit)


def _mallinfo2() -> dict[str, int]:
    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.argtypes = []
    libc.mallinfo2.restype = Mallinfo2
    value = libc.mallinfo2()
    return {name: int(getattr(value, name)) for name, _ in Mallinfo2._fields_}


def run_synthetic_probe(
    *, batches: int, warmup_batches: int, calls_per_batch: int = 256
) -> dict[str, object]:
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
            allocator_rows.append(_mallinfo2())
    slopes = {
        name: fit_slope_gib_per_10k(
            batch_indices, [row[name] for row in allocator_rows]
        )
        for name in ("arena", "uordblks", "fordblks", "keepcost")
    }
    return {
        "schema_version": 1,
        "matscipy": metadata.version("matscipy"),
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
```

Add an argparse CLI accepting `--batches`, `--warmup-batches`, and `--output`. Atomically write JSON before returning status 0 for `passed=true` or status 1 for `passed=false`.

```python
def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=10_000)
    parser.add_argument("--warmup-batches", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_synthetic_probe(
        batches=args.batches, warmup_batches=args.warmup_batches
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1
```

- [ ] **Step 4: Promote the verified real-OMat `mallinfo2()` harness.**

Create `scripts/diagnostics/omat_mallinfo_probe.py` from verified `/tmp/omat_mallinfo_probe.py` (SHA-256 `eb45f7b634e6c2d5deadf707dce4fe7533f43bde754c9eb4d99b44820530247b`) and make only this functional addition:

```python
critical = ("uordblks", "arena", "heap_private_dirty_kib")
summary["limit_gib_per_10k_batches"] = 0.02
summary["passed"] = all(
    classify_slope(summary["mean_slopes_gib_per_10k"][field])
    for field in critical
)
```

Keep its exact real-data settings: `r_max=6.0`, batch 256, four spawned persistent workers, shard-window atom-balanced sampling, eight open/window shards, 1,000 warm-up batches, and 10,000 measured batches. Exit nonzero only after writing `summary.json` when `passed` is false.

- [ ] **Step 5: Implement the generic descendant-memory monitor.**

`scripts/diagnostics/monitor_process_tree_memory.py` must recursively read `/proc/{pid}/task/{pid}/children`, parse `smaps_rollup`, parse only mapping headers ending in `[heap]`, and append one JSON object per live descendant per interval. Its parsing core is:

```python
MAP_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")


def parse_literal_heap(text: str) -> dict[str, int]:
    result = {
        "heap_rss_kib": 0,
        "heap_pss_kib": 0,
        "heap_private_dirty_kib": 0,
    }
    in_heap = False
    for line in text.splitlines():
        if MAP_HEADER.match(line):
            in_heap = line.endswith("[heap]")
        elif in_heap and line.startswith("Rss:"):
            result["heap_rss_kib"] += int(line.split()[1])
        elif in_heap and line.startswith("Pss:"):
            result["heap_pss_kib"] += int(line.split()[1])
        elif in_heap and line.startswith("Private_Dirty:"):
            result["heap_private_dirty_kib"] += int(line.split()[1])
    return result


def descendants(root_pid: int, proc_root: Path) -> list[int]:
    found: list[int] = []
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        children_path = proc_root / str(pid) / "task" / str(pid) / "children"
        try:
            children = [int(value) for value in children_path.read_text().split()]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            children = []
        found.append(pid)
        pending.extend(child for child in children if child not in found)
    return found
```

Its CLI is:

```text
--root-pid INTEGER --output PATH --interval-seconds FLOAT
--stop-file PATH --proc-root PATH(default=/proc)
```

Every row contains `observed_at`, `elapsed_s`, `root_pid`, `pid`, `ppid`, `name`, `cmdline`, `rss_kib`, `pss_kib`, `private_dirty_kib`, `heap_rss_kib`, `heap_pss_kib`, and `heap_private_dirty_kib`. Stop when the root exits or the stop file appears; flush every line.

- [ ] **Step 6: Run focused and formatting gates.**

```bash
/home/bonan/work/train-downsample/.venv/bin/python -m pytest tests/unit/test_matscipy_memory_diagnostics.py -q
/home/bonan/work/train-downsample/.venv/bin/python -m py_compile scripts/diagnostics/matscipy_neighbour_allocator.py scripts/diagnostics/omat_mallinfo_probe.py scripts/diagnostics/monitor_process_tree_memory.py tests/unit/test_matscipy_memory_diagnostics.py
git diff --check
```

Expected: all tests/checks pass.

- [ ] **Step 7: Commit the diagnostics.**

```bash
git add scripts/diagnostics/matscipy_neighbour_allocator.py scripts/diagnostics/omat_mallinfo_probe.py scripts/diagnostics/monitor_process_tree_memory.py tests/unit/test_matscipy_memory_diagnostics.py
git commit -m "test: add matscipy memory regression probes"
```

---

### Task 2: Preserve the stock-runtime RED evidence

**Files:**
- Read: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/`
- Create: `/scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/red/`

**Interfaces:**
- Consumes: Task 1's committed allocator probe.
- Produces: `red/stock-synthetic.json`, stdout/stderr, package freeze, and source commit.

- [ ] **Step 1: Create a fresh RED artifact root and record provenance.**

Outside the sandbox on Cata:

```bash
test ! -e /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1
mkdir -p /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/red
git -C /home/bonan/.codex/worktrees/a86d/mace-official rev-parse HEAD > /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/red/mace-source-commit.txt
/opt/ohpc/pub/userlibs/uv/latest/uv pip list --python /scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/bin/python > /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/red/stock-runtime-packages.txt
```

- [ ] **Step 2: Run stock matscipy and require the intended RED failure.**

Use 10,000 batches, 1,000 warm-up batches, and 256 calls/batch. Capture exit 1 as the expected test result, not infrastructure failure. Assert:

```text
matscipy == 1.2.0
passed == false
uordblks_gib_per_10k_batches > 1.0
exit status == 1
no traceback, killed, or error signature
```

- [ ] **Step 3: Compare with the retained real-OMat stock baseline.**

Read the verified stock summary beneath `/scratch/bonan/train-downsample-single-layer/b7ffd37-ec634e6-mem46-clean2/diagnostics/mallinfo2-20260826-v1`; record its `uordblks=1.630030`, arena `1.629867`, and literal heap private-dirty `1.629204 GiB/10k` beside the synthetic result. Do not rerun the real-OMat stock probe.

---

### Task 3: Commit the exact patch and provenance verifier

**Files:**
- Create: `patches/matscipy/1.2.0-free-resize-dims.patch`
- Create: `scripts/diagnostics/verify_matscipy_backport.py`
- Modify: `tests/unit/test_matscipy_memory_diagnostics.py`

**Interfaces:**
- Produces: `verify_source_tree(source_root: Path) -> dict[str, object]`.
- Enforces: pristine hash, exact one-line native delta, no other sdist-file delta.

- [ ] **Step 1: Add failing verifier tests.**

Create a temporary source fixture with `c/tools.c.pristine`, patched `c/tools.c`, and a pristine file-hash manifest. Assert acceptance of only `+    free(dims);`; add negative cases for a second C edit and a changed Python file.

```python
TOOLS_FIXTURE = """void *resize_array(PyObject *py_arr, npy_intp newsize) {
    npy_intp *dims = malloc(sizeof(npy_intp));
    PyObject *retval = PyArray_Resize(
        (PyArrayObject *) py_arr, &newshape, 1, NPY_CORDER);
    if (!retval) return NULL;
    return PyArray_DATA((PyArrayObject *) py_arr);
}
"""


def make_pristine_source_fixture(tmp_path: Path, module) -> Path:
    source = tmp_path / "matscipy-1.2.0"
    (source / "c").mkdir(parents=True)
    (source / "matscipy").mkdir()
    (source / "c/tools.c").write_text(TOOLS_FIXTURE, encoding="utf-8")
    (source / "c/tools.c.pristine").write_text(TOOLS_FIXTURE, encoding="utf-8")
    (source / "matscipy/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    module.EXPECTED_TOOLS_SHA256 = module.sha256(source / "c/tools.c.pristine")
    (source / "pristine-files.sha256").write_text(
        f"{module.sha256(source / 'c/tools.c')}  c/tools.c\n"
        f"{module.sha256(source / 'matscipy/__init__.py')}  matscipy/__init__.py\n",
        encoding="utf-8",
    )
    return source


def apply_expected_free_dims(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "    if (!retval) return NULL;",
            "    free(dims);\n    if (!retval) return NULL;",
        ),
        encoding="utf-8",
    )


def test_backport_verifier_rejects_every_extra_source_delta(tmp_path: Path) -> None:
    module = _load(
        "verify_matscipy_backport",
        "scripts/diagnostics/verify_matscipy_backport.py",
    )
    source = make_pristine_source_fixture(tmp_path, module)
    apply_expected_free_dims(source / "c/tools.c")
    assert module.verify_source_tree(source)["changed_files"] == ["c/tools.c"]

    (source / "matscipy/__init__.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected source delta"):
        module.verify_source_tree(source)


def test_backport_verifier_rejects_second_native_edit(tmp_path: Path) -> None:
    module = _load(
        "verify_matscipy_backport",
        "scripts/diagnostics/verify_matscipy_backport.py",
    )
    source = make_pristine_source_fixture(tmp_path, module)
    apply_expected_free_dims(source / "c/tools.c")
    with (source / "c/tools.c").open("a", encoding="utf-8") as handle:
        handle.write("/* extra change */\n")
    with pytest.raises(ValueError, match="exact upstream patch"):
        module.verify_source_tree(source)
```

Run the focused test and expect import failure because the verifier does not exist.

- [ ] **Step 2: Add the exact upstream patch.**

`patches/matscipy/1.2.0-free-resize-dims.patch`:

```diff
diff --git a/c/tools.c b/c/tools.c
--- a/c/tools.c
+++ b/c/tools.c
@@ -83,6 +83,7 @@ resize_array(PyObject *py_arr, npy_intp newsize)
     PyObject *retval;
     retval = PyArray_Resize((PyArrayObject *) py_arr, &newshape, 1, NPY_CORDER);
+    free(dims);
     if (!retval)  return NULL;
     Py_DECREF(retval);

     return PyArray_DATA((PyArrayObject *) py_arr);
```

- [ ] **Step 3: Implement exact source verification.**

The verifier must require pristine/patched tools copies, require tools SHA-256 `a37451489ce42c12ce4b734d9168732b2031e2892c45868682d9ac5925150b75`, accept only the inserted `free(dims);`, compare all other regular sdist files against `pristine-files.sha256`, ignore only the build-only `.git/` directory and explicitly named provenance copies/manifests, and atomically write upstream commit/source/patch hashes and allowed deltas.

```python
EXPECTED_TOOLS_SHA256 = (
    "a37451489ce42c12ce4b734d9168732b2031e2892c45868682d9ac5925150b75"
)
EXPECTED_INSERTION = "    free(dims);"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_manifest(manifest_path: Path, source_root: Path) -> list[str]:
    expected = {
        line.split(None, 1)[1]: line.split(None, 1)[0]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    }
    ignored = {"c/tools.c.pristine", "pristine-files.sha256"}
    observed = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(source_root).parts
        and path.relative_to(source_root).as_posix() not in ignored
    }
    if observed != set(expected):
        raise ValueError(
            f"unexpected source paths: missing={set(expected) - observed}, "
            f"added={observed - set(expected)}"
        )
    return sorted(
        relative
        for relative, expected_hash in expected.items()
        if sha256(source_root / relative) != expected_hash
    )


def verify_source_tree(source_root: Path) -> dict[str, object]:
    pristine = source_root / "c/tools.c.pristine"
    patched = source_root / "c/tools.c"
    if sha256(pristine) != EXPECTED_TOOLS_SHA256:
        raise ValueError("pristine c/tools.c SHA-256 mismatch")
    pristine_lines = pristine.read_text(encoding="utf-8").splitlines()
    patched_lines = patched.read_text(encoding="utf-8").splitlines()
    added = [line for line in patched_lines if line not in pristine_lines]
    restored = [line for line in patched_lines if line != EXPECTED_INSERTION]
    if added != [EXPECTED_INSERTION] or restored != pristine_lines:
        raise ValueError("c/tools.c does not contain the exact upstream patch")
    changed = compare_manifest(source_root / "pristine-files.sha256", source_root)
    if changed != ["c/tools.c"]:
        raise ValueError(f"unexpected source delta: {changed}")
    return {
        "upstream_commit": "db167a0925a687ea4ebe5785d007cc5a4cc6241f",
        "pristine_tools_sha256": EXPECTED_TOOLS_SHA256,
        "changed_files": changed,
        "inserted_line": EXPECTED_INSERTION,
    }
```

- [ ] **Step 4: Run the gate and commit.**

```bash
/home/bonan/work/train-downsample/.venv/bin/python -m pytest tests/unit/test_matscipy_memory_diagnostics.py -q
git diff --check
git add patches/matscipy/1.2.0-free-resize-dims.patch scripts/diagnostics/verify_matscipy_backport.py tests/unit/test_matscipy_memory_diagnostics.py
git commit -m "build: record matscipy leak backport"
```

---

### Task 4: Build the provenance-marked wheel from official source

**Files:**
- Create: `/scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/source/`
- Create: `/scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/dist/`
- Create: `/scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/provenance/`

**Interfaces:**
- Consumes: Task 3's patch/verifier.
- Produces: one Linux x86_64 CPython-3.12 wheel and SHA-256.

- [ ] **Step 1: Download to a new filename and verify before extraction.**

Run outside the sandbox; use `HTTPS_PROXY=http://localhost:7890` only if direct access fails. Require size `10368600` and SHA-256 `1d0691be7e541357ec7a3363e1ed2d7289e5ef8942fbc94064567124d1fb7444`. On mismatch, preserve it as `failed-download-sha-or-size-mismatch.tar.gz`, record the mismatch, and stop without extraction; refuse if that evidence filename already exists.

- [ ] **Step 2: Extract once and snapshot pristine content.**

Extract into the fresh root, copy `c/tools.c` to `c/tools.c.pristine`, create a sorted `pristine-files.sha256`, and assert the tools hash and `Version: 1.2.0` in pristine `PKG-INFO`.

- [ ] **Step 3: Apply only the committed patch and create build-only version metadata.**

Apply the patch with zero fuzz. Then initialize a Git repository inside the extracted build tree; list `c/tools.c.pristine` and `pristine-files.sha256` in build-only `.git/info/exclude`; commit the patched sdist files with command-local identity `MACE OMat Build <mace-omat-build@invalid>`; and create annotated tag `v1.2.0+omatfix1`. Require an empty `git status --porcelain` and `python3 discover_version.py` output exactly `1.2.0+omatfix1`. The `.git/` directory is build metadata and is excluded from the sdist-file-delta comparison; all package files except `c/tools.c` remain byte-identical.

- [ ] **Step 4: Verify the source before compilation.**

```bash
/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/bin/python scripts/diagnostics/verify_matscipy_backport.py \
  --source-root /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/source/matscipy-1.2.0 \
  --output /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/provenance/source-verification.json
```

Expected: one native insertion, exact hashes/commit, no other sdist-file delta.

- [ ] **Step 5: Build exactly one wheel with uv.**

```bash
/opt/ohpc/pub/userlibs/uv/latest/uv build --wheel \
  --python /scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/bin/python \
  --build-constraints /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/provenance/build-constraints.txt \
  --out-dir /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/dist \
  /scratch/bonan/matscipy-builds/20260826-matscipy-1.2.0-omatfix1-v1/source/matscipy-1.2.0
```

Write `numpy==2.5.1` to `build-constraints.txt` before the command so the native extension builds against the runtime's exact NumPy family. Require one wheel whose METADATA name/version are `matscipy` and `1.2.0+omatfix1`.

- [ ] **Step 6: Seal build provenance.**

Record wheel filename/hash, compilers, `ldd` for `_matscipy*.so`, Meson/ninja/backend versions, archive/patch/source hashes, upstream commit, build command, timestamps, and disassembly showing `resize_array` calls both `malloc` and `free`.

---

### Task 5: Create an isolated patched dependency runtime

**Files:**
- Read only: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/`
- Create: `/scratch/bonan/mace-runtimes/20260826-py312-torch211-cu130-e3nn044-cueq0110-matscipy120omatfix1/`
- Create: its `environment/` directory.

**Interfaces:**
- Consumes: Task 4 wheel.
- Produces: dependency-only runtime and `environment/matscipy-wheel.sha256`.

- [ ] **Step 1: Refuse reuse and copy the proven runtime.**

Use `cp -a --reflink=auto` into the exact temporary sibling `/scratch/bonan/mace-runtimes/20260826-py312-torch211-cu130-e3nn044-cueq0110-matscipy120omatfix1.building-v1`; verify the parent package freeze before installing and refuse an existing target. Do not mutate the parent.

- [ ] **Step 2: Install only the validated wheel with uv outside the sandbox.**

Resolve the single wheel into task-specific shell variable `MATSCIPY_OMATFIX_WHEEL`, assert it names exactly one regular file, then run `uv pip install --python /scratch/bonan/mace-runtimes/20260826-py312-torch211-cu130-e3nn044-cueq0110-matscipy120omatfix1.building-v1/bin/python --no-deps --reinstall "$MATSCIPY_OMATFIX_WHEEL"`. Reject zero/multiple matches.

- [ ] **Step 3: Assert the exact dependency boundary.**

Run `uv pip check` and require Python 3.12.11, torch 2.11.0+cu130/CUDA 13.0/NCCL 2.28.9, e3nn 0.4.4, CuEq/CuEq-Torch 0.11.0, CuEq ops torch cu13 0.10.0, matscipy 1.2.0+omatfix1, NumPy 2.5.1, SciPy 1.18.0, and ASE 3.29.0. Require MACE/project distributions absent, no `cu12` distribution, and dependency imports beneath the runtime root.

- [ ] **Step 4: Publish atomically with provenance.**

Write parent-runtime inventory, wheel/hash, package freeze, `uv pip check`, module paths, ELF dependencies, and source verification to `environment/`. Rename exactly once to the final runtime; refuse an existing target.

---

### Task 6: Prove synthetic GREEN and neighbour parity

**Files:**
- Create: patched runtime `qualification/synthetic-green/`
- Create: patched runtime `qualification/neighbour-parity/`

**Interfaces:**
- Consumes: stock/patched runtimes and Task 1 probe.
- Produces: GREEN slopes and exact `i`, `j`, `S` parity.

- [ ] **Step 1: Run patched synthetic GREEN.**

Use the exact 10,000/1,000/256 settings. Require status 0, version `1.2.0+omatfix1`, finite checksum, and every critical slope `<0.02`. Reject a patched median runtime more than 5% slower after three matched repetitions.

- [ ] **Step 2: Emit outputs for a fixed ASE corpus under both runtimes.**

```python
from ase.build import bulk, molecule

periodic = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
partial = molecule("H2O")
partial.set_cell([8.0, 9.0, 10.0])
partial.set_pbc([True, False, True])
nonperiodic = molecule("CH3OH")
nonperiodic.set_cell([12.0, 12.0, 12.0])
nonperiodic.set_pbc(False)
```

For cutoff 6.0, persist `neighbour_list("ijS", atoms, 6.0)` arrays/dtypes/shapes and exception class/message for invalid quantity `"x"`.

- [ ] **Step 3: Require exact parity.**

Use `np.array_equal` for all arrays and require exact ordering, dtype, shape, exception class, and exception message. Reject any difference.

---

### Task 7: Run the full real-OMat allocator GREEN gate

**Files:**
- Create: patched runtime `qualification/omat-mallinfo-10k/`

**Interfaces:**
- Consumes: `/data/Share/omat-aselmdb`, patched runtime, committed MACE overlay.
- Produces: four-worker events, metadata, summary, logs, stock comparison.

- [ ] **Step 1: Run the promoted probe outside the sandbox.**

Use `--root=/data/Share/omat-aselmdb --batches=10000 --warmup-batches=1000 --workers=4 --batch-size=256 --sample-every=256`.

- [ ] **Step 2: Verify worker and allocator bounds.**

Require four distinct worker files, no replacement PID, at least three post-warm-up samples/worker, 10,000 batches, no error signature, exact patched metadata, and mean `uordblks`, arena, and literal heap private-dirty slopes `<0.02`. Require absolute `fordblks` and `keepcost` slopes `<0.02`.

- [ ] **Step 3: Record stock/patched reduction.**

Compare with stock `1.630030`, `1.629867`, `1.629204`; every patched critical slope must be below 1.25% of stock.

---

### Task 8: Add fail-closed runtime admission and 10,000-update mode

**Files:**
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/runtime.py`
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer.py`
- Modify: `/home/bonan/work/train-downsample/scripts/run_single_layer_omat24.py`
- Modify: `/home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24.sbatch`
- Modify: `/home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_1x4090.sbatch`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_runtime.py`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer.py`

**Interfaces:**
- Produces: exact matscipy runtime admission.
- Produces: `build_matscipy_fix_qualification_arguments(...) -> list[str]`.
- Changes: host memory to 46,000 MB/rank and 92,000 MB/two-rank node.

- [ ] **Step 1: Write failing runtime/resource/budget tests.**

Extend qualified inventory with version/distribution `matscipy=1.2.0+omatfix1` and a matscipy module path below `RUNTIME_ROOT`. Add negative tests for stock/missing/outside-root matscipy. Require `cata_memory_mb_per_rank == 46_000`, total 92,000, `#SBATCH --mem=92000M`, and 4090 `#SBATCH --mem=46000M`. Require the new builder to emit exactly `--max_train_batches_per_epoch=10000`, batch 256, workers 4, CuEq/compile, no distributed/restart flags.

```python
def test_runtime_rejects_stock_or_outside_root_matscipy() -> None:
    stock = _qualified_inventory()
    stock["versions"]["matscipy"] = "1.2.0"
    stock["distributions"]["matscipy"] = "1.2.0"
    with pytest.raises(ValueError, match="matscipy"):
        validate_runtime_inventory(
            stock,
            runtime_root=RUNTIME_ROOT,
            mace_source_root=MACE_SOURCE_ROOT,
            project_root=PROJECT_ROOT,
            expected_gpu_count=2,
        )

    leaked = _qualified_inventory()
    leaked["module_paths"]["matscipy"] = "/tmp/matscipy/__init__.py"
    with pytest.raises(ValueError, match="matscipy import path"):
        validate_runtime_inventory(
            leaked,
            runtime_root=RUNTIME_ROOT,
            mace_source_root=MACE_SOURCE_ROOT,
            project_root=PROJECT_ROOT,
            expected_gpu_count=2,
        )


def test_matscipy_fix_qualification_is_exactly_10k_updates(tmp_path: Path) -> None:
    args = build_matscipy_fix_qualification_arguments(
        variant=VARIANTS["with-self"],
        train_root=Path("/train"),
        valid_root=Path("/valid"),
        statistics_file=Path("/stats.json"),
        run_root=tmp_path / "qualification",
    )
    assert "--max_train_batches_per_epoch=10000" in args
    assert "--batch_size=256" in args
    assert "--num_workers=4" in args
    assert "--enable_cueq=True" in args
    assert "--compile_training" in args
    assert "--distributed" not in args
    assert "--restart_latest" not in args
```

- [ ] **Step 2: Verify RED.**

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer_runtime.py tests/full_retrain/test_single_layer.py -q
```

Expected: failures for absent admission/builder and old memory values.

- [ ] **Step 3: Implement matscipy admission.**

Add matscipy exact version/distribution/module path to runtime inventory. Require `MATSCIPY_WHEEL_SHA256` in the sbatch launcher and exact equality with `$RUNTIME_ROOT/environment/matscipy-wheel.sha256` before training state is created.

```python
EXPECTED_VERSIONS["matscipy"] = "1.2.0+omatfix1"
DEPENDENCY_MODULES = (
    "torch", "e3nn", "cuequivariance", "cuequivariance_torch", "matscipy"
)

# collect_runtime_inventory()
import matscipy
inventory["versions"]["matscipy"] = metadata.version("matscipy")
inventory["module_paths"]["matscipy"] = matscipy.__file__
```

```bash
: "${MATSCIPY_WHEEL_SHA256:?MATSCIPY_WHEEL_SHA256 must be set}"
test -r "$RUNTIME_ROOT/environment/matscipy-wheel.sha256"
test "$(cut -d ' ' -f 1 "$RUNTIME_ROOT/environment/matscipy-wheel.sha256")" = "$MATSCIPY_WHEEL_SHA256"
```

- [ ] **Step 4: Implement qualification mode/resources.**

Add `matscipy_fix_qualification_updates: int = 10_000`; change rank memory to 46,000. Build a with-self one-GPU/no-DDP/no-restart mode from `_base_arguments` with batch 256/four workers and exact 10,000-update bound. Add CLI choice `matscipy-fix-qualification`. Change sbatch memory to 92,000/46,000 MB.

```python
def build_matscipy_fix_qualification_arguments(
    *,
    variant: SingleLayerVariant,
    train_root: Path,
    valid_root: Path,
    statistics_file: Path,
    run_root: Path,
) -> list[str]:
    if variant != VARIANTS["with-self"]:
        raise ValueError("matscipy-fix qualification requires with-self")
    arguments = _base_arguments(
        variant=variant,
        train_root=train_root,
        valid_root=valid_root,
        statistics_file=statistics_file,
        run_root=run_root,
        num_workers=4,
        batch_per_rank=256,
        max_num_epochs=1,
    )
    arguments.append("--max_train_batches_per_epoch=10000")
    return arguments
```

- [ ] **Step 5: Run complete gates.**

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer_runtime.py tests/full_retrain/test_single_layer.py tests/full_retrain/test_submit.py -q
.venv/bin/python -m pytest tests/full_retrain -q
bash -n scripts/cata/train_single_layer_omat24.sbatch
bash -n scripts/cata/train_single_layer_omat24_1x4090.sbatch
```

Expected: all pass.

- [ ] **Step 6: Freeze the unversioned operational delta.**

Because the project root is not Git, write unified diffs, before/after SHA-256s, test logs, and project-protocol hash into build provenance. Append the accepted decision to `docs/project/decisions.md`, allocator proof to `docs/project/findings.md`, and a new immutable entry to `docs/project/experiments.md`; never overwrite D037.

---

### Task 9: Complete the 10,000-update single-GPU qualification

**Files:**
- Create: patched runtime `qualification/login-4090-10k-with-self/`

**Interfaces:**
- Consumes: patched runtime, immutable overlays, real OMat, batch 256/four workers.
- Produces: exactly 10,000 finite updates, memory JSONL, diagnostics, model/checkpoint, verdict.

- [ ] **Step 1: Verify the login-node RTX 4090 is free.**

Require no compute process, less than 1 GiB used, and no competing MACE process. If occupied, do not share or kill it; wait/recheck.

- [ ] **Step 2: Stage immutable overlays and validate provenance.**

Stage the exact MACE commit and hashed project delta with no editable metadata. Run runtime admission for one `RTX 4090` and exact wheel hash before launch.

- [ ] **Step 3: Launch training and monitoring together.**

Use mode `matscipy-fix-qualification`, with-self, batch 256, workers 4, real train/validation/statistics, fresh run/cache roots, CuEq/compile, and no checkpoint reuse. Monitor the launcher descendant tree every 30 seconds and retain GPU telemetry until exit.

- [ ] **Step 4: Verify training and memory.**

Require exactly 10,000 optimizer updates, finite metrics, one graph, stable four spawned workers, checkpoint/model, and no traceback/OOM/killed/recompile/worker error. Require each post-warm-up worker heap slope `<0.02 GiB/10k updates`, aggregate heap slope `<0.08`, and no prior-style cgroup growth. Reject throughput more than 5% below the earlier 4090 batch-256/four-worker result unless three repeats attribute it to external load.

- [ ] **Step 5: Seal `qualification.json`.**

Include exact hashes/commands, update and finite-metric assertions, per-process slopes, throughput comparison, GPU peaks, graph count, and `passed=true`. Production is forbidden without an independently asserted passing file.

---

### Task 10: Relaunch and monitor paired two-rank production

**Files:**
- Create: fresh immutable deployment under `/scratch/bonan/train-downsample-single-layer/`
- Create: fresh with-self/without-self run and cache roots
- Append: `/home/bonan/work/train-downsample/docs/project/experiments.md`

**Interfaces:**
- Consumes: Task 9 passing verdict.
- Produces: paired job IDs/allocations, health evidence, first-epoch validation, MACE-OMat-0 comparison.

- [ ] **Step 1: Audit/cancel only superseded pending single-layer jobs.**

Resolve name, command, exported run root/variant/runtime, tasks, GPUs, CPU, memory via `scontrol`. Cancel only pending jobs belonging to superseded single-layer deployments. Record IDs/states. Never cancel `546570` or unrelated jobs.

- [ ] **Step 2: Submit both fresh variants with matched controls.**

Each uses one node, two tasks, two job-wide GPUs, eight CPUs/task, 92,000 MB/node, batch 256/rank, four workers/rank, target epoch 1, fresh initialization, patched runtime/wheel, isolated roots, job-specific port, and `srun --gpu-bind=none`.

- [ ] **Step 3: Verify requests and allocations immediately.**

Record job ID, purpose, requested tasks/GPUs/CPUs/memory/time, node, allocated GPU IDs, state/reason, runtime/wheel/deployment hashes, and run root. Require `ReqTRES`, `TresPerJob`, `MinMemoryNode`, task count, visible GPUs/model to match; cancel/resubmit mismatch before training state.

- [ ] **Step 4: Monitor through 10,000 updates/job.**

Require one graph/rank, finite loss, no recompile, bounded CUDA/host memory, four stable workers/rank, low loader wait, and bounded heap. Compare variants and prior batch-256 gates.

- [ ] **Step 5: Monitor first completed epoch and compare accuracy.**

Report energy MAE (meV/atom), force MAE (meV/Å), stress MAE (meV/Å³), weighted loss, epoch duration, throughput, peak memory, loader wait. Compare variants at matched counts with released MACE-OMat-0-medium `16.5122 / 73.4624 / 3.45276` and retained full-retrain curve.

- [ ] **Step 6: Continue restartable epochs 2–10.**

Submit one `--restart_latest` stage at a time, preserving hashes/resources. Require one new epoch/checkpoint each stage; stop/preserve evidence on non-finite metrics, renewed memory slope, runtime drift, or restart inconsistency.

- [ ] **Step 7: Append production evidence.**

Append IDs, allocations, commands/hashes, validation, throughput/memory, and checkpoint hashes to D037 as a dated superseding attempt; preserve all earlier job records and supersession reasons.

---

## Final Verification Checklist

- [ ] Stock 1.2.0 gives synthetic RED and retained real-OMat slope near 1.63 GiB/10k.
- [ ] Official source, exact patch/version, wheel/build/ELF provenance are sealed.
- [ ] Parent runtime is unchanged; patched runtime is dependency-only and passes exact-stack checks.
- [ ] Neighbour outputs/order/dtypes/shapes/exceptions are identical.
- [ ] Synthetic and real-OMat patched critical slopes are `<0.02 GiB/10k`.
- [ ] With-self qualification completes 10,000 updates with bounded heap, one graph, finite metrics, and no material slowdown.
- [ ] Both jobs use two ranks/GPUs, eight CPUs/rank, 46,000 MB/rank, batch 256, four workers, fresh state, and exact runtime admission.
- [ ] Completed validation is compared with both paired variant and MACE-OMat-0/full-retrain evidence; durable records are append-only.
