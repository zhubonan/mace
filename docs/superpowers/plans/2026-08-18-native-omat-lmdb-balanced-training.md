# Native OMat LMDB Balanced Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train MACE directly from all official extracted OMat24 ASE-LMDB
shards through a lazy native reader and deterministic fixed-structure,
atom-balanced batching without repacking data or changing model files.

**Architecture:** `OMatLMDBDataset` discovers all manifest-listed or recursive
family shards, maps global indices through cumulative lengths, decodes official
zlib/orjson rows directly, and builds graphs online. A separate
`ShardWindowAtomBalancedBatchSampler` loads `natoms` for only eight shuffled
shards at a time, forms fixed-size batches, and aligns similar total-atom work
across DDP ranks while keeping LMDB access bounded and local.

**Tech Stack:** Python 3.12, PyTorch 2.11, PyTorch Dataset/Sampler/DataLoader,
LMDB, NumPy, orjson, MACE `AtomicData`, pytest, configargparse, DDP, and the
existing compiled full energy/force/stress training path.

## Global Constraints

- Use the unchanged official OMat24 `.aselmdb` and `metadata_db_*.npz` files;
  do not repack, merge, recompress, or store edges.
- Preserve HDF5 and generic ASE-LMDB defaults. The native reader is opt-in via
  `--lmdb_backend=omat_direct`.
- Preserve exactly `--batch_size` structures per rank; do not introduce an
  atom-budget or variable-structure batch.
- Use `natoms` only as the initial cost proxy; graph edges remain online.
- Preserve eager/compiled, single-device/DDP, checkpoint, optimizer, EMA, and
  legacy-compatible single-file `.model` behavior.
- Keep e3nn pinned to 0.4.4 and use the qualified PyTorch 2.11+cu128 runtime
  with `PYTHONPATH=/tmp/e3nn044-overlay:.` for CUDA qualification.
- Use `apply_patch` for repository file edits. Run multiprocessing and GPU
  commands outside the sandbox.
- Follow red-green-refactor for every production behavior: first observe the
  focused test fail for the intended reason, then write the minimum code.

---

### Task 1: Production multi-family shard index and manifest contract

**Files:**
- Create: `mace/data/omat_lmdb_dataset.py`
- Modify: `mace/data/__init__.py`
- Create: `tests/unit/test_omat_lmdb_dataset.py`

**Interfaces:**
- Produces `OMatShard`, `OMatShardIndex`, `OMAT_MANIFEST_NAME`,
  `build_omat_shard_index()`, and `write_omat_manifest()`.
- `OMatShardIndex.locate(index)` returns `(shard_index, local_index)`.
- `OMatShardIndex.load_natoms(shard_index)` returns a validated one-dimensional
  `np.ndarray[np.int64]` without caching all corpus sizes.
- Later tasks consume `OMatShardIndex` in both the dataset and balanced sampler.

- [ ] **Step 1: Write failing discovery and manifest tests**

Add helpers that create `family_a/db_0.aselmdb`, matching
`metadata_db_0.npz`, a second family, and one empty shard. The tests must state
the public ordering and manifest schema directly:

```python
def test_index_discovers_multiple_families_in_relative_lexical_order(tmp_path):
    write_metadata_shard(tmp_path / "family_b", "0", [8, 10])
    write_metadata_shard(tmp_path / "family_a", "1", [4])

    index = build_omat_shard_index(tmp_path)

    assert [shard.relative_path.as_posix() for shard in index.shards] == [
        "family_a/db_1.aselmdb",
        "family_b/db_0.aselmdb",
    ]
    assert len(index) == 3
    assert index.locate(0) == (0, 0)
    assert index.locate(2) == (1, 1)
    assert index.load_natoms(1).tolist() == [8, 10]


def test_manifest_round_trip_preserves_dataset_identity_and_counts(tmp_path):
    write_metadata_shard(tmp_path / "family", "0", [4, 8])
    discovered = build_omat_shard_index(tmp_path)
    manifest = write_omat_manifest(tmp_path, discovered, dataset_id="OMat24")
    loaded = build_omat_shard_index(tmp_path, manifest=manifest)

    assert manifest.name == OMAT_MANIFEST_NAME
    assert loaded.dataset_id == "OMat24"
    assert loaded.shards == discovered.shards
    assert len(loaded) == 2
```

Also test duplicate resolved roots, negative/out-of-range indices, missing
`natoms`, metadata length mismatch, a missing shard named by the manifest, and
missing metadata accepted only for an LMDB proven empty.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:. \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/unit/test_omat_lmdb_dataset.py -q
```

Expected: collection fails because `mace.data.omat_lmdb_dataset` does not
exist. This proves the test is exercising the new production module.

- [ ] **Step 3: Implement the immutable index and manifest schema**

Use these exact public types and keys. The shown methods are complete; the
discovery function constructs the same fields after validating roots or the
manifest:

```python
OMAT_MANIFEST_NAME = "omat24-manifest.json"
OMAT_MANIFEST_FORMAT = "mace-omat-aselmdb-manifest-v1"


@dataclass(frozen=True)
class OMatShard:
    family: str
    path: Path
    relative_path: Path
    metadata_path: Path | None
    length: int


@dataclass(frozen=True)
class OMatShardIndex:
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
            natoms = np.asarray(metadata["natoms"], dtype=np.int64)
        if natoms.ndim != 1 or len(natoms) != shard.length or np.any(natoms < 1):
            raise ValueError(f"Invalid natoms metadata for {shard.path}")
        return natoms
```

Define `build_omat_shard_index(roots, *, manifest=None)` and
`write_omat_manifest(root, index, *, dataset_id)` with the exact annotations
listed in the Interfaces block; do not add alternate constructors.

The JSON object must contain `format`, `dataset_id`, `total_rows`, and a
`shards` list with `family`, POSIX `path`, optional POSIX `metadata`, and
`rows`. Write through a sibling temporary file and `Path.replace()` so an
interrupted process never leaves a partial manifest. Reject absolute or
parent-traversing manifest paths. Resolve and deduplicate discovered shards.
For an explicit root list, use each root directory name as its family and
reject duplicate family names. For a common root, use each first-level
relative directory as the family.
Use metadata lengths without loading all arrays simultaneously; validate
`load_natoms()` dtype, rank, strictly positive values, and exact shard length.

- [ ] **Step 4: Export the new index API and verify GREEN**

Add the five public names to `mace/data/__init__.py`, then rerun the focused
test. Expected: all Task-1 tests pass.

- [ ] **Step 5: Run compatibility tests and commit**

Run:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:. \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/unit/test_omat_lmdb_dataset.py \
  tests/unit/test_lmdb_database.py tests/unit/test_data.py -q
git diff --check
```

Expected: all selected tests pass and no whitespace errors. Commit:

```bash
git add mace/data/omat_lmdb_dataset.py mace/data/__init__.py \
  tests/unit/test_omat_lmdb_dataset.py
git commit -m "Add native OMat LMDB shard index"
```

---

### Task 2: Direct official-row dataset with process-local handle ownership

**Files:**
- Modify: `mace/data/omat_lmdb_dataset.py`
- Modify: `tests/unit/test_omat_lmdb_dataset.py`

**Interfaces:**
- Produces `decode_omat_row(value, *, head)` and `OMatLMDBDataset`.
- `OMatLMDBDataset.shard_index` exposes the immutable Task-1 index.
- `read_configuration(index)` is the edge-free boundary used in parity tests.
- `__getitem__(index)` returns ordinary MACE `AtomicData` with online edges.

- [ ] **Step 1: Write failing row, parity, LRU, and ownership tests**

Write official-format test rows as zlib-compressed orjson objects. Assert the
active E/F/S contract and the absence of ASE reconstruction:

```python
def test_direct_dataset_decodes_configuration_and_builds_graph(tmp_path):
    write_official_shard(tmp_path / "family", "0", atom_counts=[2])
    dataset = OMatLMDBDataset(
        tmp_path, r_max=3.0, z_table=AtomicNumberTable([1, 8])
    )

    config = dataset.read_configuration(0)
    item = dataset[0]

    assert config.atomic_numbers.tolist() == [1, 8]
    assert set(config.properties) == {"energy", "forces", "stress"}
    assert item.positions.shape == (2, 3)
    assert item.edge_index.shape[0] == 2
```

Add strict expected-versus-actual `AtomicData` equality using an independently
constructed `Configuration`. Add tests for missing required fields, weight and
head propagation, numeric key `local_index + 1`, missing rows, LRU eviction,
idempotent `close()`, pickle reset, and real `fork`/`spawn` children reporting
zero inherited active handles before their first access.

- [ ] **Step 2: Run the new cases and verify RED**

Expected failure: `OMatLMDBDataset` and `decode_omat_row` are not defined. Run
real process tests outside the sandbox.

- [ ] **Step 3: Implement direct decoding and the bounded LRU**

Implement the constructor and state contract listed in the Interfaces block
with these exact arguments: `roots`, `r_max`, `z_table`, optional `heads`,
`head="Default"`, `max_open_shards=8`, optional `manifest`, and optional
`transform`. The hot path must be exactly:

```python
def __getitem__(self, index: int) -> AtomicData:
    item = AtomicData.from_config(
        self.read_configuration(index),
        z_table=self.z_table,
        cutoff=self.r_max,
        heads=list(self.heads),
    )
    return self.transform(item) if self.transform is not None else item
```

Use `readonly=True`, `lock=False`, `readahead=True`, `meminit=False`, and
`max_readers=1024`. Store `(transaction, environment)` in an `OrderedDict`
owned by the current PID. Abort transactions before closing environments.
Clear ownership in `__getstate__`; on PID change close inherited handles and
reset counters. Do not catch row errors and return `None`: raise an error that
includes family, shard, local row, and numeric key.

Decode `numbers`, `positions`, `cell`, `pbc`, `energy`, `forces`, and `stress`
directly. Apply the Phase-42 `config_*_weight`, `config_weight`, and
`config_type` defaults. Construct graphs only with the complete
`AtomicData.from_config` call shown above.

- [ ] **Step 4: Verify GREEN, live pilot parity, and commit**

Run the focused tests outside the sandbox, then run a live 1,000-record plus
batch-128 parity check against the immutable HDF5 pilot using the new
production class. Expected: tensor-exact parity and source hashes unchanged.

Commit:

```bash
git add mace/data/omat_lmdb_dataset.py tests/unit/test_omat_lmdb_dataset.py
git commit -m "Read official OMat LMDB rows directly"
```

---

### Task 3: Bounded fixed-structure atom-balanced DDP sampler

**Files:**
- Create: `mace/data/atom_balanced_sampler.py`
- Modify: `mace/data/__init__.py`
- Create: `tests/unit/test_atom_balanced_sampler.py`

**Interfaces:**
- Produces `ShardWindowAtomBalancedBatchSampler`.
- Consumes `OMatShardIndex.load_natoms()` and global shard offsets.
- Exposes `set_epoch(epoch)`, `__iter__()`, and exact rank-local `__len__()`.

- [ ] **Step 1: Write failing invariant tests**

Use small synthetic shards with deliberately skewed sizes. Cover fixed batch
length, determinism, changed epoch order, no duplication, DDP disjointness,
equal steps, global drop bound, window carry, and atom balance:

```python
def test_sampler_preserves_fixed_batches_and_ddp_coverage(index):
    samplers = [
        ShardWindowAtomBalancedBatchSampler(
            index,
            batch_size=4,
            seed=17,
            num_replicas=2,
            rank=rank,
            window_shards=2,
        )
        for rank in range(2)
    ]
    batches = [list(sampler) for sampler in samplers]

    assert len(batches[0]) == len(batches[1]) == len(samplers[0])
    assert all(len(batch) == 4 for rank_batches in batches for batch in rank_batches)
    rank_sets = [{item for batch in rank_batches for item in batch} for rank_batches in batches]
    assert rank_sets[0].isdisjoint(rank_sets[1])
    used = len(rank_sets[0] | rank_sets[1])
    assert 0 <= len(index) - used < 2 * 4


def test_sampler_reduces_simultaneous_rank_atom_imbalance(skewed_index):
    balanced = step_atom_totals(make_rank_samplers(skewed_index, balanced=True))
    random = step_atom_totals(make_rank_samplers(skewed_index, balanced=False))
    assert percentile_relative_rank_imbalance(balanced, 0.95) < \
        percentile_relative_rank_imbalance(random, 0.95)
```

Add a test that monkeypatches `load_natoms()` to track active calls and proves
the iterator never loads more than `window_shards` new shard arrays at once.
Add one DataLoader test showing a batch collates through the PyG loader.

- [ ] **Step 2: Run the sampler tests and verify RED**

Expected: import failure for `mace.data.atom_balanced_sampler`.

- [ ] **Step 3: Implement the streaming epoch algorithm**

Implement the exact public constructor
`ShardWindowAtomBalancedBatchSampler(index, batch_size, *, seed=0, epoch=0,
num_replicas=1, rank=0, window_shards=8)`. Validate all positive/rank bounds
before computing these complete length fields:

```python
full_batches = len(index) // batch_size
usable_batches = full_batches - full_batches % num_replicas
rank_batches = usable_batches // num_replicas
```

Store those values and implement:

```python
def set_epoch(self, epoch: int) -> None:
    self.epoch = int(epoch)


def __len__(self) -> int:
    return self._rank_batches
```

For each epoch, seed NumPy from `[seed, epoch]`, shuffle non-empty shard IDs,
and process `window_shards` at a time. Load only that window's sizes, combine
the prior row tail, apply a seeded random tie-break before stable size sorting,
and create complete fixed-size batches. Carry the final incomplete row batch.

Compute total atoms for every complete batch. Combine up to
`num_replicas - 1` prior complete pack-tail batches, sort by total atoms, form
adjacent `num_replicas` packs, permute rank assignment inside each pack, and
shuffle pack order within the scheduling region. Each rank yields its assigned
batch. At epoch end, discard only the incomplete row tail and incomplete DDP
pack. Assert the yielded count equals `__len__()`.

- [ ] **Step 4: Verify GREEN and stress deterministic coverage**

Run the focused sampler suite with multiple seeds, epochs, ranks, batch sizes,
empty shards, negative/zero validation, and a synthetic index large enough to
expose accidental corpus-sized Python lists. Expected: all tests pass and
maximum retained memory scales with the configured window.

- [ ] **Step 5: Run dataset/sampler regression tests and commit**

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:. \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/unit/test_atom_balanced_sampler.py \
  tests/unit/test_omat_lmdb_dataset.py tests/unit/test_data.py -q
git diff --check
git add mace/data/atom_balanced_sampler.py mace/data/__init__.py \
  tests/unit/test_atom_balanced_sampler.py
git commit -m "Add atom-balanced OMat batch sampler"
```

---

### Task 4: Opt-in CLI, dataset dispatch, and training-loop integration

**Files:**
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/tools/run_train_utils.py`
- Modify: `mace/cli/run_train.py`
- Modify: `mace/tools/train.py`
- Modify: `tests/unit/test_arg_parser.py`
- Modify: `tests/unit/test_lmdb_database.py`
- Create: `tests/unit/test_omat_lmdb_training_integration.py`

**Interfaces:**
- Adds CLI/config keys `lmdb_backend`, `batch_sampling`,
  `lmdb_shard_window`, and `lmdb_max_open_shards`.
- `load_dataset_for_path` with `lmdb_backend="ase"` and
  lmdb_max_open_shards=8)` dispatches native OMat input explicitly.
- The object passed as `train_sampler` may be a sample sampler or batch sampler
  but must provide `set_epoch()` when it controls epoch order.

- [ ] **Step 1: Write failing parser and dispatch tests**

Extend the critical default test:

```python
assert args.lmdb_backend == "ase"
assert args.batch_sampling == "random"
assert args.lmdb_shard_window == 8
assert args.lmdb_max_open_shards == 8
```

Parse and assert an explicit native configuration. Add bad-choice and
non-positive integer cases. In `test_lmdb_database.py`, monkeypatch the native
constructor and prove `load_dataset_for_path()` selects it only for
`lmdb_backend="omat_direct"`; the default must still construct
`LMDBDataset`.

- [ ] **Step 2: Write failing DataLoader and epoch tests**

Create a two-family native dataset. Exercise single-rank and two-rank loader
construction and assert:

```python
assert train_loader.batch_sampler is train_sampler
assert train_loader.batch_size is None
assert all(batch.num_graphs == requested_batch_size for batch in train_loader)
```

Patch a sampler with a recording `set_epoch()` and run two minimal epochs
through `tools.train`; assert calls are `[start_epoch, start_epoch + 1]` in
single-device and distributed modes. Assert clear startup errors for balanced
sampling with generic ASE-LMDB, multiple training heads, multiple combined
datasets, LBFGS/non-drop-last semantics, or magnetic augmentation.

- [ ] **Step 3: Run focused tests and verify RED**

Expected failures: parser attributes are absent, dispatch ignores the backend,
and the current DataLoader supplies ordinary `batch_size`/`sampler` arguments.

- [ ] **Step 4: Add exact opt-in CLI arguments**

Place data-backend options beside `--num_workers`:

```python
parser.add_argument(
    "--lmdb_backend",
    choices=("ase", "omat_direct"),
    default="ase",
)
parser.add_argument(
    "--batch_sampling",
    choices=("random", "shard_window_atom_balanced"),
    default="random",
)
parser.add_argument("--lmdb_shard_window", type=positive_int, default=8)
parser.add_argument("--lmdb_max_open_shards", type=positive_int, default=8)
```

Add a tested `positive_int()` argparse converter rather than accepting zero
and failing after dataset construction.

- [ ] **Step 5: Implement native dispatch and loader construction**

Pass backend/max-handle options to every `load_dataset_for_path()` call. For
native mode, construct `data.OMatLMDBDataset` with the head, heads, cutoff,
element table, and explicit handle bound.

In `run_train.py`, require one head and one `OMatLMDBDataset` when balanced
sampling is selected. Construct the batch sampler with rank/world size even
for world size one. Build the PyG DataLoader in mutually exclusive branches:

```python
if balanced_batch_sampler is not None:
    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=balanced_batch_sampler,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_sampler = balanced_batch_sampler
else:
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=train_sampler is None and not args.lbfgs,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
```

Change the epoch hook in `tools.train` from `if distributed` to
`if train_sampler is not None and hasattr(train_sampler, "set_epoch")` so
single-device balanced order changes each epoch. Keep barriers conditioned on
distributed training.

- [ ] **Step 6: Verify GREEN and full CPU compatibility**

Run the new integration tests, argument tests, generic LMDB tests, HDF5 data
tests, multi-file workflow tests, and compile-training CPU tests. Expected:
native/balanced mode passes while every existing default assertion remains
unchanged.

- [ ] **Step 7: Commit the opt-in integration**

```bash
git add mace/tools/arg_parser.py mace/tools/run_train_utils.py \
  mace/cli/run_train.py mace/tools/train.py tests/unit/test_arg_parser.py \
  tests/unit/test_lmdb_database.py \
  tests/unit/test_omat_lmdb_training_integration.py
git commit -m "Integrate native OMat LMDB training"
```

---

### Task 5: Stable dataset relocation and reproducible manifest generation

**Files:**
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/prepare_omat24_root.py`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/tests/test_prepare_omat24_root.py`
- Create at execution time: `/mnt/hdd2/bonan/datasets/omat24-aselmdb/omat24-manifest.json`

**Interfaces:**
- The script defaults to audit-only and requires `--apply` for mutation.
- It consumes Task-1 discovery/manifest functions.
- It performs a same-filesystem rename and creates the old-path compatibility
  symlink only after all preflight gates pass.

- [ ] **Step 1: Write failing dry-run, apply, rollback, and refusal tests**

Use temporary directories and assert:

```python
def test_apply_renames_source_and_leaves_compatibility_symlink(tmp_path):
    source, target, link = make_fake_corpus(tmp_path)
    result = prepare_root(source, target, link, apply=True)
    assert result.shards == 2
    assert target.is_dir()
    assert link.is_symlink()
    assert link.resolve() == target.resolve()
    assert (target / OMAT_MANIFEST_NAME).is_file()
```

Refuse a non-empty target, a source symlink, cross-filesystem device IDs,
unexpected row/shard counts, an existing incompatible link, or a target below
the source. Inject symlink failure and assert the source directory is renamed
back. Dry-run must create nothing.

- [ ] **Step 2: Verify RED and implement the explicit relocation utility**

Expose `prepare_root(source: Path, target: Path, compatibility_link: Path, *,
expected_rows: int, expected_shards: int, apply: bool) -> PreparationResult`.
`PreparationResult` is a frozen dataclass containing `source`, `target`,
`compatibility_link`, `families`, `shards`, `rows`, `logical_bytes`,
`same_filesystem`, and `applied`.

Preflight discovery and counts before mutation. Create the target parent only
in apply mode. Use `source.rename(target)` on the same `st_dev`; create a
relative symlink from the old path to the target. On any post-rename error,
remove only the just-created compatibility link and manifest, then rename the
target back. Generate the manifest atomically in the final root and re-open it
through the production reader before reporting success.

- [ ] **Step 3: Run tests and an audit-only command**

Run the focused script tests, then outside the sandbox run:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:. \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  .profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/prepare_omat24_root.py \
  --source /mnt/hdd2/bonan/omat-descriptors/.staging \
  --target /mnt/hdd2/bonan/datasets/omat24-aselmdb \
  --compatibility-link /mnt/hdd2/bonan/omat-descriptors/.staging \
  --expected-rows 100824585 --expected-shards 5300
```

Expected: audit-only exit 0, 11 families, 5,300 shards, 100,824,585 rows, same
filesystem, no target/link mutation.

- [ ] **Step 4: Apply the approved move and verify source identity**

Repeat the exact command with `--apply` outside the sandbox. Then verify both
paths resolve to the target, `du -sb` is unchanged within manifest-file size,
the manifest reopens, the Phase-42 representative HDF5/LMDB hashes remain
unchanged, and the old-path live parity verifier still passes through the
compatibility symlink.

- [ ] **Step 5: Commit only the reproducible utility and tests**

Do not add the external dataset or generated manifest to Git.

```bash
git add .profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/prepare_omat24_root.py \
  .profiling/mace-hdf5-gpu-2026-08-13/phase44/tests/test_prepare_omat24_root.py
git commit -m "Add reproducible OMat LMDB relocation audit"
```

---

### Task 6: Complete-corpus sampler and multiworker CPU qualification

**Files:**
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/audit_balanced_sampler.py`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/benchmark_native_omat_lmdb.py`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/tests/test_phase44_analysis.py`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/REPORT.md`

**Interfaces:**
- Sampler audit emits JSON with exact coverage/drop, rank steps, atom-balance,
  memory, and deterministic checksums.
- CPU benchmark emits startup, item, DataLoader, RSS/FD, shard-open, and
  locality metrics for 0/8/16 workers.

- [ ] **Step 1: Write failing analyzer tests against tiny fixtures**

Require stable JSON fields:

```python
assert summary["rows"] == expected_rows
assert summary["used"] + summary["dropped"] == expected_rows
assert summary["dropped"] < world_size * batch_size
assert len(set(summary["rank_steps"])) == 1
assert summary["balanced_p95_rank_imbalance"] < \
    summary["random_p95_rank_imbalance"]
assert summary["peak_window_rows"] <= summary["largest_window_rows"]
```

The CPU analyzer must reject unequal sampled-index streams, non-finite rates,
unbounded active environments, or fewer than three order-balanced runs.

- [ ] **Step 2: Implement bounded analyzers and verify fixture GREEN**

Prove exact sample coverage with a verifier-only bitset stored in a temporary
`np.memmap` file of `ceil(rows / 8)` bytes. Reject a bit already set as a
duplicate, then assert every expected retained bit and only the allowed final
drop remain. This disk-backed verifier is not part of sampler memory. Also
record process RSS before/after and sampler peak arrays. Compare standard
distributed random batches against balanced batches using the identical
metadata and seed.

- [ ] **Step 3: Run the complete metadata-only audit**

Outside the sandbox, audit world sizes one and four, batch size 128, window
sizes 4/8/16, and at least two epochs. Required gates:

- exact deterministic coverage within the global drop bound;
- equal steps and disjoint ranks;
- lower p95 simultaneous-rank relative atom imbalance than random;
- memory bounded by the largest active window; and
- no corpus-sized Python index list.

- [ ] **Step 4: Run multi-family CPU and worker benchmarks**

Use a reproducible family/shard selection large enough to cross window
boundaries and include size skew. Run three alternating repetitions at
0/8/16 workers, record active/open environment counts per worker, and compare
random versus balanced scheduling. All multiprocessing commands run outside
the sandbox.

- [ ] **Step 5: Write the measured report and commit compact evidence**

Report scientific scope, exact commands, complete-corpus coverage, imbalance,
memory, throughput, and limitations. Retain scripts, tests, compact JSON, and
the report; exclude bulky raw traces and runtime environments.

```bash
git add .profiling/mace-hdf5-gpu-2026-08-13/phase44
git commit -m "Qualify native OMat LMDB input pipeline"
```

---

### Task 7: Local GPU, four-rank production, checkpoint, and model-format gates

**Files:**
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/run_gpu_matrix.sh`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/verify_phase44.py`
- Modify: `.profiling/mace-hdf5-gpu-2026-08-13/phase44/REPORT.md`
- Modify only if a regression is found: relevant production source/tests from
  Tasks 1--4, using a new red-green cycle.

**Interfaces:**
- Reuses the qualified compiled full energy/force/stress benchmark and DDP
  wrapper; does not introduce a second training implementation.
- Final verifier fails closed on parity, fallback, unstable graphs, sampler
  disagreement, incomplete steps, checksum mismatch, or source mutation.

- [ ] **Step 1: Write the fail-closed artifact verifier first**

Require exact source revision/runtime identity and assert:

```python
assert run["lmdb_backend"] == "omat_direct"
assert run["batch_sampling"] == "shard_window_atom_balanced"
assert run["batch_size_per_rank"] == 128
assert run["is_compiled"] is True
assert run["compilation_error"] is None
assert run["dynamo_counters_after_warmup"] == run["dynamo_counters_final"]
assert run["sampler"]["equal_rank_steps"] is True
assert run["sampler"]["disjoint_rank_samples"] is True
assert all(math.isfinite(loss) for loss in run["losses"])
```

Also gate tensor-exact input parity on the matched pilot, optimizer checksum
agreement across ranks, worker handle bounds, checkpoint reload, and absence
of `module.`/`base_model.` prefixes in the final `.model` owner.

- [ ] **Step 2: Run matched local RTX 5090 controls**

Outside the sandbox, run three order-balanced repetitions for:

1. HDF5 plus current distributed/random batching;
2. native LMDB plus random batching; and
3. native LMDB plus shard-window atom-balanced batching.

Use IEEE FP32, batch 128, 8 and 16 workers, production hybrid model,
four warmups, at least 256 timed variable-shape steps, and the same selected
sample stream where comparison permits. Record batch atom/edge counts, fetch
wait, optimizer time, end-to-end period, GPU memory, power/SM telemetry,
compilation state, and loss.

- [ ] **Step 3: Enforce the local acceptance gates**

Native input must retain exact input parity. Balanced native throughput must
not be more than 2% below native random throughput, and its p95 rank/work
imbalance proxy must improve. Compilation must remain one stable graph without
fallback. If any gate fails, diagnose with the systematic-debugging workflow
and add a failing regression before modifying source.

- [ ] **Step 4: Run strict two-rank then four-rank qualification**

Use the exact committed bundle on the remote RTX 5090 machine. First require a
two-rank pass, then run four ranks against multiple OMat families with
batch 128 per rank. Record global sample coverage, rank atom/edge totals,
collective wait, step latency, structures/s, loss parity, parameter checksums,
peak memory, file descriptors, worker RSS, active LMDB handles, Dynamo graph
count, and checkpoint/restart behavior. Do not touch an existing production
job until both gates pass.

- [ ] **Step 5: Verify legacy-compatible final model output**

Complete at least one checkpoint/save cycle through compiled DDP. Load the
resulting single `.model` in the e3nn 0.4.4 compatibility runtime, reject
compiled/DDP state prefixes, and run finite energy/force/virial/stress
inference. Compare ownership and state keys with the existing export
regression.

- [ ] **Step 6: Run final source and artifact verification**

Run focused tests, the full relevant CPU suite, real fork/spawn tests, live
parity, Python compilation, shell syntax, `git diff --check`, and
`verify_phase44.py`. Inspect the complete diff against the pre-feature commit
and request code review before a production recommendation.

- [ ] **Step 7: Update the report and commit the qualification milestone**

State exact gains and limitations without extrapolating single-shard results
to the full corpus. If all gates pass, recommend the opt-in native/balanced
path for OMat24 training while retaining existing defaults.

```bash
git add mace tests .profiling/mace-hdf5-gpu-2026-08-13/phase44
git commit -m "Qualify native OMat LMDB training"
```
