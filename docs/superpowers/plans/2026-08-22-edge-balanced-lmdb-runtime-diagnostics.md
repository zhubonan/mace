# Edge-Balanced OMat LMDB Runtime Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact cutoff-specific edge-cost profiles, fixed-cardinality edge-and-atom-balanced native OMat batching, and opt-in production-safe training diagnostics without changing existing training or model artifacts.

**Architecture:** Keep official OMat LMDB shards and metadata immutable. A new cost-profile module validates and lazily loads one aligned `nedges` array per shard, a separate sampler consumes bounded `natoms + nedges` windows, and a separate diagnostics module owns buffered JSONL, sparse rank aggregation, and bounded deep-capture state. The existing atom-only sampler and diagnostics-off training path remain unchanged.

**Tech Stack:** Python 3.12, NumPy, LMDB, matscipy neighbour lists, PyTorch 2.11, PyTorch DataLoader/DDP/Profiler/NVTX, pytest.

## Global Constraints

- Preserve exactly `--batch_size` structures in every rank-local batch.
- Preserve deterministic, disjoint, without-replacement DDP coverage and equal optimizer-step counts.
- Keep official OMat LMDB shards and `metadata_db_*.npz` files unchanged.
- Store exact directed edge counts only; never store edge indices, vectors, shifts, or graph topology.
- Validate source manifest SHA-256, dataset identity, cutoff via `float.hex()`, neighbour semantics, shard ordering, and row counts before DataLoader construction.
- Retain the existing bounded shard-window and per-worker LMDB-handle limits.
- Keep `random` and `shard_window_atom_balanced` behavior unchanged.
- Add no required Python dependency; CUDA PyTorch supplies NVTX and PyTorch Profiler.
- `--training_diagnostics=off` creates no timers, files, profiler state, ranges, or collectives.
- Never place cost-profile or diagnostics state in checkpoints or the canonical single-file `.model` artifact.

---

### Task 1: Cost-profile schema and fail-closed validation

**Files:**
- Create: `mace/data/omat_cost_profile.py`
- Modify: `mace/data/omat_lmdb_dataset.py`
- Modify: `mace/data/__init__.py`
- Create: `tests/unit/test_omat_cost_profile.py`

**Interfaces:**
- Produces `OMAT_COST_PROFILE_FORMAT = "mace-omat-edge-cost-profile-v1"` and `OMAT_COST_PROFILE_MANIFEST = "omat-edge-cost-profile.json"`.
- Produces immutable `OMatCostShard(relative_shard_path: Path, cost_path: Path | None, rows: int, total_atoms: int, total_edges: int)`.
- Produces immutable `OMatCostProfile(root: Path, manifest_path: Path, profile_id: str, manifest_sha256: str, cutoff: float, shards: tuple[OMatCostShard, ...])` with `load_nedges(shard_index: int) -> np.ndarray`.
- Produces `load_omat_cost_profile(root: Path | str, index: OMatShardIndex, cutoff: float) -> OMatCostProfile`.
- Extends `OMatShardIndex` with immutable `manifest_path`, `manifest_sha256`, optional `cost_profile`, `with_cost_profile(...)`, and `load_nedges(...)`.

- [ ] **Step 1: Write failing schema tests** for a valid profile and missing/partial/stale/corrupt profiles.

```python
def test_cost_profile_validates_identity_contract_and_loads_lazily(profile_fixture):
    index, profile_root = profile_fixture(cutoff=5.0, nedges=[[2, 8], []])
    associated = index.with_cost_profile(profile_root, cutoff=5.0)
    assert associated.cost_profile.cutoff.hex() == float(5.0).hex()
    assert associated.load_nedges(0).tolist() == [2, 8]
    assert associated.load_nedges(1).dtype == np.int64

@pytest.mark.parametrize("mutation,match", [
    ("source_hash", "source manifest"),
    ("cutoff", "cutoff"),
    ("shard_path", "shard ordering"),
    ("rows", "row count"),
])
def test_cost_profile_rejects_incompatible_manifest(profile_fixture, mutation, match):
    index, profile_root = profile_fixture(mutation=mutation)
    with pytest.raises(ValueError, match=match):
        index.with_cost_profile(profile_root, cutoff=5.0)
```

- [ ] **Step 2: Run RED**.

Run: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -m pytest -q tests/unit/test_omat_cost_profile.py`

Expected: collection fails because `mace.data.omat_cost_profile` and `OMatShardIndex.with_cost_profile` do not exist.

- [ ] **Step 3: Implement immutable manifest/header validation**. Read only JSON plus `.npy` headers at startup; validate full array values only in `load_nedges`.

```python
@dataclass(frozen=True)
class OMatCostProfile:
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
            raw = payload["nedges"]
        _validate_cost_array(raw, shard)
        return np.asarray(raw, dtype=np.int64)
```

- [ ] **Step 4: Run GREEN and the existing dataset tests**.

Run: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -m pytest -q tests/unit/test_omat_cost_profile.py tests/unit/test_omat_lmdb_dataset.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 1 files**.

```bash
git add mace/data/omat_cost_profile.py mace/data/omat_lmdb_dataset.py mace/data/__init__.py tests/unit/test_omat_cost_profile.py
git commit -m "Add exact OMat edge cost profiles"
```

### Task 2: Atomic shard-parallel profile generation

**Files:**
- Modify: `mace/data/omat_cost_profile.py`
- Create: `mace/cli/omat_cost_profile.py`
- Modify: `setup.cfg`
- Modify: `tests/unit/test_omat_cost_profile.py`
- Create: `tests/unit/test_omat_cost_profile_cli.py`

**Interfaces:**
- Produces `generate_cost_shard(index: OMatShardIndex, profile_root: Path | str, cutoff: float, shard_index: int, *, force: bool = False) -> Path | None`.
- Produces `finalize_cost_profile(index: OMatShardIndex, profile_root: Path | str, cutoff: float) -> Path`.
- Adds `mace_omat_edge_profile = mace.cli.omat_cost_profile:main` with `--dataset-root`, `--manifest`, `--profile-root`, `--cutoff`, repeatable `--shard-index`, `--all-shards`, `--finalize`, and `--force`.

- [ ] **Step 1: Write failing exactness, atomicity, resume, and CLI tests**.

```python
def test_generate_cost_shard_matches_atomic_data_neighbour_count(real_omat_fixture):
    index, dataset = real_omat_fixture()
    path = generate_cost_shard(index, dataset.profile_root, 2.5, 0)
    with np.load(path, allow_pickle=False) as payload:
        observed = payload["nedges"]
    expected = np.asarray([dataset[i].edge_index.shape[1] for i in range(len(dataset))])
    np.testing.assert_array_equal(observed, expected)

def test_generate_cost_shard_resumes_only_valid_complete_results(real_omat_fixture):
    index, dataset = real_omat_fixture()
    first = generate_cost_shard(index, dataset.profile_root, 2.5, 0)
    mtime = first.stat().st_mtime_ns
    assert generate_cost_shard(index, dataset.profile_root, 2.5, 0) == first
    assert first.stat().st_mtime_ns == mtime
```

- [ ] **Step 2: Run RED** and confirm missing generation APIs.

- [ ] **Step 3: Implement direct row decoding, exact `get_neighborhood` counting, atomic `.npz` writes, validated resume, and final manifest publication**.

```python
edge_index, _, _, _ = get_neighborhood(
    configuration.positions,
    cutoff=cutoff,
    pbc=configuration.pbc,
    cell=configuration.cell,
    true_self_interaction=False,
)
nedges[local_index] = edge_index.shape[1]
```

Write shard results through a same-directory temporary file, `flush()`/`os.fsync()`, then `Path.replace()`. Publish the profile manifest only after every shard result validates.

- [ ] **Step 4: Run GREEN**.

Run: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -m pytest -q tests/unit/test_omat_cost_profile.py tests/unit/test_omat_cost_profile_cli.py`

Expected: PASS.

- [ ] **Step 5: Commit only Task 2 files**.

### Task 3: Fixed-cardinality edge-and-atom sampler

**Files:**
- Create: `mace/data/atom_edge_balanced_sampler.py`
- Modify: `mace/data/__init__.py`
- Create: `tests/unit/test_atom_edge_balanced_sampler.py`

**Interfaces:**
- Produces `_build_atom_edge_balanced_batches(indices, natoms, nedges, batch_size, rng) -> tuple[list[tuple[list[int], int, int]], np.ndarray, np.ndarray, np.ndarray]`.
- Produces `ShardWindowAtomEdgeBalancedBatchSampler(index, batch_size, *, seed=0, epoch=0, num_replicas=1, rank=0, window_shards=8)`.
- Produces `contract_payload() -> dict[str, object]` and `contract_digest() -> str` for cross-rank agreement.

- [ ] **Step 1: Write failing sampler tests** for cardinality, full usable coverage, disjoint ranks, equal steps, deterministic epoch order, bounded reservoirs, pack rotation, and adversarial atom/edge decorrelation.

```python
def test_edge_sampler_improves_edge_tail_without_losing_atom_safety(cost_index):
    atom_batches = collect(ShardWindowAtomBalancedBatchSampler(cost_index, 4, seed=9))
    edge_batches = collect(ShardWindowAtomEdgeBalancedBatchSampler(cost_index, 4, seed=9))
    assert max_spread(edge_batches, cost_index.load_nedges) < max_spread(
        atom_batches, cost_index.load_nedges
    )
    assert atom_spread(edge_batches) <= atom_spread(atom_batches) * 1.10
```

- [ ] **Step 2: Run RED** and confirm the new sampler is missing.

- [ ] **Step 3: Implement bounded two-cost planning and packing**. Partition planning shards lexicographically by normalized worst edge-mean deviation, atom-mean deviation, row imbalance, shard imbalance, and deterministic mask. Within each window, edge-sort rows into `batch_size` quantiles, assign each quantile once across all fixed-capacity bins against atom deficits, then sort candidate batches by `(total_edges, total_atoms)` before DDP pack formation.

- [ ] **Step 4: Run GREEN plus the unchanged atom-only suite**.

Run: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -m pytest -q tests/unit/test_atom_edge_balanced_sampler.py tests/unit/test_atom_balanced_sampler.py`

Expected: PASS and existing atom-only expectations remain unchanged.

- [ ] **Step 5: Commit only Task 3 files**.

### Task 4: CLI/DataLoader routing and distributed contract gate

**Files:**
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/cli/run_train.py`
- Modify: `mace/tools/train.py`
- Modify: `tests/unit/test_omat_lmdb_training_integration.py`
- Create: `tests/unit/test_edge_sampler_distributed_contract.py`

**Interfaces:**
- Adds `--batch_sampling=shard_window_atom_edge_balanced` and `--lmdb_cost_profile=PATH`.
- Produces `_verify_distributed_sampler_contract(train_sampler, *, distributed: bool) -> None` in `mace.tools.train`.
- The new mode requires one native direct OMat dataset and a matching profile; a profile path with any other sampling mode is rejected.

- [ ] **Step 1: Write failing routing/configuration tests**.

```python
def test_edge_sampling_associates_profile_before_loader(profile_dataset, training_args):
    dataset, profile = profile_dataset
    args = training_args(
        batch_sampling="shard_window_atom_edge_balanced",
        lmdb_cost_profile=str(profile),
    )
    _, loader, sampler = _build_train_loader_and_sampler(
        {"Default": dataset}, ["Default"], args, world_size=2, rank=1
    )
    assert isinstance(sampler, data.ShardWindowAtomEdgeBalancedBatchSampler)
    assert loader.batch_sampler is sampler

def test_profile_is_never_silently_ignored(training_args):
    with pytest.raises(ValueError, match="only valid with"):
        _build_train_loader_and_sampler(..., training_args(lmdb_cost_profile="profile"), ...)
```

- [ ] **Step 2: Run RED**.

- [ ] **Step 3: Implement parser, loader, and pre-step distributed digest agreement**. Use one fixed-size digest exchange only for the edge sampler; never run this collective for random or atom-only sampling.

- [ ] **Step 4: Run GREEN**.

Run: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -m pytest -q tests/unit/test_omat_lmdb_training_integration.py tests/unit/test_edge_sampler_distributed_contract.py`

- [ ] **Step 5: Commit only Task 4 files**.

### Task 5: Low-overhead runtime summaries

**Files:**
- Create: `mace/tools/training_diagnostics.py`
- Modify: `mace/tools/__init__.py`
- Modify: `mace/tools/train.py`
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/cli/run_train.py`
- Create: `tests/unit/test_training_diagnostics.py`

**Interfaces:**
- Produces immutable `TrainingDiagnosticsConfig(mode, interval, directory, start_step, steps, backend)`.
- Produces `TrainingDiagnostics(config, *, rank, world_size, device, distributed, clock=time.perf_counter)` with `prepare_step`, `range`, `record_step`, and `close`.
- Adds `--training_diagnostics=off|summary|deep`, `--training_diagnostics_interval=1000`, `--training_diagnostics_dir`, `--training_diagnostics_start_step`, `--training_diagnostics_steps`, and `--training_diagnostics_backend=nvtx|torch_profiler`.

- [ ] **Step 1: Write failing controlled-clock tests** for loader wait, complete period, H2D, training wall time, structures/atoms/actual edges, memory, compiler state, JSONL buffering, sparse aggregates, clean close, and disabled-path equivalence.

```python
def test_summary_records_exposed_wait_and_actual_batch_cost(tmp_path, fake_clock, batch):
    diagnostics = TrainingDiagnostics(summary_config(tmp_path), rank=0, world_size=1,
                                      device=torch.device("cpu"), distributed=False,
                                      clock=fake_clock)
    diagnostics.record_step(epoch=2, loader_wait_s=0.004, step_period_s=0.021,
                            h2d_wall_s=0.002, train_wall_s=0.015, batch=batch,
                            optimizer_enabled=True, ema_enabled=False,
                            compile_requested=True, compiled_active=True)
    diagnostics.close()
    record = read_rank_record(tmp_path)
    assert record["structures"] == batch.num_graphs
    assert record["atoms"] == batch.positions.shape[0]
    assert record["edges"] == batch.edge_index.shape[1]
```

- [ ] **Step 2: Run RED**.

- [ ] **Step 3: Implement the diagnostics module and a strict off-path branch**. Preserve the current `for batch in batch_iterator` and `take_step` call when diagnostics is `None`; use an explicit measured `next()` loop only when enabled. Buffer per-rank records and use one sparse fixed-shape tensor gather/reduction at configured intervals.

- [ ] **Step 4: Run GREEN plus train-loop regressions**.

Run: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -m pytest -q tests/unit/test_training_diagnostics.py tests/unit/test_omat_lmdb_training_integration.py tests/compile/test_compiled_training.py`

- [ ] **Step 5: Commit only Task 5 files**.

### Task 6: Bounded NVTX and PyTorch Profiler deep capture

**Files:**
- Modify: `mace/tools/training_diagnostics.py`
- Modify: `mace/tools/train.py`
- Modify: `tests/unit/test_training_diagnostics.py`

**Interfaces:**
- `TrainingDiagnostics.range(name)` emits NVTX push/pop only inside the configured capture window for backend `nvtx`, or `torch.profiler.record_function(name)` only while the built-in profiler is active.
- Built-in profiling writes one `rank-XXXXX-step-START-END.json` Chrome trace and stops after exactly `steps` optimizer steps.

- [ ] **Step 1: Write failing backend/window tests** using patched NVTX and profiler factories.

```python
def test_nvtx_ranges_are_bounded_to_declared_steps(deep_config, fake_nvtx):
    diagnostics = TrainingDiagnostics(deep_config(start=3, steps=2, backend="nvtx"), ...)
    run_synthetic_steps(diagnostics, count=7)
    assert fake_nvtx.labels == [
        "dataloader.next", "host_to_device", "forward_loss", "backward_ddp",
        "gradient_clipping", "optimizer", "ema",
    ] * 2
```

- [ ] **Step 2: Run RED**.

- [ ] **Step 3: Instrument only the enabled step path** with the approved ranges. Validate deep mode requires CUDA and a finite positive window; never instantiate NVTX and PyTorch Profiler together.

- [ ] **Step 4: Run GREEN**.

- [ ] **Step 5: Commit only Task 6 files**.

### Task 7: Compatibility, deterministic replay, and CPU distributed qualification

**Files:**
- Modify: `tests/unit/test_omat_cost_profile.py`
- Modify: `tests/unit/test_atom_edge_balanced_sampler.py`
- Modify: `tests/unit/test_omat_lmdb_training_integration.py`
- Modify: `tests/unit/test_training_diagnostics.py`
- Create: `tests/unit/test_training_diagnostics_distributed.py`
- Modify: `docs/superpowers/specs/2026-08-22-edge-balanced-lmdb-runtime-diagnostics-design.md`

**Interfaces:** No new public interfaces; this task closes compatibility and distributed gaps.

- [ ] **Step 1: Add failing regression tests** for resumed epoch determinism, empty shards, irreducible dense structures, profile relocation, diagnostics output failure, Gloo collective ordering, per-rank files, and checkpoint/model state exclusion.

- [ ] **Step 2: Run RED**, then make the smallest fixes needed for GREEN.

- [ ] **Step 3: Run the focused CPU qualification**.

Run: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -m pytest -q tests/unit/test_omat_lmdb_dataset.py tests/unit/test_omat_cost_profile.py tests/unit/test_omat_cost_profile_cli.py tests/unit/test_atom_balanced_sampler.py tests/unit/test_atom_edge_balanced_sampler.py tests/unit/test_omat_lmdb_training_integration.py tests/unit/test_edge_sampler_distributed_contract.py tests/unit/test_training_diagnostics.py tests/unit/test_training_diagnostics_distributed.py`

Expected: PASS with no CUDA execution.

- [ ] **Step 4: Review `git diff --check`, formatter/linter output, and default CLI help**.

- [ ] **Step 5: Commit the compatibility qualification**.

### Task 8: Real GPU/DDP and full-corpus acceptance gates

**Files:**
- Create only isolated profiling reports/scripts below `.profiling/mace-hdf5-gpu-2026-08-13/phase46/`; do not mix them into product commits unless explicitly requested.

**Interfaces:** Uses the product interfaces from Tasks 1–7 without adding checkpoint/model state.

- [ ] **Step 1: Generate and verify a small real OMat profile**; compare every sampled sidecar count to online `AtomicData` edges.
- [ ] **Step 2: Replay a complete epoch** for atom-only and edge-aware modes; report median/p95/p99/max batch atoms, edges, and simultaneous-rank spreads plus identical coverage/step counts.
- [ ] **Step 3: Run compiled four-rank worst-edge and densest-structure optimizer steps** outside the sandbox; require finite loss, no OOM, no eager fallback, and no graph growth.
- [ ] **Step 4: Compare diagnostics off/summary throughput** and correlate one bounded NVTX or PyTorch Profiler capture with reported loader wait.
- [ ] **Step 5: Run matched 1/2/4-GPU weak-scaling controls** with constant structures per rank; report per-GPU throughput relative to the one-GPU control. Do not call this strong scaling.
- [ ] **Step 6: Commit only reviewed product/test/docs changes**, preserving all pre-existing profiling edits and artifacts.

## Self-review record

- Spec coverage: every required profile, sampler, DDP, diagnostics, compatibility, and qualification behavior maps to Tasks 1–8.
- Placeholder scan: no `TBD`, `TODO`, “implement later,” unnamed error handling, or undefined neighboring interface remains.
- Type/signature consistency: `OMatShardIndex.with_cost_profile` produces the association consumed by `ShardWindowAtomEdgeBalancedBatchSampler`; `TrainingDiagnosticsConfig` is constructed by CLI routing and consumed only by `train`/`train_one_epoch`.
- Default-path audit: the atom-only sampler is not modified; diagnostics-off passes `None` and retains the existing unmeasured loop.
