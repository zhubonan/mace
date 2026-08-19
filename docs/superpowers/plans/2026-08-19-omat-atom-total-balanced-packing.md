# OMat Atom-Total Balanced Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent rare OMat LMDB training OOMs by balancing total atoms among
fixed-cardinality rank batches before DDP step assignment.

**Architecture:** Keep the existing deterministic shuffled shard windows and
bounded row/DDP carries. Replace contiguous sorted slicing with vectorized
serpentine stripes so every batch samples the active window's atom-count
distribution; retain total-atom sorting only for grouping similarly weighted
batches into DDP steps.

**Tech Stack:** Python 3.12, NumPy, PyTorch `Sampler`, pytest, and the existing
OMat shard metadata/index.

## Global Constraints

- Every rank batch contains exactly `batch_size` structures.
- Balance only by `natoms`; do not add edge metadata or change graph building.
- Preserve deterministic `(seed, epoch)` behavior, disjoint ranks, equal rank
  steps, and epoch coverage with fewer than `batch_size * world_size` drops.
- Preserve the `window_shards` metadata bound and bounded row/DDP carry.
- Do not change the CLI, model, loss, optimizer, checkpoint, or `.model` format.
- Use red-green-refactor and preserve unrelated dirty-worktree changes.

---

### Task 1: Reproduce homogeneous-batch concentration

**Files:**
- Modify: `tests/unit/test_atom_balanced_sampler.py`

**Interfaces:**
- Consumes `ShardWindowAtomBalancedBatchSampler` as currently exposed.
- Produces an atom-total regression test independent of batch emission order.

- [ ] **Step 1: Add the failing regression test**

```python
def test_sampler_balances_total_atoms_between_batches(tmp_path):
    sizes = list(range(1, 65))
    index = make_index(tmp_path, [sizes])
    samplers = make_rank_samplers(index, window_shards=1)
    rank_batches = [list(sampler) for sampler in samplers]
    totals = [
        sum(sizes[item] for item in batch)
        for batches in rank_batches
        for batch in batches
    ]
    assert totals == [130] * 16
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:. \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/unit/test_atom_balanced_sampler.py::test_sampler_balances_total_atoms_between_batches -q
```

Expected: failure because contiguous sorted batches range from 10 to 250 atoms
instead of all totaling 130.

### Task 2: Pack fixed-cardinality batches by atom sum

**Files:**
- Modify: `mace/data/atom_balanced_sampler.py`
- Test: `tests/unit/test_atom_balanced_sampler.py`

**Interfaces:**
- Add private `_serpentine_batches(indices, sizes, batch_size)` returning
  `(list[tuple[list[int], int]], tail_indices, tail_sizes)`.
- Keep the public sampler constructor and yielded `list[int]` API unchanged.

- [ ] **Step 1: Select the bounded carry before sorting**

Use the sampler RNG to choose `len(indices) % batch_size` carried positions.
Sort only complete rows by `(natoms, random tie-break)` so dropped end-of-epoch
rows are not systematically the largest structures.

- [ ] **Step 2: Implement vectorized serpentine stripes**

```python
batch_count = len(complete_indices) // batch_size
striped_indices = complete_indices.reshape(batch_size, batch_count).copy()
striped_sizes = complete_sizes.reshape(batch_size, batch_count).copy()
striped_indices[1::2] = striped_indices[1::2, ::-1]
striped_sizes[1::2] = striped_sizes[1::2, ::-1]
batches = striped_indices.T
batch_totals = striped_sizes.sum(axis=0)
```

Convert only emitted rows to Python lists. Do not retain a corpus-sized index.

- [ ] **Step 3: Run focused and full sampler tests and verify GREEN**

Run the regression alone, then the entire sampler test file. Expected: all
tests pass and `peak_window_shards`/`peak_window_rows` remain within bounds.

### Task 3: Qualify distributions and integration

**Files:**
- Modify only if a discovered regression requires it:
  `tests/unit/test_atom_balanced_sampler.py`

**Interfaces:**
- No new public interface.

- [ ] **Step 1: Run native-LMDB integration regressions**

Run unit tests for the sampler, OMat dataset, and OMat training integration.
Expected: zero failures.

- [ ] **Step 2: Audit one complete OMat metadata epoch**

Replay seed 20260729, epoch 4, batch 64, four ranks, and eight-shard windows.
Record median/p95/p99/maximum batch atoms, simultaneous-rank spread, coverage,
drop count, and `peak_window_shards`. Confirm step 1,915 no longer produces the
5,376--5,632-atom homogeneous batches.

- [ ] **Step 3: Review before remote training**

Inspect the focused diff and leave production resubmission blocked until a
four-rank compiled GPU preflight replays the known fatal step plus the largest
predicted batches without fallback, graph growth, non-finite loss, or OOM.
