# OMat DataLoader Spawn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MACE DataLoaders with workers use `spawn` so a CUDA-initialized trainer is not forked into every input worker.

**Architecture:** Centralize worker kwargs in `mace.cli.run_train` and reuse them at every loader construction site. Preserve the zero-worker path exactly and validate memory behavior with a production-shaped real-OMat run.

**Tech Stack:** Python 3.12, PyTorch 2.11 DataLoader, pytest, OMat ASE-LMDB, Slurm/CUDA 13.

## Global Constraints

- Do not alter, cancel, or restart active production jobs during implementation.
- Use `uv` for Python dependency commands; GPU execution occurs outside the sandbox.
- GPU MACE runs must retain CuEquivariance.
- Change only DataLoader process creation in this iteration.

---

### Task 1: Specify worker process semantics

**Files:**
- Modify: `tests/unit/test_omat_lmdb_training_integration.py`
- Modify: `mace/cli/run_train.py`

**Interfaces:**
- Produces: `_data_loader_worker_kwargs(num_workers: int) -> dict`
- Behavior: returns `{"num_workers": 0}` for zero workers and `{"num_workers": n, "multiprocessing_context": "spawn"}` for positive workers.

- [ ] **Step 1: Write failing tests** asserting the helper contract and that the native balanced loader exposes the spawn start method.
- [ ] **Step 2: Run the focused tests** and confirm failure because the helper/context is absent.
- [ ] **Step 3: Implement the helper and route every `run_train.py` DataLoader through it.**
- [ ] **Step 4: Re-run focused tests** and confirm they pass.

### Task 2: Verify repository behavior

**Files:**
- Verify: `mace/cli/run_train.py`
- Verify: `tests/unit/test_omat_lmdb_training_integration.py`

- [ ] **Step 1: Run Python compilation and whitespace checks.**
- [ ] **Step 2: Run the complete focused native-OMat and DataLoader unit suites.**
- [ ] **Step 3: Run the full non-GPU test suite.**

### Task 3: Measure the real memory effect

**Files:**
- Read only: `/home/bonan/work/train-downsample`
- Create remotely: a fresh immutable deployment and isolated qualification output on Cata.

- [ ] **Step 1: Verify the login-node GPU and competing processes.**
- [ ] **Step 2: Stage the tested source without mutating prior deployments.**
- [ ] **Step 3: Run a batch-256/four-worker real-OMat spawn qualification with CuEquivariance and compiled training.**
- [ ] **Step 4: Compare cgroup anonymous/file memory, trainer and worker RSS/PSS, throughput, loader wait, graph count, and errors against the current fork evidence.**
- [ ] **Step 5: Record whether the result supports production deployment or requires the next isolated hypothesis.**

