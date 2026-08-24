# Dedicated CUDA 13 MACE Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the current one-layer OMat24 throughput, build a dependency-only CUDA 13 MACE runtime, qualify batch/DDP performance, and launch restartable paired training jobs.

**Architecture:** Preserve current telemetry before stopping the invalid e3nn-0.6 jobs. Build an immutable uv-managed dependency runtime with exact CUDA-13 pins and no MACE source, inject immutable MACE/project sources through `PYTHONPATH`, qualify one- and two-GPU paths, then launch one-epoch restartable stages.

**Tech Stack:** Bash, uv, Python 3.12.11, PyTorch 2.11.0+cu130, e3nn 0.4.4, CuEquivariance 0.11.0, Slurm, RTX 5090, native OMat LMDB.

## Global Constraints

- Runtime contains dependencies only; MACE and project sources come from `PYTHONPATH`.
- Pin e3nn 0.4.4, torch 2.11.0+cu130, cuequivariance 0.11.0, cuequivariance-torch 0.11.0, and cuequivariance-ops-torch-cu13 0.10.0.
- Use `uv`; run GPU and environment installation commands outside the sandbox.
- Begin each paired production job with two GPUs.
- Use fresh initialization and fresh run/cache roots; never resume from e3nn-0.6 jobs.
- Run batch candidates 16, 32, and 64 per rank with workers 4 before production selection.

---

### Task 1: Audit and retire superseded jobs

**Files:**
- Create remotely: `/scratch/bonan/train-downsample-single-layer/superseded-e3nn060-20260824/audit.json`
- Create remotely: `/scratch/bonan/train-downsample-single-layer/superseded-e3nn060-20260824/resolved/`

**Interfaces:**
- Consumes: Slurm jobs `546078` and `546079` plus their run roots.
- Produces: immutable performance/provenance evidence and terminal cancelled states.

- [ ] Capture `scontrol`, `sacct`, `nvidia-smi`, telemetry tails, resolved JSON,
  diagnostics aggregates, optimizer rows, CPU affinity, GPU mapping, and file hashes.
- [ ] Compute global structures/s from diagnostic step period and world size; compare
  with telemetry utilization, clocks, power, loader wait, and rank balance.
- [ ] Record a single bottleneck hypothesis supported by the evidence.
- [ ] Cancel only jobs `546078` and `546079`, then verify terminal Slurm states.

### Task 2: Build the dependency-only runtime

**Files:**
- Create remotely: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/`
- Create remotely: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/requirements.in`
- Create remotely: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/environment/`

**Interfaces:**
- Consumes: MACE runtime requirements from `setup.cfg` and the approved exact pins.
- Produces: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/bin/python` and frozen provenance manifests.

- [ ] Create a new Python 3.12.11 venv with `uv venv`; refuse an existing target.
- [ ] Install torch 2.11.0 from the official cu130 index, then install exact e3nn
  and CuEquivariance pins plus the remaining MACE runtime dependencies.
- [ ] Run `uv pip check` and freeze exact packages; assert that neither `mace-torch`
  nor `omat-downsample` is installed and no cu12 package is present.
- [ ] Write Python, platform, driver, torch/CUDA/NCCL, package version, import path,
  and file-hash manifests into `environment/`.

### Task 3: Qualify CUDA, conversion, and serialization

**Files:**
- Create remotely: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/qualification/one-gpu/`

**Interfaces:**
- Consumes: dependency runtime, MACE source commit `f75328d`, and project overlay.
- Produces: one-GPU pass manifest proving CUDA/CuEq/model correctness.

- [ ] Submit a one-GPU RTX-5090 smoke job that asserts exact imports and devices.
- [ ] Construct both one-interaction variants, convert e3nn to CuEq, run finite
  energy/force/stress forward/backward, convert back, serialize, and reload.
- [ ] Require exit code 0 and hashes for the converted and portable model artifacts.

### Task 4: Select production batch size

**Files:**
- Create remotely: `/scratch/bonan/mace-runtimes/20260824-py312-torch211-cu130-e3nn044-cueq0110/qualification/batch-sweep/`

**Interfaces:**
- Consumes: real OMat LMDB, workers 4, batch candidates 16/32/64 per rank.
- Produces: a common selected batch and measured throughput report.

- [ ] Run equal warm-up/measured update budgets on one GPU from identical seeds.
- [ ] Record structures/s, atoms/s, edges/s, step percentiles, loader wait, GPU
  utilization/power/clocks, allocator peak, graph count, and finite losses.
- [ ] Select the fastest stable batch that leaves operational memory headroom.

### Task 5: Run paired two-rank gates

**Files:**
- Modify: `/home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24.sbatch`
- Test: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer.py`

**Interfaces:**
- Consumes: selected batch, workers 4, runtime root, e3nn-0.4.4 import contract.
- Produces: two successful 20-update compiled-DDP gate manifests.

- [ ] Add a failing protocol assertion for the dependency-runtime root, CUDA 13,
  exact versions, no overlay, and restartable production stage arguments.
- [ ] Update the launcher to use the new runtime and exact fail-closed assertions.
- [ ] Run the protocol test red, implement the minimum launcher/protocol change,
  then run it green and run the full protocol gate.
- [ ] Submit fresh with-self and without-self two-rank gates and verify 20 updates,
  one graph, finite loss, device isolation, checkpoint, export, and runtime manifest.

### Task 6: Launch restartable paired production stages

**Files:**
- Create remotely: separate fresh with-self/without-self production roots.
- Update: `/home/bonan/work/train-downsample/docs/project/experiments.md`

**Interfaces:**
- Consumes: qualified runtime, selected batch, passed paired gates.
- Produces: two monitored fresh one-epoch stage-one jobs and reproducible evidence.

- [ ] Render stage-one full-epoch commands with no restart argument and fresh roots.
- [ ] Submit both jobs with two tasks/two GPUs, audit resolved resources and runtime.
- [ ] Verify native loader, compiled DDP, one graph, finite live metrics, throughput,
  and checkpoint/export completion.
- [ ] Record runtime ID, package hashes, source commit, job IDs, run roots, batch,
  throughput, and comparison baselines in EXP-A4-0018.
