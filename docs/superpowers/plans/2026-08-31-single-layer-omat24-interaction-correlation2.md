# Single-Layer OMat24 Interaction-Correlation-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, qualify, and submit one fresh 20-epoch full-OMat24 128-channel without-self model whose interaction product and final scalar contraction both use correlation 2.

**Architecture:** Add one immutable correlation candidate and a dedicated renderer that reuses the accepted 128-channel without-self protocol while overriding only the interaction correlation and isolated names/paths. Extend CPU construction provenance to verify both product correlations and the exact 6,251,674 trainable count, then launch through a fail-closed two-rank Slurm script from a new sealed deployment.

**Tech Stack:** Python 3.12, MACE 0.3.17, e3nn 0.4.4, PyTorch 2.11/CUDA 13, CuEquivariance 0.11, pytest, Bash, Slurm.

## Global Constraints

- Sole interaction: `RealAgnosticDensityInteractionBlock`; no self/residual connection; `num_interactions=1`.
- Hidden irreps: `128x0e + 128x1o + 128x2e`; `max_ell=3`.
- Interaction correlation is exactly 2 and readout correlation remains exactly 2.
- Expected trainable parameters are exactly 6,251,674.
- Start fresh with seed `20260729`; run epochs 0--19 in one job; do not pass a restart option.
- Preserve full OMat24 data/statistics, optimizer, EMA, scheduler, normalization, validation, loss weights, and IEEE float32 controls from the accepted 128-channel without-self run.
- Use batch 256/rank, validation batch 64, four workers/rank, two RTX 5090 ranks, eight CPUs/rank, and 92,000 MB/node.
- Every two-rank `srun` uses `--gpu-bind=mask_gpu:0x3,0x3`.
- Use fresh run/cache/log roots and do not alter active jobs `553748` or `553749`.

---

### Task 1: Add the immutable correlation-2 protocol and exact provenance

**Files:**
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer.py`
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer_width.py`
- Modify: `/home/bonan/work/train-downsample/scripts/run_single_layer_omat24.py`
- Modify: `/home/bonan/work/train-downsample/scripts/verify_single_layer_width_model.py`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer.py`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_width.py`

**Interfaces:**
- Produces: `INTERACTION_CORRELATION_CANDIDATES: dict[int, int]` mapping correlation 2 to 6,251,674.
- Produces: `build_cata_interaction_correlation_arguments(..., interaction_correlation: int) -> list[str]`.
- Extends: `build_width_model(channels: int, interaction_correlation: int = 3)` and `width_model_provenance(channels: int, interaction_correlation: int = 3)`.
- Produces CLI mode `cata-interaction-correlation` and verifier option `--interaction-correlation`.

- [ ] **Step 1: Write failing renderer/provenance tests**

Assert that the new renderer contains:

```python
assert "--hidden_irreps=128x0e + 128x1o + 128x2e" in args
assert "--interaction_first=RealAgnosticDensityInteractionBlock" in args
assert "--correlation=2" in args
assert "--readout_correlation=2" in args
assert "--max_num_epochs=20" in args
assert "--use_last_readout_only=True" in args
assert not any("restart" in arg for arg in args)
```

Compare against the accepted fresh 128-channel without-self argument set after excluding only name, output paths, and `--correlation`. Reject correlation 3, non-128 widths, workers other than 4, and batch other than 256.

Construct provenance with `width_model_provenance(128, interaction_correlation=2)` and assert:

```python
assert payload["trainable_parameters"] == 6_251_674
assert payload["interaction_first"] == "RealAgnosticDensityInteractionBlock"
assert payload["uses_self_interaction"] is False
assert payload["num_interactions"] == 1
assert payload["correlation"] == 2
assert payload["readout_correlation"] == 2
```

- [ ] **Step 2: Run RED**

```bash
cd /home/bonan/work/train-downsample
PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
  .venv/bin/python -m pytest \
  tests/full_retrain/test_single_layer.py \
  tests/full_retrain/test_single_layer_width.py -q
```

Expected: failures because the candidate, renderer, and parameterized provenance do not exist.

- [ ] **Step 3: Implement the minimal protocol**

Add the immutable mapping:

```python
INTERACTION_CORRELATION_CANDIDATES = {2: 6_251_674}
```

Allow `_base_arguments` to accept an optional `interaction_correlation`, defaulting to the existing specification value. Implement the dedicated builder with the fixed without-self variant, width 128, twenty epochs, correlation 2, distributed Slurm arguments, and production batch/worker guards. Add CLI mode `cata-interaction-correlation` with required `--interaction-correlation=2` and reject width/stage/target-epoch arguments.

- [ ] **Step 4: Parameterize exact CPU construction**

Pass `interaction_correlation` into `ScaleShiftMACE(correlation=...)`, select the exact expected count from either the existing width candidate or the new correlation candidate, and include `interaction_correlation` in stable JSON. Preserve the default correlation-3 behavior and counts for widths 96/64. Extend the verifier CLI with `--interaction-correlation`, accepting only 2 when `--channels=128`.

- [ ] **Step 5: Run GREEN and compilation**

```bash
cd /home/bonan/work/train-downsample
PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
  .venv/bin/python -m pytest \
  tests/full_retrain/test_single_layer.py \
  tests/full_retrain/test_single_layer_width.py -q
PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
  .venv/bin/python -m py_compile \
  src/omat_downsample/full_retrain/single_layer.py \
  src/omat_downsample/full_retrain/single_layer_width.py \
  scripts/run_single_layer_omat24.py \
  scripts/verify_single_layer_width_model.py
```

Expected: all focused tests pass and compilation exits zero.

---

### Task 2: Add a fail-closed fresh 20-epoch launcher

**Files:**
- Create: `/home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_interaction_correlation2.sbatch`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_width.py`

**Interfaces:**
- Consumes the existing pinned project/source/runtime/data/hash environment variables.
- Requires `EXPECTED_PARAMETERS=6251674`, `NUM_WORKERS=4`, and `BATCH_SIZE=256`.
- Executes CLI mode `cata-interaction-correlation`.

- [ ] **Step 1: Write the launcher contract test**

Require exact Slurm controls, parameter verifier invocation, fresh-root rejection, qualification admission, and all three masked `srun` calls. Assert the script contains:

```text
#SBATCH --ntasks=2
#SBATCH --gpus=5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=92000M
[[ "$EXPECTED_PARAMETERS" == "6251674" ]]
--channels=128
--interaction-correlation=2
cata-interaction-correlation
epoch-19.pt
```

Assert no restart text or target-epoch option exists and that verifier/hash checks precede run/cache creation.

- [ ] **Step 2: Run RED**

Run `test_single_layer_width.py`; expect failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

Mirror the accepted width launcher admission, runtime, telemetry, GPU binding, and postconditions. Run:

```bash
"$RUNTIME_ROOT/bin/python" \
  "$PROJECT_ROOT/scripts/verify_single_layer_width_model.py" \
  --channels=128 --interaction-correlation=2
```

before creating mutable state. Validate exact JSON topology/count, then train with the dedicated mode. Require finite initial validation, epoch-0 optimizer rows, one compiled DDP path, no checkpoint loading, and the exact epoch-19 checkpoint.

- [ ] **Step 4: Verify syntax and focused tests**

```bash
bash -n /home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_interaction_correlation2.sbatch
cd /home/bonan/work/train-downsample
PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
  .venv/bin/python -m pytest \
  tests/full_retrain/test_single_layer.py \
  tests/full_retrain/test_single_layer_width.py -q
```

Expected: syntax and tests pass.

---

### Task 3: Run regression and seal a qualified deployment

**Files:**
- Append: `/home/bonan/work/train-downsample/docs/project/experiments.md`
- Create remotely: `/scratch/bonan/train-downsample-single-layer/e3d8580-6509ea2-interaction-corr2-c128-v1/`

- [ ] **Step 1: Run the full regression**

```bash
cd /home/bonan/work/train-downsample
PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
  .venv/bin/python -m pytest tests/full_retrain -q
```

Expected: zero failures.

- [ ] **Step 2: Build and upload a metadata-free archive**

Archive `src/omat_downsample`, the renderer/verifier scripts, the new launcher, and required Cata helpers. Exclude `__pycache__`, bytecode, tests, editable metadata, runs, and caches. Record local and remote SHA-256.

- [ ] **Step 3: Qualify remotely**

Under the proven runtime and MACE source, run Python compilation, launcher `bash -n`, exact model provenance, and the dry renderer. Require 6,251,674 parameters, both correlations 2, full-data paths, twenty epochs, fresh initialization, batch 256, workers 4, and no restart.

- [ ] **Step 4: Record evidence**

Append source/runtime/archive/provenance hashes and dry-render assertions to the experiment log without overwriting prior entries.

---

### Task 4: Submit and verify production

- [ ] **Step 1: Inspect current Cata capacity and active width jobs**

Query jobs `553748` and `553749`; do not cancel, reprioritize, or alter them.

- [ ] **Step 2: Submit the sealed job**

Use a unique run root, cache root, stdout, and stderr. Submit with two typed RTX 5090s, two tasks, 16 CPUs, 92,000 MB, and expected count 6,251,674.

- [ ] **Step 3: Verify the resolved request**

Require `gres/gpu=2`, `JOB_GRES=gpu:5090:2`, two tasks, eight CPUs/task, 92,000 MB, zero restarts, the sealed launcher path, and unique output paths. Replace only this newly submitted job if its request is wrong.

- [ ] **Step 4: Verify live startup or pending reason**

If allocated, require exact provenance, both GPUs visible to both ranks, fresh initialization, one compiled graph, finite optimizer steps, bounded RSS, and no fatal signature. If pending, report the validated reason without changing the protocol.

- [ ] **Step 5: Report**

Report job ID, resolved resources, state, exact parameter reduction, and deployment hash. Do not claim accuracy before trained validation exists.
