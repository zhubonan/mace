# Single-Layer OMat24 Epoch-20 Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the with-self and without-self single-layer OMat24 models from their scheduler-faithful epoch-9 checkpoints through epoch 19, submitting with-self now and preparing without-self behind a completed-source gate.

**Architecture:** Extend the project-owned single-layer renderer with one exact continuation mode, add a pure checkpoint-preparation module that advances `ReduceLROnPlateau` once with epoch-9 validation, and launch from a sealed remote deployment with fail-closed resource and provenance checks. The source ten-epoch run roots remain immutable; each continuation receives a new run root containing an audited epoch-10 alias.

**Tech Stack:** Python 3.12, PyTorch 2.11, pytest, MACE 0.3.17, Slurm/sbatch, SHA-256 provenance, CUDA 13, CuEquivariance.

## Global Constraints

- Twenty total epochs means upstream epochs 0--19; the continuation executes epochs 10--19 exactly once with `--max_num_epochs=20`.
- Preserve two RTX 5090 ranks, batch 256/rank, four workers/rank, eight CPUs/rank, and 92,000 MB/node.
- Preserve Adam/AMSGrad state, optimizer LR `0.005`, `ReduceLROnPlateau(factor=0.5, patience=1)`, and every non-path training argument from jobs `551283`/`551284`.
- Advance the serialized scheduler once with epoch-9 validation before restart; fail if the resulting LR differs from `0.005`.
- Record that MACE's checkpoint lacks EMA shadow state; restart EMA from the saved EMA-averaged model identically for both variants.
- Never modify the completed ten-epoch run roots or reuse a failed/partial continuation root.
- Do not create the without-self checkpoint alias or run root before job `551284` is `COMPLETED 0:0` with finite epoch-9 validation and checkpoint artifacts.

---

### Task 1: Render the exact epoch-20 continuation command

**Files:**
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer.py`
- Modify: `/home/bonan/work/train-downsample/scripts/run_single_layer_omat24.py`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer.py`

**Interfaces:**
- Consumes: `SingleLayerVariant`, `_base_arguments(...)`, `SINGLE_LAYER_OMAT24`.
- Produces: `build_cata_continuation_arguments(...) -> list[str]` and CLI mode `cata-continuation`.

- [ ] **Step 1: Write failing renderer tests**

Add imports and tests asserting both variants render the identical continuation boundary:

```python
from omat_downsample.full_retrain.single_layer import (
    build_cata_continuation_arguments,
)

@pytest.mark.parametrize("variant_key", tuple(VARIANTS))
def test_cata_continuation_runs_epochs_10_through_19(
    tmp_path: Path, variant_key: str
) -> None:
    args = build_cata_continuation_arguments(
        variant=VARIANTS[variant_key],
        train_root=Path("/train"),
        valid_root=Path("/valid"),
        statistics_file=Path("/stats"),
        run_root=tmp_path / variant_key,
        num_workers=4,
        batch_per_rank=256,
    )
    assert "--max_num_epochs=20" in args
    assert "--restart_latest" in args
    assert "--distributed" in args
    assert "--launcher=slurm" in args
    assert not any(a.startswith("--max_train_batches_per_epoch") for a in args)
```

Add a subprocess dry-run test requiring JSON mode `cata-continuation`, target epoch `20`, batch 256, workers 4, and no mutation of the run root. Add rejection tests for workers other than 4 and batch sizes other than 256.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer.py -q
```

Expected: collection/import failure because `build_cata_continuation_arguments` and mode `cata-continuation` do not exist.

- [ ] **Step 3: Implement the minimal renderer**

Add `cata_continuation_max_epochs: int = 20` to `SingleLayerOMat24Spec`, then add:

```python
def build_cata_continuation_arguments(
    *,
    variant: SingleLayerVariant,
    train_root: Path,
    valid_root: Path,
    statistics_file: Path,
    run_root: Path,
    num_workers: int,
    batch_per_rank: int,
) -> list[str]:
    """Render the exact two-rank epoch-9 to epoch-20 continuation."""
    if (
        num_workers != SINGLE_LAYER_OMAT24.cata_num_workers
        or batch_per_rank != SINGLE_LAYER_OMAT24.cata_batch_per_rank
    ):
        raise ValueError(
            "continuation requires num_workers=4 and batch_per_rank=256"
        )
    arguments = _base_arguments(
        variant=variant,
        train_root=train_root,
        valid_root=valid_root,
        statistics_file=statistics_file,
        run_root=run_root,
        num_workers=num_workers,
        batch_per_rank=batch_per_rank,
        max_num_epochs=SINGLE_LAYER_OMAT24.cata_continuation_max_epochs,
    )
    arguments.extend(("--restart_latest", "--distributed", "--launcher=slurm"))
    return arguments
```

Extend `run_single_layer_omat24.py` choices with `cata-continuation`, import the builder, forbid `--target-epoch` for this mode, and dispatch to the new builder. Keep the existing `cata` behavior unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all tests in `test_single_layer.py` pass.

---

### Task 2: Prepare a scheduler-faithful restart checkpoint

**Files:**
- Create: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer_continuation.py`
- Create: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_continuation.py`
- Create: `/home/bonan/work/train-downsample/scripts/prepare_single_layer_omat24_continuation.py`

**Interfaces:**
- Produces: `extract_epoch_validation(results_path: Path, epoch: int = 9) -> float`.
- Produces: `prepare_restart(source_checkpoint: Path, source_results: Path, destination_checkpoint: Path, audit_path: Path, variant: str) -> dict[str, object]`.
- Produces CLI arguments `--source-checkpoint`, `--source-results`, `--destination-checkpoint`, `--audit-path`, and `--variant`.

- [ ] **Step 1: Write failing checkpoint tests**

Build tiny real Torch checkpoints with six optimizer groups and a real
`ReduceLROnPlateau(factor=0.5, patience=1)`. Test:

```python
def test_prepare_restart_advances_epoch9_metric_without_changing_lr(
    tmp_path: Path,
) -> None:
    source, results = make_epoch9_fixture(tmp_path, loss=0.0063928887)
    destination = tmp_path / "new" / "model_epoch-10.pt"
    audit = tmp_path / "new" / "restart-audit.json"
    payload = prepare_restart(
        source_checkpoint=source,
        source_results=results,
        destination_checkpoint=destination,
        audit_path=audit,
        variant="with-self",
    )
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    assert [g["lr"] for g in checkpoint["optimizer"]["param_groups"]] == [0.005] * 6
    assert checkpoint["lr_scheduler"]["best"] == pytest.approx(0.0063928887)
    assert checkpoint["lr_scheduler"]["last_epoch"] == 10
    assert payload["ema_state_restored"] is False
```

Also test rejection of missing keys, destination collisions, non-finite/missing/duplicate epoch-9 validation rows, optimizer LR other than `0.005`, scheduler factor/patience mismatch, scheduler-induced LR change, and variants outside `with-self`/`without-self`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer_continuation.py -q
```

Expected: collection/import failure because `single_layer_continuation.py` does not exist.

- [ ] **Step 3: Implement validation and scheduler advancement**

Implement `extract_epoch_validation` by parsing JSONL records with
`mode == "eval"`, `head == "Default"`, and `epoch == 9`, requiring exactly
one finite loss. In `prepare_restart`, load the source on CPU, validate exact
checkpoint keys, six finite optimizer LRs equal to `0.005`, and scheduler
`factor == 0.5`, `patience == 1`.

Use a real scheduler to preserve PyTorch semantics:

```python
parameters = [torch.nn.Parameter(torch.zeros(())) for _ in optimizer_lrs]
optimizer = torch.optim.SGD(
    [{"params": [parameter], "lr": lr} for parameter, lr in zip(parameters, optimizer_lrs)]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=1
)
scheduler.load_state_dict(checkpoint["lr_scheduler"])
scheduler.step(validation_loss)
advanced_lrs = [float(group["lr"]) for group in optimizer.param_groups]
if advanced_lrs != [0.005] * len(advanced_lrs):
    raise ValueError("epoch-9 scheduler advancement unexpectedly changed LR")
```

Deep-copy the checkpoint, replace each optimizer group's LR and the scheduler
state, then write through a same-directory temporary file and `os.replace`.
Write atomic audit JSON containing source/destination hashes, before/after
scheduler fields, validation loss, variant, epoch boundary 9-to-10, and
`"ema_state_restored": false`.

- [ ] **Step 4: Add the thin CLI and verify GREEN**

The CLI must call `prepare_restart` only; it prints the audit JSON and exits
nonzero on validation failure. Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer_continuation.py -q
.venv/bin/python -m py_compile \
  src/omat_downsample/full_retrain/single_layer_continuation.py \
  scripts/prepare_single_layer_omat24_continuation.py
```

Expected: focused tests pass and compilation exits 0.

---

### Task 3: Add the fail-closed two-rank continuation launcher

**Files:**
- Create: `/home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_continue_to20.sbatch`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_continuation.py`

**Interfaces:**
- Consumes prepared `RUN_ROOT/checkpoints/<tag>_epoch-10.pt` and `RUN_ROOT/restart-audit.json`.
- Consumes the existing runtime/source/project qualification environment variables used by job `551283`.
- Executes CLI mode `cata-continuation`.

- [ ] **Step 1: Write the failing launcher contract test**

Assert the launcher contains these exact controls:

```python
def test_continuation_launcher_is_fail_closed_and_two_rank() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/cata/train_single_layer_omat24_continue_to20.sbatch").read_text()
    for line in (
        "#SBATCH --ntasks=2",
        "#SBATCH --gpus=5090:2",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=92000M",
        '[[ "$NUM_WORKERS" == "4" ]]',
        '[[ "$BATCH_SIZE" == "256" ]]',
        '[[ "$TARGET_EPOCH" == "20" ]]',
        '"$PROJECT_ROOT/scripts/run_single_layer_omat24.py" cata-continuation',
        'grep -q "Loading checkpoint: $RESTART_ALIAS" "$LOG_FILE"',
    ):
        assert line in script
```

Also assert no `cp` or checkpoint mutation occurs in the batch job, the audit
hash is checked before creating result/log/model directories, DDP uses
`--gpu-bind=none`, runtime admission runs, and output validation rejects epoch
0 optimizer rows while requiring epoch 10 rows.

- [ ] **Step 2: Run the focused test and verify RED**

Run the Task 2 pytest command. Expected: failure because the launcher is absent.

- [ ] **Step 3: Implement the launcher**

Mirror the proven fresh-ten launcher resource/runtime/CUDA visibility checks.
Require `PROJECT_ROOT`, `MACE_SOURCE_ROOT`, `MACE_COMMIT`, `RUNTIME_ROOT`,
`TRAIN_ROOT`, `VALID_ROOT`, `STATISTICS_FILE`, `RUN_ROOT`, `CACHE_ROOT`,
`VARIANT`, `SOURCE_JOB_ID`, `RESTART_ALIAS`, `RESTART_AUDIT`,
`RESTART_AUDIT_SHA256`, `NUM_WORKERS=4`, `BATCH_SIZE=256`, and
`TARGET_EPOCH=20`.

Before creating mutable output directories, verify:

```bash
test -r "$RESTART_ALIAS"
test -r "$RESTART_AUDIT"
test "$(sha256sum "$RESTART_AUDIT" | cut -d ' ' -f 1)" = "$RESTART_AUDIT_SHA256"
"$RUNTIME_ROOT/bin/python" -c \
  'import json,sys; a=json.load(open(sys.argv[1])); assert a["source_epoch"]==9; assert a["restart_epoch"]==10; assert a["optimizer_lrs_after"]==[0.005]*6; assert a["scheduler_after"]["factor"]==0.5; assert a["scheduler_after"]["patience"]==1; assert a["ema_state_restored"] is False' \
  "$RESTART_AUDIT"
```

Launch two ranks through the existing master-port helper and runtime admission,
then execute `run_single_layer_omat24.py cata-continuation`. Postconditions
require the exact restart alias load, compiled DDP activation, finite epoch-10
validation, optimizer rows beginning at epoch 10, and an epoch-19 checkpoint.

- [ ] **Step 4: Verify launcher syntax and tests**

Run:

```bash
bash -n /home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_continue_to20.sbatch
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest \
  tests/full_retrain/test_single_layer.py \
  tests/full_retrain/test_single_layer_continuation.py -q
```

Expected: Bash exits 0 and both focused suites pass.

---

### Task 4: Seal and qualify the remote continuation deployment

**Files:**
- Create remotely: `/scratch/bonan/train-downsample-single-layer/<source>-continue20-v1/`
- Create remotely: deployment manifest, source archive, dry-run JSON files, and SHA-256 inventory.

**Interfaces:**
- Consumes the tested project files from Tasks 1--3 and immutable runtime/source roots from job `551283`.
- Produces a sealed deployment root and a prepared with-self run root.

- [ ] **Step 1: Run the full relevant local regression**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain -q
```

Expected: all `tests/full_retrain` tests pass with zero failures.

- [ ] **Step 2: Stage immutable project files and verify hashes**

Copy only the tested project package/scripts into a new remote deployment,
record local and remote SHA-256 for every file, and refuse an existing target.
Run remote `py_compile`, `bash -n`, and both variant dry renders with the pinned
runtime and exact `PYTHONPATH`. Expected dry renders contain
`--max_num_epochs=20`, `--restart_latest`, batch 256, workers 4, DDP, and only
the expected `interaction_first` difference.

- [ ] **Step 3: Prepare and audit the with-self epoch-10 alias**

Use source checkpoint:

```text
/scratch/bonan/train-downsample-single-layer/e3d8580-6509ea2-q19b0864-matscipyfix-continuous10-v1/runs/with-self-fresh-continuous-10/checkpoints/single-layer-omat24-with-self_run-20260729_epoch-9.pt
```

and its results JSONL. Run the preparation CLI into a new continuation run
root. Verify source remains byte-identical; audit must show source loss
`0.0063928887`, before LR `0.005`, after LR `0.005`, scheduler best advanced to
the epoch-9 loss, and restart epoch 10.

- [ ] **Step 4: Run without-self readiness and dry-preparation checks**

Query `sacct -j 551284`. If it is not `COMPLETED|0:0`, preserve only the tested
without-self dry render and report `waiting_for_source`; do not create its run
root or alias. If complete, require finite epoch-9 validation/checkpoint and run
the same preparation/audit into a separate without-self run root.

---

### Task 5: Submit and verify production state

**Files:**
- Create remotely: Slurm log and live run artifacts under the sealed continuation deployment.
- Append locally: `.planning/2026-08-23-single-layer-mace-omat24-feasibility/progress.md` with job IDs and verified state.

**Interfaces:**
- Consumes sealed deployment and prepared run root from Task 4.
- Produces one active with-self Slurm job and a ready/waiting without-self state.

- [ ] **Step 1: Submit with-self with exact resources**

Submit one job using two RTX 5090 GPUs, two tasks, eight CPUs/task, 92,000 MB,
and the prepared with-self audit/alias. Do not add a node pin unless live Cata
availability requires one and the resolved GPU visibility contract remains
valid.

- [ ] **Step 2: Verify the resolved allocation**

Run `scontrol show job -dd <job_id>` and require two tasks, two RTX 5090 GPUs,
16 CPUs, 92,000 MB, zero restarts, the sealed launcher path, and unique output
paths. Cancel and replace only this new job if the resolved request is wrong.

- [ ] **Step 3: Verify restart and initial optimizer progress**

Wait for allocation, then require the exact epoch-10 alias load, scheduler audit
printout, compiled DDP activation, optimizer epoch 10 rows, finite loss, one
compiled graph, bounded RSS, and no traceback/OOM/worker/NCCL/recompilation
signature. If pending for resources, report the validated pending state rather
than altering unrelated jobs.

- [ ] **Step 4: Handle without-self according to readiness**

If job `551284` completed during implementation, prepare and submit its
continuation with the identical workflow. Otherwise leave its sealed dry render
and readiness command in place and report that no without-self production job
was submitted yet.

- [ ] **Step 5: Record final evidence**

Append source checkpoint hashes, scheduler before/after fields, deployment
hash, job IDs, Slurm allocation, and first live optimizer/diagnostic evidence to
the persistent progress file. Re-run `git status --short` and preserve unrelated
user changes.
