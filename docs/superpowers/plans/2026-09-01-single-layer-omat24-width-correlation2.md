# Single-Layer OMat24 Correlation-2 Width Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement, qualify, and submit fresh full-OMat24 20-epoch correlation-2 jobs for 96- and 64-channel without-self single-layer MACE models.

**Architecture:** Replace the correlation-only candidate table with an exact width/correlation-pair mapping while preserving the accepted 128-channel path. Extend the renderer and CPU provenance through that mapping, then add a dedicated fail-closed two-candidate Slurm launcher and deploy it from one sealed archive.

**Tech Stack:** Python 3.12, MACE 0.3.17, e3nn 0.4.4, PyTorch 2.11/CUDA 13, CuEquivariance 0.11, pytest, Bash, Slurm.

## Global Constraints

- Exact candidates are (128, 2): 6,251,674, (96, 2): 3,569,690, and (64, 2): 1,635,226 trainable parameters.
- Use one RealAgnosticDensityInteractionBlock, no self path, one interaction, hidden Lmax=2, max_ell=3, interaction correlation 2, and readout correlation 2.
- Start fresh with seed 20260729; run epochs 0--19 in one job without restart.
- Preserve full OMat24, optimizer, EMA, scheduler, normalization, loss weights, IEEE float32, and compiled DDP controls.
- Use batch 256/rank, validation batch 64, four workers/rank, two RTX 5090 ranks, eight CPUs/rank, and 92,000 MB/node.
- Every two-rank srun uses gpu-bind mask 0x3,0x3.
- Use fresh run/cache/log roots and do not modify existing jobs.

---

### Task 1: Exact candidates, renderer, and provenance

**Files:**
- Modify: /home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer.py
- Modify: /home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer_width.py
- Modify: /home/bonan/work/train-downsample/scripts/run_single_layer_omat24.py
- Modify: /home/bonan/work/train-downsample/scripts/verify_single_layer_width_model.py
- Test: /home/bonan/work/train-downsample/tests/full_retrain/test_single_layer.py
- Test: /home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_width.py

**Interfaces:**
- Produces: INTERACTION_CORRELATION_CANDIDATES: dict[tuple[int, int], int].
- Extends: build_cata_interaction_correlation_arguments with channels 128, 96, or 64 and interaction correlation 2.
- Preserves: omitted channels in cata-interaction-correlation resolve to 128.
- Extends: width_model_provenance(channels, interaction_correlation).

- [ ] **Step 1: Write failing candidate and renderer tests**

    assert INTERACTION_CORRELATION_CANDIDATES == {
        (128, 2): 6_251_674,
        (96, 2): 3_569_690,
        (64, 2): 1_635_226,
    }
    for channels in (96, 64):
        args = build_cata_interaction_correlation_arguments(
            train_root=Path("/train"),
            valid_root=Path("/valid"),
            statistics_file=Path("/stats.json"),
            run_root=Path("/run"),
            num_workers=4,
            batch_per_rank=256,
            channels=channels,
            interaction_correlation=2,
        )
        assert (
            f"--hidden_irreps={channels}x0e + "
            f"{channels}x1o + {channels}x2e"
        ) in args
        assert "--correlation=2" in args
        assert "--readout_correlation=2" in args
        assert "--max_num_epochs=20" in args
        assert not any("restart" in arg for arg in args)

Reject channels outside 64, 96, and 128, correlation other than 2, workers
other than 4, and batch other than 256.

- [ ] **Step 2: Run RED**

    cd /home/bonan/work/train-downsample
    PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
      .venv/bin/python -m pytest \
      tests/full_retrain/test_single_layer.py \
      tests/full_retrain/test_single_layer_width.py -q

Expected: failures because 96/64 correlation-2 candidates are not admitted.

- [ ] **Step 3: Implement exact mapping and renderer**

    INTERACTION_CORRELATION_CANDIDATES = {
        (128, 2): 6_251_674,
        (96, 2): 3_569_690,
        (64, 2): 1_635_226,
    }

    candidate = WIDTH_CANDIDATES.get(channels)
    hidden_irreps = (
        SINGLE_LAYER_OMAT24.hidden_irreps
        if channels == 128
        else candidate.hidden_irreps
    )

Derive argparse correlation choices from mapping-key item 1. In
cata-interaction-correlation use channels = args.channels or 128 and reject
stage and target-epoch. Name 96/64 models with c96-correlation2 or
c64-correlation2 while preserving the 128-channel name.

- [ ] **Step 4: Write failing construction tests**

    expected = {96: 3_569_690, 64: 1_635_226}
    for channels, count in expected.items():
        payload = width_model_provenance(
            channels, interaction_correlation=2
        )
        assert payload["trainable_parameters"] == count
        assert payload["uses_self_interaction"] is False
        assert payload["num_interactions"] == 1
        assert payload["correlation"] == 2
        assert payload["readout_correlation"] == 2
        assert payload["max_ell"] == 3

- [ ] **Step 5: Implement pair-based provenance and verifier**

    pair = (channels, interaction_correlation)
    if pair in INTERACTION_CORRELATION_CANDIDATES:
        hidden_irreps = (
            SINGLE_LAYER_OMAT24.hidden_irreps
            if channels == 128
            else WIDTH_CANDIDATES[channels].hidden_irreps
        )
        return hidden_irreps, INTERACTION_CORRELATION_CANDIDATES[pair]

The verifier admits only mapped pairs and preserves correlation-3 widths.

- [ ] **Step 6: Run GREEN and compilation**

    cd /home/bonan/work/train-downsample
    PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
      .venv/bin/python -m pytest \
      tests/full_retrain/test_single_layer.py \
      tests/full_retrain/test_single_layer_width.py -q
    PYTHONPYCACHEPREFIX=/tmp/corr2-width-pyc \
    PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
      .venv/bin/python -m py_compile \
      src/omat_downsample/full_retrain/single_layer.py \
      src/omat_downsample/full_retrain/single_layer_width.py \
      scripts/run_single_layer_omat24.py \
      scripts/verify_single_layer_width_model.py

Expected: focused tests and compilation exit zero.

---

### Task 2: Fail-closed two-candidate launcher

**Files:**
- Create: /home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_interaction_correlation2_width.sbatch
- Modify: /home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_width.py

**Interfaces:**
- Consumes CHANNELS in {96, 64} and exact EXPECTED_PARAMETERS.
- Executes cata-interaction-correlation with channels and correlation 2.
- Produces candidate-specific epoch-19 checkpoints.

- [ ] **Step 1: Write failing launcher contract**

    launcher = (
        root / "scripts/cata/"
        "train_single_layer_omat24_interaction_correlation2_width.sbatch"
    ).read_text()
    for text in (
        "#SBATCH --ntasks=2",
        "#SBATCH --gpus=5090:2",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=92000M",
        '96) [[ "$EXPECTED_PARAMETERS" == "3569690" ]] ;;',
        '64) [[ "$EXPECTED_PARAMETERS" == "1635226" ]] ;;',
        "--interaction-correlation=2",
        '--channels="$CHANNELS"',
        "cata-interaction-correlation",
        "epoch-19.pt",
    ):
        assert text in launcher
    assert "restart" not in launcher.lower()

Also assert qualification-content admission, exact provenance, existing
run/cache rejection, and all three masked srun calls precede mutation.

- [ ] **Step 2: Run RED**

    cd /home/bonan/work/train-downsample
    PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
      .venv/bin/python -m pytest \
      tests/full_retrain/test_single_layer_width.py -q

Expected: FileNotFoundError for the missing launcher.

- [ ] **Step 3: Implement launcher**

    case "$CHANNELS" in
        96) [[ "$EXPECTED_PARAMETERS" == "3569690" ]] ;;
        64) [[ "$EXPECTED_PARAMETERS" == "1635226" ]] ;;
        *) echo "CHANNELS must be 96 or 64" >&2; exit 2 ;;
    esac

    RUN_NAME="single-layer-omat24-without-self-c${CHANNELS}-correlation2"
    FINAL_CHECKPOINT="$RUN_ROOT/checkpoints/${RUN_NAME}_run-20260729_epoch-19.pt"

Verify the model before mutable state. Preserve qualification, runtime
admission, telemetry, IEEE controls, three masked srun calls, fresh-state
rejection, finite epoch-0 validation, compiled DDP, and no checkpoint loading.

- [ ] **Step 4: Run launcher and focused verification**

    bash -n /home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_interaction_correlation2_width.sbatch
    cd /home/bonan/work/train-downsample
    PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
      .venv/bin/python -m pytest \
      tests/full_retrain/test_single_layer.py \
      tests/full_retrain/test_single_layer_width.py -q

Expected: syntax and focused tests pass.

---

### Task 3: Regression and sealed qualification

**Files:**
- Append: /home/bonan/work/train-downsample/docs/project/experiments.md
- Create remotely: a fresh e3d8580 correlation-2-width deployment.

- [ ] **Step 1: Run full regression**

    cd /home/bonan/work/train-downsample
    PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official \
      .venv/bin/python -m pytest tests/full_retrain -q

Expected: zero failures.

- [ ] **Step 2: Build and upload metadata-free archive**

Include src/omat_downsample, renderer/verifier, new launcher,
choose_master_port.sh, monitor_full_omat_medium.sh, and
run_mace_ieee_fp32.py. Exclude tests, bytecode, metadata, runs, and caches.
Record local and remote SHA-256.

- [ ] **Step 3: Qualify sealed deployment**

Under the proven runtime/source, run compilation, Bash syntax, exact
provenance for 96/64 correlation 2, and both dry renders. Assert exact counts,
full data paths, twenty epochs, batch 256, workers 4, both correlations 2, and
no restart.

- [ ] **Step 4: Record evidence**

Append source/runtime/archive/qualification hashes, counts, tests, and
dry-render assertions to docs/project/experiments.md.

---

### Task 4: Submit and verify production

- [ ] **Step 1: Inspect Cata capacity and active jobs**

Query active jobs without cancelling, reprioritizing, or changing them.

- [ ] **Step 2: Submit both sealed jobs**

Submit 96 channels with EXPECTED_PARAMETERS=3569690 and 64 channels with
EXPECTED_PARAMETERS=1635226. Give each unique run/cache/stdout/stderr roots.

- [ ] **Step 3: Verify resolved requests**

Require gres/gpu=2, JOB_GRES gpu:5090:2, two tasks, eight CPUs/task,
92,000 MB, zero restarts, sealed launcher, and unique output paths.

- [ ] **Step 4: Verify startup or pending reason**

If allocated, require exact provenance, visible GPUs, fresh initialization,
one compiled graph, finite optimizer rows, bounded RSS, and no fatal signature.
If pending, report the scheduler reason without changing protocol.

- [ ] **Step 5: Report**

Report job IDs, resources, states, parameter reductions, deployment hash, and
whether trained validation exists.
