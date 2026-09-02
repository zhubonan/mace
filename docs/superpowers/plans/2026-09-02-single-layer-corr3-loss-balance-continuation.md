# Single-Layer Correlation-3 Loss-Balance Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare and qualify, without submission, 128/96/64-channel correlation-3 without-self OMat24 continuations from epoch 19 through epoch 24 at 10/10/10 energy/force/stress weights.

**Architecture:** Add an exact three-width renderer and runner mode, a separate audited epoch-19-to-20 restart handoff, and a dedicated fail-closed Slurm launcher that permits MACE's final export reload.

**Tech Stack:** Python 3.12, PyTorch checkpoint/scheduler state, MACE 0.3.17, Bash, Slurm, pytest, pinned CUDA-13/e3nn-0.4.4 runtime.

## Global Constraints

- Only widths 128, 96, and 64; without-self, one interaction, correlation 3, readout correlation 2, `max_ell=3`, hidden `L_max=2`.
- Continue epochs 20--24 with `--max_num_epochs=25` and `--restart_latest`.
- Use `--energy_weight=10.0`, `--forces_weight=10.0`, and `--stress_weight=10.0`.
- Exact completed epoch-19 source hashes are in `docs/superpowers/specs/2026-09-02-single-layer-corr3-loss-balance-continuation-design.md`.
- Preserve six source LRs: 0.0025 (128) and 0.00125 (96/64), with scheduler factor/patience 0.5/1.
- Use 2 RTX 5090 ranks, 8 CPUs/rank, 92,000 MB/node, batch 256/rank, 4 workers/rank, 3 days, and `--gpu-bind=mask_gpu:0x3,0x3`.
- Build and qualify only; do not invoke `sbatch`, cancel jobs, or touch active jobs.

---

### Task 1: Render the exact three candidates

**Files:**
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer.py`
- Modify: `/home/bonan/work/train-downsample/scripts/run_single_layer_omat24.py`
- Modify: `/home/bonan/work/train-downsample/scripts/verify_single_layer_width_model.py`
- Test: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer.py`
- Test: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_width.py`

**Interfaces:** add `LOSS_BALANCE_CONTINUATION_CANDIDATES` and `build_cata_loss_balance_continuation_arguments(..., channels: int)`. Add CLI mode `cata-loss-balance-continuation`.

- [ ] **Step 1: Write RED tests.** Parametrize widths with `(128, "single-layer-omat24-without-self", "128x0e + 128x1o + 128x2e", 6639002)`, `(96, "single-layer-omat24-without-self-c96", "96x0e + 96x1o + 96x2e", 3860186)`, and `(64, "single-layer-omat24-without-self-c64", "64x0e + 64x1o + 64x2e", 1828890)`. Assert rendered arguments contain correlation 3, readout correlation 2, original name, `--max_num_epochs=25`, all three 10.0 weights, and restart/distributed/Slurm flags. Assert invalid width, self variant, workers/batch, stage, target epoch, and interaction correlation fail. Assert the verifier accepts correlation-3 128 with 6,639,002 parameters.
- [ ] **Step 2: Run RED.** From `/home/bonan/work/train-downsample`, run `PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official .venv/bin/python -m pytest tests/full_retrain/test_single_layer.py tests/full_retrain/test_single_layer_width.py -q`; expect missing renderer/mode/provenance failures.
- [ ] **Step 3: Implement.** Define the exact map including original tag, hidden irreps, parameter count, source LR, and source SHA. The builder calls `_base_arguments` with `VARIANTS["without-self"]`, correlation 3, original name, width irreps, and 25 max epochs; replace only the three loss arguments with `--energy_weight=10.0`, `--forces_weight=10.0`, and `--stress_weight=10.0`; append `--restart_latest`, `--distributed`, `--launcher=slurm`. The runner requires only without-self, one candidate width, workers 4, and batch 256; dry-run emits channels and target epoch 25.
- [ ] **Step 4: Run GREEN and compile.** Re-run Step 2, then compile the three modified Python files with `PYTHONPYCACHEPREFIX=/tmp/loss-balance-continuation-pyc` and the project `PYTHONPATH`; expect zero failures and zero compilation errors.
- [ ] **Step 5: Commit.** Stage the three source files and two test files, then commit `feat: render loss-balanced single-layer continuations`.

### Task 2: Audit the epoch-19-to-20 checkpoint handoff

**Files:**
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer_continuation.py`
- Modify: `/home/bonan/work/train-downsample/scripts/prepare_single_layer_omat24_continuation.py`
- Test: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_continuation.py`

**Interfaces:** add `prepare_loss_balance_restart(..., channels: int, expected_source_sha256: str) -> dict[str, object]` and a `loss-balance` wrapper subcommand.

- [ ] **Step 1: Write RED tests.** For `(128, 0.0025)`, `(96, 0.00125)`, and `(64, 0.00125)`, construct an epoch-19 checkpoint plus exactly one finite epoch-19 validation row, run the new helper, and assert source/restart epochs 19/20, preserved six LRs, matching source/destination hashes, scheduler before/after state, and `loss_weights_after == {"energy": 10.0, "forces": 10.0, "stress": 10.0}`. Add rejection tests for wrong hash/width/LR, invalid epoch names, duplicate alias or audit, missing epoch-19 row, invalid factor/patience, and scheduler advancement that changes LR. Keep all existing epoch-9 tests unchanged.
- [ ] **Step 2: Run RED.** Run `PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official .venv/bin/python -m pytest tests/full_retrain/test_single_layer_continuation.py -q`; expect the new helper/subcommand to be absent.
- [ ] **Step 3: Implement.** Reuse existing atomic JSON, SHA, validation extraction, and scheduler-summary primitives without modifying `prepare_restart`. Validate checkpoint keys are exactly `model`, `optimizer`, and `lr_scheduler`; compare source digest before writes; check the width LR map; reconstruct and advance ReduceLROnPlateau once with the old epoch-19 loss; reject LR changes; save only the fresh epoch-20 alias and full audit. Dispatch the legacy and `loss-balance` wrapper forms separately.
- [ ] **Step 4: Run GREEN.** Re-run Step 2 and `tests/full_retrain/test_stages.py tests/full_retrain/test_single_layer.py -q`; expect zero failures and preserved legacy continuation behavior.
- [ ] **Step 5: Commit.** Stage the helper, wrapper, and continuation test; commit `feat: audit loss-balanced epoch twenty restarts`.

### Task 3: Add the fail-closed Cata launcher

**Files:**
- Create: `/home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_loss_balance_continue_e20_to24.sbatch`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_continuation.py`

**Interfaces:** require `CHANNELS`, `EXPECTED_PARAMETERS`, source checkpoint/result/hash, prepared alias/audit/hash, runtime/data/qualification controls, and unique run/cache roots. Produce finite epoch-20 validation and epoch-24 checkpoint.

- [ ] **Step 1: Write RED launcher test.** Assert the new script has two ranks, 5090:2, 8 CPUs/task, 92,000 MB, target epoch 25, and exact parameter cases `128:6639002`, `96:3860186`, `64:1828890`; invokes `cata-loss-balance-continuation`; checks intended alias load; requires optimizer epoch 20 and epoch-24 checkpoint; and uses mask `0x3,0x3`. Assert all source/audit/provenance checks precede `mkdir`, the 10/10/10 objective and epoch 19/20 audit are checked, and no generic ban on `Loading checkpoint:` exists.
- [ ] **Step 2: Run RED.** Run the continuation test module; expect `FileNotFoundError` for the launcher.
- [ ] **Step 3: Implement.** Accept only the exact three channels, original model tags, source hashes, source LRs, and parameter counts. Before mutable results/logs/models/telemetry/cache state, check pinned source/runtime/wheel/qualification, topology provenance, source and alias digests, and audit fields. Allow only pre-created checkpoints/audit. Reuse the accepted GPU/runtime/IEEE/monitor/DDP gates. After training require the intended alias load, native OMat data, compiled DDP, finite epoch-20 validation, and epoch-24 checkpoint; permit MACE's final export reload.
- [ ] **Step 4: Run GREEN.** Run `bash -n` on the new launcher and re-run the continuation tests; expect zero errors.
- [ ] **Step 5: Commit.** Stage the launcher and test; commit `feat: add loss-balanced continuation launcher`.

### Task 4: Qualify and hand off without submission

**Files:**
- Modify: `/home/bonan/work/train-downsample/docs/project/experiments.md`
- Create remotely: one metadata-free sealed deployment and three prepared epoch-20 aliases/audits.

- [ ] **Step 1: Regress.** Run `PYTHONPATH=/home/bonan/.codex/worktrees/a86d/mace-official .venv/bin/python -m pytest tests/full_retrain -q` from `/home/bonan/work/train-downsample`; expect zero failures and record warnings separately.
- [ ] **Step 2: Seal.** Package only source, renderer, verifier, preparation wrapper, launcher, IEEE wrapper, master-port helper, and monitor script; exclude tests, bytecode, metadata, runs, and caches. Upload to a new immutable Cata deployment and record equal local/remote SHA-256.
- [ ] **Step 3: Prepare and qualify.** Use the pinned remote runtime once per width to create a unique `loss-e10-f10-continue-e20-to24` alias/audit. Verify source/alias/audit hashes, tags, epochs, LRs, scheduler, 10/10/10 weights, Bash syntax, compilation, topology provenance, and dry render. Assert no result/log/model/telemetry/compiler cache exists.
- [ ] **Step 4: Document but do not submit.** Append all hashes, tests, resource controls, and three exact ready-to-submit commands with unique run/cache/stdout/stderr roots to `docs/project/experiments.md`; do not execute `sbatch`.
- [ ] **Step 5: Final verification and commit.** Re-run full tests, launcher syntax, and read-only remote hash/audit checks. Stage the local source/test/launcher/documentation changes and commit `docs: prepare loss-balanced continuation jobs`.
