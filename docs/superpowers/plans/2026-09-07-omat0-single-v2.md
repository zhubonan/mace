# OMat-0 Single-v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a production-ready `single-v2` foundation-transfer mode that trains one OMat-0-initialized interaction end-to-end with the standard symmetric nonlinear readout.

**Architecture:** Build a one-interaction `ScaleShiftMACE` from OMat-0's extracted first-layer configuration (`128x0e + 128x1o`), transfer only source interaction/product index zero plus radial and baseline state, and leave the standard correlation-2 readout fresh. Add a dedicated CLI mode so the existing rank-16 direct-head path remains unchanged.

**Tech Stack:** Python, PyTorch, e3nn, MACE model/config factories, pytest, Slurm bash launcher.

## Global Constraints

- Preserve OMat-0 `r_max=6.0`, Agnesi distance transform, pair repulsion, and E0/scale/shift.
- Use one interaction and hidden irreps `128x0e + 128x1o`.
- Transfer source first interaction/product only; all transferred parameters are trainable.
- Use `energy:forces:stress = 1:10:10`, EMA `0.999`, weight decay `1e-8`, and ReduceLROnPlateau factor `0.5`, patience `1`.
- Do not modify or reuse the active frozen direct-head run root.

### Task 1: Add the single-v2 model/config flag

**Files:**
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/tools/model_script_utils.py`
- Modify: `mace/modules/models.py`
- Modify: `mace/tools/scripts_utils.py`
- Test: `tests/unit/test_model_script_utils.py` and `tests/unit/test_models.py`

**Interfaces:**
- CLI field: `omat0_first_layer_readout: bool`.
- Model field: `omat0_first_layer_readout: bool`.
- Foundation build path sets `num_interactions=1`, retains source first-layer `hidden_irreps`, uses source first interaction class, and sets `use_last_readout_only=True`.

- [ ] Write failing tests asserting the flag is mutually exclusive with `direct_message_hidden_readout`, forces one interaction, and creates `readout_products` plus nonlinear `readouts` with no `direct_readouts`.
- [ ] Run the focused tests and confirm they fail because the flag is absent.
- [ ] Add the parser field and thread it through `_build_model`, the MACE constructor, and configuration extraction.
- [ ] Run the focused tests and confirm construction/config round-trip passes.
- [ ] Commit with `feat: add single-v2 foundation mode`.

### Task 2: Make foundation transfer exact for one-layer OMat-0

**Files:**
- Modify: `mace/tools/finetuning_utils.py`
- Modify: `mace/tools/scripts_utils.py`
- Test: `tests/unit/test_finetuning_utils.py`

**Interfaces:**
- `load_foundations_elements_default` transfers only source index zero into a one-interaction target and leaves the target standard readout product/readout fresh.

- [ ] Add a failing regression test using a two-interaction source and one-interaction target; assert source first interaction/product tensors copy exactly, source second-layer tensors are ignored, and target readout parameters remain different from source.
- [ ] Run the regression test and confirm it fails with current layer/readout handling.
- [ ] Add the single-v2-specific skip for source `readouts.*`, preserve target `num_interactions`, and retain shape checks for all copied tensors.
- [ ] Run transfer tests, including species-remapping and E0/scale preservation, and confirm they pass.
- [ ] Commit with `fix: transfer OMat-0 first layer for single-v2`.

### Task 3: Add local construction and parameter-count gates

**Files:**
- Modify: `tests/unit/test_models.py`
- Modify: `tests/unit/test_compile_training.py`
- Create: `tests/unit/test_single_v2.py`

**Interfaces:**
- Construction from the medium OMat-0 checkpoint must produce finite outputs, standard scalar readout modules, and an exact trainable-parameter count of `7032730` in the pinned runtime.

- [ ] Add failing tests for one-layer topology, hidden irreps, no direct head, trainability of transferred parameters, exact E0/scale/shift, and finite energy/force/stress.
- [ ] Run the tests and confirm they fail before the mode exists.
- [ ] Implement only the minimum model/config changes needed to satisfy the gates.
- [ ] Run the focused suite, compile smoke test, `py_compile`, and `git diff --check`.
- [ ] Commit with `test: gate single-v2 topology and parameter count`.

### Task 4: Add the Cata launcher and qualify a fresh run

**Files:**
- Create: `scripts/cata/train_single_v2_omat0.sbatch`
- Create: `scripts/cata/README-single-v2.md`
- Test: `tests/unit/test_cata_single_v2_launcher.py`

**Interfaces:**
- Launcher accepts a fresh run root and invokes the pinned CUDA 13 runtime with two RTX 5090 ranks, eight CPUs per rank, four workers, batch 256, and ten epochs.
- Launcher passes the single-v2 flag, source foundation checkpoint, `E0s=foundation`, matched loss/EMA/scheduler controls, and OMat-0 radial settings.

- [ ] Add failing launcher contract tests for resources, exact objective, source checkpoint, no direct-head flag, and isolated output/cache roots.
- [ ] Run the launcher tests and confirm they fail because the script is absent.
- [ ] Implement the fail-closed launcher and local dry-render/preflight checks.
- [ ] Run launcher syntax, dry-render, model construction, and focused/full test suites.
- [ ] Cancel Slurm job `561679` only after all local gates pass, then submit one fresh `single-v2` job on `gn05`.
- [ ] Record the new job ID, run root, source hash, parameter count, and initial validation metrics.
- [ ] Commit with `feat: add single-v2 Cata launcher`.

## Verification Summary

Before cancellation/submission, require focused tests, full relevant tests, Python compilation, launcher syntax, exact parameter count, finite local forward/backward, and a dry-render showing the pinned runtime and isolated paths. After submission, verify source-load provenance, one compiled DDP graph per rank, finite epoch-0 validation, and stable host memory.
