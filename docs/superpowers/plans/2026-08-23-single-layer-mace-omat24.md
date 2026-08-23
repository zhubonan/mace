# Single-Interaction MACE OMat24 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and qualify paired one-interaction OMat24 models with and without a first-layer self-connection.

**Architecture:** Retain `128x0e + 128x1o + 128x2e` after the sole density interaction and its correlation-3 product. Reshape a copy into MACE's channel/irrep layout, contract it locally with correlation 2 to `128x0e`, and pass it through the existing `16x0e` nonlinear readout while preserving full node features for diagnostics.

**Tech Stack:** Python, PyTorch, e3nn, cuequivariance, pytest, TorchScript, `torch.compile`, ASE-LMDB, Slurm DDP.

## Global Constraints

- Base all code work on `training-opt` at or after `31e0f4b`.
- Use TDD: every behavior change must first fail for the expected reason.
- Use `uv` where environment management is needed.
- Run GPU and DDP commands outside the sandbox; GPU MACE requires cuequivariance.
- Keep interaction correlation 3 and readout correlation 2.
- Test worker counts 1, 4, and 8 only.
- Begin each Cata job with two GPUs.
- Do not modify `/home/bonan/work/train-downsample/mace-official`; use the reference framework at interface level.

---

### Task 1: Establish the optimized implementation base

**Files:**
- Preserve: `docs/superpowers/specs/2026-08-23-single-layer-mace-omat24-design.md`
- Preserve: `plans/2026-08-23-single-layer-mace-omat24-v1.md`

**Interfaces:**
- Consumes: local `training-opt` branch and the three approved design commits.
- Produces: branch `codex/single-layer-mace-omat24` based on `training-opt` with the approved documents retained.

- [ ] Create the implementation branch on `training-opt` and replay the approved documentation commits without modifying user-owned work.
- [ ] Verify `git merge-base HEAD training-opt` equals `training-opt` and confirm native OMat LMDB and balanced-sampler files are present.
- [ ] Commit only if replay creates unresolved documentation changes.

### Task 2: Drive the model behavior with failing tests

**Files:**
- Create: `tests/unit/test_single_interaction_readout.py`
- Modify: `mace/modules/models.py:48-275`

**Interfaces:**
- Consumes: `EquivariantProductBasisBlock`, `reshape_irreps`, `NonLinearReadoutBlock`.
- Produces: constructor argument `readout_correlation: int = 2`, `readout_reshapes`, and `readout_products` module lists.

- [ ] Add a small model factory that constructs one-interaction density residual/non-residual models with `8x0e + 8x1o + 8x2e`, `use_last_readout_only=True`, and `readout_correlation=2`.
- [ ] Assert the first product targets the full hidden irreps, the readout product targets `8x0e`, the readout contraction has degree 2, and exactly one learned nonlinear readout exists.
- [ ] Run `uv run pytest tests/unit/test_single_interaction_readout.py -q` and verify failure because `readout_correlation` and the readout product do not exist.
- [ ] Implement the special constructor branch, backend-aware reshape, readout product, and nonlinear readout in `MACE.__init__`; leave all ordinary constructor branches unchanged.
- [ ] Update both `MACE.forward` and `ScaleShiftMACE.forward` to contract a readout-only copy using node attributes while returning the original full node features.
- [ ] Re-run the focused test and verify it passes.

### Task 3: Prove equivariance and self-path semantics

**Files:**
- Modify: `tests/unit/test_single_interaction_readout.py`

**Interfaces:**
- Consumes: the model factory and readout-product attributes from Task 2.
- Produces: regression coverage for higher-order feature consumption and paired interaction semantics.

- [ ] Add a synthetic readout-product test that changes only valid `L=1` or `L=2` components and asserts the scalar output changes while a rotated copy produces the same scalar output.
- [ ] Add paired forward tests asserting invariant energy, covariant forces, finite stress/gradients, full node-feature width, and different self-path behavior for the two density interaction classes.
- [ ] Run the new tests and verify each new assertion fails before any corrective production edit.
- [ ] Make only the minimal layout or forward corrections required for the tests to pass.
- [ ] Run the focused file and `tests/unit/test_models.py` to verify special and existing model behavior.

### Task 4: Expose and persist readout correlation

**Files:**
- Modify: `mace/tools/arg_parser.py:242-275`
- Modify: `mace/tools/model_script_utils.py:187-322`
- Modify: `mace/tools/scripts_utils.py:275-329`
- Modify: `tests/unit/test_arg_parser.py`
- Modify: `tests/unit/test_single_interaction_readout.py`

**Interfaces:**
- Consumes: `MACE(readout_correlation=2)` from Task 2.
- Produces: CLI option `--readout_correlation`, exact density-residual class resolution, and round-trip config field `readout_correlation`.

- [ ] Add failing parser, model-factory, and `extract_config_mace_model` round-trip tests.
- [ ] Verify failures show the missing CLI field, silent density-residual replacement, and missing extracted value.
- [ ] Add `--readout_correlation` with integer default 2, forward it for `MACE` and `ScaleShiftMACE`, allow `RealAgnosticDensityResidualInteractionBlock` in the `MACE` first-interaction allowlist, and extract the stored value.
- [ ] Re-run the focused CLI/config tests and existing argument-parser suite.

### Task 5: Preserve backend conversion and compilation

**Files:**
- Modify: `mace/cli/convert_e3nn_cueq.py:87-212`
- Modify: `mace/cli/convert_cueq_e3nn.py:87-228`
- Modify: `tests/test_convert_e3nn_cueq.py`
- Modify: `tests/backends/backend_parity.py`
- Modify: `tests/unit/test_compile_training.py`

**Interfaces:**
- Consumes: `readout_products` and extracted `readout_correlation`.
- Produces: bidirectional symmetric-contraction transfer for state prefixes `products` and `readout_products`.

- [ ] Add a failing conversion test that requires readout symmetric-contraction parameters and predictions to survive e3nn-to-CuEq-to-e3nn conversion.
- [ ] Generalize the transfer helper to accept a state-dict prefix and invoke it once for interaction products and once for the optional readout product with one layer, correlation 2, and scalar output.
- [ ] Run conversion tests and verify parity.
- [ ] Add or extend compiled-training coverage for the special model, then run it first failing and finally passing.
- [ ] Run TorchScript, backend parity, conversion, and compiled-training focused suites.

### Task 6: Qualify the complete CPU implementation

**Files:**
- Modify only if failures reveal defects covered by a new failing regression test.

**Interfaces:**
- Consumes: Tasks 2-5.
- Produces: CPU-qualified implementation ready for GPU testing.

- [ ] Run formatting/static checks used by the repository on modified files.
- [ ] Run all focused model, parser, config, conversion, compile, and export tests.
- [ ] Run the broader unit suite in proportion to runtime and record exact pass/fail counts.
- [ ] Inspect `git diff --check`, `git status`, and the complete diff for unrelated changes.
- [ ] Commit the CPU-qualified feature with focused tests.

### Task 7: Adapt paired OMat qualification configuration

**Files:**
- Modify or create in the approved training-framework workspace after write authorization: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/` protocol/tests.
- Modify or create in the approved training-framework workspace after write authorization: `/home/bonan/work/train-downsample/scripts/cata/` local and two-rank launch scripts.

**Interfaces:**
- Consumes: the implemented MACE checkout and established native-LMDB/compiled-DDP protocol.
- Produces: two immutable run configurations differing only in `interaction_first`, local worker sweep 1/4/8, paired 500-update sessions, and two-rank Cata jobs.

- [ ] Add failing protocol tests for exact hidden irreps, one interaction, readout correlation 2, requested interaction class, worker candidates, restart boundary 250, completion boundary 500, and Cata world size 2.
- [ ] Render paired with-self and without-self configurations with identical seeds, sampler order, normalization, optimizer, scheduler, validation, and monitoring fields.
- [ ] Add fail-closed resolved-model assertions and distinct run/checkpoint/cache namespaces.
- [ ] Run the protocol tests and commit the framework adaptation separately.

### Task 8: Run local single-GPU qualification

**Files:**
- Create runtime artifacts only under an explicit run directory outside source control.

**Interfaces:**
- Consumes: Task 7 protocol and real native OMat LMDB.
- Produces: worker-selection report and two disposable 500-update qualification records.

- [ ] Audit dataset identity, runtime, cuequivariance import, GPU visibility, disk space, and file-descriptor limits outside the sandbox.
- [ ] Run 20 warm-up plus 100 measured updates for worker counts 1, 4, and 8 on the with-self model from the same initial state and sampler position.
- [ ] Select the highest stable end-to-end throughput and record all required telemetry.
- [ ] Run each variant to update 250, checkpoint/stop, resume to 500, validate, export, and verify finite metrics and checkpoint hashes.
- [ ] Do not promote local checkpoints to Cata.

### Task 9: Qualify and launch paired two-GPU Cata jobs

**Files:**
- Create runtime artifacts in separate with-self and without-self Cata run roots.

**Interfaces:**
- Consumes: CPU and local GPU qualification evidence.
- Produces: two monitored full OMat24 jobs, each with two DDP ranks and fresh initialization.

- [ ] Run two-rank production-shaped compiled-DDP gates for both variants and select a common measured worker count and per-rank batch size.
- [ ] Verify both resolved configurations, seeds, data manifests, normalization, rank count, cache isolation, output namespaces, and absence of local checkpoint reuse.
- [ ] Submit both jobs concurrently when four GPUs are available and scheduler policy permits.
- [ ] Monitor finite metrics, learning curves, throughput, data wait, GPU memory/utilization, rank balance, compile graph count, checkpoint lineage, and restart health.
- [ ] Compare both variants at matched updates/epochs with each other and MACE-OMAT-0.

### Task 10: Final verification and handoff

**Files:**
- Update: implementation and run evidence documentation selected during Task 7.

**Interfaces:**
- Consumes: all test, local qualification, and Cata evidence.
- Produces: reproducible code/configuration handoff and comparison report.

- [ ] Run fresh focused regression verification and confirm the source commit used by both jobs.
- [ ] Record exact commands, resolved configs, environment versions, dataset identity, seeds, checkpoint hashes, and known deviations.
- [ ] Report implementation status separately from long-running training status; do not claim scientific completion until both paired runs and comparison are complete.
