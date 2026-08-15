# Compiler-Friendly DDP Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support `--compile_training --distributed` by wrapping the compiler-friendly training facade with PyTorch's existing `DistributedDataParallel`, while keeping convolution fusion off for ordinary converted training.

**Architecture:** Register the base MACE model normally inside `CompiledTrainingModel`, construct the facade lazily from the first real batch, then wrap that facade with the existing PyTorch DDP class. DDP retains parameter discovery, gradient buckets, NCCL/Gloo reduction, and backward overlap; optimizers, EMA, checkpoints, validation, and exports continue to use the unwrapped base model.

**Tech Stack:** Python 3.10+, PyTorch 2.0+ with `torch.compile`, PyTorch DDP/Gloo/NCCL, pytest, MACE HDF5 training, cuEquivariance.

## Global Constraints

- Do not add a MACE-specific DDP class or manual gradient reducer.
- Do not compile DDP or NCCL control flow; compile only the MACE training graph inside DDP.
- Keep `find_unused_parameters=False`, matching existing eager DDP behavior.
- Keep LBFGS and unsupported compiled objectives rejected.
- Preserve checkpoint keys, optimizer/EMA ownership, resume behavior, exports, and explicit `--only_cueq` semantics.
- Pass `conv_fusion=False` only from the ordinary converted training path; direct inference conversion keeps its CUDA default.
- Use PyTorch 2.11+cu128 for GPU qualification and install no new dependency.
- Preserve the unrelated changes already present in `mace/tools/compile.py` and `tests/unit/test_compile.py`; never stage them with this feature.

---

### Task 1: Make convolution fusion an explicit conversion policy

**Files:**
- Modify: `mace/cli/convert_e3nn_cueq.py:215-252`
- Modify: `mace/cli/run_train.py:870-889`
- Test: `tests/test_convert_e3nn_cueq.py`
- Test: `tests/unit/test_train_compile_integration.py`

**Interfaces:**
- Consumes: existing `convert_e3nn_cueq.run(input_model, output_model, device, return_model, layout)`.
- Produces: `convert_e3nn_cueq.run(..., conv_fusion: bool | None = None)` and `run_train._convert_e3nn_to_cueq_for_training(model, device)`.

- [ ] **Step 1: Add failing converter policy tests**

Extend the conversion stub test so the default remains device-selected and an explicit override is honored:

```python
@pytest.mark.parametrize(
    ("device", "conv_fusion", "expected"),
    [
        ("cuda", None, True),
        (torch.device("cuda"), None, True),
        ("cuda", False, False),
        ("cpu", True, True),
    ],
)
def test_conversion_selects_requested_convolution_fusion(
    monkeypatch, device, conv_fusion, expected
):
    monkeypatch.setattr(
        convert_e3nn_cueq,
        "extract_config_mace_model",
        lambda model: {
            "hidden_irreps": _HiddenIrrepsStub(),
            "correlation": 2,
            "num_interactions": 0,
        },
    )
    converted = convert_e3nn_cueq.run(
        _ConversionModelStub(), device=device, conv_fusion=conv_fusion
    )
    assert converted.cueq_config.conv_fusion is expected
```

Add a focused training-policy test:

```python
def test_training_conversion_disables_convolution_fusion(monkeypatch):
    calls = []
    source = object()
    converted = object()
    monkeypatch.setattr(run_train, "deepcopy", lambda model: model)
    monkeypatch.setattr(
        run_train,
        "run_e3nn_to_cueq",
        lambda model, **kwargs: calls.append((model, kwargs)) or converted,
    )
    assert run_train._convert_e3nn_to_cueq_for_training(source, "cuda") is converted
    assert calls == [(source, {"device": "cuda", "conv_fusion": False})]
```

- [ ] **Step 2: Run the policy tests and verify RED**

Run: `uv run pytest tests/test_convert_e3nn_cueq.py tests/unit/test_train_compile_integration.py::test_training_conversion_disables_convolution_fusion -q`

Expected: failure because `run()` has no `conv_fusion` argument and the training helper does not exist.

- [ ] **Step 3: Write the minimum explicit policy**

Add the optional converter argument, resolve `None` to the current CUDA-device default, and pass the resolved boolean into `CuEquivarianceConfig`. Add `_convert_e3nn_to_cueq_for_training(model, device)` in `run_train.py`; it deep-copies the base model and calls the converter with `conv_fusion=False`. Replace the inline training conversion call with this helper.

- [ ] **Step 4: Run focused and surrounding conversion tests**

Run: `uv run pytest tests/test_convert_e3nn_cueq.py tests/unit/test_train_compile_integration.py -q`

Expected: all tests pass; direct CUDA conversion still selects fusion and training explicitly disables it.

- [ ] **Step 5: Commit the conversion policy alone**

```bash
git add mace/cli/convert_e3nn_cueq.py mace/cli/run_train.py tests/test_convert_e3nn_cueq.py tests/unit/test_train_compile_integration.py
git commit -m "Keep cuEq convolution fusion off for training"
```

---

### Task 2: Make the compiled facade a DDP-compatible module owner

**Files:**
- Modify: `mace/tools/compile_training.py:580-720`
- Test: `tests/unit/test_compile_training.py`

**Interfaces:**
- Consumes: `CompiledTrainingModel(model, example_batch, output_args, mode, dynamic)`.
- Produces: registered `base_model`, unchanged base-compatible state operations, and `disable_compilation(reason: str | None = None) -> None`.

- [ ] **Step 1: Add failing module-tree and state tests**

Add assertions to the existing full-objective test that `base_model` appears in `named_modules()`, facade and base parameter names and object identities match, every parameter appears once, and facade state keys remain identical to base state keys. Add a test that calls `disable_compilation("distributed rank fallback")` and verifies eager state plus unchanged parameters/state keys.

- [ ] **Step 2: Run the facade tests and verify RED**

Run: `uv run pytest tests/unit/test_compile_training.py::test_compiled_force_stress_training_matches_eager_gradients_and_state_dict tests/unit/test_compile_training.py::test_compiled_training_can_disable_compilation_without_losing_base_state -q`

Expected: failure because `base_model` is not registered and `disable_compilation` is absent.

- [ ] **Step 3: Register the base model and add the state transition**

Replace tuple ownership with `self.base_model = model`, remove `_base_model_ref` and its property, and add:

```python
def disable_compilation(self, reason: str | None = None) -> None:
    self._compiled_ref = (None,)
    if reason is not None:
        self.compilation_error = reason
```

Keep `state_dict`, `load_state_dict`, `train`, and parameter/buffer delegation base-compatible. Verify that the compiled FX graph still receives the same parameter identities.

- [ ] **Step 4: Run the complete compiler-friendly unit suite**

Run: `uv run pytest tests/unit/test_compile_training.py tests/unit/test_train_compile_integration.py -q`

Expected: all CPU tests pass, including eager/compiled output and gradient parity.

- [ ] **Step 5: Commit facade ownership separately**

```bash
git add mace/tools/compile_training.py tests/unit/test_compile_training.py
git commit -m "Expose compiled training state to DDP"
```

---

### Task 3: Route compiled training through the existing DDP wrapper

**Files:**
- Modify: `mace/cli/run_train.py:91-100, 980-1065`
- Modify: `mace/tools/train.py:160-455`
- Modify: `mace/tools/arg_parser.py:79-92`
- Test: `tests/unit/test_train_compile_integration.py`
- Test: `tests/unit/test_arg_parser.py`

**Interfaces:**
- Consumes: registered `CompiledTrainingModel`, PyTorch `DistributedDataParallel`, and existing `distributed_model` eager path.
- Produces: `_compiled_training_facade`, `_synchronize_compiled_training_state`, cached DDP-wrapped compiled facade, and `distributed_device_ids` plumbing.

- [ ] **Step 1: Replace rejection expectations with DDP-routing tests**

Change unsupported-mode parametrizations so only LBFGS remains rejected. Add a `train_one_epoch` test that monkeypatches `compile_training_model`, `_synchronize_compiled_training_state`, `DistributedDataParallel`, and `take_step`; assert the facade is wrapped exactly once, the wrapper is cached, every forward enters that wrapper, and the base model still owns clipping/optimizer state. Add a run-level test proving eager DDP construction is deferred only for compiled distributed mode. Update the parser help assertion so it mentions distributed support rather than a single device.

- [ ] **Step 2: Run routing tests and verify RED**

Run: `uv run pytest tests/unit/test_train_compile_integration.py tests/unit/test_arg_parser.py -q`

Expected: distributed compile is still rejected and no deferred compiled DDP path exists.

- [ ] **Step 3: Implement DDP construction and lifecycle plumbing**

Remove only the distributed compile rejection; retain LBFGS/output validation. Compute the existing DDP device IDs once in `run_train.run`. Construct eager DDP only when distributed and not compiling, pass the device IDs into `tools.train`, and add the same optional argument to `train` and `train_one_epoch`.

After facade construction from the first batch, synchronize setup state and wrap the facade:

```python
compiled_training_model = _synchronize_compiled_training_state(
    compiled_training_model, device=device, distributed=distributed
)
if distributed:
    compiled_training_model = DistributedDataParallel(
        compiled_training_model,
        device_ids=distributed_device_ids,
    )
```

Continue calling `take_step` with the base model as `model` and cached DDP wrapper as `forward_model`. During validation, use the base model whenever compiled training is active. Ensure the compiled DDP wrapper is released before end-of-training evaluation constructs a base-model DDP wrapper.

- [ ] **Step 4: Implement rank-consistent setup and runtime fallback**

Add `_compiled_training_facade(model)` to unwrap DDP and `_synchronize_compiled_training_state(model, *, device, distributed)` to reduce a one-bit `is_compiled` flag with `ReduceOp.MIN`. If any rank is eager, call `disable_compilation("another distributed rank entered eager fallback")` on compiled peers. Call it before DDP construction and after every completed optimizer step; log only the transition.

- [ ] **Step 5: Run the complete integration suite**

Run: `uv run pytest tests/unit/test_train_compile_integration.py tests/unit/test_arg_parser.py tests/unit/test_compile_training.py -q`

Expected: all tests pass; eager distributed and single-device compile behavior remain unchanged.

- [ ] **Step 6: Commit DDP routing separately**

```bash
git add mace/cli/run_train.py mace/tools/train.py mace/tools/arg_parser.py tests/unit/test_train_compile_integration.py tests/unit/test_arg_parser.py
git commit -m "Support compiled training through DDP"
```

---

### Task 4: Prove real Gloo reducer and workflow correctness

**Files:**
- Modify: `tests/workflows/test_distributed.py`
- Create: `tests/unit/test_compile_training_ddp.py`

**Interfaces:**
- Consumes: DDP-wrapped `CompiledTrainingModel` and rank-consistent fallback from Task 3.
- Produces: two-process global-gradient/parameter parity coverage and end-to-end CLI coverage.

- [ ] **Step 1: Add a two-process reducer-discovery and parity test**

Create a top-level worker that initializes Gloo on loopback, constructs the same small float64 MACE model on both ranks, gives each rank a distinct local batch, constructs the compiled facade, wraps it in DDP, executes one full energy/force/stress backward and Adam step, and saves the base-model state for the parent. The parent asserts rank states match, DDP reducer parameter identities match base parameter identities exactly once, and the update matches a single-process global-batch reference. Always destroy the process group in `finally`.

- [ ] **Step 2: Run the new Gloo test independently**

Run outside the sandbox because distributed execution is prohibited inside: `uv run pytest tests/unit/test_compile_training_ddp.py -q`

Expected: two ranks exit successfully, reducer parameters are complete and unique, and both ranks match the global reference.

- [ ] **Step 3: Parameterize the existing CLI workflow**

Parameterize `test_run_train_distributed_cpu` over eager and compiled modes. For compiled mode add `--compile_training`, use a distinct model name, and assert the log records compiled DDP activation without recording fallback. Retain stage-one/stage-two save assertions.

- [ ] **Step 4: Run eager and compiled distributed workflows**

Run outside the sandbox: `uv run pytest tests/workflows/test_distributed.py -q`

Expected: eager and compiled two-rank Gloo jobs finish, save final models, and do not deadlock at stage transitions.

- [ ] **Step 5: Commit distributed correctness coverage**

```bash
git add tests/unit/test_compile_training_ddp.py tests/workflows/test_distributed.py
git commit -m "Test compiled training with distributed reducers"
```

---

### Task 5: Qualify checkpointing, fallback, and real CUDA DDP

**Files:**
- Modify: `tests/unit/test_train_compile_integration.py`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/compiled-ddp/README.md`

**Interfaces:**
- Consumes: full feature implementation and variable-shape HDF5 smoke data.
- Produces: resume/fallback regressions and reproducible eager-versus-compiled NCCL evidence.

- [ ] **Step 1: Add fallback and state-ownership regressions**

Add tests proving that simulated one-rank setup failure disables all facades, runtime fallback is synchronized after the completed step, and checkpoint state is taken from the base model without `module.` or `base_model.` prefixes. Include optimizer and EMA parameter-identity assertions.

- [ ] **Step 2: Run focused state and fallback tests**

Run: `uv run pytest tests/unit/test_train_compile_integration.py -q`

Expected: all setup/runtime fallback and state-ownership tests pass.

- [ ] **Step 3: Add a bounded multi-GPU launcher**

Create a launcher for identical two- and four-rank eager/compiled variable-shape HDF5 workloads using PyTorch 2.11+cu128, converted cuEquivariance with `conv_fusion=False`, batch 64/rank, full energy/force/stress, Adam+AMSGrad, clipping, EMA, two warm-ups, and at least sixteen timed steps. Record per-rank loss, timing, memory, compile/fallback state, and final parameter checksum. Use `torchrun`, validate visible GPU count, and keep source HDF5 read-only.

- [ ] **Step 4: Run real NCCL correctness and performance controls**

Run outside the sandbox:

```bash
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh eager 2
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 2
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh eager 4
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 4
```

Expected: ranks have identical final checksums, no specialization/module-load storm occurs, memory is bounded, and compiled DDP improves warmed complete-step throughput. If the host lacks GPUs, record the unavailable control and run it on the production node before deployment.

- [ ] **Step 5: Capture a bounded Nsight Systems comparison**

Profile at least four warmed steps with CUDA, NVTX, and NCCL tracing. Report kernels, NCCL collectives, compute/communication overlap, GPU-active fraction, complete step time, and structures/s. Do not infer overlap from kernel counts alone.

- [ ] **Step 6: Record qualification evidence**

Write exact environment, commands, results, medians, parity, and limitations to the compiled-DDP README and `.profiling/mace-hdf5-gpu-2026-08-13/REPORT.md`. Keep raw profiler reports out of source commits unless explicitly requested.

- [ ] **Step 7: Commit lightweight qualification sources**

```bash
git add tests/unit/test_train_compile_integration.py .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh .profiling/mace-hdf5-gpu-2026-08-13/compiled-ddp/README.md
git commit -m "Qualify compiled DDP training"
```

---

### Task 6: Final regression and review gate

**Files:**
- Verify: all files changed by Tasks 1-5
- Update: `docs/superpowers/specs/2026-08-15-compiled-training-ddp-design.md` only if evidence requires a clarified limitation
- Update: `.profiling/mace-hdf5-gpu-2026-08-13/REPORT.md`

**Interfaces:**
- Consumes: completed implementation and qualification artifacts.
- Produces: merge-ready source commits and an evidence-backed deployment recommendation.

- [ ] **Step 1: Run focused CPU suites**

Run: `uv run pytest tests/test_convert_e3nn_cueq.py tests/unit/test_compile.py tests/unit/test_compile_training.py tests/unit/test_train_compile_integration.py tests/unit/test_compile_training_ddp.py tests/unit/test_arg_parser.py -q`

Expected: all tests pass with no unexpected skips or warnings promoted to errors.

- [ ] **Step 2: Run distributed workflow and source checks**

Run distributed tests outside the sandbox, then run:

```bash
uv run pytest tests/workflows/test_distributed.py -q
python -m py_compile mace/cli/convert_e3nn_cueq.py mace/cli/run_train.py mace/tools/compile_training.py mace/tools/train.py
bash -n .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Audit implementation against the specification**

Confirm only PyTorch DDP reduces gradients; the compiled facade owns the DDP forward; every trainable parameter appears once; wrappers never alter checkpoint/export keys; training conversion disables fusion while inference defaults remain; single-device compile, eager DDP, resume, EMA, SWA, validation, and export pass; and multi-GPU throughput includes compilation and NCCL time.

- [ ] **Step 4: Request a focused code review**

Review against the pre-feature commit with priority on collective ordering, parameter ownership, fallback, checkpoint compatibility, and fusion policy. Resolve every P1/P2 finding and rerun affected gates.

- [ ] **Step 5: Commit review fixes only if needed**

```bash
git add <only files intentionally changed by review fixes>
git commit -m "Harden compiled DDP training"
```

Do not stage `mace/tools/compile.py`, `tests/unit/test_compile.py`, unrelated profiling artifacts, or pre-existing plan files unless explicitly part of a reviewed change.
