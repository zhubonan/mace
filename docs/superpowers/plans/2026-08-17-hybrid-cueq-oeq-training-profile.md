# Hybrid cuEq + OpenEquivariance Training Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether MACE's existing hybrid cuEquivariance plus OpenEquivariance model can perform correct, dynamically compiled full energy/force/stress training faster than unfused cuEquivariance.

**Architecture:** Install OpenEquivariance only into the qualified profiling runtime, add a full-double-backward qualification beside the existing compiled-training tests, and expose the already existing hybrid converter only through the retained profiling harness. Correctness, dynamic-shape stability, and optimizer-update parity gate all timing and Nsight work.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu128, e3nn 0.4.4, cuEquivariance 0.11.1, OpenEquivariance 0.6.8, pytest, HDF5, Nsight Systems, Nsight Compute.

## Global Constraints

- Use float32 only; do not enable TF32, BF16, FP16, or autocast.
- Keep cuEquivariance convolution fusion disabled.
- Keep the canonical MACE model as the owner of parameters, optimizer state, EMA state, checkpoints, and final `.model` files.
- Do not add a CLI flag or alter `--enable_cueq`, `--enable_oeq`, or `--only_cueq` behavior.
- Do not serialize OpenEquivariance, FX, compiled, experimental, or DDP wrappers.
- Require one stable dynamic graph across distinct atom and edge counts before profiling.
- Keep all diagnostic implementation changes uncommitted until the user reviews the results.
- Stop before timing if full force/stress double backward, trainable-parameter gradients, or dynamic compilation fails.

---

### Task 1: Qualify the isolated OpenEquivariance runtime

**Files:**
- Verify: `setup.cfg`
- Verify: `mace/modules/wrapper_ops.py`

**Interfaces:**
- Consumes: `.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python`
- Produces: an isolated runtime containing `openequivariance==0.6.8`, with MACE still imported from the current worktree and e3nn resolved from `/tmp/e3nn044-overlay`

- [ ] **Step 1: Record pre-install provenance**

Run outside the sandbox:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -c \
  'import importlib.util, e3nn, mace, torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "e3nn": e3nn.__version__, "mace": mace.__file__, "oeq": importlib.util.find_spec("openequivariance")})'
```

Expected: PyTorch `2.11.0+cu128`, e3nn `0.4.4`, current-worktree MACE, and no OpenEquivariance module before installation.

- [ ] **Step 2: Install the exact isolated dependency set**

Run outside the sandbox, following `AGENTS.md`:

```bash
uv pip install \
  --python .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  openequivariance==0.6.8
```

Expected additions: `openequivariance==0.6.8`, `ninja==1.13.0`, and `pybind11==3.1.0`. Do not alter `setup.cfg`, the legacy runtime, or the base environment.

- [ ] **Step 3: Run the real CUDA import and JIT-construction smoke**

Run outside the sandbox:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -c \
  'import openequivariance as oeq, torch; from e3nn import o3; p=oeq.TPProblem(o3.Irreps("2x0e"),o3.Irreps("1x0e"),o3.Irreps("2x0e"),[(0,0,0,"uvu",True)] ,shared_weights=False,internal_weights=False,irrep_dtype=oeq.torch_to_oeq_dtype(torch.float32),weight_dtype=oeq.torch_to_oeq_dtype(torch.float32)); m=oeq.TensorProductConv(p,deterministic=False).cuda(); print({"oeq": oeq.__version__, "device": torch.cuda.get_device_name(), "module": type(m).__name__})'
```

Expected: an RTX 5090 device and a constructed `TensorProductConv`. If the installed API differs, inspect `inspect.signature` and update only the diagnostic smoke; do not modify MACE to hide an incompatible package API.

- [ ] **Step 4: Verify the complete post-install runtime provenance**

Run outside the sandbox:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python -c \
  'import importlib.metadata, e3nn, mace, openequivariance, torch; print({"torch":torch.__version__,"cuda":torch.version.cuda,"e3nn":e3nn.__version__,"cuequivariance":importlib.metadata.version("cuequivariance"),"openequivariance":openequivariance.__version__,"mace":mace.__file__,"device":torch.cuda.get_device_name(),"dtype":str(torch.get_default_dtype())})'
```

Expected: exact qualified versions, current-worktree MACE, RTX 5090, and
`torch.float32`. Task 4 will retain the same provenance atomically beside each
profile artifact.

- [ ] **Step 5: Review checkpoint without committing**

Run:

```bash
git status --short
```

Expected: installing the isolated package creates no repository changes. Do not commit.

---

### Task 2: Prove eager full-double-backward and optimizer parity

**Files:**
- Modify: `tests/unit/test_compile_training.py`
- Inspect: `mace/cli/convert_e3nn_hybrid.py`
- Inspect: `mace/cli/convert_e3nn_cueq.py`

**Interfaces:**
- Consumes: `_create_model`, `_create_batch`, `_batch_to`, `_objective`, and `_assert_parameter_gradients_close` from `tests/unit/test_compile_training.py`
- Produces: `test_hybrid_full_force_stress_optimizer_parity_cuda`, the mandatory eager higher-derivative feasibility gate

- [ ] **Step 1: Add imports and the failing/characterization test**

Add these imports to `tests/unit/test_compile_training.py`:

```python
from mace.cli.convert_e3nn_cueq import run as convert_e3nn_to_cueq
from mace.cli.convert_e3nn_hybrid import run as convert_e3nn_to_hybrid
from tests.helpers import OEQ_AVAILABLE
```

Add this GPU test after the existing compiled CUDA optimizer test:

```python
@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.oeq
def test_hybrid_full_force_stress_optimizer_parity_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")
    if not CUET_AVAILABLE or not OEQ_AVAILABLE:
        pytest.skip("hybrid dependencies are not installed")

    with tools.torch_tools.default_dtype(torch.float32):
        source = _create_model()
        baseline = convert_e3nn_to_cueq(
            deepcopy(source), device="cuda", conv_fusion=False
        ).cuda()
        hybrid = convert_e3nn_to_hybrid(
            deepcopy(source), device="cuda", return_model=True
        ).cuda()
        baseline_batch = _batch_to(_create_batch(), "cuda")
        hybrid_batch = _batch_to(baseline_batch, "cuda")
        baseline_optimizer = torch.optim.Adam(
            baseline.parameters(), lr=1.0e-3, amsgrad=True
        )
        hybrid_optimizer = torch.optim.Adam(
            hybrid.parameters(), lr=1.0e-3, amsgrad=True
        )

        baseline_outputs = baseline(
            baseline_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        hybrid_outputs = hybrid(
            hybrid_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        for key in ("energy", "forces", "virials", "stress"):
            assert_close(baseline_outputs[key], hybrid_outputs[key])

        _objective(baseline_outputs).backward()
        _objective(hybrid_outputs).backward()
        _assert_parameter_gradients_close(baseline, hybrid)
        for name, parameter in baseline.named_parameters():
            assert parameter.grad is not None, name
        baseline_optimizer.step()
        hybrid_optimizer.step()
        baseline_parameters = dict(baseline.named_parameters())
        hybrid_parameters = dict(hybrid.named_parameters())
        assert baseline_parameters.keys() == hybrid_parameters.keys()
        for name, baseline_parameter in baseline_parameters.items():
            hybrid_parameter = hybrid_parameters[name]
            assert_close(baseline_parameter, hybrid_parameter)
            baseline_state = baseline_optimizer.state[baseline_parameter]
            hybrid_state = hybrid_optimizer.state[hybrid_parameter]
            assert baseline_state.keys() == hybrid_state.keys()
            for state_name, baseline_value in baseline_state.items():
                hybrid_value = hybrid_state[state_name]
                if isinstance(baseline_value, torch.Tensor):
                    assert_close(baseline_value, hybrid_value)
                else:
                    assert baseline_value == hybrid_value
```

- [ ] **Step 2: Run the test and classify the result**

Run outside the sandbox:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/unit/test_compile_training.py::test_hybrid_full_force_stress_optimizer_parity_cuda -vv
```

Expected gate: PASS with non-missing gradients for every trainable parameter. A failure from `TensorProductConv` backward-backward, missing weight gradients, or unsupported dtype is a scientific stop result. Record the exact traceback and do not weaken the objective.

- [ ] **Step 3: If the failure is only parameter-name representation, prove the mapping**

Use this diagnostic before changing the comparison:

```python
baseline_names = tuple(dict(baseline.named_parameters()))
hybrid_names = tuple(dict(hybrid.named_parameters()))
assert baseline_names == hybrid_names, (baseline_names, hybrid_names)
```

Expected: identical optimizer-owned parameter names. If names differ, compare the converter's explicit source-to-target mapping and add a named mapping table; never rely on `zip(model.parameters())`.

- [ ] **Step 4: Run existing backend and compiled-training regressions**

Run outside the sandbox:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/backends/test_oeq.py tests/unit/test_compile_training.py -q
```

Expected: existing tests pass; GPU-marked tests execute rather than skip when their dependencies are present.

- [ ] **Step 5: Review checkpoint without committing**

Run `git diff --check` and retain the red/green result. Do not commit.

---

### Task 3: Prove dynamic compiled full-update parity

**Files:**
- Modify: `tests/unit/test_compile_training.py`
- Inspect: `mace/tools/compile_training.py`

**Interfaces:**
- Consumes: the qualified hybrid model from Task 2 and `mace_compile.compile_training_model`
- Produces: `test_hybrid_compiled_force_stress_dynamic_update_parity_cuda`

- [ ] **Step 1: Write the dynamic compiled parity test**

Add:

```python
@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.oeq
def test_hybrid_compiled_force_stress_dynamic_update_parity_cuda():
    if not torch.cuda.is_available() or not CUET_AVAILABLE or not OEQ_AVAILABLE:
        pytest.skip("hybrid CUDA dependencies are unavailable")
    with tools.torch_tools.default_dtype(torch.float32):
        source = _create_model()
        reference = convert_e3nn_to_hybrid(
            deepcopy(source), device="cuda", return_model=True
        ).cuda()
        candidate = convert_e3nn_to_hybrid(
            deepcopy(source), device="cuda", return_model=True
        ).cuda()
        trace_batch = _batch_to(_create_batch(num_graphs=2), "cuda")
        validation_batch = _batch_to(_create_batch(num_graphs=3), "cuda")
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }
        compiled = mace_compile.compile_training_model(
            candidate,
            trace_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=validation_batch,
        )
        assert compiled.is_compiled, compiled.compilation_error

        for raw_batch in (trace_batch, validation_batch):
            reference_batch = _batch_to(raw_batch, "cuda")
            candidate_batch = _batch_to(raw_batch, "cuda")
            reference_outputs = reference(
                reference_batch,
                training=True,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
            )
            candidate_outputs = compiled(candidate_batch)
            for key in ("energy", "forces", "virials", "stress"):
                assert_close(reference_outputs[key], candidate_outputs[key])
            _objective(reference_outputs).backward()
            _objective(candidate_outputs).backward()
            _assert_parameter_gradients_close(reference, candidate)
            for model in (reference, candidate):
                model.zero_grad(set_to_none=True)
            assert compiled.is_compiled, compiled.compilation_error
```

- [ ] **Step 2: Run and verify the dynamic gate**

Run outside the sandbox:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/unit/test_compile_training.py::test_hybrid_compiled_force_stress_dynamic_update_parity_cuda -vv
```

Expected: one stable compiled facade across both shapes. Any AOTAutograd/custom-op error or eager fallback stops the hybrid experiment before profiling.

- [ ] **Step 3: Extend the test to one optimizer update after dynamic validation**

Append this one-update check after the dynamic-validation loop:

```python
        reference_optimizer = torch.optim.Adam(
            reference.parameters(), lr=1.0e-3, amsgrad=True
        )
        candidate_optimizer = torch.optim.Adam(
            candidate.parameters(), lr=1.0e-3, amsgrad=True
        )
        reference_batch = _batch_to(validation_batch, "cuda")
        candidate_batch = _batch_to(validation_batch, "cuda")
        reference_outputs = reference(
            reference_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        candidate_outputs = compiled(candidate_batch)
        reference_loss = _objective(reference_outputs)
        candidate_loss = _objective(candidate_outputs)
        assert_close(reference_loss, candidate_loss)
        reference_loss.backward()
        candidate_loss.backward()
        _assert_parameter_gradients_close(reference, candidate)
        reference_optimizer.step()
        candidate_optimizer.step()

        reference_parameters = dict(reference.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        assert reference_parameters.keys() == candidate_parameters.keys()
        for name, reference_parameter in reference_parameters.items():
            candidate_parameter = candidate_parameters[name]
            assert_close(reference_parameter, candidate_parameter)
            reference_state = reference_optimizer.state[reference_parameter]
            candidate_state = candidate_optimizer.state[candidate_parameter]
            assert reference_state.keys() == candidate_state.keys()
            for state_name, reference_value in reference_state.items():
                candidate_value = candidate_state[state_name]
                if isinstance(reference_value, torch.Tensor):
                    assert_close(reference_value, candidate_value)
                else:
                    assert reference_value == candidate_value
        assert compiled.is_compiled, compiled.compilation_error
```

Expected: `assert_close` passes with float32 defaults and no missing optimizer state.

- [ ] **Step 4: Run the two-rank gate**

Extend `.profiling/mace-hdf5-gpu-2026-08-13/scripts/compiled_ddp_benchmark.py` in Task 4 first, then run:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_MODEL_BACKEND=hybrid \
  MACE_DDP_MODEL_PRESET=small \
  MACE_DDP_BATCH_SIZE=8 \
  MACE_DDP_WARMUP_STEPS=2 \
  MACE_DDP_TIMED_STEPS=4 \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 2
```

Expected: both ranks remain compiled and end with identical checksums. If only one local GPU is available, retain this as a remote qualification gate and do not substitute a one-rank claim.

- [ ] **Step 5: Review checkpoint without committing**

Run `git diff --check`; retain results and stop if any mandatory gate failed.

---

### Task 4: Add profiling-only backend selection and capture matched traces

**Files:**
- Modify: `.profiling/mace-hdf5-gpu-2026-08-13/scripts/compiled_ddp_benchmark.py`
- Modify: `.profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/{cueq,hybrid}/rank-0.json`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/{cueq,hybrid}/trace.nsys-rep`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/{cueq,hybrid}/trace.sqlite`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/{cueq,hybrid}/environment.json`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/timing/{cueq,hybrid}-b{64,128}/rank-0.json`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/smoke/{cueq,hybrid}/rank-0.json`

**Interfaces:**
- Consumes: `convert_e3nn_to_cueq` and `convert_e3nn_to_hybrid`
- Produces: `--model-backend={cueq,hybrid}`, launcher controls for workers and CUDA-profiler capture, and JSON field `model_backend`

- [ ] **Step 1: Add the profiling-only argument**

Add to `_parse_args()`:

```python
parser.add_argument(
    "--model-backend",
    choices=("cueq", "hybrid"),
    default="cueq",
)
```

Add `model_backend: str` to `_build_model`, and replace its final conversion with:

```python
if model_backend == "hybrid":
    from mace.cli.convert_e3nn_hybrid import run as convert_e3nn_to_hybrid

    with conversion_context:
        return convert_e3nn_to_hybrid(
            model, device=device, return_model=True
        )
with conversion_context:
    return convert_e3nn_to_cueq(
        model,
        device=device,
        conv_fusion=False,
    )
```

Pass `args.model_backend` from `main()` and record it in `rank-0.json`.
Add `import e3nn`, `import importlib.metadata`, and `import subprocess`. After creating
`args.output_dir` and before model construction, have rank zero emit the
following provenance beside the result:

```python
environment = {
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "e3nn": e3nn.__version__,
    "cuequivariance": importlib.metadata.version("cuequivariance"),
    "openequivariance": importlib.metadata.version("openequivariance"),
    "device": torch.cuda.get_device_name(),
    "default_dtype": str(torch.get_default_dtype()),
}
if rank == 0:
    environment_path = args.output_dir / "environment.json"
    temporary_path = environment_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(environment_path)
```

- [ ] **Step 2: Forward the environment-controlled launcher option**

Add the worker option to the existing command:

```bash
--model-backend="${MACE_DDP_MODEL_BACKEND:-cueq}" \
--num-workers="${MACE_DDP_NUM_WORKERS:-0}" \
```

Add the profiler switch next to the existing `--dump-fx-only` handling:

```bash
if [[ "${MACE_DDP_CUDA_PROFILER_CAPTURE:-0}" == "1" ]]; then
    extra_args+=(--cuda-profiler-capture)
fi
```

Expected: existing commands remain cuEq by default.

- [ ] **Step 3: Run a syntax and one-step provenance smoke**

Run outside the sandbox:

```bash
.profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m py_compile \
  .profiling/mace-hdf5-gpu-2026-08-13/scripts/compiled_ddp_benchmark.py

env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_MODEL_BACKEND=cueq \
  MACE_DDP_MODEL_PRESET=small \
  MACE_DDP_BATCH_SIZE=8 \
  MACE_DDP_NUM_WORKERS=0 \
  MACE_DDP_WARMUP_STEPS=2 \
  MACE_DDP_TIMED_STEPS=1 \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/smoke/cueq" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1

env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_MODEL_BACKEND=hybrid \
  MACE_DDP_MODEL_PRESET=small \
  MACE_DDP_BATCH_SIZE=8 \
  MACE_DDP_NUM_WORKERS=0 \
  MACE_DDP_WARMUP_STEPS=2 \
  MACE_DDP_TIMED_STEPS=1 \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/smoke/hybrid" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1
```

Expected: both rank JSON files state the correct backend and active compiled
state; both environment JSON files state the same source commit and float32
policy.

- [ ] **Step 4: Run matched production timing controls**

Run each command outside the sandbox in an independent process:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_MODEL_BACKEND=cueq MACE_DDP_MODEL_PRESET=production \
  MACE_DDP_BATCH_SIZE=64 MACE_DDP_NUM_WORKERS=4 MACE_DDP_WARMUP_STEPS=8 \
  MACE_DDP_TIMED_STEPS=24 \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/timing/cueq-b64" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1

env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_MODEL_BACKEND=hybrid MACE_DDP_MODEL_PRESET=production \
  MACE_DDP_BATCH_SIZE=64 MACE_DDP_NUM_WORKERS=4 MACE_DDP_WARMUP_STEPS=8 \
  MACE_DDP_TIMED_STEPS=24 \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/timing/hybrid-b64" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1

env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_MODEL_BACKEND=cueq MACE_DDP_MODEL_PRESET=production \
  MACE_DDP_BATCH_SIZE=128 MACE_DDP_NUM_WORKERS=4 MACE_DDP_WARMUP_STEPS=8 \
  MACE_DDP_TIMED_STEPS=24 \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/timing/cueq-b128" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1

env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_MODEL_BACKEND=hybrid MACE_DDP_MODEL_PRESET=production \
  MACE_DDP_BATCH_SIZE=128 MACE_DDP_NUM_WORKERS=4 MACE_DDP_WARMUP_STEPS=8 \
  MACE_DDP_TIMED_STEPS=24 \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/timing/hybrid-b128" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1
```

Expected: stable graph counters, no fallback, finite losses, identical batch-shape sequences, and comparable peak memory.

- [ ] **Step 5: Capture eight post-warmup Nsight steps**

Run the cuEq control from the repository root:

```bash
env \
  PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/cueq" \
  MACE_DDP_MODEL_BACKEND=cueq \
  MACE_DDP_MODEL_PRESET=production \
  MACE_DDP_BATCH_SIZE=128 \
  MACE_DDP_NUM_WORKERS=4 \
  MACE_DDP_WARMUP_STEPS=8 \
  MACE_DDP_TIMED_STEPS=8 \
  MACE_DDP_CUDA_PROFILER_CAPTURE=1 \
  /usr/local/cuda/bin/nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/cueq/trace" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1

/usr/local/cuda/bin/nsys export \
  --force-overwrite=true \
  --type=sqlite \
  --output="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/cueq/trace.sqlite" \
  "$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/cueq/trace.nsys-rep"
```

Run the hybrid candidate in a clean process with the same settings:

```bash
env \
  PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  MACE_DDP_OUTPUT_DIR="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/hybrid" \
  MACE_DDP_MODEL_BACKEND=hybrid \
  MACE_DDP_MODEL_PRESET=production \
  MACE_DDP_BATCH_SIZE=128 \
  MACE_DDP_NUM_WORKERS=4 \
  MACE_DDP_WARMUP_STEPS=8 \
  MACE_DDP_TIMED_STEPS=8 \
  MACE_DDP_CUDA_PROFILER_CAPTURE=1 \
  /usr/local/cuda/bin/nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/hybrid/trace" \
  bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 1

/usr/local/cuda/bin/nsys export \
  --force-overwrite=true \
  --type=sqlite \
  --output="$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/hybrid/trace.sqlite" \
  "$PWD/.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/hybrid/trace.nsys-rep"
```

Export each report with `nsys export --type=sqlite`. Expected: exactly eight timed-step NVTX ranges and non-empty kernel tables.

- [ ] **Step 6: Collect bounded Nsight Compute counters only if OEQ is hot**

If OpenEquivariance accounts for at least 5% of GPU kernel time, replay one representative kernel using the established privileged container method with `LaunchStats`, `Occupancy`, `SpeedOfLight`, and `MemoryWorkloadAnalysis` sections.

Expected: achieved occupancy, SM throughput, and DRAM throughput. If replay fails, retain the error and do not infer counters.

- [ ] **Step 7: Review checkpoint without committing**

Run syntax, focused tests, `git diff --check`, and inspect `git status`. Do not commit.

---

### Task 5: Analyze the hybrid result and enforce the stop decision

**Files:**
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/hybrid/REPORT.md`
- Modify: `.planning/2026-08-13-mace-hdf5-gpu-profiling/findings.md`
- Modify: `.planning/2026-08-13-mace-hdf5-gpu-profiling/progress.md`

**Interfaces:**
- Consumes: parity outputs, JSON timing records, Nsight SQLite reports, and optional NCU counters
- Produces: one of `reject`, `continue as diagnostic opt-in`, or `prepare production design`

- [ ] **Step 1: Calculate matched metrics from SQLite and JSON**

Report median/mean/p95 step time, structures/s, peak memory, kernels/step, kernel time/step, launch API time, duty, and family time for cuEq and hybrid at both batch sizes.

- [ ] **Step 2: Attribute the OpenEquivariance effect**

Separate OpenEquivariance opaque kernels, cuEq segmented polynomial, GEMM, scatter/gather/index, Triton fusion, and residual native kernels. State absolute milliseconds before percentages.

- [ ] **Step 3: Apply the decision rule**

Use:

```text
reject:
  any correctness, missing-gradient, dynamic-shape, DDP, or compatibility failure

continue as diagnostic opt-in:
  all correctness gates pass and steady-state median improves reproducibly,
  but gain is below 5% or distributed evidence is incomplete

prepare production design:
  all gates pass, batch-64 and batch-128 median improve by at least 5%,
  no material memory regression, and two-rank checksums agree
```

- [ ] **Step 4: Run final artifact verification**

Assert that every cited JSON/report exists, every profiled record has `is_compiled=true`, all trace SQLite kernel tables are non-empty, and the report contains exact source/package versions.

- [ ] **Step 5: Hand results to the user without committing prototypes**

Present the decision, the measured speedup or failure boundary, and clickable trace/report paths. Keep all diagnostic changes uncommitted pending review.
