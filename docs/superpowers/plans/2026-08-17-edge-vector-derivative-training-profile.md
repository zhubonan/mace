# Edge-Vector Derivative Training Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether differentiating MACE energy with respect to edge-vector leaves and explicitly assembling forces, virials, and stress reduces the compiled full-training graph and steady-state step time without changing scientific outputs or saved models.

**Architecture:** First implement and verify pure analytical edge-gradient assembly against MACE's existing position/displacement autograd on periodic and duplicated-edge controls. Then add a private trace-time context used only by the internal compiled facade, validate full parameter double backward and dynamic shapes, and profile it against the current compiled derivative route.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu128, e3nn 0.4.4, cuEquivariance 0.11.1, pytest, ASE, HDF5, Nsight Systems, Nsight Compute.

## Global Constraints

- Use float32 for production-size qualification and float64 only for analytical identity unit tests.
- Do not enable TF32, BF16, FP16, or autocast.
- Keep cuEquivariance convolution fusion disabled.
- The eager model API and behavior must remain unchanged.
- The canonical model must remain the sole owner of parameters, optimizer/EMA state, checkpoints, and final single-file `.model` exports.
- The edge-vector route must exist only in the private compiled-training lowering and must not add persistent parameters or buffers.
- Require two-shape dynamic compilation, full force/stress-loss parameter gradients, optimizer-state parity, and two-rank checksum parity before profiling.
- Keep all prototype changes uncommitted until the user reviews the results.
- Stop if periodic virial/stress parity, double backward, dynamic shapes, or saved-model ownership fails.

---

### Task 1: Implement and prove the analytical edge-gradient identities

**Files:**
- Create: `mace/tools/edge_derivative.py`
- Create: `tests/unit/test_edge_derivative.py`
- Inspect: `mace/modules/utils.py:83-116`
- Inspect: `mace/modules/utils.py:270-336`
- Inspect: `mace/modules/utils.py:394-410`

**Interfaces:**
- Produces: `assemble_edge_forces(edge_gradient, edge_index, num_atoms)`
- Produces: `assemble_edge_virials_stress(edge_gradient, vectors, edge_index, node_batch, cell, num_graphs, compute_stress)`
- Consumes later: the trace-time output replacement in Task 2

- [ ] **Step 1: Write the force-assembly failing test**

Create `tests/unit/test_edge_derivative.py` with:

```python
import torch
from torch.testing import assert_close

from mace.tools.edge_derivative import assemble_edge_forces


def test_edge_force_assembly_matches_position_gradient_with_duplicates():
    positions = torch.tensor(
        [[0.1, -0.2, 0.3], [1.2, 0.4, -0.7], [-0.3, 1.1, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0], [1, 0, 2, 1, 1]], dtype=torch.long
    )
    sender, receiver = edge_index
    vectors = positions[receiver] - positions[sender]
    energy = (
        0.4 * vectors.square().sum(dim=-1)
        + 0.1 * vectors[:, 0] * vectors[:, 1]
    ).sum()
    position_gradient, edge_gradient = torch.autograd.grad(
        energy, (positions, vectors), retain_graph=True
    )

    actual = assemble_edge_forces(
        edge_gradient, edge_index, num_atoms=positions.shape[0]
    )
    assert_close(actual, -position_gradient)
    assert_close(actual.sum(dim=0), torch.zeros(3, dtype=actual.dtype))
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:$PWD \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python \
  -m pytest tests/unit/test_edge_derivative.py::test_edge_force_assembly_matches_position_gradient_with_duplicates -vv
```

Expected: FAIL because `mace.tools.edge_derivative` does not exist.

- [ ] **Step 3: Implement the minimal force assembly**

Create `mace/tools/edge_derivative.py`:

```python
import torch


def assemble_edge_forces(
    edge_gradient: torch.Tensor,
    edge_index: torch.Tensor,
    num_atoms: int,
) -> torch.Tensor:
    sender, receiver = edge_index[0], edge_index[1]
    forces = torch.zeros(
        (num_atoms, 3),
        dtype=edge_gradient.dtype,
        device=edge_gradient.device,
    )
    forces = forces.index_add(0, sender, edge_gradient)
    forces = forces.index_add(0, receiver, -edge_gradient)
    return forces
```

- [ ] **Step 4: Run the force test and verify GREEN**

Expected: PASS in float64, including the translational force sum.

- [ ] **Step 5: Write the periodic virial/stress failing test**

Add a two-graph test that constructs triclinic cells, nonzero integer unit
shifts, duplicated directed edges, and a differentiable symmetric displacement
using `get_symmetric_displacement`. Compute the reference with:

```python
deformed_positions, shifts, displacement = get_symmetric_displacement(
    positions=positions,
    unit_shifts=unit_shifts,
    cell=cell,
    edge_index=edge_index,
    num_graphs=2,
    batch=node_batch,
    displacement=displacement,
)
vectors = (
    deformed_positions[edge_index[1]]
    - deformed_positions[edge_index[0]]
    + shifts
)
energy = (0.2 * vectors.square().sum(dim=-1) + vectors[:, 0] ** 3).sum()
position_gradient, displacement_gradient, edge_gradient = torch.autograd.grad(
    energy, (positions, displacement, vectors), retain_graph=True
)
```

Then assert:

```python
forces = assemble_edge_forces(edge_gradient, edge_index, positions.shape[0])
virials, stress = assemble_edge_virials_stress(
    edge_gradient,
    vectors.detach(),
    edge_index,
    node_batch,
    cell,
    num_graphs=2,
    compute_stress=True,
)
assert_close(forces, -position_gradient)
assert_close(virials, -displacement_gradient)
volume = torch.linalg.det(cell).abs().view(-1, 1, 1)
assert_close(stress, displacement_gradient / volume)
```

- [ ] **Step 6: Run the virial test and verify RED**

Expected: FAIL because `assemble_edge_virials_stress` is missing.

- [ ] **Step 7: Implement graphwise virial and stress assembly**

Add:

```python
from mace.tools.scatter import scatter_sum


def assemble_edge_virials_stress(
    edge_gradient: torch.Tensor,
    vectors: torch.Tensor,
    edge_index: torch.Tensor,
    node_batch: torch.Tensor,
    cell: torch.Tensor,
    num_graphs: int,
    compute_stress: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    edge_graph = node_batch[edge_index[0]]
    displacement_gradient = scatter_sum(
        torch.einsum("ei,ej->eij", vectors, edge_gradient),
        edge_graph,
        dim=0,
        dim_size=num_graphs,
    )
    displacement_gradient = 0.5 * (
        displacement_gradient + displacement_gradient.transpose(-1, -2)
    )
    virials = -displacement_gradient
    if not compute_stress:
        return virials, None
    volume = torch.linalg.det(cell.view(-1, 3, 3)).abs().view(-1, 1, 1)
    stress = displacement_gradient / volume
    stress = torch.where(
        torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
    )
    return virials, stress
```

- [ ] **Step 8: Run all analytical identity tests and verify GREEN**

Run the entire new test file. Expected: force, virial, stress, duplicates,
periodic shifts, triclinic cells, multi-graph volumes, and translational sum all
pass in float64.

- [ ] **Step 9: Review checkpoint without committing**

Run `git diff --check` and Python syntax validation. Do not commit.

---

### Task 2: Add a private trace-time edge-derivative output context

**Files:**
- Modify: `mace/tools/edge_derivative.py`
- Modify: `mace/tools/compile_training.py`
- Modify: `tests/unit/test_edge_derivative.py`

**Interfaces:**
- Produces: `trace_edge_derivative_outputs()` context manager
- Produces: private `derivative_mode: Literal["positions", "edges"]` argument on `mace.tools.compile_training.compile_training_model`
- Does not modify: `mace.tools.compile.compile_training_model`, CLI arguments, eager model signatures, or serialized modules

- [ ] **Step 1: Write the context-restoration failing test**

Add:

```python
from mace.modules import models
from mace.tools.edge_derivative import trace_edge_derivative_outputs


def test_edge_derivative_trace_context_restores_model_globals():
    original_prepare = models.prepare_graph
    original_outputs = models.get_outputs
    with trace_edge_derivative_outputs():
        assert models.prepare_graph is not original_prepare
        assert models.get_outputs is not original_outputs
    assert models.prepare_graph is original_prepare
    assert models.get_outputs is original_outputs
```

- [ ] **Step 2: Run the restoration test and verify RED**

Expected: FAIL because the context manager is absent.

- [ ] **Step 3: Implement the minimal trace-only context**

In `mace/tools/edge_derivative.py`, add a context manager that temporarily
patches the Python globals referenced by `ScaleShiftMACE.forward`:

```python
from contextlib import contextmanager
import threading
from typing import Any


_TRACE_PATCH_LOCK = threading.Lock()


@contextmanager
def trace_edge_derivative_outputs():
    from mace.modules import models

    with _TRACE_PATCH_LOCK:
        original_prepare = models.prepare_graph
        original_outputs = models.get_outputs
        trace_state: dict[str, Any] = {}

        def prepare_graph(data, **kwargs):
            context = original_prepare(data, **kwargs)
            vectors = context.vectors.detach().requires_grad_(True)
            trace_state["edge_index"] = data["edge_index"]
            trace_state["node_batch"] = data["batch"]
            trace_state["num_graphs"] = context.num_graphs
            return context._replace(
                vectors=vectors,
                lengths=torch.linalg.vector_norm(vectors, dim=1, keepdim=True),
            )

        def get_outputs(
            energy,
            positions,
            cell,
            displacement,
            vectors=None,
            magmoms=None,
            training=False,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
            compute_hessian=False,
            compute_edge_forces=False,
            compute_magforces=False,
        ):
            if (
                vectors is None
                or magmoms is not None
                or compute_hessian
                or compute_edge_forces
                or compute_magforces
            ):
                raise RuntimeError(
                    "edge derivative training supports only energy/force/virial/stress"
                )
            edge_gradient = torch.autograd.grad(
                energy,
                vectors,
                grad_outputs=torch.ones_like(energy),
                retain_graph=training,
                create_graph=training,
                allow_unused=False,
            )[0]
            forces = assemble_edge_forces(
                edge_gradient, trace_state["edge_index"], positions.shape[0]
            ) if compute_force else None
            virials = stress = None
            if compute_virials or compute_stress:
                virials, stress = assemble_edge_virials_stress(
                    edge_gradient,
                    vectors,
                    trace_state["edge_index"],
                    trace_state["node_batch"],
                    cell,
                    trace_state["num_graphs"],
                    compute_stress,
                )
            return forces, virials, stress, None, None, None

        models.prepare_graph = prepare_graph
        models.get_outputs = get_outputs
        try:
            yield
        finally:
            models.prepare_graph = original_prepare
            models.get_outputs = original_outputs
```

The mutable trace state is valid only during one synchronous `make_fx` call;
the emitted FX graph owns no reference to it. The module-level lock serializes
the temporary global patch so concurrent compilation cannot observe it.

- [ ] **Step 4: Run restoration and exception-restoration tests**

Add a second test that raises inside the context and verifies both globals are
restored. Expected: both pass.

- [ ] **Step 5: Write the default-preservation and private-mode failing tests**

In `tests/unit/test_compile_training.py`, add one test asserting the existing
call remains position-based, and one calling the internal function:

```python
from mace.tools.compile_training import compile_training_model as compile_internal

edge_compiled = compile_internal(
    model,
    trace_batch,
    output_args=output_args,
    mode="default",
    dynamic=True,
    validation_batch=validation_batch,
    derivative_mode="edges",
)
assert edge_compiled.is_compiled, edge_compiled.compilation_error
```

Expected RED: `compile_training_model()` rejects the unknown keyword.

- [ ] **Step 6: Thread the private mode through the internal lowering only**

Add `derivative_mode: Literal["positions", "edges"] = "positions"` to internal
`compile_training_model`, `CompiledTrainingModel.__init__`, `make_training_fx`,
and `_lower_training_fx`. Store it on the facade. Wrap only `make_fx` with:

```python
derivative_context = (
    trace_edge_derivative_outputs()
    if derivative_mode == "edges"
    else nullcontext()
)
with (
    _fx_duck_shape(False),
    _traceable_legacy_spherical_harmonics(model),
    derivative_context,
):
    fx_model = make_fx(
        flat_model,
        decomposition_table=mace_decomp_table(),
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        _error_on_data_dependent_ops=True,
    )(*trace_inputs)
```

Reject values outside `{"positions", "edges"}` before tracing. Do not add the
argument to `mace/tools/compile.py` or any CLI path.

- [ ] **Step 7: Ensure eager fallback remains canonical**

Add a test that calls `disable_compilation()` on an edge-mode facade and checks
that `_eager_forward` uses the original position/displacement derivative model.
Expected: canonical eager outputs remain unchanged; the private mode affects
only successfully lowered FX execution.

- [ ] **Step 8: Run focused CPU tests and verify GREEN**

Run analytical tests, context restoration, default behavior, invalid mode, and
fallback tests. Expected: all pass without changing serialized state keys.

- [ ] **Step 9: Review checkpoint without committing**

Run syntax, focused tests, and `git diff --check`. Do not commit.

---

### Task 3: Prove full objective, optimizer, dynamic-shape, and DDP parity

**Files:**
- Modify: `tests/unit/test_compile_training.py`
- Modify: `tests/unit/test_compile_training_ddp.py`
- Inspect: `mace/tools/train.py:42-72`

**Interfaces:**
- Consumes: internal `derivative_mode="edges"` facade
- Produces: complete one-rank and two-rank qualification before profiling

- [ ] **Step 1: Write the one-rank full-objective parity test**

Parameterize the existing CUDA optimizer/EMA parity test over
`derivative_mode=("positions", "edges")`. For edge mode, call the internal
compile function; leave the public path unchanged. Compare energy, forces,
virials, stress, every named parameter gradient, one Adam+AMSGrad update, every
optimizer-state tensor, and EMA shadow parameters using float32 defaults.

- [ ] **Step 2: Run and verify the one-shape gate**

Run outside the sandbox on the RTX 5090. Expected: edge mode remains compiled
and all outputs/updates pass `assert_close`.

- [ ] **Step 3: Write the periodic and variable-shape production test**

Extend the retained production HDF5 test to build the 9,063,204-parameter model,
trace on one batch-64 shape, validate on a distinct shape, and execute both in
edge mode. Assert:

```python
assert facade.is_compiled, facade.compilation_error
assert facade.dynamic_shape_signatures[0] != facade.dynamic_shape_signatures[1]
assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
```

Compare full outputs and named parameter gradients against the current
position/displacement compiled route.

- [ ] **Step 4: Run the production one-rank gate**

Expected: no fallback, no literal edge specialization, finite values, and
float32 parity on both shapes. Stop before timing if it fails.

- [ ] **Step 5: Parameterize the DDP parity harness privately**

Add a profiling-only `--derivative-mode={positions,edges}` to
`compiled_ddp_benchmark.py`, and have it import the internal compile function
only when `edges` is selected. Record the mode in JSON.

- [ ] **Step 6: Run the two-rank full-update gate**

Run two ranks with the small model, then the production preset if resources
permit. Expected: all ranks remain compiled, losses agree, and post-update
checksums are identical. If the local host exposes one GPU, dispatch the exact
source bundle to the established remote qualification task without changing
production training jobs.

- [ ] **Step 7: Verify canonical ownership and legacy export**

Run the existing final-artifact subprocess regression after an edge-mode update.
Expected: the saved object is the canonical e3nn 0.4.4 model, contains no
`base_model.`/`module.` prefixes or edge-derivative state, and loads under the
separate PyTorch 2.10/e3nn 0.4.4 runtime.

- [ ] **Step 8: Run the focused regression suite**

Run `tests/unit/test_edge_derivative.py`, compiled-training unit tests, DDP
tests, and integration/export tests. Expected: all selected tests pass.

- [ ] **Step 9: Review checkpoint without committing**

Retain exact commands and outputs. Do not commit.

---

### Task 4: Capture matched production timing and Nsight profiles

**Files:**
- Modify: `.profiling/mace-hdf5-gpu-2026-08-13/scripts/compiled_ddp_benchmark.py`
- Modify: `.profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/edge-derivative/{positions,edges}/rank-0.json`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/edge-derivative/{positions,edges}/trace.nsys-rep`
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/edge-derivative/{positions,edges}/trace.sqlite`

**Interfaces:**
- Consumes: qualified private derivative mode from Task 3
- Produces: matched batch-64 and batch-128 timing and kernel attribution

- [ ] **Step 1: Record graph and compilation diagnostics**

Add JSON fields for derivative mode, FX node count, trace wall time, compile
wall time, dynamic signatures, Dynamo counters, peak host RSS during trace, and
peak GPU allocation.

- [ ] **Step 2: Run matched 24-step controls**

Run position and edge modes in independent clean processes at batch 64 and 128,
with four workers, eight warmups, and 24 timed steps. Keep all model, optimizer,
seed, data, and output settings identical.

Expected: identical batch shapes, stable graphs, finite losses, and no fallback.

- [ ] **Step 3: Capture eight post-warmup Nsight steps for each mode**

Use the same CUDA Profiler API capture boundary and Nsight options as Phase 33.
Export `.nsys-rep` to SQLite and assert exactly eight timed ranges.

- [ ] **Step 4: Compare the intended kernel families**

Calculate absolute time and launches for GEMM, cuEq segmented polynomial,
scatter/gather/index, Triton fusion, native elementwise, optimizer, and CUDA
launch APIs. Confirm GEMM and segmented-polynomial time remain within run noise;
attribute any change in coordinate gather/scatter and derivative bookkeeping.

- [ ] **Step 5: Collect bounded NCU counters only for a new hot assembly kernel**

If an edge-assembly kernel exceeds 5% of GPU time, capture `LaunchStats`,
`Occupancy`, `SpeedOfLight`, and `MemoryWorkloadAnalysis`. Otherwise retain the
Nsight Systems result and avoid counter replay overhead.

- [ ] **Step 6: Review checkpoint without committing**

Verify artifacts, syntax, tests, and `git diff --check`. Do not commit.

---

### Task 5: Analyze the edge-derivative result and decide its future

**Files:**
- Create: `.profiling/mace-hdf5-gpu-2026-08-13/phase34/edge-derivative/REPORT.md`
- Modify: `.planning/2026-08-13-mace-hdf5-gpu-profiling/findings.md`
- Modify: `.planning/2026-08-13-mace-hdf5-gpu-profiling/progress.md`

**Interfaces:**
- Consumes: all correctness, dynamic, DDP, export, timing, Nsight, and optional NCU evidence
- Produces: `reject`, `continue as private diagnostic`, or `prepare production design`

- [ ] **Step 1: Report correctness before performance**

Include maximum absolute/relative differences for every output, named parameter
gradient, parameter update, optimizer-state tensor, and DDP checksum. State any
fallback or graph-count change explicitly.

- [ ] **Step 2: Report compilation and steady-state costs separately**

Compare FX node count, trace/compile time, host RSS, GPU allocation, median and
p95 step time, structures/s, launches, kernel time, and duty at both batch sizes.

- [ ] **Step 3: Apply the decision rule**

```text
reject:
  any scientific, double-backward, dynamic-shape, DDP, export, or ownership failure

continue as private diagnostic:
  all gates pass but median speedup is below 5%, inconsistent across batches,
  or compile-memory cost materially regresses

prepare production design:
  all gates pass, both batch medians improve by at least 5%, kernel attribution
  confirms reduced derivative bookkeeping, and memory/compile costs are acceptable
```

- [ ] **Step 4: Verify every retained artifact and claim**

Run a machine-readable check for file existence, non-empty SQLite kernel tables,
`is_compiled=true`, stable graph counters, exact source/runtime versions, and
report values recomputed from JSON/SQLite.

- [ ] **Step 5: Present the result without committing the prototype**

Provide the decision, measured ceiling realized, remaining bottleneck, and
clickable report/trace paths. Await explicit user direction before retaining,
reworking, committing, or deleting the prototype.
