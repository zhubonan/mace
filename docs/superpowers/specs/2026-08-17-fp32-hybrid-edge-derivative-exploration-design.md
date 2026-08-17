# FP32 hybrid-kernel and edge-derivative exploration design

## Objective

Evaluate two independent ways to accelerate full MACE energy/force/stress
training while preserving conservative forces, dynamic variable-shape
compilation, distributed-training compatibility, and legacy single-file model
exports:

1. Hybrid cuEquivariance plus OpenEquivariance training.
2. Edge-vector derivative leaves with explicit force and virial assembly.

This phase is diagnostic. It will produce correctness evidence, timing, Nsight
profiles, and a production recommendation. It will not enable a new CLI default,
change model serialization, or lower precision. TF32 and BF16 are explicitly
out of scope.

## Shared constraints

- Use PyTorch 2.11, e3nn 0.4.4, float32, and the qualified CUDA 12 runtime.
- Keep cuEquivariance convolution fusion disabled.
- Use the existing variable-shape OMat HDF5 workload and production
  9,063,204-parameter model.
- Keep the canonical MACE model as the owner of parameters, optimizer state,
  EMA state, checkpoints, and final `.model` files.
- Do not serialize OpenEquivariance wrappers, compiled FX modules, experimental
  derivative helpers, or DDP wrappers.
- Require one stable dynamic graph across at least two distinct atom and edge
  shapes before profiling.
- Treat any eager fallback, missing gradient, non-finite value, or unsupported
  higher derivative as a failed experiment rather than a reason to weaken the
  parity gate.

## Experiment A: hybrid cuEq plus OpenEquivariance

### Boundary

Install OpenEquivariance only in the isolated profiling runtime. Construct a
hybrid copy of the canonical model using the existing MACE converter:

- cuEquivariance remains responsible for linear layers, fully connected tensor
  products, and symmetric contractions.
- OpenEquivariance `TensorProductConv` replaces only the channelwise tensor
  product plus destination scatter.
- cuEquivariance convolution fusion remains off.

The canonical model and export path remain unchanged. The first implementation
may use a profiling-only model builder or a private experimental switch; it must
not add a user-visible training option.

### Correctness gates

Compare the existing unfused-cuEq route and the hybrid route from identical
weights and identical batches. Test in this order:

1. Energy, forces, virials, and stress.
2. The complete weighted energy/force/stress loss.
3. Gradients for every trainable parameter, including missing-gradient and
   layout checks.
4. One optimizer update, including parameters and Adam/AMSGrad state.
5. Two sequential variable-shape updates.
6. Compiler state: one stable graph, no fallback, and no graph-count growth
   after warmup.
7. Two-rank full-update parity and identical post-update checksums if the
   one-rank dynamic gate passes.

The initial numerical gate uses `torch.testing.assert_close` float32 defaults
(`rtol=1.3e-6`, `atol=1e-5`). Any larger discrepancy must be attributed to a
specific operation and explicitly approved; tolerances will not be relaxed
solely to obtain a performance result.

### Profiling gates

Only after correctness passes:

- Benchmark batches 64 and 128 using the same warmup and timed-step counts as
  the retained Phase 33 controls.
- Capture eight post-warmup optimizer steps with Nsight Systems.
- Compare synchronized step time, structures/s, peak memory, graph counters,
  total launches, GPU kernel time, device duty, and kernel-family time.
- Attribute whether OpenEquivariance reduces cuEquivariance segmented
  polynomial work, scatter/gather/index work, or merely moves work to another
  opaque kernel.
- If the fused kernel becomes material, collect bounded Nsight Compute
  occupancy and memory-throughput counters using the established privileged
  profiling method.

### Stop conditions

Stop this experiment without profiling if OpenEquivariance lacks the required
backward-backward, parameter gradients, dynamic-shape behavior, or compatible
CUDA support. Do not replace the failed path with an inference-only benchmark.

## Experiment B: edge-vector derivative leaf

### Mathematical design

The canonical model defines energy from atomic positions through directed edge
vectors

\[
\mathbf e_{ij}=\mathbf r_j-\mathbf r_i+\mathbf s_{ij}.
\]

The experimental compiled lowering will make edge vectors differentiable
leaves, compute

\[
\mathbf g_{ij}=\frac{\partial E}{\partial \mathbf e_{ij}},
\]

and assemble atomic forces with the exact chain rule:

\[
\mathbf F_i=
\sum_{(i,j)}\mathbf g_{ij}
-\sum_{(j,i)}\mathbf g_{ji}.
\]

The implementation will derive the signs from the actual MACE sender/receiver
convention and validate them against direct position differentiation. It will
not assume that stored edges are unique, half-listed, or symmetrically paired.

Virials will be assembled from the same edge gradients and edge vectors using
the convention that reproduces the current displacement derivative. Stress
will continue to apply the current sign, cell-volume normalization, symmetry,
and invalid-value guard.

### Boundary

The edge-vector representation exists only inside the compiled-training
lowering. The eager model API, input batches, learned parameters, state dict,
checkpoint code, inference behavior, and final serialized model remain
unchanged.

The first prototype must isolate three responsibilities:

1. Construct differentiable edge vectors from the existing positions, cell,
   unit shifts, edge indices, batch mapping, and displacement.
2. Evaluate model energy from those edge vectors without silently
   reconstructing a second position-gradient path.
3. Assemble forces, virials, and stress from the edge gradients.

If the current model cannot accept precomputed edge vectors at a clean internal
boundary, the prototype may introduce a private functional override inside the
compiled facade. It must not mutate the canonical modules or add persistent
buffers.

### Correctness gates

Use direct position/displacement differentiation as the reference and test:

1. Non-periodic structures with explicitly directed edges.
2. Periodic structures with nonzero integer shifts and triclinic cells.
3. Edge lists containing both directions and deliberately duplicated edges.
4. Multiple structures per batch with different cell volumes.
5. Energy, forces, virials, stress, and translational force sum.
6. Complete force/stress-loss parameter gradients.
7. One Adam/AMSGrad update and optimizer-state parity.
8. Two distinct variable-shape batches with one stable compiled graph.
9. Two-rank full-update parity and identical post-update checksums if the
   one-rank dynamic gate passes.

The reference and candidate must use the same model owner and weights. Gradient
comparison is by parameter name, not by positional iteration alone.

### Profiling gates

Only after correctness passes:

- Benchmark and profile batches 64 and 128 against the current compiled
  position/displacement derivative route.
- Compare graph node count, compile time, peak host and GPU memory, synchronized
  step time, launches, kernel duty, and kernel-family time.
- Pay particular attention to gather/scatter/index, Triton fusion, and residual
  elementwise families. GEMM and segmented-polynomial time should remain
  scientifically unchanged.
- Retain the candidate only if it reduces measured steady-state wall time and
  does not create graph specialization or excessive compilation memory.

### Stop conditions

Stop if the edge-vector boundary changes the energy function, requires a
different neighbor list, cannot reproduce periodic virial/stress, breaks
parameter double backward, or requires changing saved model structure.

## Sequencing

The experiments are independent and will be executed in this order:

1. OpenEquivariance environment and full-double-backward feasibility.
2. Hybrid correctness and, if qualified, Nsight profiling.
3. Edge-vector analytical parity tests.
4. Private compiled-lowering prototype and dynamic-shape gate.
5. Edge-vector Nsight profiling if qualified.
6. Side-by-side recommendation with no precision changes mixed into either
   result.

This order obtains an early answer on the existing hybrid implementation before
making the more invasive derivative-lowering prototype.

## Deliverables

- Exact environment and source provenance.
- Machine-readable parity and timing records.
- Retained `.nsys-rep` and SQLite exports for every qualified candidate.
- Any bounded Nsight Compute reports required for a new hot opaque kernel.
- A report separating correctness failures, steady-state speed, compilation
  overhead, memory, and compatibility risk.
- A recommendation to reject, continue as an opt-in implementation, or prepare
  a separately reviewed production change for each direction.

No production feature will be committed as part of this diagnostic phase.
