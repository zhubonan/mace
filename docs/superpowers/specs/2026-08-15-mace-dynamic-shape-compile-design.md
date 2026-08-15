# Dynamic-Shape Compiler Qualification for MACE Training

> **Status: approved for implementation.** The user approved adopting
> NequIP's two-shape validation architecture and using its failure diff to
> locate and repair the MACE operation that specializes production edge
> counts.

## Objective

Make `--compile_training` safe and effective for production MACE models with
variable numbers of structures, atoms, and edges. The production OMat
`max_ell=3` energy/force/stress model must retain symbolic graph dimensions in
the lowered FX program, execute changing HDF5 batches without compiler
fallback, and remain compatible with the existing compiled DDP path.

This work does not introduce shape bucketing, arbitrary FX-literal rewriting,
or a separate distributed implementation.

## Evidence and failure boundary

The production model fails before Inductor. Its symbolic `make_fx` output
already contains a trace-batch edge count such as `30422` in the
`edge_index`, edge-feature placeholders, and downstream `expand` operations.
The next batch with a different edge count therefore fails a shape guard and
the facade permanently falls back to eager execution.

The failure is reproducible with PyTorch 2.10 and 2.11, and remains present in
the native e3nn model before cuEquivariance conversion. It is therefore not a
DDP, Inductor, or cuEquivariance-conversion problem. Controlled MACE traces
place the architecture boundary at `max_ell > 1`, although an isolated e3nn
spherical-harmonic derivative does not reproduce the specialization. The
remaining cause is in the composite high-order MACE conservative graph.

Current NequIP 0.19 uses the same overall
`make_fx -> torch.compile(dynamic=True)` strategy but adds two important
properties:

- e3nn TorchScript FX code generation is disabled while constructing models;
- the complete force/stress graph is traced twice with different frame, atom,
  and edge counts, and compilation is rejected unless the two FX programs are
  structurally identical.

An RTX 5090 test using PyTorch 2.11+cu128 confirmed that NequIP's `l_max=2`
force/stress FX graph keeps its edge dimension symbolic and processes unseen
changing shapes with a stable compiled cache. Applying the same inspection to
native MACE still produces a literal edge extent. NequIP therefore provides a
validation pattern and a working comparison graph, not a compiler flag that
automatically fixes MACE.

## Approved architecture

### Two-shape lowering gate

Dynamic compilation will require two eligible real training batches with
different graph-derived dimensions. `train_one_epoch` will prefetch the first
two batches only when it needs to construct a new compiled facade. Both
batches remain in their original order and receive exactly one optimizer
step; prefetching changes neither sampling nor training semantics.

`compile_training_model` and `make_training_fx` will accept the second batch as
a dynamic-shape validation input. Each batch is lowered independently through
the same functional model and decomposition table with FX duck shaping
disabled. Before `torch.compile` is called, the two lowered programs are
compared using a deterministic structural representation that preserves true
architecture constants while normalizing irrelevant generated names.

The gate must verify all graph-derived input families, including:

- structures: `ptr`, per-graph cells, displacement, volume, and graph labels;
- atoms: positions, node attributes, batch indices, charges, and optional
  atom-wise fields;
- edges: `edge_index`, shifts, unit shifts, and edge-derived intermediates.

It is not sufficient merely to observe different input sizes. The lowered
program must express those extents through symbolic size operations wherever
they affect runtime tensor shapes.

If fewer than two eligible batches exist, or the two batches do not differ in
any graph-derived dimension, dynamic compilation fails closed with a precise
setup reason. Ordinary eager training remains available.

### Failure diagnostics

When two traces differ, the compiler setup will not pass a static program to
Inductor. It will report the first meaningful structural divergence and its
dataflow context, including:

- input dimension family and the two observed sizes;
- FX node index, operation, arguments, result shape, and users;
- the corresponding nodes from both traces;
- stack or module metadata when PyTorch provides it;
- optional complete trace dumps under an explicitly requested diagnostic
  output directory.

Normal training must not create unsolicited files. Error text remains bounded
and suitable for one-line rank-zero logging. Distributed setup uses the
existing readiness consensus so every rank selects the same compiled or eager
execution policy.

### Source-level symbolic repair

The two-shape diff is then used to identify the first MACE or dependency
operation that refines a symbolic atom/edge dimension to a singleton. The fix
must occur at that source or lowering boundary. Acceptable repairs include:

- keeping tensor extents as `SymInt`/symbolic tensor sizes instead of
  converting them to Python integers;
- replacing a scripted/generated region with an unscripted FX-compatible
  equivalent during compile-oriented model construction;
- correcting a fake/meta/decomposition implementation that declares a
  dynamic extent as static;
- isolating and independently lowering a generated region only when its
  input/output symbolic relationships are explicitly preserved.

The implementation must not scan the final FX text and replace integers based
only on equality with the trace-batch atom or edge count. Such values can also
be legitimate architecture constants, and an incorrect replacement could
silently corrupt tensor algebra.

The repaired production traces must pass the same two-shape structural gate.
The first accepted graph, rather than a post-processed approximation, is the
program supplied to `torch.compile`.

## Training and DDP lifecycle

The existing compiled-DDP design remains unchanged except for two-batch
qualification:

1. Obtain two real local training batches before constructing the facade.
2. Lower and compare their complete conservative graphs.
3. Compile only after the dynamic-shape gate passes.
4. Materialize forward and backward before DDP installs reducer hooks.
5. Establish readiness consensus across ranks.
6. Wrap the compiled facade in standard PyTorch DDP.
7. Run both prefetched batches and every subsequent batch exactly once.

If any rank cannot establish a safe dynamic graph, all ranks use the facade's
eager path. Optimizer, EMA, gradient clipping, checkpoint, and model-export
ownership remain with the existing base model.

Runtime compiler failure remains a defensive fallback, but a trace-shape
mismatch must be caught during setup rather than on the second production
batch.

## Tests

Implementation follows red-green-refactor development.

1. Unit tests construct two valid batches with different graph, atom, and edge
   counts and verify that a compiler-friendly `max_ell=1` model passes the
   structural gate.
2. A synthetic static-size operation verifies that the gate rejects the graph
   before `torch.compile` is called and reports the first useful divergence.
3. Training-loop tests prove the first two batches are prefetched, kept in
   order, and each receives one optimizer step in single-device and DDP modes.
4. One-batch and same-shape loaders verify explicit fail-closed behavior.
5. The production `max_ell=3`, correlation-3 model supplies the mandatory
   regression: two batches differ only in graph-derived sizes and both lowered
   programs retain symbolic edge placeholders.
6. Full energy/force/stress output, loss, and parameter-gradient parity is
   checked against eager execution on both shapes.
7. Optimizer, AMSGrad, EMA, gradient clipping, checkpoint, and resume parity
   remain covered after the new setup lifecycle.
8. DDP tests verify rank-consistent setup and identical parameters after
   changing-shape updates.

## Performance qualification

Passing the symbolic gate establishes correctness, not speed. Qualification
uses the existing production HDF5 benchmark and PyTorch 2.11+cu128 runtime:

- one-GPU changing-shape batch-64 timing against the qualified eager baseline;
- Dynamo counters proving no exact-edge-count recompilation storm;
- Nsight Systems kernel/API attribution after warm-up;
- two- and four-rank DDP parity and sustained throughput;
- peak GPU memory and compilation startup cost;
- complete optimizer-step time and structures per second.

The compiled path is recommended for production only if it remains active and
improves sustained post-warm-up throughput. A lower kernel count alone is not
sufficient.

## Compatibility and non-goals

- `--compile_training` remains opt-in.
- Eager single-device and eager DDP behavior do not change.
- cuEquivariance convolution fusion remains disabled for variable-shape
  training.
- The explicit `--only_cueq` save behavior remains unchanged.
- No shape buckets or compile-cache-by-edge-count mechanism are added.
- No arbitrary FX integer substitution is permitted.
- Inference AOTInductor/export dynamic-shape bounds are outside this training
  change.
