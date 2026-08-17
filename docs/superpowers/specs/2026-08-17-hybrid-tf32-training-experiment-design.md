# Hybrid TF32 training experiment design

## Objective

Measure the speed and numerical effect of enabling TensorFloat-32 matrix
multiplication in the current compiled hybrid cuEquivariance plus
OpenEquivariance training path.

This is a diagnostic qualification. It will not enable TF32 by default, add a
public training option, change parameter or optimizer dtypes, introduce
autocast, modify model serialization, or include BF16. A production-policy
change requires a separate review after the experiment.

## Fixed workload

- Source: the current `training-opt` branch and hybrid implementation.
- Runtime: PyTorch 2.11.0+cu128, CUDA 12.8, e3nn 0.4.4,
  cuEquivariance 0.11.1, and the installed OpenEquivariance runtime.
- Hardware: local RTX 5090.
- Model: the production 9,063,204-parameter MACE model.
- Backend: hybrid cuEquivariance plus OpenEquivariance with cuEquivariance
  convolution fusion disabled.
- Data: the existing variable-shape production OMat HDF5 shard.
- Objective: complete energy, force, and stress loss, followed by outer
  backward, gradient clipping, Adam with AMSGrad, and EMA.
- Batch size: 128 structures with four persistent DataLoader workers.
- Compilation: the existing dynamic compiled-training facade with no eager
  fallback and no graph-count growth after warmup.

The FP32 and TF32 cases must use identical model parameters, optimizer state,
EMA state, random seeds, batch order, and batch shapes.

## Precision policies

### IEEE FP32 control

Use `torch.set_float32_matmul_precision("highest")` and explicitly disable
CUDA matmul TF32. Model parameters, inputs, outputs, gradients, optimizer
state, geometry, reductions, and loss remain float32.

### TF32 candidate

Use `torch.set_float32_matmul_precision("high")` and explicitly enable CUDA
matmul TF32. No tensor is converted to another dtype. cuDNN TF32 policy will
be recorded and held identical unless a profiled MACE operation demonstrably
uses cuDNN; the intended independent variable is matrix multiplication.

The harness must record the effective PyTorch precision setting and backend
flags in every result. It must restore process-global precision flags after
an in-process diagnostic, although the preferred timing runs use isolated
processes.

## Stage 1: one-step numerical gate

Start both policies from the same canonical model and optimizer snapshot and
run the same batch. Compare:

1. Total energy, per-structure energy, forces, virials, and stress.
2. Total and component energy/force/stress losses.
3. Every populated trainable parameter gradient by parameter name.
4. Missing-gradient inventory and gradient shape/stride inventory.
5. Parameters and Adam/AMSGrad state after one optimizer update.
6. EMA state after the same update.

Record maximum absolute difference, relative L2 difference, cosine
similarity, and percentile error distributions where meaningful. Relative
errors near zero-valued references must not be used alone.

The candidate fails immediately on a non-finite value, missing or extra
gradient, changed tensor shape/stride, compiler fallback, or graph-count
growth. The initial gradient acceptance gate is relative L2 difference no
larger than `1e-3`; all larger output, update, or optimizer-state differences
must be reported rather than hidden by relaxed elementwise tolerances.

## Stage 2: steady-state performance

Run each policy in three clean isolated processes. Each repetition uses eight
warmup steps and 50 timed variable-shape optimizer steps. Alternate policy
order across repetitions to reduce thermal and background-load bias.

Record:

- synchronized wall and CUDA-event optimizer-step samples;
- median, mean, p95, and structures per second;
- peak allocated and reserved GPU memory;
- batch atom and edge counts;
- Dynamo graph, frame, and fallback counters before and after timed steps;
- effective precision flags and imported package provenance.

The primary speed result is the ratio of medians pooled by repetition. Report
the per-repetition ratios and dispersion; do not select only the fastest run.
DataLoader wait remains outside the optimizer-core timer, with full
start-to-start throughput reported separately.

## Stage 3: short matched trajectory

Run 256 matched optimizer steps for each policy from the same initial state
and through the same batch sequence. Evaluate a fixed validation subset at
the start and end with the model in its ordinary float32 inference policy.

Record total and component training-loss curves plus validation energy,
force, and stress errors. The TF32 trajectory qualifies locally when:

- all values remain finite;
- no compiler fallback or new graph appears;
- the loss trajectory has no precision-specific instability;
- each final validation metric is within 1% of the FP32 control.

This is a short-horizon accuracy gate, not proof of converged foundation-model
accuracy. A default or production recommendation still requires a longer
training comparison.

## Stage 4: Nsight attribution

Only after Stages 1 and 2 pass, capture exactly eight post-warmup optimizer
steps for one representative repetition of each policy with Nsight Systems.
Compilation, model construction, and DataLoader startup remain outside the
capture.

Compare:

- total optimizer-range wall time and GPU kernel time;
- GEMM calls and elapsed time;
- SIMT SGEMM versus tensor-core GEMM families;
- OpenEquivariance, cuEquivariance, Triton, scatter/index, and optimizer
  families;
- kernel duty and sub-10-microsecond launch contribution.

The attribution must show whether the speedup comes from GEMM tensor-core
selection. If non-GEMM families change materially, investigate that difference
before assigning the result to TF32.

Nsight Compute is optional because the Phase-35 replay path fails preparation
for these GEMMs. Use the supported Nsight Systems GB20x GPU metrics only when
lower-level utilization is needed, and do not label sampled SM activity as
achieved occupancy.

## Expected result and decision rule

The retained pre-hybrid profile reduced GEMM time by 28.6% and optimizer-step
median by 6.9%. The current hybrid batch-128 step spends about 8.14 ms in GEMM
out of a 22.81 ms median. If the GEMM reduction transfers, the expected
end-to-end improvement is approximately 8--10%.

Recommend a later opt-in production policy only if all correctness and
short-trajectory gates pass and the median optimizer step improves by at least
5% in at least two of three repetitions. A smaller or unstable gain remains a
diagnostic result and does not justify a user-visible option.

## Artifacts and deliverables

Retain results under
`.profiling/mace-hdf5-gpu-2026-08-13/phase36/`:

- exact environment and source provenance;
- per-run JSON timing and numerical records;
- first-step parity and optimizer-state summaries;
- 256-step loss and validation summaries;
- one FP32 and one TF32 `.nsys-rep` plus SQLite export;
- reproducible analyzers and a claim verifier;
- a report separating numerical accuracy, short-horizon accuracy, optimizer
  throughput, input throughput, GPU attribution, and limitations.

Generated Inductor caches and temporary runtime state are not commit-worthy
artifacts. Existing uncommitted profiling and planning files remain untouched.
