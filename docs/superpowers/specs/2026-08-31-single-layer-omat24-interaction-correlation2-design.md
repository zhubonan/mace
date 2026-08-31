# Single-Layer OMat24 Interaction-Correlation-2 Design

## Objective

Train one fresh 128-channel, without-self, single-interaction MACE model on
full OMat24 for twenty epochs. Relative to the completed 128-channel
without-self reference, change only the first interaction product correlation
from 3 to 2. Keep the separate invariant contraction used by the nonlinear
scalar readout at correlation 2.

## Considered approaches

1. **Fresh controlled correlation-2 run (selected).** Start from the same seed
   and preserve every non-correlation argument. This isolates the effect of
   reducing the interaction product degree and retains valid optimizer and
   scheduler semantics.
2. Continue from the correlation-3 checkpoint after changing topology. This
   is rejected because product tensors and optimizer state do not represent
   the new model, so the comparison would not be controlled.
3. Reduce both interaction and readout correlations. This is rejected because
   the requested final scalar contraction remains correlation 2 and changing
   it would confound two architectural effects.

## Scientific topology

- `RealAgnosticDensityInteractionBlock` is the sole interaction.
- There is no residual or self-connection.
- `num_interactions=1`.
- Hidden irreps are `128x0e + 128x1o + 128x2e`.
- Spherical harmonics use `max_ell=3`.
- The first product basis uses `correlation=2`.
- The readout-local invariant product independently uses
  `readout_correlation=2`.
- `use_last_readout_only=True` and `MLP_irreps=16x0e` retain the established
  nonlinear scalar readout.

The readout product is not a second neighbor interaction: it performs no edge
aggregation and only contracts the retained first-layer equivariant features
to scalar invariants.

## Exact model size

CPU construction against the selected MACE source and e3nn 0.4.4 gives exactly
6,251,674 trainable parameters:

| Module group | Trainable parameters |
|---|---:|
| Interaction | 5,956,234 |
| First product basis | 220,032 |
| Readout product | 61,952 |
| Node embedding | 11,392 |
| Nonlinear readout | 2,064 |
| **Total** | **6,251,674** |

The completed correlation-3 reference has 6,639,002 trainable parameters, so
the selected change removes 387,328 parameters (5.83%). A production verifier
must construct the model and fail unless it observes the exact topology and
count above.

## Training and resource protocol

Everything except the interaction correlation remains identical to the fresh
128-channel without-self protocol:

- full OMat24 native LMDB training data and frozen validation/statistics;
- fresh initialization with seed `20260729`;
- epochs 0 through 19 in one job, with no checkpoint restart;
- batch 256/rank, validation batch 64, and four workers/rank;
- two RTX 5090 ranks, eight CPUs/rank, and 92,000 MB/node;
- Adam/AMSGrad, EMA, loss weights, normalization, clipping, and
  `ReduceLROnPlateau(factor=0.5, patience=1)` unchanged;
- IEEE float32, CuEquivariance, compiled DDP, and
  `--gpu-bind=mask_gpu:0x3,0x3` unchanged.

The job receives unique run, compiler-cache, and Slurm-log roots. It may queue
behind the active 96/64-channel jobs; their state must not be altered.

## Implementation and qualification

Add a dedicated immutable candidate and renderer rather than weakening the
existing width-sweep constraints. Extend the CPU provenance verifier so the
interaction and readout correlations are independently checked. Add a fresh
fail-closed launcher or a narrowly parameterized launcher that accepts only
this exact 128/correlation-2 candidate.

Before submission, require:

- focused renderer and exact-topology/count tests;
- the complete `tests/full_retrain` regression suite;
- launcher `bash -n`;
- a metadata-free sealed project archive;
- remote construction yielding exactly 6,251,674 trainable parameters;
- a dry render containing `--correlation=2`,
  `--readout_correlation=2`, twenty epochs, no restart, and the established
  full-data/runtime paths.

After allocation, require both GPUs to be visible to both ranks, one compiled
graph/rank, finite initial validation and optimizer records, bounded host/GPU
memory, and no fatal or recompilation signature.

## Comparison

Report epoch-aligned validation loss and energy, force, and stress MAEs against
the completed 128-channel without-self correlation-3 model. Co-report
throughput, GPU memory, host memory, and the 5.83% parameter reduction. No
accuracy outcome is predeclared; the run measures whether third-order
interaction products justify their additional parameters and cost.

## Failure handling

- Never reuse a failed or partial run root.
- Preserve a failed attempt and submit a replacement only after identifying
  the cause.
- Do not change batch, worker count, epochs, optimizer, scheduler, GPU type,
  or the final readout correlation to make the run fit.
- If the three-day limit is insufficient, continue from a scheduler-faithful
  checkpoint while preserving optimizer and scheduler state; do not launch
  independent one-epoch jobs.
