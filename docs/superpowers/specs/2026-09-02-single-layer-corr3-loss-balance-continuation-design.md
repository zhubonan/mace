# Single-Layer Correlation-3 Loss-Balance Continuation Design

## Purpose

Prepare three isolated, scheduler-faithful OMat24 continuations from the
completed without-self, one-interaction, correlation-3 single-layer models.
The continuations change the energy/force balance from 1:10 to 1:1 by raising
the energy weight, while keeping the force and stress weights unchanged. They
will be prepared and qualified but not submitted in this change.

## Fixed experiment boundary

Each continuation resumes its completed epoch-19 checkpoint and executes five
additional upstream epochs, 20 through 24, with `max_num_epochs=25`.

| Width | Source job | Source checkpoint | SHA-256 |
|---:|---:|---|---|
| 128 | 553602 | `/scratch/bonan/train-downsample-single-layer/e3d8580-cont20-fe344bc4-v3/runs/without-self-continuation-to20/checkpoints/single-layer-omat24-without-self_run-20260729_epoch-19.pt` | `43a7b4f42317b5dc84ec963d2f925a1a0c186c26a234ff0258468054c3c905ec` |
| 96 | 553748 | `/scratch/bonan/train-downsample-single-layer/e3d8580-6509ea2-width-sweep-c96-c64-v2/runs/c96/checkpoints/single-layer-omat24-without-self-c96_run-20260729_epoch-19.pt` | `f3785a6d0e4c8a9479b43da3fc25092ea27f1ce141feae61f142814a97866b39` |
| 64 | 553749 | `/scratch/bonan/train-downsample-single-layer/e3d8580-6509ea2-width-sweep-c96-c64-v2/runs/c64/checkpoints/single-layer-omat24-without-self-c64_run-20260729_epoch-19.pt` | `8a26f88e1ce2e050ae33dd9140ae2d73de14b9dab4d716f960b61dd4c9a67f72` |

The 128-channel source is the completed correlation-3 reference, not the
still-running 128-channel correlation-2 run.

## Objective transition

The previous objective was:

```text
energy_weight=1.0
forces_weight=10.0
stress_weight=10.0
```

The continuation objective is:

```text
energy_weight=10.0
forces_weight=10.0
stress_weight=10.0
```

Thus energy and force have equal weight. The model topology, normalization,
stress weight, optimizer type, AMSGrad state, EMA handling, float32/TF32
controls, data paths, batch 256/rank, four DataLoader workers/rank, and two
RTX-5090 ranks remain unchanged.

## Scheduler-faithful restart

Each source checkpoint is copied only into a fresh continuation run root as an
epoch-20 restart alias. The restart-preparation utility must verify the source
hash and checkpoint schema, extract the epoch-19 validation loss from the
source result file, and advance the saved `ReduceLROnPlateau` state exactly
once using that old-objective epoch-19 loss. It then records before/after
optimizer LRs and scheduler state in immutable restart audit JSON.

The new 10:10:10 objective takes effect only after this handoff, at epoch 20.
No historical validation row is reinterpreted under the new loss weights.

## Launcher and admission contract

One dedicated three-width launcher accepts only channels 128, 96, or 64 with
the exact source-checkpoint identity and source epoch 19. It fails before
creating mutable output when any source, scheduler, runtime, qualification,
GPU, resource, objective-weight, or run/cache-path check differs from the
contract.

Each invocation uses a distinct `loss-e10-f10-continue-e20-to24` run root,
compiler cache, Slurm stdout, and stderr path. The launcher requires two typed
RTX 5090 GPUs, two ranks, eight CPUs/rank, 92,000 MB/node, and a three-day
allocation. It must observe the intended restart-alias load before training,
compiled DDP activation, finite epoch-20 validation, and an epoch-24
checkpoint.

The post-training MACE export reload of its final checkpoint is permitted.
Only a missing or incorrect initial restart-alias load is an admission
failure; a generic ban on `Loading checkpoint:` would incorrectly reject a
completed run, as happened for jobs 553748 and 553749.

## Verification and delivery

Tests will first define the exact argument rendering, source-width mapping,
loss weights, epochs 20--24, scheduler audit, fresh-state rejection, and
post-training reload behavior. The implementation will pass focused and full
`tests/full_retrain` suites, Python compilation, launcher syntax, local
topology/dry-render checks, and a sealed remote source qualification. The
result is a prepared, unsubmitted deployment and three ready-to-submit Slurm
commands.
