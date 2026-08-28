# Single-Layer OMat24 Epoch-20 Continuation Design

## Objective

Continue both single-layer OMat24 experiments from their completed epoch-9
checkpoints through epoch 19, yielding twenty total trained epochs per variant.
Submit the with-self continuation after qualification. Prepare the identical
without-self continuation, but do not stage or submit it until job `551284`
has completed normally and produced a finite epoch-9 validation row and an
epoch-9 checkpoint.

## Scientific boundary

The continuation preserves the established comparison:

- with-self uses `RealAgnosticDensityResidualInteractionBlock`;
- without-self uses `RealAgnosticDensityInteractionBlock`;
- all data, model, optimizer, batching, DDP, runtime, and validation settings
  remain unchanged from jobs `551283` and `551284`;
- each continuation runs upstream epochs 10 through 19 exactly once;
- the continuation is isolated in a new immutable deployment and separate run
  root so the completed ten-epoch evidence remains unchanged.

The production shape remains two RTX 5090 ranks, batch 256 per rank, four
DataLoader workers per rank, eight CPU cores per rank, and 46 GB of host memory
per rank (92,000 MB per node).

## Restart semantics

MACE derives the starting epoch from the checkpoint filename. For each
variant, copy the source `epoch-9.pt` checkpoint into the new run root under an
`epoch-10.pt` restart alias. Set `--max_num_epochs=20` and
`--restart_latest`; this begins at epoch 10 and executes epochs 10--19.

Before writing the alias, audit the source checkpoint:

- it contains `model`, `optimizer`, and `lr_scheduler`;
- every optimizer parameter group has LR `0.005`;
- the scheduler is compatible with
  `ReduceLROnPlateau(factor=0.5, patience=1)`;
- the source checkpoint hash and source epoch-9 validation row are recorded.

The checkpoint is saved after epoch-9 validation but before the next scheduler
step. Advance a copy of its scheduler state once using the finite epoch-9
validation loss, reproducing the scheduler state that uninterrupted training
would have at the start of epoch 10. This must leave LR at `0.005` for the
with-self checkpoint and update scheduler `best` from `0.0064259283` to the
epoch-9 loss `0.0063928887`. The without-self values are discovered and
asserted after job `551284` completes rather than predicted.

MACE checkpoints do not contain the Exponential Moving Average shadow state.
The continuation therefore reinitializes EMA from the saved checkpoint model,
whose parameters were saved under MACE's EMA-average context. This limitation
is recorded in provenance and applied identically to both variants.

## Components

1. A pure restart-preparation helper reads the source result and checkpoint,
   validates the epoch boundary, advances the serialized scheduler state, and
   writes the restart alias atomically with before/after audit JSON and
   SHA-256 records.
2. A continuation argument renderer reuses the exact single-layer production
   arguments, changes only the run directories and epoch/restart boundary, and
   emits `--max_num_epochs=20 --restart_latest`.
3. A fail-closed Slurm launcher verifies runtime, source, qualification,
   resources, checkpoint audit, and an unused run root before starting DDP.
4. A post-launch verifier requires loading the exact epoch-10 alias, optimizer
   records beginning at epoch 10, one compiled graph, finite metrics, and no
   fatal DataLoader, CUDA, DDP, or recompilation signatures.

## Variant sequencing

The with-self source is the completed job `551283` epoch-9 checkpoint and may
be prepared and submitted immediately after all tests and audits pass.

The without-self continuation uses the same code and launcher. Its preparation
has a hard readiness gate requiring job `551284` to be `COMPLETED` with exit
code `0:0`, a finite epoch-9 validation row, and a readable epoch-9 checkpoint.
Until then, only the renderer and dry-run configuration are prepared; no
checkpoint alias, production run root, or Slurm job is created.

## Failure handling

- Refuse a source checkpoint whose hash, epoch, optimizer LR, scheduler shape,
  or validation result does not match the audit.
- Refuse any pre-existing continuation run root or restart alias.
- Never alter the ten-epoch source run roots or their checkpoints.
- Stop before submission if scheduler advancement changes LR unexpectedly.
- After submission, preserve and report any failed job rather than reusing its
  run root.

## Verification and acceptance

Tests first cover the epoch-9-to-10 alias, scheduler advancement, LR
preservation, epoch range 10--19, existing-root refusal, non-finite validation,
wrong LR, and without-self readiness gate. The implementation then passes the
focused tests, repository regression tests relevant to single-layer rendering,
launcher syntax checks, and remote dry-run audits.

With-self submission is accepted when Slurm allocates the intended resources,
the log names the exact epoch-10 alias, diagnostics show optimizer epoch 10,
losses are finite, and the compiled graph count is one. Without-self
preparation is accepted when the identical dry-run passes; its eventual
submission requires the readiness gate described above.
