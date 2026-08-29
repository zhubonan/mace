# Single-Layer OMat24 96/64-Channel Width Sweep Design

## Objective

Train two fresh without-self single-layer MACE models on full OMat24 for twenty
epochs, changing only the hidden channel multiplicity from the established
128-channel run to 96 and 64. The experiment determines how much width can be
removed without losing the accuracy advantage of the without-self topology.

## Scientific boundary

The two candidates use:

- `RealAgnosticDensityInteractionBlock` as the sole interaction;
- `num_interactions=1` and no first-layer self/residual connection;
- hidden irreps `Cx0e + Cx1o + Cx2e`, where `C` is 96 or 64;
- spherical harmonics through `max_ell=3`;
- interaction correlation 3 and readout correlation 2;
- the existing nonlinear last-readout path with `MLP_irreps=16x0e`;
- identical full OMat24 training and validation data, statistics, normalization,
  seed, loss weights, optimizer, EMA, scheduler, batching, workers, precision,
  CuEquivariance, compilation, and diagnostics.

Both runs start from fresh random initialization and execute epochs 0--19 in
one uninterrupted Slurm job. They do not load, resize, or reuse weights from
the 128-channel model. Using the same seed preserves a paired architectural
comparison; the different tensor shapes naturally produce different random
parameter streams.

## Expected model sizes

CPU construction against the selected MACE source and e3nn 0.4.4 gives:

| Width | Trainable parameters | Reduction from 128 |
|---:|---:|---:|
| 128 | 6,639,002 | baseline |
| 96 | 3,860,186 | 41.9% |
| 64 | 1,828,890 | 72.5% |

The production renderer and preflight must instantiate each candidate and
fail unless its exact trainable parameter count matches this table. Checkpoint
tensor totals are not used for this gate because they include non-trainable
product-basis buffers.

## Training protocol

Every non-width training argument remains identical to the accepted
without-self run:

- full OMat24 native LMDB training set and frozen optimizer-validation set;
- batch 256 per rank and validation batch 64;
- four DataLoader workers per rank;
- two distributed ranks and two RTX 5090 GPUs;
- eight CPU cores and 46,000 MB host memory per rank;
- float32 IEEE execution, CuEquivariance, and compiled DDP;
- Adam with AMSGrad, weight decay `1e-8`, clipping 10, and the established
  initial learning-rate configuration;
- EMA decay 0.999;
- `ReduceLROnPlateau(factor=0.5, patience=1)`;
- twenty epochs in a single job with all checkpoints retained.

The launcher uses the verified two-rank Cata GPU visibility contract
`--gpu-bind=mask_gpu:0x3,0x3`, because current Slurm task auto-binding exposes
only one GPU per rank when `--gpu-bind=none` is used.

## Implementation components

1. Extend the immutable single-layer specification with width candidates 96
   and 64 and exact expected trainable parameter counts.
2. Add a width-sweep renderer that changes only `hidden_irreps`, run name, and
   isolated output paths while reusing the accepted without-self arguments.
3. Add CPU model-construction provenance that verifies topology, parameter
   count, and the absence of the residual first interaction.
4. Add a fail-closed two-rank Slurm launcher for fresh 20-epoch training using
   the proven pinned CUDA 13/MACE/e3nn/CuEquivariance/matscipy runtime.
5. Seal a new deployment with source/archive hashes, dry renders, runtime
   admission evidence, and distinct run/cache/log roots for each width.
6. Submit the 96- and 64-channel jobs independently. They may run concurrently
   when four RTX 5090s are available; otherwise Slurm may queue either job
   without changing its resources or protocol.

## Qualification and verification

Before production submission:

- renderer tests prove width is the only scientific argument difference;
- unsupported widths fail closed;
- model-construction tests assert 3,860,186 and 1,828,890 trainable parameters;
- both configurations construct and execute a finite CPU forward/backward
  smoke test where practical;
- launcher syntax and resource contracts pass;
- remote dry renders reproduce the pinned runtime and full-data paths.

After allocation, each job is accepted only after it shows:

- two typed RTX 5090s visible to both ranks;
- exact runtime/source/project admission;
- fresh initialization with no checkpoint load;
- the expected trainable parameter count;
- one compiled training graph per rank;
- finite optimizer diagnostics, low loader wait, and bounded host memory;
- no traceback, CUDA OOM, DataLoader worker exit, NCCL failure, or unexpected
  graph recompilation.

## Comparison and acceptance

Validation loss and energy, force, and stress MAEs are reported for every
completed epoch against the 128-channel without-self baseline. The primary
selection metric is force MAE; energy and stress MAEs and weighted validation
loss are co-reported to prevent a misleading single-metric choice.

No smaller model is declared acceptable until epoch-19 validation is
available. The final report includes parameter reduction, throughput, GPU and
host memory, and relative accuracy degradation. The acceptable degradation
threshold is left as an explicit selection decision rather than silently
encoded into job submission.

## Failure handling

- Never reuse a failed or partial production run root.
- Preserve failed jobs and logs as evidence and submit replacements only from
  a fresh root after identifying the cause.
- Do not alter or stop the active 128-channel continuation jobs.
- Do not reduce epochs, change batch size, relax GPU typing, or change the
  optimizer to make a candidate fit the scheduler limit.
- If a job cannot complete twenty epochs within the three-day limit, continue
  it from a scheduler-faithful checkpoint in one restart job while preserving
  optimizer and scheduler state; do not split training into independent
  one-epoch jobs.
