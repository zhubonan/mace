# Single-Layer OMat24 Correlation-2 Width Sweep Design

## Objective

Train two fresh without-self, single-interaction MACE models on full OMat24
for twenty epochs. The candidates use 96 and 64 hidden channels. Relative to
their active correlation-3 width counterparts, change only the first
interaction product correlation from 3 to 2. Keep the separate nonlinear
scalar readout contraction at correlation 2.

## Considered approaches

1. **Dedicated exact-pair width sweep (selected).** Extend the immutable
   correlation-2 protocol to admit only \`(96, 2)\` and \`(64, 2)\`, construct and
   verify each model, and submit independent fresh jobs. This preserves a
   controlled comparison and fail-closed production admission.
2. Generalize the 128-channel correlation-2 launcher to accept arbitrary
   widths. This is rejected because it would permit unqualified scientific
   combinations and make exact parameter admission less clear.
3. Continue the correlation-3 width checkpoints after changing their product
   basis. This is rejected because the topology and optimizer tensors differ;
   it would not be a fresh or controlled comparison.

## Scientific topology

Both candidates use:

- \`RealAgnosticDensityInteractionBlock\` as the sole interaction;
- no residual or self-connection;
- \`num_interactions=1\`;
- hidden irreps of \`Cx0e + Cx1o + Cx2e\`, where \`C\` is 96 or 64;
- spherical harmonics \`max_ell=3\`;
- interaction product \`correlation=2\`;
- readout-local invariant product \`readout_correlation=2\`;
- \`use_last_readout_only=True\` and \`MLP_irreps=16x0e\`.

The readout contraction is not a second neighbor interaction. It contracts
the retained first-layer equivariant features into scalar invariants for the
nonlinear energy readout.

## Exact parameter counts

CPU construction against the pinned MACE source and e3nn 0.4.4 gives:

| Candidate | Correlation 3 | Correlation 2 | Reduction |
|---|---:|---:|---:|
| 96 channels | 3,860,186 | 3,569,690 | 290,496 (7.53%) |
| 64 channels | 1,828,890 | 1,635,226 | 193,664 (10.59%) |

Correlation-2 module groups are:

| Module group | 96 channels | 64 channels |
|---|---:|---:|
| Interaction | 3,360,394 | 1,503,882 |
| First product basis | 155,808 | 97,728 |
| Readout product | 43,392 | 26,880 |
| Node embedding | 8,544 | 5,696 |
| Nonlinear readout | 1,552 | 1,040 |
| **Total** | **3,569,690** | **1,635,226** |

The production verifier must construct each model and fail unless it observes
the exact topology and count.

## Training and resource protocol

Everything except the interaction correlation remains identical to the active
fresh correlation-3 width jobs:

- full OMat24 native LMDB training data and frozen validation/statistics;
- fresh initialization with seed \`20260729\`;
- epochs 0 through 19 in one job, with no checkpoint restart;
- batch 256/rank, validation batch 64, and four workers/rank;
- two RTX 5090 ranks, eight CPUs/rank, and 92,000 MB/node;
- Adam/AMSGrad, EMA, loss weights, normalization, clipping, and
  \`ReduceLROnPlateau(factor=0.5, patience=1)\` unchanged;
- IEEE float32, CuEquivariance, compiled DDP, and
  \`--gpu-bind=mask_gpu:0x3,0x3\` unchanged.

Each candidate receives unique run, compiler-cache, stdout, and stderr roots.
Existing correlation-3 and 128-channel correlation-2 jobs remain untouched.

## Implementation boundary

- Represent the two exact width/correlation pairs and expected counts in one
  immutable mapping.
- Extend CPU provenance to admit only those pairs while preserving current
  correlation-3 width and 128-channel correlation-2 behavior.
- Add a dedicated dry-render/execute CLI path that requires width 96 or 64,
  without-self, correlation 2, batch 256, and four workers.
- Add one narrowly parameterized Slurm launcher that accepts only the two exact
  candidates and enforces their expected parameter counts and isolated names.
- Retain qualification JSON, MACE source commit, patched matscipy wheel,
  runtime, full-data, GPU-binding, IEEE-float32, finite-validation, compiled
  graph, no-checkpoint-loading, and epoch-19 postconditions.

## Testing and qualification

Use test-first development for:

- exact candidate mappings and rejected combinations;
- renderer equivalence to the correlation-3 width protocol after excluding
  only correlation and isolated names/paths;
- exact model construction, both product correlations, and parameter counts;
- CLI admission and dry-render provenance;
- launcher resources, admission ordering, cache/run freshness, all three
  masked \`srun\` calls, candidate-specific names, and final checkpoints.

Before submission, require the focused tests, complete \`tests/full_retrain\`
suite, Python compilation, launcher \`bash -n\`, a metadata-free sealed archive,
remote exact construction for both widths, and remote dry renders containing
twenty epochs with no restart.

## Submission and monitoring

Submit two independent jobs from the same sealed deployment. Verify each
resolves to two typed RTX 5090 GPUs, two tasks, eight CPUs/task, 92,000 MB,
three days, zero restarts, the sealed launcher, and unique output paths.

If allocated, require exact provenance, both GPUs visible, fresh
initialization, one compiled graph, finite optimizer/validation rows, bounded
memory, and no fatal signature. If pending, report the scheduler reason without
changing the protocol.

## Failure handling

- Never reuse a failed or partial run or compiler-cache root.
- Preserve failed attempts and use a fresh deployment/run suffix after fixing
  the identified cause.
- Do not alter width, correlations, batch, workers, optimizer, scheduler,
  epochs, GPU type, or memory request to make a job start.
- If a job cannot complete within three days, continue from a
  scheduler-faithful checkpoint with optimizer and scheduler state intact;
  never split it into independent one-epoch jobs.
