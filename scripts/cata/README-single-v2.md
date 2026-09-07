# single-v2 Cata run

`train_single_v2_omat0.sbatch` trains a fresh one-interaction model from the
medium OMat-0 checkpoint. It copies only the source first interaction and
product, then trains those parameters together with a fresh standard
correlation-2 nonlinear readout.

The launcher requires these environment variables:

```text
MACE_SOURCE_ROOT MACE_COMMIT RUNTIME_ROOT FOUNDATION_MODEL FOUNDATION_SHA256
TRAIN_ROOT VALID_ROOT STATISTICS_FILE RUN_ROOT CACHE_ROOT
```

`RUN_ROOT` and `CACHE_ROOT` must be unique and absent before submission. The
launcher requests two RTX 5090 GPUs, two ranks, eight CPUs per rank, four data
workers per rank, and 92 GB host memory. Its loss, EMA, weight decay, and
ReduceLROnPlateau settings match the accepted single-v1 recipe.
