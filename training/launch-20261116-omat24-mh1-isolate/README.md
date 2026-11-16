# Upstream-main isolation run for MACE-MH-1 on OMat24

This recipe compares the production MH-1 branch against upstream `main` plus
only two selected features:

1. Compiled training from [PR 1801](https://github.com/ACEsuit/mace/pull/1801).
2. The standalone `GlobalRandomBatchSampler` port.

It deliberately omits the production branch's custom DDP loss reduction,
accepted-update LR warmup, reference-target/loss-spike guards, native OMat
LMDB reader, direct-foundation loader, Per-L kernels, and custom checkpoint
format.

The upstream LMDB reader cannot consume the OMat24 manifest directory
directly. `prepare_remote.sh` therefore creates manifest-ordered symlinks to
every `.aselmdb` shard in `data/omat-aselmdb-flat`. This preserves row order
while keeping the reader itself upstream.

Launch order:

```bash
bash scripts/submit-production.sh preflight
bash scripts/submit-production.sh dry-run
bash scripts/submit-production.sh submit
```

The preflight uses the 10,000-row validation HDF5 pack and stops after model
construction and cuEq conversion. The dry run opens the complete OMat24 LMDB
set on all four ranks. Production then runs 20 epochs at global batch 512
(128 per rank), exactly 196,923 complete updates per epoch, dropping the final
nine rows.
