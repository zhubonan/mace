# Phase 44: Native OMat24 LMDB input qualification

Date: 2026-08-18

## Outcome

The production input path can read the unchanged official OMat24
`.aselmdb` corpus directly. It does not reconstruct `ase.Atoms`, repack rows,
or store neighbour graphs. `AtomicData.from_config` still constructs the
cutoff-dependent graph online.

The new path is opt-in:

```text
--lmdb_backend=omat_direct
--batch_sampling=shard_window_atom_balanced
--lmdb_shard_window=8
--lmdb_max_open_shards=8
```

Existing HDF5 and ASE-LMDB defaults are unchanged. Batches retain exactly
`--batch_size` structures per rank. This report qualifies the full-corpus
sampler invariants and the local CPU input path; GPU and compiled-DDP
qualification remain separate Phase-44 gates.

## Architecture

`OMatLMDBDataset` builds a lightweight immutable index from the checked-in
manifest, maps a global row to a shard in logarithmic time, and opens LMDB
environments lazily through a process-local bounded LRU. Official zlib/orjson
rows are decoded directly into a MACE `Configuration`. Each worker owns its
own transactions and environments; fork, spawn, and pickle reset inherited
ownership.

`ShardWindowAtomBalancedBatchSampler` shuffles shards deterministically for
each epoch, loads `natoms` for eight shards at a time, sorts rows by atom
count with a seeded tie-break, and forms fixed 128-structure batches. Complete
batches are sorted by total atoms and assigned in adjacent four-rank packs.
Only an incomplete row batch and incomplete DDP pack are carried across a
window. `set_epoch()` changes the deterministic epoch order.

The extracted corpus is now visible at:

```text
/mnt/hdd2/bonan/datasets/omat24-aselmdb
```

The former staging path is a relative compatibility symlink to this location.
The move preserved all 11 families, 5,300 shards, 100,824,585 rows, and the
representative shard SHA256
`55436e16c28427a8f95f05afb924225ae52e5fc5acd96d62fa2dd7bec6a5ade6`.

## Full-corpus sampler audit

The audit ran world sizes 1 and 4, windows 4/8/16, and epochs 0/1. All twelve
runs used batch size 128 and passed these invariants:

- 100,824,576 rows retained and exactly nine rows dropped;
- no duplicate sample and disjoint rank streams;
- equal step counts on all ranks (196,923/rank at world size 4);
- deterministic output for a fixed seed and epoch;
- retained metadata bounded by the active shard window and at most 127 carry
  rows.

The world-size-4 p95 simultaneous-rank relative atom imbalance was:

| Window | Epoch 0 | Epoch 1 | Random epoch 0 | Random epoch 1 |
|---:|---:|---:|---:|---:|
| 4 | 4.654% | 4.436% | 36.269% | 35.866% |
| 8 | 0.953% | 0.957% | 36.269% | 35.866% |
| 16 | 0.000% | 0.000% | 36.269% | 35.866% |

Window 8 is the recommended initial setting. It captures nearly all of the
load-balance benefit without doubling the active metadata region again. In
epoch 0 its largest eight-shard window contained 295,546 rows; the observed
peak was 295,652 rows including carry.

The world-size-4/window-8/epoch-0 audit was repeated independently and
reproduced checksum
`e28e0e285da8aa4de247a125031b14aa331aa9dc4fc21e5773265b93e71a98af`.

The verifier intentionally uses a 12,603,074-byte disk-backed coverage bitset
and an 806,596,680-byte disk-backed whole-corpus `natoms` array. Those are
audit-only structures, not production sampler memory. Peak audit RSS includes
four rank samplers in one process and must not be interpreted as per-rank
training memory.

## CPU worker and locality benchmark

The local benchmark used an Intel Core i9-14900KF, Python 3.12.11,
PyTorch 2.11.0+cu128, LMDB 2.3.0, NumPy 2.5.2, cutoff 6 A, batch 128, and
2,048 graph-built structures per run. Three repetitions rotated the first
worker count among 0/8/16 and alternated policy order. Worker variants within
one policy/repetition consumed the identical row checksum.

| Sampling | Workers | Median structures/s | Full range | Median first batch | Full range |
|---|---:|---:|---:|---:|---:|
| Random | 0 | 1,591 | 45-1,713 | 0.077 s | 0.073-3.112 s |
| Random | 8 | 6,044 | 73-6,510 | 0.272 s | 0.244-15.117 s |
| Random | 16 | 6,556 | 82-6,815 | 0.273 s | 0.257-24.908 s |
| Balanced | 0 | 1,269 | 156-1,934 | 0.146 s | 0.065-1.377 s |
| Balanced | 8 | 4,345 | 251-6,448 | 0.171 s | 0.163-6.048 s |
| Balanced | 16 | 4,134 | 374-5,178 | 0.364 s | 0.168-5.446 s |

The wide ranges are real page-cache and shard-open effects. The random stream
touched 1,566-1,581 shards and made 2,046-2,047 shard transitions for 2,048
samples. The balanced stream touched only 6-7 shards and made 708-922
transitions. When a worker variant ran first for its policy, random loading
took 25-45 seconds at 0-82 structures/s, while balanced loading took 5-13
seconds at 156-374 structures/s. Cache-warm multiworker runs for both policies
exceeded 4,100 structures/s.

All runs kept active LMDB environments at or below the configured limit of
eight. The reported cumulative open counter can be much larger for random
sampling because evicted environments are reopened; it is not the number of
simultaneously live handles. Parent-process file-descriptor deltas with
workers mostly reflect DataLoader pipes and worker processes.

The sampling policies use different rows, so their raw CPU structures/s are
not a model-training speed comparison: graph cost varies strongly with atom
count. The result establishes bounded handles, worker correctness, and much
better shard locality. The matched GPU experiment must decide whether load
balancing improves end-to-end training throughput.

## Reproduction

Runtime prefix for every Python command:

```bash
env PYTHONPATH=/tmp/e3nn044-overlay:. \
  .profiling/mace-hdf5-gpu-2026-08-13/runtime/pytorch-211-cu128/bin/python
```

Example complete-corpus audit:

```bash
<runtime> .profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/audit_balanced_sampler.py \
  --root /mnt/hdd2/bonan/datasets/omat24-aselmdb \
  --batch-size 128 --world-size 4 --window-shards 8 \
  --seed 123 --epoch 0 --temporary-directory /tmp \
  --output .profiling/mace-hdf5-gpu-2026-08-13/phase44/audit-w4-window8-epoch0.json
```

CPU matrix:

```bash
<runtime> .profiling/mace-hdf5-gpu-2026-08-13/phase44/scripts/benchmark_native_omat_lmdb.py \
  --root /mnt/hdd2/bonan/datasets/omat24-aselmdb \
  --samples 2048 --batch-size 128 --window-shards 8 \
  --max-open-shards 8 --r-max 6 --seed 7123 --repetitions 3 \
  --output .profiling/mace-hdf5-gpu-2026-08-13/phase44/cpu-worker-matrix-r2.json
```

Compact source artifacts are retained in this directory. Raw decoded rows,
temporary memmaps, runtime environments, and bulky traces are not retained.

## Recommendation and remaining gate

Use `window_shards=8` for the next GPU qualification. The native reader and
sampler meet the CPU correctness, coverage, locality, and bounded-resource
requirements. Do not yet replace production HDF5 training solely from these
CPU numbers: the required next result is a matched RTX 5090 full
energy/force/stress comparison of HDF5/random, native/random, and
native/balanced at 8 and 16 workers, followed by strict two-rank and four-rank
compiled-DDP checkpoint/model-format gates.
