# Native OMat LMDB and Atom-Balanced Training Design

> **Status: approved for implementation.** The user approved consuming the
> unchanged official OMat24 ASE-LMDB files through a native MACE reader and
> retaining exactly `--batch_size` structures per rank while balancing batch
> atom counts.

## Objective

Add an opt-in production data path that trains directly from the extracted
official OMat24 `.aselmdb` shards without repacking them, constructing
`ase.Atoms`, or storing neighbor graphs. The data path must scale to all
100,824,585 structures in 5,300 shards across 11 archive families and must
use the existing per-record `natoms` metadata to reduce batch-size and DDP
rank imbalance without changing the number of structures in a batch.

HDF5 and the existing generic ASE-LMDB backend remain supported and retain
their current defaults. Model computation, loss normalization, optimizer
behavior, checkpoints, and the single-file `.model` format do not change.

## Evidence and constraints

The Phase-42 profiling-only reader established exact parity between the
official OMat24 ASE-LMDB source and the matched HDF5 pilot for all 1,000
records plus a collated batch of 128. Direct official decoding improved
standalone random-read throughput by 2.57--3.18x and compiled RTX 5090
end-to-end training throughput by 29.8--45.6% at 8/16 workers.

The complete extracted corpus occupies about 188 GiB. Its official metadata
contains one `natoms` array per non-empty shard. A read-only audit found:

- 100,824,585 rows and 5,300 LMDB shards;
- 5,269 per-shard metadata files, with metadata absent only for validated
  empty shards;
- 140 distinct atom counts from 1 to 236;
- mean and median sizes of 18.68 and 12 atoms;
- 90th, 95th, and 99th percentiles of 60, 72, and 88 atoms; and
- 94.3% of adjacent records sharing the same atom count, but a mean equal-size
  run of only 17.6 records.

Uniform record-level shuffling across thousands of files defeats a small
LMDB handle cache and causes HDD seeks. Pure shard-local batching preserves
I/O locality but does not control atom-count variation. A global 100.8-million
row sort gives strong size balance but violates the bounded-memory and storage
locality requirements. The approved design therefore balances within bounded
shard windows.

## Stable source location

The extracted corpus will move by same-filesystem rename from:

```text
/mnt/hdd2/bonan/omat-descriptors/.staging
```

to:

```text
/mnt/hdd2/bonan/datasets/omat24-aselmdb
```

The old path will become a compatibility symlink to the stable path. The
operation must not copy, rewrite, recompress, or delete an LMDB shard. Before
and after the rename, validation must record the 11 family names, shard and
row counts, representative source hashes, total logical size, and the fact
that the target and compatibility link resolve to the same filesystem
objects.

The stable root will contain a small human-readable manifest that records the
dataset identity, official format, family ordering, shard counts, row counts,
source provenance, and the fact that graph construction remains online. The
manifest is an index and discovery aid, not a repacked dataset.

## Native reader

### Public boundary

Introduce a production `OMatLMDBDataset` under `mace.data`. Training selects
it explicitly with `--lmdb_backend=omat_direct`; the option accepts `ase` or
`omat_direct` and defaults to `ase`. The existing generic `LMDBDataset`
therefore remains the default for `.lmdb` and `.aselmdb` inputs. This avoids
silently changing behavior for user-created ASE databases.

The native backend accepts either the stable OMat24 root, one archive-family
directory, or an explicit list of roots. A stable-root manifest takes
precedence. Without a manifest, discovery is recursive and lexicographically
sorted by family-relative shard path. Duplicate resolved shard paths are an
error.

The first production version supports the official OMat energy/force/stress
schema. Each decoded row must contain atomic numbers, positions, cell, PBC,
energy, forces, and stress. Configuration and property weights are read from
the official key-value fields with the same defaults as the parity-qualified
prototype. Unsupported or missing active properties fail with the shard and
row key in the error; they are never silently zero-filled.

### Global index

Dataset construction reads manifest counts or per-shard metadata lengths into
an immutable cumulative index. It does not open non-empty LMDB environments
and does not retain the 100.8-million-entry `natoms` vector in parent memory.
Missing per-shard metadata is accepted only after a brief read-only validation
proves the shard is empty.

A global integer index resolves to `(shard_index, local_index)` by cumulative
binary search. The LMDB lookup uses the official one-based numeric ASCII key,
so it is O(1) within the selected shard and does not enumerate keys.

### Row materialization

Workers decompress and decode official zlib/orjson rows directly into a MACE
`Configuration`. They do not construct `ase.Atoms` or call
`config_from_atoms`. `AtomicData.from_config` then constructs cutoff-dependent
edges and tensors online. No edge index, edge vector, shift, or MACE-specific
serialized object is stored in the source database.

### Process-local handles

Each worker owns a bounded LRU of read-only LMDB environments and transactions
opened with `readonly=True` and `lock=False`. The default maximum is eight,
matching the default sampler shard window. Handles are never inherited as
active ownership across pickle, `fork`, `spawn`, or PID change. Explicit
close and destruction are idempotent.

## Fixed-structure atom-balanced sampler

### Invariants

Introduce `ShardWindowAtomBalancedBatchSampler` with these hard invariants:

- every emitted rank-local batch contains exactly `batch_size` structures;
- all ranks emit the same number of batches and optimizer steps;
- a retained sample appears at most once in an epoch;
- rank sample sets are disjoint;
- only the final global incomplete batch and up to `world_size - 1` complete
  batches may be dropped to satisfy fixed batch and equal-rank-step semantics;
- `seed`, `epoch`, dataset identity, rank, and world size determine the exact
  sequence;
- `set_epoch()` changes the permutation without changing coverage rules; and
- memory is bounded by the active shard window and carry buffers, not total
  corpus size.

This is shuffled epoch sampling without replacement. It does not implement
independent sampling with replacement.

### Window construction

At each epoch, all non-empty shard IDs are deterministically shuffled and
partitioned into bounded windows, defaulting to eight shards. For one window,
the sampler loads only those official `natoms` arrays and constructs compact
global indices for the window's rows. Equal-size records receive a seeded
random tie-breaker before size ordering, preventing a fixed material order.

Rows are ordered by atom count and partitioned into fixed-size batches. An
incomplete tail is carried into the next window rather than dropped per shard.
The next window rebalances its rows together with that carry. At the end of
the epoch, only the global remainder allowed by the invariants is omitted.
Likewise, complete batches that do not fill a `world_size` DDP pack are
carried into the next window. They are not dropped at every window boundary.

All batches derived from one window are emitted as one bounded scheduling
region. A batch can reference multiple shards in that window, but never a
shard outside it. This limits per-worker active storage to the same shard set
that fits the reader's handle cache and avoids uniformly random access across
5,300 files.

### DDP step balancing

Each candidate batch records its total atom count. Within a window, batches
are sorted by total atoms and grouped into packs of `world_size` adjacent
batches. A deterministic random permutation assigns the batches in each pack
to ranks, and the packs themselves are shuffled before emission.

Consequently, every DDP optimizer step uses the same number of structures on
each rank and selects rank-local batches with similar predicted work. This
reduces rank stragglers while avoiding a persistent rank-to-size association.
Loss scaling continues to use the existing fixed structure count and global
DDP reduction behavior.

Atom count is the initial cost proxy because it is already available for all
100.8 million structures without graph construction. Edge count may be
measured during qualification, but no cached edges or required edge-count
sidecar is part of this implementation.

## Training integration

Add `--lmdb_backend` with choices `ase` and `omat_direct`, and add
`--batch_sampling` with choices `random` and
`shard_window_atom_balanced`. Both default to the existing behavior (`ase`
and `random`). `--lmdb_shard_window` and `--lmdb_max_open_shards` are positive
integers that both default to eight. The balanced sampling mode is initially
valid only with `--lmdb_backend=omat_direct`.

When the balanced sampler is selected, the PyG DataLoader receives it as
`batch_sampler`; it must not also receive `batch_size`, `shuffle`, `sampler`,
or `drop_last`.

The training loop calls `set_epoch()` on either the existing distributed
sampler or the new balanced batch sampler. Epoch-level checkpoint/resume
reproduces the same next-epoch sequence as the current training contract.
Mid-epoch cursor checkpointing is outside scope.

The initial integration supports one native OMat training dataset spanning
all selected archive families. Existing validation/test inputs and multi-head
non-OMat datasets continue through their current loaders. Mixing native OMat
and arbitrary datasets inside one balanced sampler is outside scope.

Average-neighbor statistics, model construction, compiled training, eager
training, DDP wrapping, optimizer state, EMA, checkpoint ownership, and final
model serialization continue to consume ordinary PyG batches and require no
model-side change.

## Error handling

Startup fails before training when the manifest is inconsistent, a shard is
missing, a non-empty shard lacks metadata, cumulative counts disagree, roots
resolve to duplicates, or backend/sampler options are incompatible. Runtime
row errors include family, shard, local index, and LMDB key.

All ranks must agree on dataset identity, row count, sampler configuration,
and number of steps before the first optimizer update. A disagreement is a
fatal distributed setup error rather than a fallback to ordinary random
sampling.

## Tests

Implementation follows red-green-refactor development and must cover:

1. Recursive and manifest-driven multi-family discovery, stable ordering,
   duplicate rejection, and missing/empty metadata rules.
2. Exact row and batch parity against matched HDF5 for the active OMat
   energy/force/stress schema.
3. No eager LMDB handles at construction plus real `fork` and `spawn`
   ownership reset, bounded LRU eviction, and idempotent cleanup.
4. Synthetic 11-family indexing and random global lookups without retaining a
   corpus-sized Python index list.
5. Fixed batch size, deterministic repeat, changed epoch order, no duplicates,
   disjoint DDP ranks, exact allowed-drop bounds, and equal rank step counts.
6. Bounded shard-window membership and carry behavior across windows.
7. Lower simultaneous-rank atom imbalance than the standard distributed
   random sampler on both synthetic skewed data and the official metadata.
8. DataLoader integration with zero and multiple workers, including real
   `fork` and `spawn` contexts.
9. Existing generic LMDB, HDF5, CLI/config, eager, compiled, DDP, checkpoint,
   and final `.model` compatibility.

## Qualification gates

Before production recommendation, run:

- a complete metadata-only sampler audit over all 100,824,585 records,
  proving deterministic coverage, drop bounds, equal rank steps, bounded
  memory, and atom-balance distributions for one and four ranks;
- a multi-family multiworker CPU benchmark recording startup, throughput,
  open/active environments, descriptors, RSS, shard opens, and random-read
  locality;
- matched one-GPU HDF5 versus native-LMDB training with identical sample and
  batch-shape sequences where feasible;
- four-rank variable-shape compiled full energy/force/stress training on the
  RTX 5090 node, with stable Dynamo graphs, finite matching losses, optimizer
  checksum agreement, peak memory, batch atom/edge counts, rank wait time,
  step latency, and structures per second; and
- checkpoint/restart plus final legacy-compatible single-file `.model`
  loading.

The sampler must reduce the 95th-percentile simultaneous-rank relative atom
imbalance compared with the ordinary distributed sampler. The native path
must retain the Phase-42 scientific parity and must not regress sustained
native-LMDB end-to-end throughput by more than 2%. Full-corpus production use
remains opt-in until these gates pass.

## Compatibility and non-goals

- Official `.aselmdb` files and per-shard metadata remain unchanged.
- No dataset repack, shard merge, ASE object construction, or stored graph is
  introduced.
- HDF5 and generic ASE-LMDB behavior remain available and unchanged by
  default.
- `batch_size` remains the number of structures per rank.
- Variable-structure atom-budget batching is not introduced.
- Edge-count prediction, cached edges, and a new MACE-native on-disk LMDB
  schema are deferred.
- Mid-epoch sampler cursor checkpoints and arbitrary mixed-dataset balancing
  are outside scope.
- Model architecture, numerical precision, loss definitions, optimizer
  parameters, checkpoint schema, and `.model` serialization do not change.
