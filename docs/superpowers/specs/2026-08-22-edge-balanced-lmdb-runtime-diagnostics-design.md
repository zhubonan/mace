# Edge-Balanced OMat LMDB Training and Runtime Diagnostics Design

> **Status: approved for implementation.**
> The user approved exact cutoff-specific edge-count sidecars, fixed structures
> per rank, and runtime diagnostics for loader waits and distributed efficiency.

## Objective

Extend the native OMat LMDB training path with an opt-in sampler that balances
both total atoms and exact directed neighbour edges while retaining exactly
`--batch_size` structures per rank. Add opt-in, production-safe diagnostics that
identify exposed DataLoader waits, rank-local workload imbalance, GPU-active
training intervals, distributed stragglers, and measured multi-GPU scaling
losses.

Official OMat LMDB files remain unchanged. Neighbour topology remains
cutoff-dependent and is constructed online. Model computation, loss scaling,
optimizer behavior, checkpoints, and the canonical single-file `.model` format
do not change.

## Requirements and non-goals

The first implementation must:

- preserve exactly `--batch_size` structures in every emitted rank-local batch;
- preserve deterministic, disjoint, without-replacement DDP sampling;
- retain the current bounded shard-window and LMDB-handle limits;
- use exact edge counts from the same neighbour-list contract as training;
- fail before training on a missing, stale, corrupt, or incompatible profile;
- keep the current atom-only sampler available and unchanged;
- add no required Python dependency for diagnostics;
- add negligible overhead when diagnostics are disabled; and
- distinguish low-overhead runtime summaries from deeper profiler attribution.

The first implementation will not store edge indices, vectors, shifts, or
graphs. It will not change batch structure count, implement dynamic gradient
accumulation, register a custom DDP communication hook, or infer true scaling
efficiency from a single multi-rank job.

## Edge-count cost profiles

### Separate derived data

Edge counts will live in a separate immutable cost-profile root rather than in
the official LMDB directory or official `metadata_db_*.npz` files. A profile is
identified by its source dataset, cutoff, neighbour semantics, and format
version. Multiple profiles for different cutoffs may coexist.

The profile manifest records:

- format and schema version;
- source dataset identifier and source OMat manifest SHA-256;
- relative source shard ordering and row counts;
- exact floating-point cutoff representation;
- directed-edge and self-interaction semantics;
- PBC/cell behavior and neighbour-list implementation identifier;
- MACE and matscipy provenance used for generation;
- relative per-shard cost-array paths and row counts; and
- profile-wide row, shard, atom, and edge summaries.

Each non-empty shard has a one-dimensional integer `nedges` array aligned with
the shard's official one-based numeric row order. Empty shards have an explicit
zero-row entry. The existing official `natoms` array remains the atom source of
truth; duplicating it in the cost profile is unnecessary.

### Generation

A standalone CPU preprocessing command reads official rows directly through
the native decoder, invokes the same neighbourhood construction used by
`AtomicData.from_config`, and records only the resulting directed edge count.
Generation is shard-parallel, resumable, and safe for independent Slurm array
tasks. Each shard result is written atomically and is accepted on resume only
after its schema, row count, and generation contract validate.

The final manifest is written only after all declared shard results pass. A
partial profile is never accepted for training. The generator never writes to
an LMDB shard and never retains a complete graph.

### Validation and loading

Training selects a profile explicitly. Startup validates the source dataset and
profile contracts from headers and manifests without decompressing every edge
array. Edge-balanced sampling requires an exact cutoff and semantics match;
there is no silent fallback to atom-only sampling.

`OMatShardIndex` gains an optional immutable cost-profile association and a
lazy per-shard cost loader. The sampler loads `natoms` and `nedges` only for the
same bounded planning reservoir it already uses. Array shape, integer type,
non-negative values, and shard length are validated when a shard enters the
reservoir. No corpus-wide cost vector is retained by every rank.

Before the first optimizer step, all ranks verify the same dataset identity,
profile identity, cutoff, sampler configuration, epoch length, and profile
manifest hash.

## Fixed-structure atom-and-edge balancing

### Public behavior

Add `shard_window_atom_edge_balanced` as a new `--batch_sampling` choice and
add `--lmdb_cost_profile=PATH`. The existing random and atom-only modes retain
their behavior and defaults. Selecting the new mode requires
`--lmdb_backend=omat_direct` and a validated cost profile. Supplying
`--lmdb_cost_profile` to another training sampling mode is a configuration
error, preventing a profile path from being silently ignored.

The fixed-cardinality and coverage invariants of
`ShardWindowAtomBalancedBatchSampler` remain mandatory: equal steps on all
ranks, disjoint rank samples, deterministic epoch permutations, bounded carry,
and only the final globally unavoidable remainder dropped.

### Planning-reservoir balance

The existing two-window planning reservoir will consider row-weighted mean
edges as well as row-weighted mean atoms when partitioning source shards into
access windows. The objective minimizes the worst normalized edge deviation
first and atom deviation second, followed by row-count and shard-count
imbalance. Profile manifest summaries provide these aggregate costs without
loading all row arrays.

This improves workload consistency between access windows while preserving the
configured shard-locality and handle bounds.

### Batch construction

Within an access window, complete rows are assigned to a fixed number of
fixed-capacity batch bins. Atom balance is a hard invariant: the sampler first
constructs the exact same atom-count slots as the atom-only striped packer and
therefore preserves its per-batch atom totals exactly. Structures with the same
atom count are then assigned to those equal-atom slots in descending edge-cost
order, always selecting the currently lightest eligible edge bin with seeded
tie-breaking. Edge balance is optimized only within this atom-total boundary;
it may never trade an atom-safe batch for a lower edge total.

The procedure must remain bounded by active-window rows and have near sorting
complexity rather than comparing every row with every batch. Tests and complete
corpus replay, rather than an assumed heuristic property, determine whether its
edge and atom tails meet the acceptance gates.

Every candidate batch records structures, total atoms, and total edges. No edge
estimate is recomputed during sampler planning.

### DDP pack formation

Candidate batches are ordered primarily by total edges and secondarily by total
atoms. Adjacent groups of `world_size` batches form one simultaneous DDP pack.
A seeded permutation assigns pack members to ranks, and packs are shuffled for
training order.

Heavy work therefore rotates among ranks while simultaneous ranks receive
similar predicted memory and compute cost. Fixed structure count preserves the
existing loss and optimizer semantics. A single unusually dense structure may
still create an irreducible tail; full-corpus qualification must report these
cases rather than hiding or dropping them.

## Runtime training diagnostics

### Operating modes

Diagnostics are opt-in and have two levels:

1. **Runtime summaries** use host clocks, existing batch tensors, buffered
   counters, and infrequent fixed-size distributed reductions. They are suitable
   for long production jobs.
2. **Deep capture** adds either bounded NVTX ranges or a scheduled PyTorch
   Profiler trace for selected steps. It is intended for short diagnostic
   windows, not continuous full-epoch collection.

CUDA-enabled PyTorch already supplies `torch.cuda.nvtx` and PyTorch Profiler.
Nsight Systems remains an external NVIDIA executable. No TensorBoard package is
required because profiler traces use the standard Chrome trace format.

The public controls are `--training_diagnostics=off|summary|deep`,
`--training_diagnostics_interval` with default 1,000 steps, and
`--training_diagnostics_dir`, which defaults below the ordinary run log root.
Deep mode additionally requires `--training_diagnostics_start_step`, a positive
`--training_diagnostics_steps`, and `--training_diagnostics_backend=nvtx` for an
external Nsight Systems capture or `torch_profiler` for a built-in Chrome trace.
The two deep backends are not run simultaneously because both may compete for
CUDA profiling facilities. The default is `off`, with no diagnostic timers,
files, ranges, profiler state, or collectives.

### Loader and step boundaries

The ordinary `for batch in loader` loop will be expressed through an explicit
iterator when diagnostics are enabled so time blocked in `next()` is visible.
The measured loader wait includes all latency exposed to the parent rank after
worker prefetching; it does not claim to be the sum of worker CPU time.

Each sampled step records:

- epoch, optimizer step, rank, and world size;
- structures, atoms, and actual directed edges;
- exposed DataLoader wait and complete step period;
- host-to-device submission wall time;
- training-step wall time through the existing loss synchronization point;
- optimizer and EMA enabled state;
- current and peak allocated/reserved CUDA memory when CUDA is active; and
- compiler-active or eager-fallback state when compiled training is requested.

Dynamic compiled training consumes its first one or two batches before the
ordinary measured loop to create and validate the compiled facade. Their
replayed optimizer records are marked `compile_prefetched=true` and
`timing_representative=false`; they are retained in rank-local JSONL for audit
but excluded from sparse rank timing aggregates. Subsequent compiled records
have the same timing semantics as eager summary records.

Asynchronous host phase timings are labelled as host wall time. The low-overhead
mode does not present Python launch intervals as GPU kernel durations.

### Parallel-efficiency summaries

At a configurable, sparse interval, all ranks reduce fixed-size aggregates for
loader wait, step period, atoms, edges, and memory. Rank zero reports minimum,
mean, maximum, coefficient of variation, and worst-rank identity. The primary
within-job indicators are:

- exposed-loader-wait fraction of the global critical-path period;
- rank workload spread for atoms and edges;
- rank step-period balance;
- fraction of intervals dominated by input wait; and
- compiled fallback or graph-growth events.

These metrics diagnose rank stragglers and input starvation but are not labelled
as DDP scaling efficiency. Production-relevant weak-scaling efficiency is
computed only from matched one-, two-, and four-GPU end-to-end controls with a
constant per-rank batch: global throughput divided by GPU count and the one-GPU
throughput. A separate fixed-global-batch matrix is required before making a
strong-scaling claim.

### Attribution ranges and deep capture

NVTX ranges identify DataLoader wait, host-to-device transfer, forward/loss,
backward/DDP reduction, gradient clipping, optimizer, and EMA boundaries.
Nsight Systems can then correlate these ranges with CUDA kernels, copies, CUDA
API calls, and NCCL without changing DDP communication semantics.

An optional bounded PyTorch Profiler schedule captures CPU operators, CUDA
kernels, tensor shapes, memory events, and distributed collectives over a small
declared step window. It exports one trace per rank. Continuous profiling and a
custom DDP communication hook are excluded because both can alter the runtime
being measured.

### Output and overhead control

Each rank writes buffered diagnostic JSONL to an isolated rank-specific file;
rank zero also writes sparse aggregate records compatible with existing run
directories. File output is flushed at bounded intervals and at clean shutdown.
The schema is versioned and records clock and timing semantics.

When diagnostics are disabled, the training loop follows its existing path and
does not create timers, ranges, profiler objects, extra collectives, or files.
When runtime summaries are enabled, per-step work is constant and records are
buffered. Deep capture requires an explicit finite step window.

## Error handling

Training fails before DataLoader construction when the cost profile is missing,
partial, stale, corrupt, mapped to different shards, or generated with a
different cutoff or neighbour contract. All ranks fail rather than allowing a
mixed atom-only/edge-aware run.

Diagnostic output failures are fatal only when diagnostics were explicitly
requested. A profiler configuration that requests CUDA activities on a
non-CUDA runtime fails with a precise configuration error. Rank-specific output
paths prevent concurrent writes to one file.

## Testing strategy

Cost-profile tests cover atomic generation, resume validation, corrupt and
partial profiles, dataset/profile identity mismatches, cutoff mismatches, empty
shards, array schema, and exact comparison with freshly constructed neighbour
graphs.

Sampler tests cover deterministic coverage, fixed cardinality, bounded
reservoirs, equal DDP steps, disjoint rank sets, epoch changes, pack rotation,
old atom-only behavior, and adversarial datasets where atoms and edges are
weakly correlated. Assertions compare edge and atom spread against the current
atom-only sampler and retain irreducible single-structure tails explicitly.

Diagnostics tests use controlled clocks and synthetic loaders to verify exposed
wait, step-period boundaries, batching statistics, buffering, output schemas,
rank aggregation, clean shutdown, disabled-path equivalence, and profiler-window
bounds. CPU distributed tests use a real two-process Gloo group to verify
collective ordering, rank-local files, and the rank-zero aggregate. Compatibility
tests cover resumed-epoch sampler reconstruction, profile-root relocation,
empty shards, irreducibly dense structures, and exclusion of diagnostics from
checkpoint/model state.

## Qualification and acceptance gates

Qualification proceeds from a small LMDB pilot to the complete OMat corpus and
then to real GPU training:

- reproduce every sidecar edge count on a sampled set with online graph
  construction;
- replay a complete epoch and report median, p95, p99, maximum, and simultaneous
  rank spread for structures, atoms, and edges;
- demonstrate strictly better p99 and maximum edge-total spread than atom-only
  sampling while preserving its per-window atom-total distribution exactly;
- retain identical sample coverage and per-rank optimizer-step count;
- run compiled four-rank worst-edge and densest-structure optimizer steps with
  finite loss and no OOM or fallback;
- compare diagnostics disabled/enabled throughput and require negligible
  runtime-summary overhead;
- verify parent loader-wait attribution against the corresponding selected
  Nsight or PyTorch Profiler deep capture; and
- run matched one-, two-, and four-GPU end-to-end weak-scaling controls with the
  production dataset, workers per rank, precision, model, and per-rank batch;
  run a separate fixed-global-batch matrix if strong scaling is claimed.

No production recommendation is made solely from sampler statistics. The final
gate is a representative variable-shape compiled DDP run that reports stable
memory, no fallback, loader-wait fraction, rank workload spread, and measured
end-to-end scaling efficiency.

## Compatibility and rollout

The feature is opt-in. Existing HDF5, generic LMDB, native random, and native
atom-only commands remain valid. Existing OMat manifests and metadata remain
readable. Cost profiles are derived artifacts that can be regenerated or moved
independently of the source LMDB root.

Checkpoint state, restart semantics, model parameters, EMA, scheduler state,
and saved `.model` files contain no cost-profile or diagnostics object. A job
may resume with an equivalent relocated profile after identity validation; its
next epoch remains deterministic for the selected sampler mode and seed.
