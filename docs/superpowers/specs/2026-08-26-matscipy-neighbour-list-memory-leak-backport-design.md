# Matscipy Neighbour-List Memory-Leak Backport Design

## Objective

Remove the linear host-memory growth in OMat24 DataLoader workers without
changing MACE neighbour semantics, the pinned dependency family, or the
production model/training configuration. Backport the exact upstream matscipy
fix into an isolated dependency-only runtime, qualify it on the reproducing
workload, and retain the current runtime as an immutable rollback artifact.

## Root cause and evidence

MACE calls matscipy 1.2.0 `neighbour_list("ijS")` once per structure. In
`c/tools.c::resize_array`, matscipy allocates a temporary NumPy dimensions
buffer using `malloc(ndim * sizeof(npy_intp))`, passes it to
`PyArray_Resize`, and never frees it. The helper is called for all three output
arrays on every geometric expansion and final shrink.

The deployed `_matscipy` binary contains the same missing-free control flow:
its exported `resize_array` symbol calls `malloc` and has no `free` call on
success or failure. The exact matscipy v1.2.0 tagged source matches the source
distribution at SHA-256
`a37451489ce42c12ce4b734d9168732b2031e2892c45868682d9ac5925150b75`.

The 10,000-batch real-OMat probe measured `1.630030 GiB/10k` growth in glibc
`uordblks`, `1.629204 GiB/10k` literal heap growth, and effectively flat
`fordblks` and `keepcost`. Thus the retained pages are live native allocations,
not Python cycles or glibc free-space fragmentation. This accounts for more
than 99.9% of the observed heap slope.

Upstream matscipy commit
`db167a0925a687ea4ebe5785d007cc5a4cc6241f` adds exactly `free(dims)` after
`PyArray_Resize`. Its commit message independently identifies approximately
0.65 KiB leaked per `neighbour_list("ijS")` call and GB/hour growth in
DataLoader workers. No tagged matscipy release newer than v1.2.0 contains the
fix.

## Selected approach

Build a provenance-marked matscipy `1.2.0+omatfix1` wheel from the official
v1.2.0 source with only the one-line upstream patch. Install it into a new,
immutable copy of the qualified CUDA-13 dependency runtime. Do not modify the
existing runtime, install MACE/project source into the runtime, change MACE's
Python neighbour wrapper, or consume unrelated unreleased matscipy changes.

The patched artifact must preserve:

- matscipy v1.2.0 Python and native source except for `free(dims)`;
- Python 3.12.11, NumPy 2.5.1, SciPy 1.18.0, and the remainder of the proven
  CUDA-13 MACE dependency stack;
- `neighbour_list("ijS")` values, shapes, dtypes, ordering, and exception
  behavior;
- a dependency-only runtime with MACE and the OMat project supplied through
  immutable `PYTHONPATH` source overlays.

The build root records the pristine-source hash, patch text/hash, upstream
commit identifier, built-wheel hash, compiler/build metadata, package freeze,
and `uv pip check` output. The runtime records its parent runtime and patched
wheel hash and is never mutated after qualification.

## Alternatives rejected

1. Install current matscipy master or the upstream fix commit directly. This
   includes unrelated changes after v1.2.0 and weakens attribution and
   reproducibility.
2. Replace matscipy with ASE or another neighbour implementation in MACE. This
   changes performance and potentially edge ordering/semantics across every
   workload for a defect already fixed by one upstream line.
3. Recycle workers during an epoch or increase host memory. This masks rather
   than removes the leak and adds lifecycle/performance complexity.
4. Call `gc.collect()` or `malloc_trim(0)`. Controlled probes proved both leave
   the retained `1.629 GiB/10k` floor unchanged.

## Test-first qualification

### RED: stock dependency

Create a standalone allocator regression using an ASE-built periodic crystal
and repeated real `neighbour_list("ijS")` calls. After allocator warm-up, fit
`mallinfo2().uordblks` versus calls and assert a strict bounded-growth limit.
Run it first against the stock matscipy 1.2.0 runtime and require it to fail for
the measured positive slope. The acceptance limit is the same
`0.02 GiB/10k calls` used by the real-OMat test. Preserve the failing result as
the RED artifact.

Also run the existing 10,000-batch real-OMat neighbour-only probe against stock
1.2.0 as the full-workload RED reference; the already verified value is
`1.630030 GiB/10k` in `uordblks`.

### GREEN: patched dependency

Apply only upstream's `free(dims)` line, build/install the wheel in the new
runtime, and rerun the exact standalone regression. It must pass with bounded
post-warm-up `uordblks`. Then rerun the same 10,000-batch real-OMat probe and
require:

- four workers complete with no error or worker replacement;
- `uordblks`, arena, and literal heap slopes each below
  `0.02 GiB/10k batches` (less than 1.25% of the stock slope);
- `fordblks` and `keepcost` remain bounded;
- identical neighbour outputs on a fixed corpus of periodic, partially
  periodic, and non-periodic structures between stock and patched runtimes,
  including sender, receiver, shifts, dtypes, and ordering;
- exact runtime/package provenance and successful `uv pip check`.

## Production-shaped qualification

After GREEN, run the existing single-GPU login-node training qualification
with the real OMat24 LMDB path, batch 256, four spawned workers, CuEquivariance,
and the selected with-self model for at least 10,000 optimizer updates. Record
each worker's RSS/PSS/private-dirty heap, trainer/cgroup memory, throughput,
loader wait, compiled graph count, and finite losses. The post-warm-up worker
heap must be bounded and throughput must not regress materially relative to
the prior CUDA-13 qualification.

Only after this gate passes may the with-self and without-self two-rank Slurm
jobs be relaunched from fresh initialization using the patched runtime. Both
jobs must record the runtime root and wheel hash before training begins.

## Failure and rollback policy

Fail closed if the patch differs from the upstream one-line change, the wheel
contains an unexpected source/version delta, output parity fails, allocator
growth exceeds the threshold, `uv pip check` fails, MACE/project becomes
installed in the runtime, or the production-shaped qualification regresses.
Do not alter or delete the current runtime or prior diagnostics. A failed
patched runtime remains an immutable failed artifact and is never used for
production.
