# Dedicated CUDA 13 MACE Runtime Design

## Objective

Create a dependency-only, immutable Cata runtime for MACE training and use it
to replace the superseded one-layer OMat24 jobs after diagnosing their low
throughput. The runtime must not contain a MACE or project installation; each
job supplies an immutable MACE source checkout and project source through
`PYTHONPATH`.

## Runtime contract

- Python 3.12.11.
- PyTorch 2.11.0+cu130 from the official PyTorch CUDA-13 index.
- e3nn 0.4.4, matching MACE's `setup.cfg` and the retraining lock.
- cuequivariance 0.11.0 and cuequivariance-torch 0.11.0.
- cuequivariance-ops-torch-cu13 0.10.0.
- Remaining MACE runtime dependencies from `setup.cfg`, installed without
  installing `mace-torch` itself.
- No CUDA-12 CuEquivariance or NVIDIA runtime packages.
- A unique versioned root under `/scratch/bonan/mace-runtimes/`; never mutate a
  qualified runtime in place.

## Performance investigation

Before building or relaunching, preserve the current jobs' resolved arguments,
per-rank diagnostics, GPU telemetry, Slurm allocation, optimizer progress, and
process placement. Compare the measured global throughput with the per-rank
batch and step period. The reported 46-49 ms period at global batch 32 implies
about 660-686 structures/s, so the immediate issue is not loader wait or
recompilation; determine whether batch 16/rank under-fills the GPUs and whether
two independent jobs are correctly isolated across devices.

The superseded e3nn-0.6 jobs must not provide checkpoints for the replacement
runs. Preserve their diagnostic evidence, cancel them after the audit, and use
fresh run roots.

## Qualification ladder

1. Verify exact package versions, import paths, `uv pip check`, and absence of
   MACE/project packages.
2. On one RTX 5090, verify CUDA matrix multiplication, e3nn model construction,
   e3nn-to-CuEquivariance conversion, force/stress backward, CuEq-to-e3nn
   conversion, and model serialization.
3. Benchmark batch 16, 32, and 64 per rank on one GPU with workers 4 using the
   real native OMat LMDB path. Select by structures/s, stable one-graph compile,
   memory, and finite loss.
4. Run paired two-rank, 20-update compiled-DDP gates at the selected common
   batch, requiring one graph, finite matched updates, isolated devices,
   successful checkpoint/export, and exact runtime provenance.
5. Launch the with-self and without-self full runs from fresh initialization as
   restartable one-epoch Slurm stages, each beginning with two GPUs.

## Production and monitoring

Each production stage uses an isolated run root and compiler cache. Completion
requires an epoch checkpoint, model export, validation metrics, runtime
manifest, and terminal exit code 0. Later stages restart only from the previous
stage's verified checkpoint. Compare the two variants at matched optimizer
steps and completed epochs against the official MACE-OMat-0 optimizer-validation
metrics.

## Failure policy

Fail closed on any version/import mismatch, CUDA-12 package, MACE installed in
the runtime, more than one compiled graph, non-finite metric, rank/device
collision, missing artifact, or checkpoint-lineage mismatch. Do not launch full
training until the one-GPU batch benchmark and both two-rank gates pass.
