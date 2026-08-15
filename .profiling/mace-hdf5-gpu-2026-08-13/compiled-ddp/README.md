# Compiled DDP qualification

This directory records bounded eager-versus-compiler-friendly DDP training
controls on variable-size HDF5 structures. The launcher keeps the source HDF5
read-only, repeats its four 12–14 atom structures through a distributed
sampler, disables cuEquivariance convolution fusion, and measures complete
energy/force/stress Adam+AMSGrad steps including NCCL and fallback consensus.

Run with a PyTorch 2.11 CUDA environment containing cuEquivariance:

```bash
export MACE_DDP_PYTHON=/path/to/environment/bin/python
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh eager 2
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 2
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh eager 4
bash .profiling/mace-hdf5-gpu-2026-08-13/scripts/run_compiled_ddp.sh compiled 4
```

Override `MACE_DDP_DATASET`, `MACE_DDP_OUTPUT_DIR`, batch size, warmups, or
timed steps through the corresponding `MACE_DDP_*` environment variables.
Each rank writes JSON containing complete-step times, loss, peak memory,
compiler/fallback state, and a parameter checksum. A valid run has identical
checksums on all ranks and no compiled fallback.

## Local availability

The local host exposes one RTX 5090 (32 GB) through PyTorch 2.10.0+cu128 and
cuEquivariance 0.9.1. Consequently, two- and four-rank NCCL qualification
cannot run locally; the launcher fails before `torchrun` when fewer GPUs than
ranks are visible. Production qualification must use the PyTorch 2.11+cu128
environment on the multi-GPU node.

A bounded one-rank functionality smoke completed on 2026-08-15 with batch 4,
zero warmups, and one recorded step. Eager and compiled losses both equaled
`347.3849792480469`; compiled execution remained active without fallback.
The eager cold step took 2771 ms and the compiled cold step took 23659 ms,
including compilation, so these numbers are not a warmed performance
comparison. Requesting two local ranks reported `visible_gpus=1` and exited
with code 3 before launching workers, as designed.
