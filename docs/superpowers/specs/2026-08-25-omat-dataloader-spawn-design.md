# OMat DataLoader Spawn Design

## Goal

Prevent CUDA-initialized MACE training processes from forking OMat DataLoader workers and amplifying the parent's allocator high-water through copy-on-write.

## Design

Add one private `run_train` helper that returns the worker-related DataLoader keyword arguments. For `num_workers > 0`, it selects Python's `spawn` multiprocessing context; for `num_workers == 0`, it omits the context because PyTorch rejects a multiprocessing context without workers. Every DataLoader constructed by `mace.cli.run_train` uses the helper, covering balanced/native OMat training, ordinary training, per-head training, validation, and post-training tests.

The public CLI remains unchanged. Existing `--num_workers` values retain their meaning, datasets already support pickling (`OMatLMDBDataset.__getstate__` and `HDF5Dataset.__getstate__`), and the active production job is not modified.

## Alternatives Considered

- Keep `fork` and lower `prefetch_factor`: may reduce transient batches but preserves the unsafe multithreaded/CUDA fork and copy-on-write amplification.
- Stream validation metrics before forking: useful independently, but broader and does not protect other large parent allocator arenas.
- Use `forkserver`: avoids direct trainer forking, but has more cluster/runtime variability and less direct test coverage than `spawn`.

## Verification

Unit tests must first fail against the current behavior and then prove that workers use `spawn` while zero-worker loaders remain context-free. The focused and full non-GPU suites must pass. A production-shaped login-node comparison uses batch 256, four workers, the real OMat24 LMDB path, and identical settings except old-fork versus new-spawn source; it records cgroup anonymous/file memory, per-process RSS/PSS, throughput, loader wait, compiled graph count, and errors after warm-up. The fix is accepted only if spawn materially lowers worker/cgroup anonymous memory without errors or a material throughput regression.

