# Hybrid cuEquivariance + OpenEquivariance training result

## Decision

**Continue the hybrid route as a promising compiled-training prototype.** The
OpenEquivariance ownership incompatibility is resolved without serializing its
runtime state, dynamic full force/stress training stays compiled on variable
shapes, and the matched production HDF5 control improves median optimizer-step
time by 8.5% at batch 64 and 24.3% at batch 128.

## Environment

- Source: `b8447f72d4180d38b78c34e1c5d94b06dc050ac2`
- PyTorch: `2.11.0+cu128`; CUDA runtime: `12.8`
- e3nn: `0.4.4`
- cuEquivariance / cuEquivariance Torch: `0.11.1` / `0.11.1`
- OpenEquivariance: `0.6.8`
- GPU: NVIDIA GeForce RTX 5090
- Precision: IEEE float32; TF32, BF16, FP16, and autocast were not enabled
- cuEquivariance convolution fusion: disabled

## Gates

### Passed: isolated runtime and CUDA construction

`openequivariance==0.6.8` installed only in the existing isolated profiling
runtime, together with `ninja==1.13.0` and `pybind11==3.1.0`. A real CUDA
construction of `TensorProductConv` succeeded on the RTX 5090.

### Passed: eager full force/stress training

`test_hybrid_full_force_stress_optimizer_parity_cuda` passed against the
unfused-cuEquivariance control. At float32 default tolerances it matched:

- energy, forces, virials, and stress;
- the complete energy/force/virial/stress objective;
- every named trainable parameter gradient, with no missing gradients;
- one Adam+AMSGrad parameter update; and
- every optimizer-state tensor by parameter name.

The test completed in 4.17 seconds. This establishes that OpenEquivariance
0.6.8 supplies the required eager backward-backward and weight-gradient path
for this small MACE model.

### Passed: audited dynamic compiled graph

OpenEquivariance's generated kernel, workspace, and dummy permutation tensors
are now registered as non-persistent buffers. A device-safe `_apply` adapter
preserves OEQ's required CPU kernel and CUDA workspace/permutation placement
through `.cuda()` and `.to()`. They are explicit compiler inputs but absent
from `state_dict` and legacy saved models.

The focused buffer/device gate and dynamic compiled full-update gate pass. The
latter validates two atom/edge shapes, energy/force/virial/stress outputs,
double-backward parameter gradients, Adam+AMSGrad state/update parity, active
compilation, and no graph growth when both shapes are replayed.

Float32 atomic convolution is nondeterministic. Eager-to-eager repeated
gradients differ by up to about `1.5e-5`; Inductor also changes reduction
order. The compiled comparison therefore uses `rtol=1e-5, atol=2e-5`, selected
from a separate error-distribution diagnostic: only 2 of 21,016 elements
failed PyTorch's default tolerance, while none failed this bound. Outputs and
losses still pass the default tolerance.

## Matched production HDF5 timing

The control is the phase-34 position-derivative compiled cuEquivariance run:
same production model, seeds, variable-shape OMat shard, full
energy/force/stress objective, Adam+AMSGrad, clipping, EMA, four persistent
workers, 8 warmups, and 24 timed steps. TF32 and lower precision remain off.

| Batch | cuEq median | Hybrid median | Change | cuEq peak | Hybrid peak |
|---:|---:|---:|---:|---:|---:|
| 64 | 20.520 ms | 18.783 ms | **8.5% faster** | 3.377 GB | 1.813 GB |
| 128 | 30.140 ms | 22.812 ms | **24.3% faster** | 6.278 GB | 3.280 GB |

The batch-128 mean changes from 30.448 to 23.526 ms, corresponding to 4,204
versus 5,441 structures/s. Losses and batch shapes match the control. The
hybrid run retained one Dynamo graph with identical counters before and after
24 timed variable-shape steps and no fallback.

## Nsight Systems attribution

Matched batch-128 traces contain eight post-warmup steps.

| Metric (8 steps) | cuEq control | Hybrid | Change |
|---|---:|---:|---:|
| Total kernels | 7,023 | 7,215 | +192 (+2.7%) |
| GPU kernel time | 210.293 ms | 142.674 ms | **-32.2%** |
| Tensor-product family | 136 / 69.003 ms | 96 OEQ / 37.857 ms plus 80 cuEq / 1.770 ms | **-42.6% time** |
| GEMM | 2,184 / 65.368 ms | 2,184 / 65.087 ms | unchanged |
| Scatter/gather/index | 536 / 41.098 ms | 400 / 1.189 ms | **-97.1% time** |
| Other Triton | 2,136 / 28.942 ms | 2,264 / 30.511 ms | +5.4% time |

The hybrid kernel count increases slightly, so the win is not fewer launches
overall. OEQ replaces the costly cuEq channelwise tensor-product plus external
scatter path with fused convolution kernels. GEMM is now the largest family
(45.6% of GPU kernel time), followed by OEQ (26.5%) and other Triton (21.4%).
The trace still has 76.1% of launches below 10 us, consuming 10.8% of kernel
time, so fragmentation remains but is no longer dominated by graph scatter.

No Nsight Compute replay was run yet; Systems already gives the architectural
answer and the next lower-level target should be representative OEQ
forward/backward/double-backward kernels and the unchanged small GEMMs.

## Remaining risks and next gate

- DDP must use `broadcast_buffers=False`: NCCL cannot broadcast OEQ's required
  CPU kernel buffer. Parameters are still synchronized normally. A true
  multi-rank CUDA qualification is still required.
- DDP reports one gradient-stride mismatch for a `[1, 11392]` parameter. It is
  not a correctness failure, but its multi-rank communication impact should be
  measured.
- The hybrid converter remains experimental and the prototype is uncommitted.
  Canonical `.model` ownership is unchanged because runtime buffers are
  non-persistent, but legacy save/load/export should receive an explicit gate
  before productization.

## Artifacts

- `hybrid-b64/rank-0.json`
- `hybrid-b128/rank-0.json`
- `hybrid-b128-nsys/rank-0.json`
- `hybrid-b128-nsys/trace.nsys-rep` and `trace.sqlite`
