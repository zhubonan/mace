# Edge-vector derivative training profile

Date: 2026-08-17

## Decision

**Reject the edge-vector derivative leaf as a training optimization.** It is a
useful private diagnostic, but it does not meet the pre-agreed 5% speedup rule:
batch 64 regressed by 2.5% and batch 128 improved by only 0.7%. The prototype
and profiling-harness changes remain uncommitted.

## Environment and workload

- Source baseline: `b8447f72d4180d38b78c34e1c5d94b06dc050ac2`
- PyTorch `2.11.0+cu128`, CUDA runtime 12.8, e3nn 0.4.4,
  cuEquivariance 0.11.1, RTX 5090
- IEEE float32; TF32, BF16, FP16, and autocast disabled
- cuEquivariance convolution fusion disabled
- Production 9,063,204-parameter MACE model
- Variable-shape OMat HDF5, full energy/force/stress objective,
  Adam+AMSGrad, clipping, EMA, four persistent data workers

## Correctness gates

Analytical float64 tests passed for duplicated directed edges, periodic shifts,
triclinic cells, force/virial signs, stress normalization, multiple graph
volumes, and translational force balance. Trace-time global patch restoration
also passed, including exception cleanup.

At float32, the uncompiled edge FX route matched eager MACE outputs and every
named parameter gradient. After Inductor, the edge and established position
routes matched energy, forces, virials, stress, and all named gradients at
PyTorch default tolerances. Both modes retained one dynamic Dynamo graph on
the production HDF5 workload, with no fallback.

The earlier apparent `node_embedding.linear.weight` discrepancy (about
`1.5e-5`) was not caused by edge assembly: the established position-derivative
compiled route has the same eager-versus-Inductor reduction-order difference.
Direct compiled-edge versus compiled-position parity passes.

## Matched steady-state timing

Each control used identical seeds and batch shapes, 8 warmups, and 24 timed
steps. The optimizer-core timer excludes `DataLoader.__next__`.

| Batch | Position median | Edge median | Position mean | Edge mean | Result |
|---:|---:|---:|---:|---:|---:|
| 64 | 20.520 ms | 21.038 ms | 21.249 ms | 21.772 ms | **2.5% slower** |
| 128 | 30.140 ms | 29.928 ms | 30.448 ms | 30.206 ms | **0.7% faster** |

Peak allocation was unchanged: 3.377 versus 3.375 GB at batch 64, and 6.278
versus 6.277 GB at batch 128. Maximum matched loss difference was
`1.526e-5` (maximum relative difference `1.25e-7`).

## Nsight Systems attribution

The matched batch-128 traces contain exactly eight post-warmup steps.

| Metric (8 steps) | Positions | Edges | Change |
|---|---:|---:|---:|
| Total kernels | 7,023 | 6,847 | -176 (-2.5%) |
| GPU kernel time | 210.293 ms | 210.525 ms | +0.1% |
| GEMM launches / time | 2,184 / 65.368 ms | 2,104 / 64.738 ms | -80 / -1.0% |
| cuEq launches / time | 136 / 69.003 ms | 144 / 70.017 ms | +8 / +1.5% |
| Scatter/gather/index | 536 / 41.098 ms | 472 / 40.725 ms | -64 / -0.9% |
| Other Triton | 2,136 / 28.942 ms | 2,096 / 29.147 ms | -40 / +0.7% |
| Other native | 2,031 / 5.883 ms | 2,031 / 5.899 ms | unchanged |

The edge formulation removes 22 launches per step, mainly derivative
gather/scatter and GEMM bookkeeping, but their aggregate time is only about
0.17 ms per step. cuEquivariance variation offsets that saving, so total GPU
kernel time is unchanged. No new edge-assembly kernel exceeds 5% of GPU time;
an additional Nsight Compute replay is therefore not justified.

## Hybrid cuEquivariance + OpenEquivariance status

The separate hybrid candidate passed eager full-objective double backward and
Adam/AMSGrad update parity. It was stopped before timing because
OpenEquivariance 0.6.8 stores generated kernel bytes and workspace/permutation
tensors as ordinary attributes rather than registered buffers. MACE's audited
compiled ownership gate correctly rejected the first unaudited CPU
`uint8[28171]` kernel constant. Profiling would have measured eager fallback,
not the candidate.

## Conclusion

The experiment confirms that position/strain derivative bookkeeping is not a
material remaining bottleneck. The dominant batch-128 GPU time remains cuEq
(about 8.7 ms/step), GEMM (about 8.1 ms/step), and compound
scatter/gather/index kernels (about 5.1 ms/step). Further work should return to
those dominant families or the already-measured input-pipeline direction,
while precision lowering remains deferred as requested.

## Artifacts

- `positions-b64/rank-0.json`, `edges-b64/rank-0.json`
- `positions-b128/rank-0.json`, `edges-b128/rank-0.json`
- `positions-b128-nsys/trace.nsys-rep` and `trace.sqlite`
- `edges-b128-nsys/trace.nsys-rep` and `trace.sqlite`
- Hybrid result: `../../hybrid/REPORT.md`
