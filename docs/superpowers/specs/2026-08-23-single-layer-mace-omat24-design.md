# Single-Interaction MACE for Full OMat24 Training

## Status

Approved design. This document defines the model architecture, framework integration, qualification sequence, and full-run comparison protocol. It does not implement or launch training.

## Objective

Train a MACE model from scratch on full OMat24 with exactly one neighbor-message-passing interaction, 128 channels, hidden angular momentum through `L=2`, a residual self-connection in that interaction, local contraction of the retained equivariant features, and a nonlinear scalar energy readout. Qualify the implementation locally on one GPU, then run it on four RTX 5090 GPUs on Cata and compare it with the established MACE-OMAT-0 retrain.

## Feasibility Verdict

The architecture is feasible, but it is not expressible by the current stock one-interaction CLI path. `MACE.__init__` reduces the only product output to scalars and does not create the nonlinear last readout when `num_interactions=1` and `use_last_readout_only=true`. The CLI model factory also replaces a requested density-residual first interaction with its non-residual form.

The required operations already exist in MACE. `RealAgnosticDensityResidualInteractionBlock` supplies density-normalized neighbor messages and an element-conditioned self-connection. `EquivariantProductBasisBlock` supplies the CuEq-aware symmetric contraction needed for a local invariant readout. `NonLinearReadoutBlock` supplies the scalar MLP. The implementation should compose those existing operations rather than introduce new contraction mathematics.

Software feasibility is therefore high after a focused model-construction and serialization change. Scientific quality remains uncertain because the receptive field is one cutoff radius rather than the two-hop receptive field of the comparison model. The full run and matched validation comparison will measure that limitation.

## Model Architecture

### Fixed configuration

- One neighbor interaction: `num_interactions=1`.
- Sole interaction class: `RealAgnosticDensityResidualInteractionBlock`.
- Hidden representation: `128x0e + 128x1o + 128x2e`.
- Interaction product correlation: 3, matching the OMat retrain convention.
- Readout-only contraction correlation: 2.
- Readout contraction output: `128x0e`.
- Nonlinear readout hidden representation: `16x0e` with the existing gate and per-head scalar output.
- Spherical-harmonic edge expansion: `max_ell=3`, retained from the MACE-OMAT-0 retrain; this is distinct from hidden `max_L=2`.
- Cutoff: 6 Å.

### Data flow

The element embedding initially provides 128 even scalar features for each atom. The sole density-residual interaction computes a neighbor message and an explicit central-atom self-path. The self-path is an element-conditioned equivariant tensor product of the initial node features and the central atom attributes. Because both inputs are scalar, that path contributes only to `L=0`; `L=1,2` remain geometry-derived.

The first product-basis block combines the density-normalized neighbor message with the self-path and retains the complete `128x0e + 128x1o + 128x2e` representation. A second product-basis operation is attached only to the readout. It performs no neighbor lookup, edge convolution, or message aggregation. Its correlation-2 symmetric contraction consumes the retained representation and forms 128 even scalar invariants, including the `1o x 1o -> 0e` and `2e x 2e -> 0e` channels. The existing nonlinear readout maps those invariants through `16x0e` to per-atom energy, and graph summation produces structural energy.

The readout product is not a second interaction layer. The model has one neighbor aggregation and a receptive field of one 6 Å cutoff.

### Output behavior

The full equivariant first-layer features remain available in model diagnostics and `node_feats` output. The contracted scalar copy is used only for the energy readout. Forces and stress continue to be obtained by differentiating total energy.

## Framework Integration

### Model construction

The one-interaction behavior is selected through the already-approved `use_last_readout_only=true` semantics rather than a second overlapping architecture switch. For this case only, model construction retains the full hidden irreps after the sole interaction, constructs one readout-local scalar product with correlation 2, and constructs the existing nonlinear readout. Multi-interaction models and one-interaction models that do not select last-readout-only behavior remain unchanged.

Readout correlation must be represented explicitly as `readout_correlation` in the constructor, CLI, resolved configuration, and extraction path so saved models are reconstructible. Its default and supported production value is 2; it is not a second architecture-selection flag. The model factory must allow `RealAgnosticDensityResidualInteractionBlock` as `interaction_first` for `model=MACE` and must verify rather than silently rewrite the requested class.

The forward path must pass node attributes to the readout contraction without replacing the retained equivariant feature tensor. The scalar readout features and diagnostic node features are separate values.

### Backend and export compatibility

The readout contraction reuses `EquivariantProductBasisBlock` so e3nn and CuEq follow their established symmetric-contraction and layout paths. Conversion between e3nn and CuEq, compiled training, configuration extraction, checkpoint reload, and legacy single-file model export must include the new readout product. Existing model files without that component must continue to load with their original behavior.

Implementation work is based on the local optimized `training-opt` lineage. `/home/bonan/work/train-downsample` remains the read-only protocol and results reference; its native-LMDB, balanced batching, compiled DDP, monitoring, checkpoint, and export behavior should be adapted rather than independently recreated.

## Verification Design

### Unit and model tests

Tests must establish all of the following:

- The resolved model has exactly one neighbor interaction and that interaction is `RealAgnosticDensityResidualInteractionBlock`.
- The first product emits `128x0e + 128x1o + 128x2e` and has an active self-connection.
- The readout-only product has correlation 2, targets `128x0e`, and does not aggregate neighbors.
- Perturbing valid `L=1` or `L=2` inputs changes the scalar contraction output, proving those channels are consumed.
- The self-path is nonzero for suitable scalar embeddings and affects the retained first-layer scalar features.
- Energy is rotation invariant, forces are rotation covariant, and energy/force/stress gradients are finite.
- e3nn and CuEq outputs and gradients agree within the repository's backend tolerances.
- Eager and compiled training agree, use the expected compiled graph count, and do not recompile after warm-up across qualified dynamic batch shapes.
- Configuration extraction, checkpoint reload, CuEq conversion, and legacy export reproduce the same architecture and predictions.
- Existing two-interaction and ordinary one-interaction model tests remain unchanged and passing.

### Full-scale preflight gates

Preflight uses real native OMat LMDB records and the production graph path. It validates data keys, units, element coverage, normalization identity, finite labels, parameter keys and shapes, loss computation, backward gradients, optimizer update, checkpoint restart, and exported-model inference. These checks are engineering qualification and do not use short-run accuracy to accept or reject the architecture.

## Training and Launch Sequence

### Local single-GPU qualification

Use one local RTX 5090 outside the sandbox with cuequivariance installed. Run an end-to-end worker sweep with `num_workers=1`, `4`, and `8`, using the same production-shaped batch sequence and bounded shard-window settings. Each candidate receives 20 warm-up updates followed by 100 measured updates from the same initial model state and deterministic sampler position. Compare median and p95 data wait, end-to-end step time, structures/atoms/edges per second, GPU utilization, peak device memory, loader stalls, open-file behavior, and compile stability.

Select the worker count with the highest stable end-to-end throughput, not merely the highest loader throughput. Then run a disposable 500-update session, checkpoint and stop at update 250, resume to update 500, and perform validation. Verify finite losses and gradients, short-term loss descent, stable memory, a single qualified compiled graph after warm-up, and successful model export. Local weights are never promoted to the production run.

### Cata qualification

Recheck loader workers and batch size under Cata's shared filesystem, CPU allocation, and four-rank DDP conditions because the local worker result is only a starting point. Use production-shaped batches, per-rank cache isolation, atom/edge-balanced batching, and the established compiled-DDP gate. Fix production workers, per-rank batch size, and scheduler resources from those measurements rather than from unverified assumptions.

### Full OMat24 run

Start the Cata production run from a fresh random initialization with the recorded seed; do not resume the local checkpoint. Use the official native OMat LMDB corpus and the same train/validation/test identities as the MACE-OMAT-0 retrain. Preserve its 89-element E0 values, `avg_num_neighbors=61.964672446250916`, `mean=0`, `std=0.7735790334431056`, `r_max=6`, loss weights, optimizer, scheduler, and evaluation definitions unless a measured stability issue requires an explicitly recorded deviation.

Use the retrain's restartable checkpoint protocol and maximum ten-epoch envelope. Report comparisons at matched optimizer updates and completed epoch fractions so differing throughput does not distort the scientific comparison. This is a full from-scratch run; no subset accuracy pilot or architecture ablation is required.

## Monitoring and Comparison

Record train and validation total loss, energy-per-atom error, force error, stress error, each weighted loss component, learning rate, gradient norm, and EMA state. Operational telemetry includes structures/atoms/edges per second, data wait, step-time percentiles, GPU utilization and memory, rank atom/edge imbalance, loader stalls, open shard count, compile graph count, recompilations, checkpoint duration, and checkpoint hash lineage.

Compare the one-interaction model with MACE-OMAT-0 retrain results on identical validation and test records. Report both learning curves and final/best metrics together with the architectural differences: one versus two neighbor interactions, hidden `L=2` versus the retrain's hidden `L=1`, and the additional local correlation-2 invariant readout. There is no predeclared requirement to match MACE-OMAT-0 accuracy; the comparison quantifies the cost of removing the second message-passing hop.

The run stops or fails closed on nonfinite loss/gradients, DDP rank divergence, repeated post-warm-up recompilation, unrecoverable loader failure, invalid checkpoint lineage, missing metrics, or model-export failure. Ordinary underperformance is reported rather than treated as an infrastructure failure.

## Risks and Mitigations

### One-hop underfitting

The nonlinear invariant readout cannot reproduce a second neighbor-message hop. Mitigation is controlled full-run comparison at matched updates and identical evaluation records, with no claim that additional readout capacity restores the missing receptive field.

### Memory and contraction cost

Retaining 128 channels through `L=2` and differentiating a correlation-2 readout contraction may offset part of the saving from removing the second interaction. Mitigation is local memory/throughput measurement followed by Cata batch qualification; production batch size is not fixed before those measurements.

### I/O starvation

Online graph construction from native LMDB can starve a 5090. Mitigation is the approved `1, 4, 8` worker sweep, bounded shard windows, pinned memory, balanced batches, and separate measurement of data wait and compute time.

### Backend or serialization gaps

The new readout product is outside the current `products` sequence and could be missed by conversion or config extraction. Mitigation is to make it a first-class serialized component and gate e3nn/CuEq conversion, compiled training, reload, and legacy export before any production launch.

### Accidental architecture substitution

The current `model=MACE` factory silently replaces unsupported first-interaction selections. Mitigation is to permit the density-residual selection, record the resolved class in provenance, and fail preflight if it differs from the requested class.

## Non-Goals

- Adding a second neighbor interaction or self-loop edge.
- Running a subset-scale scientific pilot or broad architecture ablation.
- Changing OMat24 labels, splits, E0 values, or normalization.
- Fixing production worker count, batch size, or scheduler resources before local and Cata measurements.
- Modifying the reference workspace's embedded MACE checkout.

## Approval Summary

The user approved the retained `L=0,1,2` first-layer representation, local correlation-2 invariant contraction, nonlinear scalar readout, first-layer self-connection, density-residual sole interaction, local single-GPU qualification, `num_workers=1,4,8` sweep, and subsequent four-GPU Cata full run.
