# Single-Interaction MACE OMat24 Implementation

## Objective

Implement, qualify, and prepare paired full-OMat24 trainings for a one-interaction MACE with `128x0e + 128x1o + 128x2e`, a local correlation-2 invariant contraction, nonlinear scalar readout, and matched density interactions with and without a direct first-layer self-connection.

## Assumptions

- Implementation is based on local branch `training-opt` at or after `31e0f4b`.
- `use_last_readout_only=true` selects the new one-interaction invariant nonlinear readout without changing multi-interaction behavior.
- Interaction product correlation remains 3; `readout_correlation` is 2.
- The two variants use `RealAgnosticDensityResidualInteractionBlock` and `RealAgnosticDensityInteractionBlock` respectively.
- Local worker candidates are exactly 1, 4, and 8.
- Each initial Cata job uses two GPUs; scaling beyond two GPUs is deferred.
- GPU commands run outside the sandbox and use cuequivariance.

## Implementation Plan

- [ ] Establish an implementation branch on the latest optimized lineage, preserving the approved specification commits and the native OMat LMDB, balanced-batching, diagnostics, compiled-DDP, checkpoint, and export changes from `training-opt`.
- [ ] Add failing one-interaction model tests in `tests/unit/test_single_interaction_readout.py` covering retained `L=0,1,2` features, correlation-2 scalar contraction, nonlinear learned energy, both density interaction variants, direct-self-path presence/absence, rotational invariance, force covariance, finite gradients, and unchanged ordinary one- and two-interaction behavior.
- [ ] Implement the minimal invariant readout path in `mace/modules/models.py:48` and, if a focused reshape wrapper is required, `mace/modules/blocks.py:439`, preserving full node features while passing a reshaped copy through an element-conditioned `EquivariantProductBasisBlock` targeting `128x0e` before `NonLinearReadoutBlock`.
- [ ] Add failing CLI and configuration tests in `tests/unit/test_arg_parser.py`, `tests/unit/test_models.py`, and the relevant model-factory tests, then expose `--readout_correlation=2` in `mace/tools/arg_parser.py:242`, forward it through `mace/tools/model_script_utils.py:187`, permit `RealAgnosticDensityResidualInteractionBlock` for `model=MACE`, and serialize it through `mace/tools/scripts_utils.py:275`.
- [ ] Add failing conversion tests in `tests/test_convert_e3nn_cueq.py` and the reverse-conversion coverage, then extend `mace/cli/convert_e3nn_cueq.py:87` and `mace/cli/convert_cueq_e3nn.py:87` to transfer the readout-only symmetric-contraction weights as well as the interaction products.
- [ ] Run focused CPU verification for construction, forward/backward, rotation behavior, config round-trip, TorchScript, compiled-training facade compatibility, e3nn/CuEq conversion where available, and the existing model/CLI regression suites.
- [ ] Adapt the optimized OMat qualification protocol from `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/` to render paired one-interaction configurations, exact resolved-class assertions, local worker candidates 1/4/8, 250-update checkpoint restart, 500-update completion, and two-rank Cata launches without modifying the reference workspace's embedded MACE checkout.
- [ ] Run the local single-GPU worker sweep on the with-self variant with 20 warm-up and 100 measured updates per worker candidate; choose the highest stable end-to-end throughput and record data wait, step-time percentiles, structures/atoms/edges per second, GPU utilization, memory, loader stalls, open-shard behavior, and compile graph count.
- [ ] Run matched disposable 500-update single-GPU sessions for both interaction variants, restart each at update 250, and gate finite losses/gradients, short-term loss descent, stable memory, checkpoint lineage, no post-warm-up recompilation, validation, and legacy export.
- [ ] Qualify two-rank compiled DDP on Cata for both variants with a common measured worker count and per-rank batch size, then render two fresh-initialization full OMat24 jobs that may run concurrently on the four available GPUs.
- [ ] Monitor and compare both Cata runs at matched optimizer updates and epoch fractions against each other and the MACE-OMAT-0 retrain, reporting energy/force/stress metrics, learning rate, gradient norm, throughput, data wait, utilization, memory, rank imbalance, compile health, and checkpoint hashes.

## Verification Criteria

- A constructed special-case model contains exactly one neighbor interaction, emits `128x0e + 128x1o + 128x2e`, contains one correlation-2 readout contraction targeting `128x0e`, and contains one nonlinear scalar readout.
- The residual variant adds a nonzero direct central-input self-path; the non-residual variant has no such path, while both use density-normalized messages.
- Synthetic nonzero `L=1` and `L=2` inputs affect the invariant readout output, and rotated inputs preserve energy while rotating forces covariantly within existing tolerances.
- `readout_correlation=2` survives CLI resolution, configuration extraction, checkpoint reload, e3nn/CuEq conversion, and legacy export.
- Existing multi-interaction and one-interaction configurations without `use_last_readout_only=true` retain their previous module structure and predictions.
- Focused and regression tests exit successfully with no unexpected warnings; compiled training retains one qualified graph after warm-up.
- Local worker sweep produces a recorded comparison for 1, 4, and 8 workers, and both 500-update local sessions pass restart/export/finite-metric gates.
- Each initial Cata configuration resolves to two DDP ranks and its requested interaction class, starts from fresh initialization, and uses the same data, normalization, seed, sampler order, optimizer, scheduler, and evaluation protocol.

## Potential Risks and Mitigations

1. **The retained product output is flat while symmetric contraction expects channel-by-irrep layout.**
   Mitigation: reuse the existing `reshape_irreps` backend-aware layout conversion before the readout product and test both e3nn and CuEq layouts.

2. **Conversion tools currently remap only `model.products`.**
   Mitigation: generalize the symmetric-contraction transfer helper to accept a state-dict prefix and add bidirectional readout-product parity tests.

3. **Older pickled models lack the new readout-product attributes.**
   Mitigation: keep forward/config extraction compatible through guarded access and verify loading an existing model fixture.

4. **The new L=2 state and contraction may exceed expected memory or compile repeatedly.**
   Mitigation: qualify dynamic shapes locally, measure memory before selecting batch size, and fail closed on post-warm-up recompilation.

5. **Online LMDB graph construction can starve the GPU.**
   Mitigation: run the approved 1/4/8 worker sweep, retain bounded shard windows and balanced sampling, and select on end-to-end rather than loader-only throughput.

6. **Paired training doubles production allocation.**
   Mitigation: begin each job with two GPUs, allow concurrent use of the four available GPUs, preserve restartability, and avoid unrelated ablations.

## Alternative Approaches

1. Add explicit hand-written norms for `L=1,2`: cheaper but less expressive, element conditioning and backend conversion would be new custom paths, and it would not reuse MACE's established contraction machinery.
2. Keep only scalar output from the first product: stock-like and cheaper, but it does not retain and consume the standard first-layer `L=1,2` hidden representation requested for this experiment.
3. Add a second correlation-3 readout contraction: more expressive but more expensive than the approved minimum correlation 2 and confounds the intended shallow comparison.
