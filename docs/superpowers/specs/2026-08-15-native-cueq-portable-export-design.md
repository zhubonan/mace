# Native cuEquivariance Training and Portable e3nn Export

## Objective

Decouple MACE's training backend from its public saved-model format.
`--enable_cueq True` will select native cuEquivariance construction for the
training model, while every final model artifact will be a canonical e3nn model
stored on CPU. Backend-native checkpoints remain unchanged so interrupted
training can resume without conversion.

This replaces the current coupling in which `--only_cueq` simultaneously
selects the faster native construction route and suppresses conversion back to
e3nn during final saving.

## Current behavior

The current CLI exposes two related flags:

| Flags | Training construction | Final model |
|---|---|---|
| `enable_cueq=False` | e3nn | e3nn; device follows `save_cpu` |
| `enable_cueq=True`, `only_cueq=False` | e3nn, then cuEq conversion | converted back to e3nn; device follows `save_cpu` |
| `enable_cueq=True`, `only_cueq=True` | native cuEq | native cuEq; device follows `save_cpu` |

This makes a training-performance choice also determine artifact portability.
It also leaves the nominally converted training route slower than native
construction. The batch-128 profile measured 35.210 ms/step for converted
compiled training and 21.211 ms/step for native compiled training.

## Approved behavior

### Training backend

- `--enable_cueq False` constructs and trains the ordinary e3nn model.
- `--enable_cueq True` constructs the model natively with
  `CuEquivarianceConfig`; it does not build an intermediate e3nn model and does
  not call the e3nn-to-cuEq conversion routine.
- A requested cuEquivariance backend that is unavailable fails before training
  with an actionable dependency error. It must not silently train an e3nn
  model while claiming cuEquivariance acceleration.
- `--only_cueq` remains accepted during a deprecation window but no longer
  selects a separate training or saving path.
  Its parser default becomes `None`, used only as an internal sentinel to
  distinguish omission from an explicitly supplied legacy boolean.
  - An explicitly supplied value emits a deprecation warning.
  - `--only_cueq True` without `--enable_cueq True` acts as a deprecated alias
    that enables cuEquivariance, preserving the useful historical invocation.
  - `--enable_cueq True --only_cueq False` warns that the converted training
    path has been removed and proceeds with native construction.

### Checkpoints

- Optimizer/resume checkpoints keep the backend-native training state.
- No cuEq-to-e3nn conversion occurs in the checkpoint path.
- Resuming a cuEq checkpoint reconstructs the native cuEq model before loading
  its state.
- Checkpoint naming and retention behavior do not change.

### Final model artifacts

All user-facing final model files, including stage-one and stage-two/SWA
variants in both checkpoint and model output directories, follow one export
pipeline:

1. Deep-copy the trained model so export cannot mutate the live evaluation or
   checkpoint owner.
2. Merge LoRA weights when LoRA is active.
3. Normalize the backend: convert cuEquivariance or OpenEquivariance modules to
   e3nn.
4. Move the normalized model to CPU unconditionally.
5. Save the ordinary `.model` artifact.
6. Generate the optional TorchScript/compiled artifact from the same normalized
   CPU e3nn model.
7. Derive embedded configuration metadata from the normalized export model,
   not from the backend-native training model.

`--save_cpu` remains accepted temporarily for command-line compatibility, but
final artifacts are CPU-resident regardless of its value. Supplying the flag
does not alter behavior and should emit a deprecation warning.

Backend conversion or CPU normalization failures are fatal for final export.
MACE must not silently write a native backend model under the canonical final
model filename. Failure to generate the additional TorchScript artifact is
reported as a warning while preserving the successfully written ordinary e3nn
`.model` file.

## Code boundaries

The implementation should keep policy decisions out of the large final-save
block in `mace/cli/run_train.py`:

- CLI normalization resolves deprecated `only_cueq` input into the single
  `enable_cueq` backend decision before model construction, then replaces the
  sentinel with the normalized boolean state used by downstream code.
- Model construction in `mace/tools/model_script_utils.py` receives a
  `CuEquivarianceConfig` whenever normalized `enable_cueq` is true.
- A focused export helper deep-copies, merges LoRA when requested, normalizes
  cuEq/OEQ to e3nn, and moves the result to CPU.
- The existing final-save loop calls that helper once per stage and uses the
  returned model for ordinary serialization, metadata, and TorchScript export.
- Checkpoint-handler code is not routed through the export helper.

The helper must determine the actual model backend rather than trusting only a
CLI flag. This prevents attempting cuEq conversion after an e3nn-only run and
ensures a backend-native model cannot accidentally bypass normalization.

## Compatibility and migration

- Existing e3nn-only invocations retain their training behavior; their final
  models become CPU-resident even when `--save_cpu` was omitted.
- Existing `--enable_cueq True --only_cueq True` invocations retain the fast
  native training route but now produce portable final artifacts.
- Existing `--enable_cueq True` invocations change from converted to native
  construction and gain the measured native-path performance behavior.
- Existing `--only_cueq True` invocations remain functional during the
  deprecation window through alias normalization.
- Users requiring exact training continuation use the native checkpoint, not
  the converted final model.
- Published final-model filenames remain unchanged.

The deprecated flags are not removed in this change. Removal requires a later
release boundary and release-note notice.

## Correctness requirements

- A final model loaded with `torch.load(..., map_location="cpu")` contains no
  cuEquivariance or OpenEquivariance operation modules.
- Every parameter and buffer in the final model resides on CPU.
- Converted energy, forces, and stress agree with the backend-native model
  within the project's declared floating-point tolerances.
- Native checkpoint save and resume preserve optimizer ownership and parameter
  layout.
- Export does not modify the live training/evaluation model.
- e3nn-only, cuEq, LoRA, and SWA paths all use the same final-export contract.

## Test strategy

Implementation follows red-green-refactor development.

1. Parser and normalization tests establish that `enable_cueq` is the single
   backend selector and that deprecated `only_cueq` combinations warn and
   normalize as specified.
2. Construction tests prove `enable_cueq=True` directly produces native fused
   cuEq interaction modules and does not invoke post-construction conversion.
3. Export-helper tests use real small models to prove backend normalization,
   CPU residency, non-mutation of the source model, and metadata extraction
   from the export model.
4. A cuEq round-trip test compares energy, forces, and stress before and after
   export conversion.
5. Workflow tests assert that ordinary and SWA final paths are e3nn/CPU while
   native checkpoints remain usable for resume.
6. Error-path tests prove backend-conversion failures stop canonical export and
   TorchScript-only failures produce visible warnings without deleting the
   ordinary model.
7. Existing compiler-friendly CUDA parity tests remain required because native
   cuEq construction is the intended accelerated training route.

## Non-goals

- Changing checkpoint format, retention, or restart epoch semantics.
- Making the compiler-friendly training facade support DDP or LBFGS.
- Removing deprecated CLI flags in the same release.
- Saving an additional native cuEq final-model artifact; the native checkpoint
  already serves exact continuation.
- Changing numerical tolerances or enabling TF32/mixed precision.
