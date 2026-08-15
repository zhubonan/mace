# DDP Support for Compiler-Friendly MACE Training

> **Status: approved for implementation.** The user approved adapting the
> existing PyTorch `DistributedDataParallel` and compiler-friendly training
> paths. This design deliberately does not introduce a custom DDP class or a
> separate gradient-reduction implementation.

## Objective

Enable `--compile_training` together with `--distributed` for large-scale
energy/force/stress training while preserving the existing PyTorch DDP
reducer, gradient bucketing, communication/backward overlap, optimizer and EMA
ownership, checkpoint format, and eager distributed behavior.

The ordinary e3nn-to-cuEquivariance training conversion will explicitly keep
convolution fusion disabled because variable atom/edge shapes caused a measured
approximately 15x production regression. Inference conversion retains its
current CUDA fusion default, and the explicit `--only_cueq` construction/save
behavior remains unchanged.

## Current behavior and failure boundary

MACE currently rejects `--compile_training --distributed` at both the CLI and
public training API boundaries. The rejection is necessary in the current
implementation because the two training paths own separate forwards:

- Eager distributed training wraps the base MACE model with PyTorch DDP and
  calls that wrapper for every training forward.
- Compiler-friendly training constructs a `CompiledTrainingModel` facade and
  passes it separately as `forward_model` to the optimizer step.

If the rejection were simply removed, the compiled facade would be called
directly and `DistributedDataParallel.forward` would be bypassed. DDP would
not prepare its reducer for the iteration, so correct gradient synchronization
could not be assumed.

The current compiled facade also stores its base model in a tuple instead of
registering it as a child module. Although the facade delegates
`parameters()` and `named_parameters()`, DDP builds reducer buckets by walking
registered modules and their direct parameters. The tuple therefore prevents
DDP from reliably discovering the complete nested MACE parameter topology.

## Approved architecture

PyTorch DDP remains the only distributed-training implementation. The
compiler-friendly facade becomes a normal parameter-owning module facade and
is wrapped by the same DDP class already used for eager training.

The two paths differ only in the module placed inside DDP:

| Mode | Module invoked for the training forward |
|---|---|
| Eager, single device | Base MACE model |
| Eager, distributed | DDP wrapping the base MACE model |
| Compiled, single device | Compiled training facade |
| Compiled, distributed | DDP wrapping the compiled training facade |

The compiled FX graph continues to receive the original MACE parameters as
functional inputs. Registering the base model under the facade does not copy
those parameters; DDP, the optimizer, EMA, checkpoint handlers, and the
compiled graph all reference the same `Parameter` objects.

No custom reducer, manual gradient flattening, or MACE-specific DDP subclass
will be added.

## Compiled facade ownership and public state

`CompiledTrainingModel` will register the base model as a normal child module
instead of retaining it only through a tuple. This gives DDP a conventional
module tree for parameter and buffer discovery and lets its reducer install
the standard autograd hooks.

The wrapper remains an execution facade rather than a new checkpoint owner:

- Optimizers and EMA continue to use the original base-model parameters.
- Checkpoint save/load continues to receive the base model.
- Final model export continues to receive the base model.
- Facade state operations preserve the existing base-model-compatible public
  state dictionary rather than exposing a new serialized wrapper format.
- DDP's transient `module.` prefix is never written into canonical MACE
  checkpoints or final model artifacts.

Training and evaluation helpers will use an explicit unwrapping boundary so
checkpointing, validation, conversion, and export never depend on nested DDP
or compiler-wrapper implementation details.

## Construction and training lifecycle

The eager distributed path remains unchanged. It constructs DDP around the
base model before training.

The compiled distributed path must defer DDP construction until the compiled
facade exists, because DDP cannot safely accept parameters or child modules
added after its constructor has built reducer buckets. The lifecycle is:

1. Build or restore the base model, optimizer, scheduler, and EMA on every
   rank using the existing code.
2. Do not construct DDP around the base model when `compile_training` is
   active.
3. On the first real training batch, construct the compiler-friendly facade on
   each rank using that rank's batch and `dynamic=True`.
4. Establish rank consensus on whether compilation setup succeeded. If any
   rank selected eager fallback, disable compiled execution on every rank.
5. Wrap the facade with the existing PyTorch DDP constructor and the same
   process group, device, and local-rank policy used by eager training.
6. Invoke the DDP-wrapped facade for the first and all subsequent training
   forwards. Cache this wrapper across epochs.
7. Continue using the base model for optimizer parameter ownership,
   gradient clipping, EMA, checkpointing, validation, and final export.

The first batch is not dropped or replayed. It both initializes the facade and
performs the first optimizer step after DDP construction.

## Gradient and loss semantics

All gradient communication remains native DDP behavior. DDP installs hooks on
the registered base-model parameters, marks the iteration through its own
forward lifecycle, groups gradients into reducer buckets, and overlaps NCCL
all-reduce with backward as gradients become ready.

MACE's existing distributed loss normalization remains unchanged. Each local
loss scales its local sum by world size divided by the global element count;
DDP's gradient averaging then produces the gradient of the global mean. This
must be verified separately for energy, forces, virials, and stress because
their local element counts differ with variable structures.

Gradient clipping remains after `loss.backward()`. At that point DDP has
completed reduction, so every rank clips the same synchronized gradients
before performing the same optimizer and EMA update.

The first implementation will use normal DDP reduction with
`find_unused_parameters=False`, matching the current eager configuration.
Compiler tracing already rejects unused trainable placeholders. DDP static
graph mode, custom communication hooks, gradient accumulation with `no_sync`,
and optimizer-in-backward remain later optimizations rather than initial
requirements.

## Validation, checkpoint, and stage transitions

Validation does not need the training-only compiled facade. During compiled
distributed training, validation uses the synchronized base model directly;
the existing distributed samplers and metric/loss reductions remain
responsible for global evaluation results.

Checkpoint and resume behavior remains backend-native and wrapper-free:

- Base model, optimizer, scheduler, and EMA state retain their existing keys.
- Resume reconstructs the base model and optimizer before the compiled facade
  and DDP wrapper are recreated.
- Stage-one and stage-two/SWA transitions continue to update and serialize the
  base model.
- Final model conversion and saving remain independent of the transient
  compiled/DDP training wrapper.

End-of-training evaluation may construct the existing base-model DDP wrapper
after compiled training has completed, as the current stage evaluation path
already does. Two live DDP wrappers must never own reducer hooks for the same
parameters during an optimizer step.

## Rank-consistent compilation and fallback

Compilation is optional acceleration, but distributed ranks must follow a
compatible execution policy:

- After facade construction, ranks reduce a readiness flag. If any rank could
  not compile, every facade disables compiled execution before DDP wrapping.
- The DDP wrapper remains valid in fallback mode because the facade's eager
  forward uses the same registered base model and parameters.
- A runtime compiler failure records a persistent local fallback state. At the
  next safe optimizer-step boundary, ranks exchange that state and all disable
  compiled execution for subsequent steps.
- The transitional step may contain compiled execution on some ranks and eager
  execution on another only after a caught runtime failure. Required compiled
  versus eager output and gradient parity makes this scientifically
  equivalent within declared tolerances, while DDP still synchronizes the
  resulting gradients. The event is logged prominently with rank and failure
  details.
- Collective ordering for readiness and fallback checks is identical on every
  rank. No rank exits or skips a reducer collective independently.

The fallback-state collective is measured as part of the performance gate. If
its steady-state overhead is material, the implementation may replace the
per-step check with a DDP-compatible failure signal, but it may not weaken
rank-consistency guarantees.

## cuEquivariance convolution-fusion policy

The production dynamic-shape profile showed that convolution fusion in the
`uniform_1d` force/stress path repeatedly loads and unloads generated modules,
synchronizes CUDA events, and launches thousands of one-block segmented radix
sorts. The ordinary converted training path must therefore pass
`conv_fusion=False` explicitly.

This policy is scoped to training:

- General e3nn-to-cuEquivariance conversion retains its current device-based
  fusion default so fixed/bucketed inference behavior does not regress.
- `run_train` explicitly requests an unfused converted model.
- The explicit `--only_cueq` behavior remains unchanged as previously
  requested; it is not the recommended path for variable-shape force/stress
  training.
- No new command-line flag is introduced in this change.

The DDP compiler qualification uses variable-shape HDF5 data with convolution
fusion disabled.

## Compatibility and user-visible behavior

- Existing eager single-device and eager distributed commands remain
  unchanged.
- Existing single-device `--compile_training` remains opt-in and retains eager
  fallback behavior.
- `--compile_training --distributed` becomes supported for Adam-family
  training with energy, force, virial, and stress objectives.
- LBFGS and unsupported output objectives remain rejected.
- The `--compile_training` help text no longer claims single-device-only
  support.
- Checkpoint filenames, state keys, resume epoch behavior, logging formats,
  model export, and `--only_cueq` semantics do not change.
- Converted cuEquivariance training stops enabling convolution fusion
  implicitly; inference conversion behavior does not change.

## Test and qualification strategy

Implementation follows red-green-refactor development.

1. Facade ownership tests prove the registered module tree exposes every base
   parameter and buffer exactly once, uses the same object identities as the
   optimizer, and retains base-compatible state save/load behavior.
2. A two-process CPU/Gloo unit workflow wraps a minimal compiled facade in DDP
   and proves that all ranks finish with identical gradients and parameter
   updates matching a single-process concatenated global batch.
3. Integration tests remove both distributed rejection gates and prove the
   compiled facade, rather than the base model, is the object wrapped and
   invoked through DDP.
4. Variable-size distributed loss tests cover unequal per-rank graph, atom,
   force-component, and stress-component counts.
5. Failure tests prove one-rank setup fallback disables compilation everywhere
   and one-rank runtime fallback converges to a common persistent eager state
   without changing collective ordering.
6. Checkpoint, resume, optimizer, scheduler, EMA, SWA, validation, and final
   export tests prove wrappers remain transient and state keys remain stable.
7. Conversion tests prove training requests `conv_fusion=False`, while direct
   CUDA conversion for inference and explicit `--only_cueq` retain their
   existing behavior.
8. A real two-GPU and four-GPU NCCL smoke uses PyTorch 2.11+cu128,
   variable-shape HDF5 batches, full energy/force/stress training, and
   `conv_fusion=False`. It checks finite loss, identical post-step parameters,
   checkpoint resume, absence of recompilation/fallback storms, and bounded
   memory.
9. A production-shaped performance comparison reports complete optimizer-step
   time, compilation startup, CUDA kernels, NCCL collectives, overlap, GPU
   duty, and structures per second against eager DDP. The compiled path must
   improve sustained throughput after warm-up; launch-count reduction alone is
   insufficient.

## Acceptance criteria

- `--compile_training --distributed` passes two- and four-rank correctness
  tests without custom gradient-reduction code.
- Compiled DDP gradients and one-step parameters match eager DDP and a
  single-process global-batch reference within declared float32/float64
  tolerances.
- DDP reducer buckets include every expected trainable parameter exactly once,
  and no reducer hook is installed by a second live DDP wrapper.
- Variable-shape energy/force/stress training completes without convolution-
  fusion specialization/module-load storms.
- Optimizer, scheduler, EMA, SWA, checkpoint, resume, validation, and export
  behavior remain compatible.
- Setup or runtime compilation failure reaches a common eager fallback state
  across ranks without deadlock.
- The warmed compiled DDP run provides a measurable end-to-end throughput
  improvement over eager DDP on the same hardware, data, batch size, and
  objective.

## Non-goals

- Implementing a MACE-specific DDP class or manual gradient reducer.
- Compiling DDP/NCCL control flow itself; only the compiler-friendly MACE
  training graph is compiled inside the DDP wrapper.
- Enabling convolution fusion for arbitrary variable-shape training.
- Changing the explicit `--only_cueq` construction or final-save behavior.
- Supporting distributed LBFGS, new output objectives, FSDP, tensor/pipeline
  parallelism, mixed precision, TF32, or optimizer-in-backward.
- Guaranteeing a speedup before the required real multi-GPU performance gate.
