# OMat-0 Single-v2 Design

## Purpose

Define `single-v2`, a one-interaction MACE model initialized from the first
layer of OMat-0 and then trained end-to-end on OMat24. The model replaces the
experimental rank-16 direct message/hidden head with MACE's standard
element-conditioned symmetric contraction followed by the nonlinear scalar
readout used by the earlier single-layer experiments.

## Architecture

The model has one interaction and one product block:

```text
node embedding
  -> OMat-0 first radial/message interaction
  -> OMat-0 first product: 128x0e + 128x1o
  -> element-conditioned correlation-2 scalar contraction
  -> nonlinear scalar readout
  -> atomic energy + OMat-0 E0 baseline
```

The transferred OMat-0 first-layer parameters are initialized exactly from
the source checkpoint and remain trainable after initialization. This includes
the node embedding, radial basis, pair-repulsion term, first interaction, and
first product. The second OMat-0 interaction and product are not constructed.

The new scalar readout product and nonlinear readout are freshly initialized.
The rank-16 direct message/hidden module is not present, and no raw message is
sent directly to the readout. The standard readout contracts the post-product
hidden state, preserving O(3) invariance and central-element conditioning.

## Foundation transfer

The source is the medium OMat-0 checkpoint. The target must preserve its first
layer settings:

- hidden irreps `128x0e + 128x1o`;
- `r_max=6.0`;
- 10 radial basis functions;
- Agnesi distance transform;
- pair repulsion enabled;
- OMat-0 atomic energies (`E0s`);
- OMat-0 scale and shift.

Only source interaction/product index zero is transferred. The transfer path
must fail if the target changes the source radial geometry or if any copied
tensor has a shape mismatch. The source's second layer is ignored rather than
partially mapped into the one-layer target.

## Training configuration

The optimization contract matches the accepted `single-v1` recipe:

```text
energy_weight=1
forces_weight=10
stress_weight=10
optimizer=adam, amsgrad=true, beta=0.9
lr=0.01
weight_decay=1e-8
clip_grad=10
ema=true, ema_decay=0.999
scheduler=ReduceLROnPlateau, lr_factor=0.5, scheduler_patience=1
batch_size=256, valid_batch_size=64
distance_transform=Agnesi, pair_repulsion=true, r_max=6
```

The OMat-0 E0 values and scale/shift are fixed buffers. All other transferred
first-layer parameters and all new readout parameters are trainable.

## Parameter target

The expected trainable count is approximately 7,032,730:

| Component | Parameters |
| --- | ---: |
| OMat-0 first interaction/backbone | ~5,956,323 |
| OMat-0 first product (`L=0,1`) | ~1,014,976 |
| Node/radial/pair-repulsion parameters | ~11,655 |
| Correlation-2 scalar readout product | ~50,560 |
| Nonlinear scalar readout | ~2,064 |
| **Total** | **~7,032,730** |

The implementation must assert the exact count produced by the target runtime;
the estimate is not a substitute for that construction gate.

## CLI and serialization

Add an explicit `single-v2`/OMat-0-first-layer mode rather than overloading the
rank-16 direct-head flag. The mode must be mutually exclusive with
`direct_message_hidden_readout` and must require one interaction. Extracted
model configuration must round-trip the mode and preserve the standard
readout settings.

## Verification

Tests must verify:

1. one interaction and `128x0e + 128x1o` hidden output;
2. exact transfer of source node embedding, radial, interaction, and product
   tensors;
3. fresh standard scalar readout parameters;
4. every transferred parameter has `requires_grad=True` after setup;
5. E0, scale, and shift are preserved;
6. the direct rank-16 head is absent;
7. exact trainable parameter count and finite energy/force/stress outputs;
8. configuration extraction/reconstruction and compiled forward compatibility;
9. all single-v1 loss, EMA, weight-decay, scheduler, and cutoff arguments.

The production launcher must use a fresh run root and must not alter the
currently running frozen direct-head job.
