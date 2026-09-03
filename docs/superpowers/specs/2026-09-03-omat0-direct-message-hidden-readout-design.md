# OMat-0 Direct Message-Hidden Readout Design

## Goal

Add an opt-in, one-interaction `ScaleShiftMACE` architecture that reuses the
released MACE-OMat-0 first-layer weights and predicts atomic energy from a
direct, element-conditioned low-rank contraction of the first interaction
message and its first product hidden state.  It replaces the current second
`EquivariantProductBasisBlock` readout contraction.

## Scope and compatibility

- The mode is valid only when `num_interactions == 1`.
- It is incompatible with `use_last_readout_only=True`; that existing flag
  retains its present legacy one-interaction invariant readout semantics.
- Exact OMat-0 reuse fixes the copied backbone at 128 channels and hidden
  `128x0e + 128x1o`.  It copies node/radial embedding, interaction 0, product
  0, atomic baseline, and scale/shift when the existing foundation loader
  permits them.
- The new direct head and its nonlinear readout are freshly initialized. They
  are never copied from a two-interaction foundation checkpoint.
- This is central-element conditioned, not a new edgewise species-pair
  operator. Pair chemistry remains encoded by the copied interaction.

## Public configuration

Add two flags and matching `ScaleShiftMACE` constructor fields:

```text
--direct_message_hidden_readout {True,False}  # default False
--direct_readout_rank POSITIVE_INT             # default 16
```

When enabled, model construction must reject `num_interactions != 1` and
`use_last_readout_only=True` with a clear `ValueError`.  Both fields must
round-trip through `extract_config_mace_model` and model reconstruction.

## Direct head

Create `DirectMessageHiddenReadoutBlock` in `mace/modules/blocks.py`.  It
accepts the reshaped first-interaction message, the flattened post-product
hidden features, and the central one-hot `node_attrs`.

For OMat-0-compatible irreps it uses only the common `0e` and `1o` portions:

\[
q^{(l)}_{ir} = \frac{1}{\sqrt{2l+1}}\sum_\mu
  (A^{(l)}m_i^{(l)})_{r\mu}(B^{(l)}h_i^{(l)})_{r\mu},\qquad l\in\{0,1\}.
\]

`A` and `B` map 128 channels to rank `r`.  Per-element parameters select an
affine transform of the resulting `2r` invariants:

\[
q_i' = \gamma_{Z_i}\odot q_i + \beta_{Z_i}.
\]

The existing `NonLinearReadoutBlock` then applies its configured scalar
activation and maps `2r x 0e` to one energy per model head.  At rank 16 and
89 elements the direct block plus element affine table and a `32 -> 16 -> 1`
MLP is roughly 15k parameters, versus the current approximately 53k second
readout path.

The block must preserve the repository's `mul_ir` and `ir_mul` conventions,
must be TorchScript/`torch.compile` compatible, and must not use e3nn
`ElementwiseTensorProduct`.

## Model integration

The `MACE` forward loop must retain the first interaction message before it
is overwritten by `products[0]`.  In direct mode it passes that message,
`products[0]` output, and `node_attrs` to one fresh direct head, then passes
the scalar invariants to the existing nonlinear readout.  It must create no
`readout_reshapes` or `readout_products` for this mode.  Legacy one- and
multi-interaction paths must be unchanged.

The direct block should live in a dedicated `ModuleList` named
`direct_readouts`. Plain `load_foundations` already copies only source state
keys and therefore naturally leaves this fresh target-only module untouched.
The element-remapping `load_foundations_elements` child-module traversal must
explicitly skip `direct_readouts`, alongside `interactions`, `products`, and
`readouts`, so a foundation model without this module can be loaded safely.

## Verification

- Unit-test direct invariants for output shape, central-element dependence,
  rotation invariance, and force equivariance.
- TorchScript the direct one-interaction model and run one energy/force
  forward pass.
- Verify legacy `use_last_readout_only` remains unchanged and direct mode
  creates no second readout product.
- Verify parser/factory/config round-trip and invalid option combinations.
- Verify foundation loading copies the compatible backbone but leaves every
  direct-head parameter unchanged.
