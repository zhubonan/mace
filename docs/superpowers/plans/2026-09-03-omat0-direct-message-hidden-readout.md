# OMat-0 Direct Message-Hidden Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in one-interaction MACE energy readout that reuses the OMat-0 first layer and directly, element-conditionally contracts its message with its hidden state before a nonlinear scalar readout.

**Architecture:** Keep the first interaction and first `EquivariantProductBasisBlock` unchanged. Add a fresh `DirectMessageHiddenReadoutBlock` that forms rank-16 invariant dot products between common `0e` and `1o` message/hidden channels, applies a one-hot-central-element affine gate, and supplies `2 * rank` scalar features to the existing `NonLinearReadoutBlock`. The legacy `use_last_readout_only` path remains unchanged; the new mode has its own mutually-exclusive configuration flag.

**Tech Stack:** Python 3, PyTorch, e3nn, cuequivariance-compatible MACE wrappers, pytest, TorchScript, `torch.compile`.

## Global Constraints

- Add `direct_message_hidden_readout: bool = False` and `direct_readout_rank: int = 16`; reject the direct mode unless `num_interactions == 1`, `direct_readout_rank > 0`, and `use_last_readout_only is False`.
- Exact OMat-0 reuse targets `RealAgnosticDensityInteractionBlock` followed by its correlation-3 product with hidden irreps `128x0e + 128x1o`; do not add a first-layer self/residual term.
- The direct block consumes a reshaped first-interaction message, flat post-product hidden features, and central one-hot `node_attrs`; it uses only shared `0e` and `1o` irreps.
- The direct block is central-element conditioned through learned `gamma[Z]` and `beta[Z]`, not through a new edgewise species-pair table.
- Do not use `e3nn.ElementwiseTensorProduct`; preserve `mul_ir` and `ir_mul` layout behaviour and keep the block TorchScript/compile compatible.
- Foundation loading copies compatible node/radial embedding, interaction, product, and existing baseline/scale-shift state; it must leave all direct-head and final-readout parameters freshly initialized.

---

## File structure

- Modify `mace/modules/blocks.py`: define the focused direct invariant block.
- Modify `mace/modules/__init__.py`: export that block for public module construction and tests.
- Modify `mace/modules/models.py`: configure the direct mode, retain the pre-product first message, and route its invariants to the existing nonlinear readout.
- Modify `mace/tools/arg_parser.py`: expose direct-mode CLI flags.
- Modify `mace/tools/model_script_utils.py`: pass direct-mode arguments into MACE and `ScaleShiftMACE` construction.
- Modify `mace/tools/scripts_utils.py`: serialize direct-mode configuration for model reconstruction.
- Modify `mace/tools/finetuning_utils.py`: exempt `direct_readouts` from element-remapping foundation child traversal.
- Modify `tests/unit/test_single_interaction_readout.py`: cover model construction, invariance/equivariance, TorchScript, CLI factory, and config reconstruction.
- Modify `tests/unit/test_arg_parser.py`: cover direct-mode defaults and parsed values.
- Modify `tests/unit/test_finetuning_utils.py`: prove copied backbone/fresh direct head behaviour.
- Create `tests/unit/test_direct_message_hidden_readout.py`: isolate direct-block shape, central-element conditioning, layout, and invalid-irreps behaviour.

### Task 1: Implement and unit-test the element-conditioned direct invariant block

**Files:**
- Create: `tests/unit/test_direct_message_hidden_readout.py`
- Modify: `mace/modules/blocks.py`
- Modify: `mace/modules/__init__.py`

**Interfaces:**
- Consumes: `message: Tensor[N, C, D_message]` in `reshape_irreps` layout, `hidden: Tensor[N, hidden_irreps.dim]`, and `node_attrs: Tensor[N, num_elements]`.
- Produces: `DirectMessageHiddenReadoutBlock.forward(message, hidden, node_attrs) -> Tensor[N, 2 * rank]`, ordered as rank invariants for `0e` followed by rank invariants for `1o`.
- Defines: `DirectMessageHiddenReadoutBlock(message_irreps: o3.Irreps, hidden_irreps: o3.Irreps, num_elements: int, rank: int, cueq_config: Optional[CuEquivarianceConfig] = None)`.

- [ ] **Step 1: Write failing direct-block tests**

Add a factory using OMat-compatible message and hidden irreps, and assert the element-specific affine table changes the output for identical geometric features:

```python
def _block(rank: int = 3):
    return modules.DirectMessageHiddenReadoutBlock(
        message_irreps=o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o"),
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        num_elements=2,
        rank=rank,
    )


def test_direct_block_output_shape_and_central_element_conditioning():
    block = _block()
    message = torch.randn(2, 4, 16)
    hidden = torch.randn(2, 16)
    attrs = torch.eye(2)
    with torch.no_grad():
        block.gamma[1].fill_(2.0)
        block.beta[1].fill_(0.25)
    result = block(message, hidden, attrs)
    assert result.shape == (2, 6)
    assert not torch.allclose(result[0], result[1])
```

Also add tests which: (a) call the block twice with identical inputs and one-hot species and expect identical outputs, (b) assert rank zero and hidden irreps missing `0e` or `1o` raise `ValueError`, and (c) script the block using `e3nn.util.jit.compile` and compare eager/scripted output.

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_direct_message_hidden_readout.py -q
```

Expected: collection fails because `modules.DirectMessageHiddenReadoutBlock` does not exist.

- [ ] **Step 3: Add `DirectMessageHiddenReadoutBlock`**

In `mace/modules/blocks.py`, directly after `NonLinearReadoutBlock`, add a `@compile_mode("script")` module with the declared constructor. Its implementation must:

```python
self.message_weight_0 = torch.nn.Parameter(torch.empty(rank, message_mul))
self.message_weight_1 = torch.nn.Parameter(torch.empty(rank, message_mul))
self.hidden_weight_0 = torch.nn.Parameter(torch.empty(rank, hidden_mul))
self.hidden_weight_1 = torch.nn.Parameter(torch.empty(rank, hidden_mul))
self.gamma = torch.nn.Parameter(torch.ones(num_elements, 2 * rank))
self.beta = torch.nn.Parameter(torch.zeros(num_elements, 2 * rank))
torch.nn.init.xavier_uniform_(self.message_weight_0)
torch.nn.init.xavier_uniform_(self.message_weight_1)
torch.nn.init.xavier_uniform_(self.hidden_weight_0)
torch.nn.init.xavier_uniform_(self.hidden_weight_1)
```

Derive and store the `0e`/`1o` offsets and multiplicities from the supplied irreps; reject missing irreps and unequal message/hidden multiplicities. Convert the message to `[N, C, 2l+1]` for each block, respecting `get_layout(cueq_config)`. Slice and reshape each hidden block in the same layout. Form each rank projection and normalized invariant without an e3nn tensor product:

```python
message_rank = torch.einsum("ncd,rc->nrd", message_l, message_weight)
hidden_rank = torch.einsum("ncd,rc->nrd", hidden_l, hidden_weight)
invariant = (message_rank * hidden_rank).sum(dim=-1) / math.sqrt(2 * ell + 1)
q = torch.cat([q0, q1], dim=-1)
return q * (node_attrs @ self.gamma) + (node_attrs @ self.beta)
```

Use `torch.narrow`/integer offsets rather than data-dependent indexing. Export the class from `mace/modules/__init__.py` and add it to `__all__`.

- [ ] **Step 4: Extend direct-block tests for both feature layouts**

When `CUET_AVAILABLE`, add a second block with `CuEquivarianceConfig(enabled=True, layout="ir_mul")` and explicitly feed message/hidden tensors rearranged from `mul_ir` to `ir_mul`. Assert the result agrees with the `mul_ir` block after copying its state dict. Mark this test skipped when cuEquivariance is unavailable. This protects the same layout distinction exercised by the repository’s gate and LES blocks.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/unit/test_direct_message_hidden_readout.py -q
```

Expected: all direct-block tests pass, including scripted output parity.

- [ ] **Step 6: Commit the focused block**

```bash
git add mace/modules/blocks.py mace/modules/__init__.py tests/unit/test_direct_message_hidden_readout.py
git commit -m "feat: add direct message hidden readout block"
```

### Task 2: Integrate the direct mode into one-interaction MACE and configuration

**Files:**
- Modify: `mace/modules/models.py`
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/tools/model_script_utils.py`
- Modify: `mace/tools/scripts_utils.py`
- Modify: `tests/unit/test_single_interaction_readout.py`
- Modify: `tests/unit/test_arg_parser.py`

**Interfaces:**
- Consumes: constructor and CLI fields `direct_message_hidden_readout` and `direct_readout_rank`.
- Produces: `model.direct_readouts: torch.nn.ModuleList`, empty in legacy mode and containing exactly one direct block in direct mode; `model.direct_message_hidden_readout` and `model.direct_readout_rank` are serializable fields.
- Preserves: `model.readout_products` is populated only by legacy `use_last_readout_only=True` models.

- [ ] **Step 1: Write failing model/CLI/config tests**

Extend the one-interaction helper with `direct_message_hidden_readout` and `direct_readout_rank` keyword arguments. Add these assertions:

```python
def test_direct_mode_uses_message_hidden_head_not_second_product():
    model = _build_model(
        "RealAgnosticDensityInteractionBlock",
        direct_message_hidden_readout=True,
        direct_readout_rank=3,
    )
    assert len(model.direct_readouts) == 1
    assert len(model.readout_products) == 0
    assert model.readouts[0].linear_1.irreps_in == o3.Irreps("6x0e")


def test_direct_mode_rejects_ambiguous_or_multilayer_configuration():
    kwargs = _scale_shift_model_kwargs()
    with pytest.raises(ValueError, match="num_interactions == 1"):
        modules.ScaleShiftMACE(
            **kwargs, num_interactions=2, direct_message_hidden_readout=True
        )
    with pytest.raises(ValueError, match="use_last_readout_only"):
        modules.ScaleShiftMACE(
            **kwargs,
            num_interactions=1,
            use_last_readout_only=True,
            direct_message_hidden_readout=True,
        )
```

Reuse `_create_batch()` to verify rotated structures have equal direct-model energy and appropriately rotated forces. Add a TorchScript test, config extraction/reconstruction test, parser default/custom-flag test, and factory test using a `Namespace` with both new fields.

- [ ] **Step 2: Run integration tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_single_interaction_readout.py tests/unit/test_arg_parser.py -q
```

Expected: direct-mode constructor and parser attributes are missing.

- [ ] **Step 3: Add model construction and forward routing**

In `MACE.__init__` in `mace/modules/models.py`, add the two fields immediately next to `use_last_readout_only`/`readout_correlation`; validate the constraints before constructing interaction/readout modules. Always initialise:

```python
self.direct_readouts = torch.nn.ModuleList()
```

For direct mode, append one `DirectMessageHiddenReadoutBlock` using `inter.target_irreps`, the first `hidden_irreps_out`, `num_elements`, the configured rank, and `cueq_config`; append the existing `readout_cls` with `o3.Irreps(f"{2 * direct_readout_rank}x0e")`. Do not append a `readout_reshape` or `readout_product`.

In `MACE.forward`, rename the value returned by the existing interaction call
to `interaction_message`, then preserve it before the assignment that currently
overwrites `node_feats`:

```python
hidden_feats = product(interaction_message, sc, node_attrs_slice)
if i == 0 and len(self.direct_readouts) == 1:
    direct_feats = self.direct_readouts[0](
        interaction_message, hidden_feats, node_attrs_slice
    )
node_feats = hidden_feats
```

Select `direct_feats` as the only input to the sole nonlinear readout in direct mode. Leave existing `node_feats_concat`, legacy readout-product handling, multi-interaction behaviour, contributions, and force differentiation unchanged.

- [ ] **Step 4: Add CLI/config plumbing**

In `mace/tools/arg_parser.py`, define both flags beside `use_last_readout_only` and `readout_correlation` with defaults `False` and `16`; parse rank with `positive_int`. In every `MACE`/`ScaleShiftMACE` constructor call in `mace/tools/model_script_utils.py`, forward both fields. In `extract_config_mace_model`, include their values whenever present, using defaults for models saved before this change.

- [ ] **Step 5: Run focused integration tests**

Run:

```bash
uv run pytest tests/unit/test_single_interaction_readout.py tests/unit/test_arg_parser.py -q
```

Expected: legacy and direct tests pass; the direct model has rotationally invariant energy, equivariant forces, no second readout product, and a reconstructible configuration.

- [ ] **Step 6: Commit model integration**

```bash
git add mace/modules/models.py mace/tools/arg_parser.py mace/tools/model_script_utils.py mace/tools/scripts_utils.py tests/unit/test_single_interaction_readout.py tests/unit/test_arg_parser.py
git commit -m "feat: add one-layer direct message readout mode"
```

### Task 3: Make element-remapped foundation transfer fresh-head-safe and verify the OMat-0-compatible path

**Files:**
- Modify: `mace/tools/finetuning_utils.py`
- Modify: `tests/unit/test_finetuning_utils.py`
- Modify: `tests/unit/test_compile_training.py`

**Interfaces:**
- Consumes: `load_foundations_elements(target_direct_model, source_foundation, table)` where the source has no `direct_readouts` module.
- Produces: a target with matching first-layer embedding/interaction/product parameters copied and unchanged direct head/MLP parameters.
- Preserves: legacy source/target foundation loading and `include_readouts` semantics.

- [ ] **Step 1: Write the failing fresh-head transfer test**

Extend the foundation test helper so it can build a one-interaction legacy source over `[1, 6, 8]` and an otherwise-identical direct target over `[1, 8]`, both with `8x0e + 8x1o` hidden irreps. Clone the target direct-block state before calling `load_foundations_elements(target, source, table=target_table, load_readout=False)`. Then assert:

```python
target = load_foundations_elements(target, source, table=target_table)
assert torch.allclose(
    target.interactions[0].linear_up.weight,
    source.interactions[0].linear_up.weight,
)
assert torch.allclose(
    target.products[0].linear.weight,
    source.products[0].linear.weight,
)
for key, value in direct_before.items():
    assert torch.allclose(target.direct_readouts[0].state_dict()[key], value)
```

Finish with `water_batch(target_table, cutoff=R_MAX)` and assert finite energy and force. Add a compile-training unit test that creates a direct model with `compile_training=True`, performs one optimizer step, and checks finite loss/gradients; use the same minimal data/parameter pattern already present in `tests/unit/test_compile_training.py`.

- [ ] **Step 2: Run loader and compile tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_finetuning_utils.py tests/unit/test_compile_training.py -q
```

Expected: the element-remapping child-module traversal attempts to access `direct_readouts` from the legacy source, or direct-mode setup is not yet accepted by the compile test.

- [ ] **Step 3: Exempt the fresh direct head from transfer**

In `load_foundations_elements` in `mace/tools/finetuning_utils.py`, extend the handled child-module set exactly as follows:

```python
_handled_attrs = {"interactions", "products", "readouts", "direct_readouts"}
```

Do not add a direct-head copy branch. This is intentional: the direct head’s input and architecture do not exist in the two-interaction OMat-0 checkpoint. Plain `load_foundations` already copies only matching source state keys and therefore also leaves this target-only head fresh.

- [ ] **Step 4: Run all direct and regression tests**

Run:

```bash
uv run pytest tests/unit/test_direct_message_hidden_readout.py tests/unit/test_single_interaction_readout.py tests/unit/test_arg_parser.py tests/unit/test_finetuning_utils.py tests/unit/test_compile_training.py -q
```

Expected: all pass. Then run the repository’s backend parity suite for the supported direct configuration:

```bash
uv run pytest tests/backends/backend_parity.py -q
```

Expected: either parity passes or an explicitly identified cueq/oeq limitation is fixed before merging.

- [ ] **Step 5: Perform the local OMat-0 checkpoint smoke test**

Using `/home/bonan/work/foundation-fe/mace-omat-0-medium.model` and the dedicated MACE runtime, construct a one-interaction direct model with `hidden_irreps="128x0e + 128x1o"`, `interaction_first="RealAgnosticDensityInteractionBlock"`, correlation 3, rank 16, and `use_last_readout_only=False`. Load the foundation with `load_foundations`, then verify:

```python
assert torch.allclose(
    target.interactions[0].linear_up.weight,
    source.interactions[0].linear_up.weight,
)
assert torch.allclose(
    target.products[0].linear.weight,
    source.products[0].linear.weight,
)
assert all(torch.isfinite(x).all() for x in [out["energy"], out["forces"]])
```

Record the target parameter count, direct-head parameter count, one-batch forward latency, and energy/force finiteness in the run log. Do not submit an OMat24 training job until this smoke test passes.

- [ ] **Step 6: Commit foundation compatibility and verification**

```bash
git add mace/tools/finetuning_utils.py tests/unit/test_finetuning_utils.py tests/unit/test_compile_training.py
git commit -m "fix: preserve fresh direct readout during foundation loading"
```

## Plan self-review

- Spec coverage: Task 1 implements direct cross-only invariants, low-rank weights, central-element conditioning, layout support, and TorchScript. Task 2 wires the opt-in mode, preserves legacy semantics, adds the nonlinear MLP, and covers parser/config/force behaviour. Task 3 makes OMat-0 transfer explicit and verifies compile/runtime compatibility.
- Placeholder scan: no deferred implementation markers or undefined work items remain.
- Type consistency: Task 1 defines `DirectMessageHiddenReadoutBlock` and its `forward` inputs; Task 2 creates `model.direct_readouts` and supplies exactly those inputs; Task 3 references that `ModuleList` when proving loader freshness.
