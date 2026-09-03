import inspect
import re

import pytest
import torch
from e3nn import o3
from e3nn.util.jit import compile as e3nn_compile

from mace import modules
from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig


def _block(rank: int = 3):
    return modules.DirectMessageHiddenReadoutBlock(
        message_irreps=o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o"),
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        num_elements=2,
        rank=rank,
    )


def test_direct_block_channel_projections_use_readable_matmul():
    source = inspect.getsource(modules.DirectMessageHiddenReadoutBlock.forward)
    compact_source = re.sub(r"\s+", "", source)

    assert "[nodes, channels, irrep_dim] @ [channels, rank] -> [nodes, rank, irrep_dim]" in source
    for features, weights in (
        ("message_0", "self.message_weight_0"),
        ("message_1", "self.message_weight_1"),
        ("hidden_0", "self.hidden_weight_0"),
        ("hidden_1", "self.hidden_weight_1"),
    ):
        assert (
            f"torch.matmul({features}.transpose(1,2),{weights}.T).transpose(1,2)"
            in compact_source
        )
    assert 'torch.einsum("ncd,rc->nrd"' not in source


def test_direct_block_matches_explicit_0e_and_1o_projection_reference():
    block = _block(rank=2)
    message = torch.arange(128, dtype=torch.float32).reshape(2, 4, 16)
    hidden = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with torch.no_grad():
        block.message_weight_0.copy_(
            torch.tensor([[1.0, -1.0, 2.0, 0.5], [-2.0, 1.0, 0.0, 1.5]])
        )
        block.message_weight_1.copy_(
            torch.tensor([[0.5, 1.0, -1.0, 2.0], [1.0, -0.5, 2.0, -1.0]])
        )
        block.hidden_weight_0.copy_(
            torch.tensor([[2.0, 0.0, -1.0, 1.0], [-1.0, 1.0, 0.5, 2.0]])
        )
        block.hidden_weight_1.copy_(
            torch.tensor([[1.0, -1.0, 0.5, 2.0], [0.0, 1.5, -2.0, 1.0]])
        )
        block.gamma.copy_(
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -0.5]])
        )
        block.beta.copy_(
            torch.tensor([[0.25, -0.5, 1.0, -1.5], [-2.0, 0.5, -1.0, 2.0]])
        )

    message_0 = message[:, :, :1]
    message_1 = message[:, :, 1:4]
    hidden_0 = hidden[:, :4].reshape(-1, 4, 1)
    hidden_1 = hidden[:, 4:16].reshape(-1, 4, 3)

    message_rank_0 = (
        message_0.unsqueeze(1) * block.message_weight_0[None, :, :, None]
    ).sum(dim=2)
    hidden_rank_0 = (
        hidden_0.unsqueeze(1) * block.hidden_weight_0[None, :, :, None]
    ).sum(dim=2)
    q0 = (message_rank_0 * hidden_rank_0).sum(dim=-1)
    message_rank_1 = (
        message_1.unsqueeze(1) * block.message_weight_1[None, :, :, None]
    ).sum(dim=2)
    hidden_rank_1 = (
        hidden_1.unsqueeze(1) * block.hidden_weight_1[None, :, :, None]
    ).sum(dim=2)
    q1 = (message_rank_1 * hidden_rank_1).sum(dim=-1) / 3.0**0.5
    expected = torch.cat([q0, q1], dim=-1) * (attrs @ block.gamma) + (attrs @ block.beta)

    torch.testing.assert_close(block(message, hidden, attrs), expected)


def test_direct_block_output_shape_and_central_element_conditioning():
    block = _block()
    # Keep geometric features identical across samples so any output change
    # is attributable to central-species conditioning via gamma/beta.
    message_row = torch.randn(1, 4, 16)
    hidden_row = torch.randn(1, 16)
    message = message_row.repeat(2, 1, 1)
    hidden = hidden_row.repeat(2, 1)
    attrs = torch.eye(2)
    with torch.no_grad():
        block.gamma[1].fill_(2.0)
        block.beta[1].fill_(0.25)
    result = block(message, hidden, attrs)
    assert result.shape == (2, 6)
    assert not torch.allclose(result[0], result[1])


def test_direct_block_rejects_rank_zero():
    with pytest.raises(ValueError):
        modules.DirectMessageHiddenReadoutBlock(
            message_irreps=o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o"),
            hidden_irreps=o3.Irreps("4x0e + 4x1o"),
            num_elements=2,
            rank=0,
        )


def test_direct_block_is_deterministic_for_identical_inputs():
    block = _block()
    message = torch.randn(1, 4, 16)
    hidden = torch.randn(1, 16)
    attrs = torch.tensor([[1.0, 0.0]])
    assert torch.equal(block(message, hidden, attrs), block(message, hidden, attrs))


@pytest.mark.parametrize(
    "message_irreps,hidden_irreps",
    [
        ("4x1o + 4x2e + 4x3o", "4x0e + 4x1o"),
        ("4x0e + 4x2e + 4x3o", "4x0e + 4x1o"),
        ("4x0e + 4x1o", "4x1o"),
        ("4x0e + 4x1o", "4x0e"),
        ("4x0e + 4x1o", "3x0e + 4x1o"),
    ],
)
def test_direct_block_rejects_missing_or_mismatched_irreps(message_irreps, hidden_irreps):
    with pytest.raises(ValueError):
        modules.DirectMessageHiddenReadoutBlock(
            message_irreps=o3.Irreps(message_irreps),
            hidden_irreps=o3.Irreps(hidden_irreps),
            num_elements=2,
            rank=3,
        )


def test_direct_block_scripts_with_e3nn_jit():
    block = _block()
    scripted = e3nn_compile(block)
    message = torch.randn(2, 4, 16)
    hidden = torch.randn(2, 16)
    attrs = torch.eye(2)
    torch.testing.assert_close(block(message, hidden, attrs), scripted(message, hidden, attrs))


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuEquivariance unavailable")
def test_direct_block_supports_ir_mul_layout():
    mul_ir = _block()
    config = CuEquivarianceConfig(enabled=True, layout="ir_mul")
    ir_mul = modules.DirectMessageHiddenReadoutBlock(
        message_irreps=mul_ir.message_irreps,
        hidden_irreps=mul_ir.hidden_irreps,
        num_elements=2,
        rank=3,
        cueq_config=config,
    )
    ir_mul.load_state_dict(mul_ir.state_dict())
    message = torch.randn(2, 4, 16)
    hidden = torch.randn(2, 16)
    attrs = torch.eye(2)
    # Convert mul_ir message to ir_mul by transposing each irrep block.
    message_ir = torch.cat(
        [
            message[:, :, start : start + dim].transpose(1, 2)
            for start, dim in [(0, 1), (1, 3), (4, 5), (9, 7)]
        ],
        dim=1,
    )
    hidden_ir = torch.cat(
        [
            hidden[:, start : start + 4 * dim]
            .reshape(2, 4, dim)
            .transpose(1, 2)
            .reshape(2, -1)
            for start, dim in [(0, 1), (4, 3)]
        ],
        dim=1,
    )
    torch.testing.assert_close(mul_ir(message, hidden, attrs), ir_mul(message_ir, hidden_ir, attrs))
