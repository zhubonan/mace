"""Tests for the one-interaction invariant nonlinear MACE readout."""

from argparse import Namespace

import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn.util import jit

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.model_script_utils import _build_model as _build_model_from_args
from mace.tools.scripts_utils import extract_config_mace_model


def _build_model(interaction_first: str) -> modules.ScaleShiftMACE:
    return modules.ScaleShiftMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticDensityResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[interaction_first],
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(2),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=3,
        readout_correlation=2,
        use_last_readout_only=True,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _build_legacy_two_interaction_model() -> modules.ScaleShiftMACE:
    return modules.ScaleShiftMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=modules.RealAgnosticResidualInteractionBlock,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(2),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=3,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _create_batch() -> tuple[dict[str, torch.Tensor], np.ndarray]:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    positions = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5],
        ]
    )
    configurations = [
        data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=coords,
            properties={},
            property_weights={},
        )
        for coords in (positions, (rotation @ positions.T).T)
    ]
    table = tools.AtomicNumberTable([1, 8])
    atomic_data = [
        data.AtomicData.from_config(configuration, z_table=table, cutoff=5.0)
        for configuration in configurations
    ]
    batch = next(
        iter(
            torch_geometric.dataloader.DataLoader(
                atomic_data, batch_size=2, shuffle=False
            )
        )
    ).to_dict()
    return batch, rotation


@pytest.mark.parametrize(
    "interaction_first",
    [
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
    ],
)
def test_single_interaction_builds_local_invariant_nonlinear_readout(
    interaction_first,
):
    model = _build_model(interaction_first)

    assert len(model.interactions) == 1
    assert len(model.products) == 1
    assert model.products[0].linear.irreps_out == o3.Irreps(
        "8x0e + 8x1o + 8x2e"
    )
    assert len(model.readout_products) == 1
    assert model.readout_products[0].linear.irreps_out == o3.Irreps("8x0e")
    assert model.readout_correlation == 2
    assert len(model.readouts) == 1
    assert isinstance(model.readouts[0], modules.NonLinearReadoutBlock)


def test_density_residual_variant_alone_enables_direct_self_connection():
    with_self = _build_model("RealAgnosticDensityResidualInteractionBlock")
    without_self = _build_model("RealAgnosticDensityInteractionBlock")

    assert with_self.products[0].use_sc is True
    assert without_self.products[0].use_sc is False


@pytest.mark.parametrize(
    "interaction_first",
    [
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
    ],
)
def test_single_interaction_forward_is_invariant_and_keeps_full_node_features(
    interaction_first,
):
    torch.manual_seed(7)
    model = _build_model(interaction_first)
    batch, rotation = _create_batch()

    output = model(batch, training=True, compute_force=True)

    assert output["node_feats"].shape == (6, o3.Irreps("8x0e + 8x1o + 8x2e").dim)
    assert torch.allclose(output["energy"][0], output["energy"][1], atol=1.0e-8)
    rotation_tensor = torch.as_tensor(
        rotation, dtype=output["forces"].dtype, device=output["forces"].device
    )
    expected_rotated_forces = output["forces"][:3] @ rotation_tensor.T
    assert torch.allclose(
        expected_rotated_forces, output["forces"][3:], atol=1.0e-7, rtol=1.0e-7
    )
    assert torch.isfinite(output["energy"]).all()
    assert torch.isfinite(output["forces"]).all()


@pytest.mark.parametrize("irrep_index", [1, 2])
def test_readout_contraction_consumes_higher_order_features(irrep_index):
    torch.manual_seed(11)
    model = _build_model("RealAgnosticDensityResidualInteractionBlock")
    irreps = o3.Irreps("8x0e + 8x1o + 8x2e")
    features = torch.zeros((2, irreps.dim))
    features[:, irreps.slices()[irrep_index]] = torch.randn(
        (2, irreps[irrep_index].dim)
    )
    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    reshaped = model.readout_reshapes[0](features)
    contracted = model.readout_products[0](reshaped, None, node_attrs)

    assert contracted.shape == (2, 8)
    assert not torch.allclose(contracted, torch.zeros_like(contracted))


def test_single_interaction_readout_is_torchscript_compatible():
    model = _build_model("RealAgnosticDensityResidualInteractionBlock")

    compiled = jit.compile(model)

    assert len(compiled.readout_products) == 1
    assert len(compiled.readout_reshapes) == 1


def test_forward_accepts_legacy_model_without_readout_contraction_modules():
    model = _build_legacy_two_interaction_model()
    del model.readout_reshapes
    del model.readout_products
    batch, _ = _create_batch()

    output = model(batch, training=False, compute_force=False)

    assert torch.isfinite(output["energy"]).all()


def test_readout_correlation_round_trips_through_extracted_config():
    model = _build_model("RealAgnosticDensityResidualInteractionBlock")

    config = extract_config_mace_model(model)
    reconstructed = modules.ScaleShiftMACE(**config)

    assert config["readout_correlation"] == 2
    assert reconstructed.readout_correlation == 2
    assert len(reconstructed.readout_products) == 1


def test_mace_factory_preserves_requested_density_residual_first_interaction():
    args = Namespace(
        model="MACE",
        interaction_first="RealAgnosticDensityResidualInteractionBlock",
        pair_repulsion=False,
        distance_transform="None",
        correlation=3,
        gate="silu",
        MLP_irreps="4x0e",
        std=1.0,
        radial_MLP="[8, 8]",
        radial_type="bessel",
        embedding_specs=None,
        use_embedding_readout=False,
        use_last_readout_only=True,
        use_agnostic_product=False,
        readout_correlation=2,
    )
    model_config = dict(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticDensityResidualInteractionBlock"
        ],
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
    )

    model = _build_model_from_args(args, model_config, None, ["Default"])

    assert isinstance(
        model.interactions[0], modules.RealAgnosticDensityResidualInteractionBlock
    )
    assert args.interaction_first == "RealAgnosticDensityResidualInteractionBlock"
