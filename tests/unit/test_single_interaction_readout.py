"""Tests for the one-interaction invariant nonlinear MACE readout."""

import inspect
from argparse import Namespace

import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn.util import jit

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.model_script_utils import (
    _build_model as _build_model_from_args,
    configure_model,
)
from mace.tools.scripts_utils import extract_config_mace_model


def _build_model(
    interaction_first: str,
    *,
    direct_message_hidden_readout: bool = False,
    direct_readout_rank: int = 16,
    atomic_inter_scale: float = 1.0,
    atomic_inter_shift: float = 0.0,
) -> modules.ScaleShiftMACE:
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
        use_last_readout_only=not direct_message_hidden_readout,
        direct_message_hidden_readout=direct_message_hidden_readout,
        direct_readout_rank=direct_readout_rank,
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
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


def test_foundation_direct_mode_replaces_the_second_interaction() -> None:
    foundation = _build_legacy_two_interaction_model()
    foundation_config = extract_config_mace_model(foundation)
    args = Namespace(
        model="FoundationMACE",
        direct_message_hidden_readout=True,
        direct_readout_rank=2,
    )

    target = _build_model_from_args(args, {}, foundation_config, ["Default"])

    assert target.num_interactions.item() == 1
    assert isinstance(target.interactions[0], modules.RealAgnosticDensityInteractionBlock)
    assert target.direct_message_hidden_readout is True
    assert target.use_last_readout_only is False


def test_foundation_single_v2_mode_uses_standard_scalar_readout() -> None:
    foundation = _build_legacy_two_interaction_model()
    foundation_config = extract_config_mace_model(foundation)
    args = Namespace(
        model="FoundationMACE",
        direct_message_hidden_readout=False,
        direct_readout_rank=16,
        omat0_first_layer_readout=True,
    )

    target = _build_model_from_args(args, {}, foundation_config, ["Default"])

    assert target.num_interactions.item() == 1
    assert target.omat0_first_layer_readout is True
    assert target.direct_message_hidden_readout is False
    assert target.use_last_readout_only is True
    assert len(target.direct_readouts) == 0
    assert len(target.readout_products) == 1
    assert len(target.readouts) == 1


def test_foundation_direct_mode_uses_imported_scale_without_loader_scan() -> None:
    class NoIterationLoader:
        def __iter__(self):
            raise AssertionError("direct foundation mode must not scan the train set")

    source = _build_model(
        "RealAgnosticDensityInteractionBlock",
        direct_message_hidden_readout=True,
        direct_readout_rank=2,
        atomic_inter_scale=0.5,
        atomic_inter_shift=0.25,
    )
    args = Namespace(
        model="MACE",
        scaling="rms_forces_scaling",
        mean=None,
        std=None,
        compute_energy=True,
        compute_forces=True,
        compute_dipole=False,
        compute_polarizability=False,
        compute_magforces=False,
        error_table="PerAtomRMSE",
        loss="stress",
        foundation_model_elements=True,
        embedding_specs=None,
        foundation_filter_elements=True,
        finetune_dipoles_polarizabilities=False,
        default_dtype="float32",
        direct_message_hidden_readout=True,
        direct_readout_rank=2,
    )

    target, _ = configure_model(
        args,
        NoIterationLoader(),
        np.zeros(2),
        source,
        ["Default"],
        tools.AtomicNumberTable([1, 8]),
        head_configs=None,
    )

    assert target.scale_shift.scale.item() == 0.5
    assert target.scale_shift.shift.item() == 0.25


def test_foundation_single_v2_mode_uses_imported_scale_without_loader_scan() -> None:
    class NoIterationLoader:
        def __iter__(self):
            raise AssertionError("single-v2 foundation mode must not scan the train set")

    source = _build_model(
        "RealAgnosticDensityInteractionBlock",
        atomic_inter_scale=0.5,
        atomic_inter_shift=0.25,
    )
    args = Namespace(
        model="MACE",
        scaling="rms_forces_scaling",
        mean=None,
        std=None,
        compute_energy=True,
        compute_forces=True,
        compute_dipole=False,
        compute_polarizability=False,
        compute_magforces=False,
        error_table="PerAtomRMSE",
        loss="stress",
        foundation_model_elements=True,
        embedding_specs=None,
        foundation_filter_elements=True,
        finetune_dipoles_polarizabilities=False,
        default_dtype="float32",
        direct_message_hidden_readout=False,
        direct_readout_rank=16,
        omat0_first_layer_readout=True,
    )

    target, _ = configure_model(
        args,
        NoIterationLoader(),
        np.zeros(2),
        source,
        ["Default"],
        tools.AtomicNumberTable([1, 8]),
        head_configs=None,
    )

    assert target.scale_shift.scale.item() == 0.5
    assert target.scale_shift.shift.item() == 0.25


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
    kwargs = dict(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=modules.RealAgnosticDensityInteractionBlock,
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
    with pytest.raises(ValueError, match="RealAgnosticDensityInteractionBlock"):
        residual_kwargs = dict(kwargs)
        residual_kwargs["interaction_cls_first"] = (
            modules.RealAgnosticDensityResidualInteractionBlock
        )
        modules.ScaleShiftMACE(
            **residual_kwargs,
            num_interactions=1,
            direct_message_hidden_readout=True,
        )


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


def test_direct_single_interaction_forward_is_invariant_with_equivariant_forces():
    torch.manual_seed(7)
    model = _build_model(
        "RealAgnosticDensityInteractionBlock",
        direct_message_hidden_readout=True,
        direct_readout_rank=3,
    )
    batch, rotation = _create_batch()
    output = model(batch, training=True, compute_force=True)
    assert torch.allclose(output["energy"][0], output["energy"][1], atol=1.0e-8)
    rotation_tensor = torch.as_tensor(
        rotation, dtype=output["forces"].dtype, device=output["forces"].device
    )
    expected_rotated_forces = output["forces"][:3] @ rotation_tensor.T
    assert torch.allclose(
        expected_rotated_forces, output["forces"][3:], atol=1.0e-7, rtol=1.0e-7
    )


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


def test_direct_single_interaction_readout_is_torchscript_compatible():
    model = _build_model(
        "RealAgnosticDensityInteractionBlock",
        direct_message_hidden_readout=True,
        direct_readout_rank=3,
    )
    compiled = jit.compile(model)
    assert len(compiled.direct_readouts) == 1
    assert len(compiled.readout_products) == 0


def test_forward_accepts_legacy_model_without_readout_contraction_modules():
    model = _build_legacy_two_interaction_model()
    del model.readout_reshapes
    del model.readout_products
    batch, _ = _create_batch()

    output = model(batch, training=False, compute_force=False)

    assert torch.isfinite(output["energy"]).all()


def test_forward_accepts_legacy_model_without_direct_mode_state():
    """A checkpoint saved before direct mode had no direct-mode attributes."""
    model = _build_legacy_two_interaction_model()
    for name in (
        "direct_message_hidden_readout",
        "direct_readout_rank",
        "direct_readouts",
    ):
        if hasattr(model, name):
            delattr(model, name)
    batch, _ = _create_batch()

    output = model(batch, training=True, compute_force=True)

    assert torch.isfinite(output["energy"]).all()
    assert torch.isfinite(output["forces"]).all()


def test_readout_correlation_round_trips_through_extracted_config():
    model = _build_model("RealAgnosticDensityResidualInteractionBlock")

    config = extract_config_mace_model(model)
    reconstructed = modules.ScaleShiftMACE(**config)

    assert config["readout_correlation"] == 2
    assert reconstructed.readout_correlation == 2
    assert len(reconstructed.readout_products) == 1


def test_direct_readout_config_round_trip():
    model = _build_model(
        "RealAgnosticDensityInteractionBlock",
        direct_message_hidden_readout=True,
        direct_readout_rank=3,
    )
    config = extract_config_mace_model(model)
    reconstructed = modules.ScaleShiftMACE(**config)
    assert config["direct_message_hidden_readout"] is True
    assert config["direct_readout_rank"] == 3
    assert reconstructed.direct_message_hidden_readout is True
    assert reconstructed.direct_readout_rank == 3
    assert len(reconstructed.direct_readouts) == 1


def test_mace_new_direct_options_follow_existing_optional_positions():
    option_names = list(inspect.signature(modules.MACE.__init__).parameters)[16:]
    assert option_names == [
        "pair_repulsion",
        "apply_cutoff",
        "use_reduced_cg",
        "use_so3",
        "use_agnostic_product",
        "use_last_readout_only",
        "use_embedding_readout",
        "distance_transform",
        "edge_irreps",
        "use_edge_irreps_first",
        "radial_MLP",
        "radial_type",
        "heads",
        "cueq_config",
        "embedding_specs",
        "oeq_config",
        "lammps_mliap",
        "readout_cls",
        "keep_last_layer_irreps",
        "readout_correlation",
        "direct_message_hidden_readout",
        "direct_readout_rank",
    ]


def test_foundation_model_build_applies_requested_direct_readout_options():
    source = _build_model("RealAgnosticDensityInteractionBlock")
    foundation_config = extract_config_mace_model(source)
    args = Namespace(
        model="FoundationMACE",
        direct_message_hidden_readout=True,
        direct_readout_rank=3,
    )

    model = _build_model_from_args(
        args, foundation_config, foundation_config, ["Default"]
    )

    assert model.direct_message_hidden_readout is True
    assert model.direct_readout_rank == 3
    assert model.use_last_readout_only is False
    assert len(model.direct_readouts) == 1


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
        direct_message_hidden_readout=False,
        direct_readout_rank=16,
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
