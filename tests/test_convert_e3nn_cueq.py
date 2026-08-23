import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import get_optimization_defaults, o3, set_optimization_defaults

from mace import data, modules, tools
from mace.cli import convert_cueq_e3nn, convert_e3nn_cueq
from mace.tools import torch_geometric


class _HiddenIrrepsStub:
    @staticmethod
    def slices():
        return [slice(0, 1)]


class _ConversionModelStub(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.products = []
        self.interactions = []
        self.cueq_config = config.get("cueq_config")

    def to(self, *args, **kwargs):
        return self


@pytest.mark.parametrize(
    ("device", "conv_fusion", "expected"),
    [
        ("cuda", None, True),
        (torch.device("cuda"), None, True),
        ("cuda", False, False),
        ("cpu", True, True),
    ],
)
def test_conversion_selects_requested_convolution_fusion(
    monkeypatch, device, conv_fusion, expected
):
    monkeypatch.setattr(
        convert_e3nn_cueq,
        "extract_config_mace_model",
        lambda model: {
            "hidden_irreps": _HiddenIrrepsStub(),
            "correlation": 2,
            "num_interactions": 0,
        },
    )

    converted = convert_e3nn_cueq.run(
        _ConversionModelStub(), device=device, conv_fusion=conv_fusion
    )

    assert converted.cueq_config.conv_fusion is expected


def test_conversion_round_trip_preserves_invariant_readout_contraction():
    torch.manual_seed(19)
    source = modules.ScaleShiftMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.RealAgnosticDensityResidualInteractionBlock,
        interaction_cls_first=modules.RealAgnosticDensityResidualInteractionBlock,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(2),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=2,
        readout_correlation=2,
        use_last_readout_only=True,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )

    cueq_model = convert_e3nn_cueq.run(
        source, device="cpu", return_model=True, conv_fusion=False
    )
    round_trip = convert_cueq_e3nn.run(
        cueq_model, device="cpu", return_model=True
    )

    source_state = source.state_dict()
    round_trip_state = round_trip.state_dict()
    readout_keys = [
        key
        for key in source_state
        if key.startswith("readout_products.0.symmetric_contractions")
    ]
    assert readout_keys
    for key in readout_keys:
        torch.testing.assert_close(source_state[key], round_trip_state[key])

    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.9, 0.1]],
    )
    atomic_data = data.AtomicData.from_config(
        data.config_from_atoms(atoms),
        z_table=tools.AtomicNumberTable([1, 8]),
        cutoff=5.0,
    )
    batch = next(
        iter(
            torch_geometric.dataloader.DataLoader(
                [atomic_data], batch_size=1, shuffle=False
            )
        )
    ).to_dict()
    expected = source(batch, training=False, compute_force=True)
    cueq_output = cueq_model(batch, training=False, compute_force=True)
    actual = round_trip(batch, training=False, compute_force=True)

    for key in ("energy", "forces"):
        torch.testing.assert_close(cueq_output[key], expected[key])
        torch.testing.assert_close(actual[key], expected[key])


def test_cueq_to_e3nn_result_is_pickleable_in_script_jit_mode(tmp_path):
    defaults = get_optimization_defaults()
    if "jit_mode" not in defaults:
        pytest.skip("e3nn release does not expose jit_mode")

    source = modules.ScaleShiftMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.RealAgnosticDensityResidualInteractionBlock,
        interaction_cls_first=modules.RealAgnosticDensityResidualInteractionBlock,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(2),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=2,
        readout_correlation=2,
        use_last_readout_only=True,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    cueq_model = convert_e3nn_cueq.run(
        source, device="cpu", return_model=True, conv_fusion=False
    )

    try:
        set_optimization_defaults(jit_mode="script")
        round_trip = convert_cueq_e3nn.run(
            cueq_model, device="cpu", return_model=True
        )

        assert get_optimization_defaults()["jit_mode"] == "script"
        torch.save(round_trip, tmp_path / "round-trip.model")
    finally:
        set_optimization_defaults(jit_mode=defaults["jit_mode"])
