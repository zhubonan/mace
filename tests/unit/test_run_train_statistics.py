from types import SimpleNamespace

from mace.cli.run_train import _apply_statistics_to_head_config


def test_statistics_file_preserves_direct_foundation_e0_request() -> None:
    head = SimpleNamespace(
        atomic_numbers="[1, 8]",
        mean=None,
        std=None,
        avg_num_neighbors=None,
        compute_avg_num_neighbors=True,
        E0s="foundation",
        atomic_energies_dict=None,
    )
    statistics = {
        "atomic_numbers": "[1, 8]",
        "mean": 0.25,
        "std": 0.5,
        "avg_num_neighbors": 7.0,
        "atomic_energies": "{1: -1.0, 8: -2.0}",
    }

    _apply_statistics_to_head_config(
        head,
        statistics,
        preserve_foundation_e0s=True,
    )

    assert head.mean == 0.25
    assert head.std == 0.5
    assert head.avg_num_neighbors == 7.0
    assert head.compute_avg_num_neighbors is False
    assert head.E0s == "foundation"
    assert head.atomic_energies_dict is None
