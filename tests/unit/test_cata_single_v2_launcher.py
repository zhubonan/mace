from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "cata"
    / "train_single_v2_omat0.sbatch"
)


def test_pair_repulsion_uses_store_true_cli_flag():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "--pair_repulsion=true" not in launcher
    assert ("--pair_repulsion " + chr(92)) in launcher
