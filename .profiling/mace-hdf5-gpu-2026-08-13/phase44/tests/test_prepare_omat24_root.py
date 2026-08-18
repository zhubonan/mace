###########################################################################################
# Tests for the audited OMat24 dataset-root relocation utility
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_omat24_root import prepare_root  # noqa: E402

from mace.data import OMAT_MANIFEST_NAME, build_omat_shard_index  # noqa: E402


def make_fake_corpus(tmp_path):
    source = tmp_path / "hidden" / ".staging"
    for family_name, sizes in (("family_a", [2, 4]), ("family_b", [3])):
        family = source / family_name
        family.mkdir(parents=True)
        (family / "db_0.aselmdb").touch()
        np.savez(
            family / "metadata_db_0.npz",
            natoms=np.asarray(sizes, dtype=np.int64),
        )
    target = tmp_path / "datasets" / "omat24-aselmdb"
    return source, target, source


def call_prepare(source, target, link, *, apply=False, rows=3, shards=2):
    return prepare_root(
        source,
        target,
        link,
        expected_rows=rows,
        expected_shards=shards,
        apply=apply,
    )


def test_dry_run_audits_without_mutation(tmp_path):
    source, target, link = make_fake_corpus(tmp_path)

    result = call_prepare(source, target, link)

    assert result.families == 2
    assert result.shards == 2
    assert result.rows == 3
    assert result.same_filesystem is True
    assert result.applied is False
    assert source.is_dir()
    assert not source.is_symlink()
    assert not target.exists()
    assert not (source / OMAT_MANIFEST_NAME).exists()


def test_apply_renames_source_and_leaves_compatibility_symlink(tmp_path):
    source, target, link = make_fake_corpus(tmp_path)

    result = call_prepare(source, target, link, apply=True)

    assert result.shards == 2
    assert target.is_dir()
    assert link.is_symlink()
    assert link.resolve() == target.resolve()
    assert (target / OMAT_MANIFEST_NAME).is_file()
    reopened = build_omat_shard_index(target)
    assert len(reopened) == 3
    assert reopened.dataset_id == "OMat24"


def test_apply_rolls_back_when_symlink_creation_fails(tmp_path, monkeypatch):
    source, target, link = make_fake_corpus(tmp_path)

    def fail_symlink(*args, **kwargs):
        raise OSError("injected symlink failure")

    monkeypatch.setattr(Path, "symlink_to", fail_symlink)

    with pytest.raises(OSError, match="injected"):
        call_prepare(source, target, link, apply=True)

    assert source.is_dir()
    assert not source.is_symlink()
    assert not target.exists()
    assert not (source / OMAT_MANIFEST_NAME).exists()


def test_refuses_nonempty_target(tmp_path):
    source, target, link = make_fake_corpus(tmp_path)
    target.mkdir(parents=True)
    (target / "unrelated").touch()

    with pytest.raises(ValueError, match="target"):
        call_prepare(source, target, link)


def test_refuses_to_overwrite_a_preexisting_manifest(tmp_path):
    source, target, link = make_fake_corpus(tmp_path)
    (source / OMAT_MANIFEST_NAME).write_text("user-owned", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        call_prepare(source, target, link)


def test_refuses_source_symlink(tmp_path):
    real_source, target, _ = make_fake_corpus(tmp_path)
    link = tmp_path / "source-link"
    link.symlink_to(real_source, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        call_prepare(link, target, link)


def test_refuses_cross_filesystem_move(tmp_path, monkeypatch):
    source, target, link = make_fake_corpus(tmp_path)
    source_device = source.stat().st_dev

    def fake_device(path):
        return source_device if Path(path) == source else source_device + 1

    monkeypatch.setattr("prepare_omat24_root._device_id", fake_device)

    with pytest.raises(ValueError, match="filesystem"):
        call_prepare(source, target, link)


@pytest.mark.parametrize(
    "rows,shards,match",
    [(4, 2, "row"), (3, 3, "shard")],
)
def test_refuses_unexpected_counts(tmp_path, rows, shards, match):
    source, target, link = make_fake_corpus(tmp_path)

    with pytest.raises(ValueError, match=match):
        call_prepare(source, target, link, rows=rows, shards=shards)


def test_refuses_existing_incompatible_link(tmp_path):
    source, target, _ = make_fake_corpus(tmp_path)
    link = tmp_path / "public" / "omat24"
    link.parent.mkdir()
    link.symlink_to(tmp_path)

    with pytest.raises(ValueError, match="compatibility"):
        call_prepare(source, target, link)


def test_refuses_target_below_source(tmp_path):
    source, _, link = make_fake_corpus(tmp_path)
    target = source / "nested" / "target"

    with pytest.raises(ValueError, match="below"):
        call_prepare(source, target, link)
