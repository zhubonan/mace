###########################################################################################
# Tests for opt-in native OMat LMDB training integration
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import importlib
import zlib
from types import SimpleNamespace

import lmdb
import numpy as np
import orjson
import pytest
import torch
from torch.utils.data import ConcatDataset

from mace import data, tools
from mace.cli import run_train
from mace.cli.run_train import _build_train_loader_and_sampler

train_tools = importlib.import_module("mace.tools.train")


def test_native_omat_root_is_routed_without_ase_probe(tmp_path):
    family = tmp_path / "family"
    family.mkdir()
    (family / "db_0.aselmdb").touch()

    assert run_train._is_native_omat_input("omat_direct", [str(tmp_path)])
    assert not run_train._is_native_omat_input("ase", [str(tmp_path)])


def test_split_input_paths_routes_native_roots_as_one_dataset(tmp_path):
    roots = []
    for family_name in ("family_a", "family_b"):
        family = tmp_path / family_name
        family.mkdir()
        (family / "db_0.aselmdb").touch()
        roots.append(str(family))

    ase_files, non_ase_files = run_train._split_input_paths(
        "omat_direct", roots
    )

    assert ase_files == []
    assert non_ase_files == [roots]


def test_split_input_paths_accepts_native_validation_root(tmp_path):
    family = tmp_path / "family"
    family.mkdir()
    (family / "db_0.aselmdb").touch()

    ase_files, non_ase_files = run_train._split_input_paths(
        "omat_direct", [str(tmp_path)]
    )

    assert ase_files == []
    assert non_ase_files == [[str(tmp_path)]]


def write_native_dataset(tmp_path, rows=4):
    family = tmp_path / "family"
    family.mkdir()
    shard = family / "db_0.aselmdb"
    environment = lmdb.open(str(shard), subdir=False, map_size=16 << 20)
    with environment.begin(write=True) as transaction:
        for row in range(rows):
            payload = {
                "numbers": [1],
                "positions": [[float(row), 0.0, 0.0]],
                "cell": np.eye(3).tolist(),
                "pbc": [False, False, False],
                "energy": float(row),
                "forces": [[0.0, 0.0, 0.0]],
                "stress": [0.0] * 6,
            }
            transaction.put(
                str(row + 1).encode("ascii"), zlib.compress(orjson.dumps(payload))
            )
    environment.close()
    np.savez(family / "metadata_db_0.npz", natoms=np.ones(rows, dtype=np.int64))
    return data.OMatLMDBDataset(
        tmp_path,
        r_max=1.5,
        z_table=tools.AtomicNumberTable([1]),
        heads=["Default"],
    )


def training_args(**overrides):
    values = {
        "batch_sampling": "shard_window_atom_balanced",
        "batch_size": 2,
        "seed": 17,
        "lmdb_shard_window": 2,
        "distributed": False,
        "lbfgs": False,
        "data_aug_magmom": False,
        "pin_memory": False,
        "num_workers": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("distributed,world_size,rank", [(False, 1, 0), (True, 2, 1)])
def test_balanced_native_loader_uses_fixed_structure_batch_sampler(
    tmp_path, distributed, world_size, rank
):
    dataset = write_native_dataset(tmp_path, rows=8)
    args = training_args(distributed=distributed)

    train_set, train_loader, train_sampler = _build_train_loader_and_sampler(
        {"Default": dataset},
        ["Default"],
        args,
        world_size=world_size,
        rank=rank,
    )

    assert train_set is dataset
    assert train_loader.batch_sampler is train_sampler
    assert train_loader.batch_size is None
    assert all(batch.num_graphs == args.batch_size for batch in train_loader)
    dataset.close()


@pytest.mark.parametrize(
    "train_sets,heads,overrides,match",
    [
        ({"Default": [object()]}, ["Default"], {}, "OMatLMDBDataset"),
        ({}, ["a", "b"], {}, "one training head"),
        ({"Default": [object()]}, ["Default"], {"lbfgs": True}, "LBFGS"),
        (
            {"Default": [object()]},
            ["Default"],
            {"data_aug_magmom": True},
            "magnetic",
        ),
    ],
)
def test_balanced_sampling_rejects_unsupported_training_modes(
    train_sets, heads, overrides, match
):
    with pytest.raises(ValueError, match=match):
        _build_train_loader_and_sampler(
            train_sets,
            heads,
            training_args(**overrides),
            world_size=1,
            rank=0,
        )


def test_balanced_sampling_rejects_combined_native_datasets(tmp_path):
    dataset = write_native_dataset(tmp_path)
    combined = ConcatDataset([dataset, dataset])

    with pytest.raises(ValueError, match="one native OMat dataset"):
        _build_train_loader_and_sampler(
            {"Default": combined},
            ["Default"],
            training_args(),
            world_size=1,
            rank=0,
        )
    dataset.close()


class RecordingSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


@pytest.mark.parametrize("distributed", [False, True])
def test_train_advances_any_epoch_aware_sampler(monkeypatch, distributed):
    sampler = RecordingSampler()
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters())
    losses = iter([3.0, 2.0, 1.0])
    monkeypatch.setattr(train_tools, "evaluate", lambda **kwargs: (next(losses), {}))
    monkeypatch.setattr(train_tools, "valid_err_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_tools,
        "train_one_epoch",
        lambda **kwargs: kwargs["compiled_training_model"],
    )
    if distributed:
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
        monkeypatch.setattr(
            torch.distributed, "broadcast", lambda *args, **kwargs: None
        )
    checkpoint_handler = SimpleNamespace(save=lambda **kwargs: None)
    scheduler = SimpleNamespace(step=lambda **kwargs: None)

    train_tools.train(
        model=model,
        loss_fn=torch.nn.Identity(),
        train_loader=[],
        valid_loaders={"valid": object()},
        optimizer=optimizer,
        lr_scheduler=scheduler,
        start_epoch=3,
        max_num_epochs=5,
        patience=10,
        checkpoint_handler=checkpoint_handler,
        logger=SimpleNamespace(log=lambda metrics: None),
        eval_interval=1,
        output_args={
            "energy": True,
            "forces": False,
            "virials": False,
            "stress": False,
        },
        device=torch.device("cpu"),
        log_errors="",
        distributed=distributed,
        train_sampler=sampler,
    )

    assert sampler.epochs == [3, 4]
