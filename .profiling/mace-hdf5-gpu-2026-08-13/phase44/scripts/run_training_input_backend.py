#!/usr/bin/env python3
"""Adapt the retained compiled training benchmark to native OMat LMDB input."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from mace import data
from mace.tools import compile as mace_compile


class PrefixShardIndex:
    """A verifier-only first-shard prefix without copying official metadata."""

    def __init__(self, index, length: int) -> None:
        length = int(length)
        if not index.shards or length < 1 or length > index.shards[0].length:
            raise ValueError("record limit must fit within the first OMat shard")
        self._index = index
        self.roots = index.roots
        self.dataset_id = f"{index.dataset_id}:first-shard-prefix-{length}"
        self.shards = (replace(index.shards[0], length=length),)
        self.cumulative_ends = (length,)

    def __len__(self) -> int:
        return self.cumulative_ends[0]

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return 0, index

    def load_natoms(self, shard_index: int):
        if shard_index != 0:
            raise IndexError(shard_index)
        return self._index.load_natoms(0)[: len(self)]


class NativeOMatDatasetAdapter(Dataset):
    """Expose the production native reader through the retained harness API."""

    def __init__(self, file_path, r_max, z_table, atomic_dataclass=None, **kwargs):
        del file_path, atomic_dataclass
        root = os.environ.get("MACE_PHASE44_LMDB_ROOT")
        if not root:
            raise ValueError("MACE_PHASE44_LMDB_ROOT is required for native input")
        self.dataset = data.OMatLMDBDataset(
            Path(root),
            r_max=r_max,
            z_table=z_table,
            heads=kwargs.get("heads", ["Default"]),
            head=kwargs.get("head", "Default"),
            max_open_shards=int(os.environ.get("MACE_PHASE44_MAX_OPEN_SHARDS", "8")),
        )
        record_limit = int(os.environ.get("MACE_PHASE44_RECORD_LIMIT", "0"))
        self.shard_index = (
            PrefixShardIndex(self.dataset.shard_index, record_limit)
            if record_limit
            else self.dataset.shard_index
        )

    def __len__(self):
        return len(self.shard_index)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.dataset[index]


class RepeatedBalancedDistributedSampler(Sampler[int]):
    """Flatten fixed batches from each repeated native dataset for DataLoader."""

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
        *,
        batch_size=None,
        window_shards=None,
    ) -> None:
        del shuffle, drop_last
        if not isinstance(dataset, ConcatDataset) or not dataset.datasets:
            raise TypeError("Balanced input requires a non-empty ConcatDataset")
        if num_replicas is None or rank is None:
            raise ValueError("num_replicas and rank are required")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = int(
            batch_size
            if batch_size is not None
            else os.environ.get("MACE_DDP_BATCH_SIZE", "128")
        )
        self.window_shards = int(
            window_shards
            if window_shards is not None
            else os.environ.get("MACE_PHASE44_WINDOW_SHARDS", "8")
        )
        first = dataset.datasets[0]
        if not hasattr(first, "shard_index"):
            raise TypeError("Balanced input requires datasets with a shard_index")
        self._source_length = len(first)
        self._shard_index = first.shard_index
        if any(
            len(source) != self._source_length
            or getattr(source, "shard_index", None) != self._shard_index
            for source in dataset.datasets
        ):
            raise ValueError("Repeated balanced datasets must share one shard index")
        prototype = self._sampler(repeat=0)
        self._rank_batches_per_repeat = len(prototype)
        full_batches = len(dataset) // self.batch_size
        usable_batches = full_batches - full_batches % self.num_replicas
        self._target_rank_batches = usable_batches // self.num_replicas
        if self._target_rank_batches and not self._rank_batches_per_repeat:
            raise ValueError("Each repeated source must contain one complete DDP pack")

    def _sampler(self, *, repeat: int):
        return data.ShardWindowAtomBalancedBatchSampler(
            self._shard_index,
            batch_size=self.batch_size,
            seed=self.seed,
            epoch=self.epoch * len(self.dataset.datasets) + repeat,
            num_replicas=self.num_replicas,
            rank=self.rank,
            window_shards=self.window_shards,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        yielded_batches = 0
        repeat = 0
        while yielded_batches < self._target_rank_batches:
            source_index = repeat % len(self.dataset.datasets)
            offset = source_index * self._source_length
            for batch in self._sampler(repeat=repeat):
                if yielded_batches == self._target_rank_batches:
                    return
                for local_index in batch:
                    yield offset + int(local_index)
                yielded_batches += 1
            repeat += 1

    def __len__(self) -> int:
        return self._target_rank_batches * self.batch_size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotate_result(
    *,
    output_dir: Path,
    benchmark: Path,
    lmdb_backend: str,
    batch_sampling: str,
) -> None:
    rank = int(os.environ["RANK"])
    result_path = output_dir / f"rank-{rank}.json"
    record = json.loads(result_path.read_text(encoding="utf-8"))
    source_revision = os.environ.get("MACE_PHASE44_SOURCE_REVISION")
    if not source_revision:
        raise ValueError("MACE_PHASE44_SOURCE_REVISION is required")
    record.update(
        {
            "lmdb_backend": lmdb_backend,
            "batch_sampling": batch_sampling,
            "source_revision": source_revision,
            "benchmark_sha256": _sha256(benchmark),
            "adapter_sha256": _sha256(Path(__file__).resolve()),
            "repetition": int(os.environ.get("MACE_PHASE44_REPETITION", "0")),
            "lmdb_root": os.environ.get("MACE_PHASE44_LMDB_ROOT"),
            "lmdb_shard_window": int(os.environ.get("MACE_PHASE44_WINDOW_SHARDS", "8")),
            "lmdb_max_open_shards": int(
                os.environ.get("MACE_PHASE44_MAX_OPEN_SHARDS", "8")
            ),
            "record_limit": int(os.environ.get("MACE_PHASE44_RECORD_LIMIT", "0")),
        }
    )
    result_path.write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    lmdb_backend = os.environ.get("MACE_PHASE44_LMDB_BACKEND", "hdf5")
    batch_sampling = os.environ.get("MACE_PHASE44_BATCH_SAMPLING", "random")
    if lmdb_backend not in {"hdf5", "omat_direct"}:
        raise ValueError(f"Unsupported input backend: {lmdb_backend}")
    if batch_sampling not in {"random", "shard_window_atom_balanced"}:
        raise ValueError(f"Unsupported batch sampling: {batch_sampling}")
    if lmdb_backend == "hdf5" and batch_sampling != "random":
        raise ValueError("Atom-balanced sampling requires native OMat LMDB input")

    benchmark = (
        Path(__file__).resolve().parents[2] / "scripts/compiled_ddp_benchmark.py"
    )
    namespace = runpy.run_path(str(benchmark))
    namespace["main"].__globals__["_atomic_numbers"] = lambda path: list(
        range(1, 84)
    ) + list(range(89, 95))
    if lmdb_backend == "omat_direct":
        data.HDF5Dataset = NativeOMatDatasetAdapter
    if batch_sampling == "shard_window_atom_balanced":
        namespace["main"].__globals__[
            "DistributedSampler"
        ] = RepeatedBalancedDistributedSampler

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with mace_compile.disable_e3nn_codegen():
        namespace["main"]()
    output_dir = Path(os.environ["MACE_DDP_OUTPUT_DIR"])
    _annotate_result(
        output_dir=output_dir,
        benchmark=benchmark,
        lmdb_backend=lmdb_backend,
        batch_sampling=batch_sampling,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
