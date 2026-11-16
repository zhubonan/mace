"""Globally shuffled optimizer batches, partitioned across ranks afterwards."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


def _integer(value: int, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


class GlobalRandomBatchSampler(Sampler[list[int]]):
    """Select global batches before striding their members across ranks.

    ``batch_size`` is per rank. Only complete global batches are emitted;
    no indices are repeated to pad the tail. Each iterator holds one CPU
    int64 permutation tensor, converting only local batches to Python lists.
    The configuration digest checks rank agreement, not dataset contents.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int = 0,
        epoch: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.dataset_size = _integer(dataset_size, "dataset_size")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "seed")
        self.num_replicas = _integer(num_replicas, "num_replicas")
        self.rank = _integer(rank, "rank")
        if self.dataset_size < 0:
            raise ValueError("dataset_size must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        epoch = _integer(epoch, "epoch")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if self.seed + epoch >= 2**64:
            raise ValueError("seed + epoch must be less than 2**64")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dataset_size // (self.batch_size * self.num_replicas)

    def __iter__(self) -> Iterator[list[int]]:
        if len(self) == 0:
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(
            self.dataset_size, generator=generator, device="cpu", dtype=torch.int64
        )
        global_size = self.batch_size * self.num_replicas
        for start in range(0, len(self) * global_size, global_size):
            global_indices = permutation[start : start + global_size]
            yield global_indices[self.rank :: self.num_replicas].tolist()

    def contract_payload(self) -> dict[str, object]:
        return {
            "format": "mace-global-random-sampler-contract-v1",
            "sampler": "global_random",
            "dataset_rows": self.dataset_size,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "seed": self.seed,
            "epoch": self.epoch,
            "rank_batches": len(self),
            "dropped_rows": self.dataset_size % (self.batch_size * self.num_replicas),
            "tail_policy": "drop_incomplete_global_batch",
            "permutation": "torch-cpu-randperm-seed-plus-epoch-v1",
            "torch_version": torch.__version__,
        }

    def contract_digest(self) -> str:
        payload = json.dumps(
            self.contract_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
