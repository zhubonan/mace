###########################################################################################
# Distributed startup contract tests for edge-balanced OMat sampling
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from types import SimpleNamespace

import pytest
import torch

from mace.tools.train import _verify_distributed_sampler_contract


def test_sampler_contract_gate_is_inactive_for_non_distributed_or_other_samplers(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("unexpected collective"),
    )

    _verify_distributed_sampler_contract(
        SimpleNamespace(contract_digest=lambda: "a" * 64),
        distributed=False,
        device=torch.device("cpu"),
    )
    _verify_distributed_sampler_contract(
        SimpleNamespace(), distributed=True, device=torch.device("cpu")
    )


def test_sampler_contract_gate_accepts_identical_fixed_size_digests(monkeypatch):
    sampler = SimpleNamespace(contract_digest=lambda: "ab" * 32)

    def all_gather(outputs, local):
        assert local.dtype == torch.uint8
        assert local.numel() == 32
        for output in outputs:
            output.copy_(local)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)

    _verify_distributed_sampler_contract(
        sampler, distributed=True, device=torch.device("cpu")
    )


def test_sampler_contract_gate_fails_all_ranks_on_mismatch(monkeypatch):
    sampler = SimpleNamespace(contract_digest=lambda: "ab" * 32)

    def all_gather(outputs, local):
        outputs[0].copy_(local)
        outputs[1].fill_(0)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)

    with pytest.raises(RuntimeError, match="edge-sampler contract mismatch"):
        _verify_distributed_sampler_contract(
            sampler, distributed=True, device=torch.device("cpu")
        )
