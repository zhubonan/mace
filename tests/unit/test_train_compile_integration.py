import copy
import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close
from torch_ema import ExponentialMovingAverage

from mace.cli import run_train
from mace.tools import compile as mace_compile

train_tools = importlib.import_module("mace.tools.train")


class _Batch:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def to(self, device: torch.device):
        self.values = self.values.to(device)
        return self

    def to_dict(self):
        return {"values": self.values}


class _EnergyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float64))

    def forward(self, data, **kwargs):
        del kwargs
        return {"energy": self.weight * data["values"]}


class _ForwardOnly(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        object.__setattr__(self, "base_model", base_model)

    def forward(self, data):
        return self.base_model(data)

    def parameters(self, recurse=True):
        del recurse
        raise AssertionError("compiled forward wrapper must not own clipping")


class _EnergyLoss(torch.nn.Module):
    def forward(self, pred, ref):
        del ref
        return pred["energy"].square().mean()


class _Logger:
    def log(self, metrics):
        del metrics


def _assert_optimizer_states_close(left, right):
    for left_group, right_group in zip(left.param_groups, right.param_groups):
        for left_param, right_param in zip(left_group["params"], right_group["params"]):
            assert left.state[left_param].keys() == right.state[right_param].keys()
            for key, left_value in left.state[left_param].items():
                right_value = right.state[right_param][key]
                if torch.is_tensor(left_value):
                    assert_close(left_value, right_value)
                else:
                    assert left_value == right_value


def _call_public_train_with_invalid_compile_configuration(
    *, distributed=False, lbfgs=False, unsupported_output=None
):
    model = _EnergyModel()
    optimizer = (
        torch.optim.LBFGS(model.parameters())
        if lbfgs
        else torch.optim.Adam(model.parameters())
    )
    output_args = {
        "energy": True,
        "forces": True,
        "virials": False,
        "stress": False,
    }
    if unsupported_output is not None:
        output_args[unsupported_output] = True
    train_tools.train(
        model=model,
        loss_fn=_EnergyLoss(),
        train_loader=[],
        valid_loaders={},
        optimizer=optimizer,
        lr_scheduler=None,
        start_epoch=0,
        max_num_epochs=0,
        patience=1,
        checkpoint_handler=None,
        logger=_Logger(),
        eval_interval=1,
        output_args=output_args,
        device=torch.device("cpu"),
        log_errors="",
        distributed=distributed,
        compile_training=True,
    )


def test_take_step_uses_forward_wrapper_but_eager_model_owns_training_state():
    eager_model = _EnergyModel()
    candidate_model = copy.deepcopy(eager_model)
    eager_optimizer = torch.optim.Adam(eager_model.parameters(), lr=0.01, amsgrad=True)
    candidate_optimizer = torch.optim.Adam(
        candidate_model.parameters(), lr=0.01, amsgrad=True
    )
    eager_ema = ExponentialMovingAverage(eager_model.parameters(), decay=0.9)
    candidate_ema = ExponentialMovingAverage(candidate_model.parameters(), decay=0.9)
    loss_fn = _EnergyLoss()
    output_args = {"forces": False, "virials": False, "stress": False}

    train_tools.take_step(
        model=eager_model,
        loss_fn=loss_fn,
        batch=_Batch(torch.tensor([1.0, -2.0], dtype=torch.float64)),
        optimizer=eager_optimizer,
        ema=eager_ema,
        output_args=output_args,
        max_grad_norm=0.2,
        device=torch.device("cpu"),
    )
    train_tools.take_step(
        model=candidate_model,
        forward_model=_ForwardOnly(candidate_model),
        loss_fn=loss_fn,
        batch=_Batch(torch.tensor([1.0, -2.0], dtype=torch.float64)),
        optimizer=candidate_optimizer,
        ema=candidate_ema,
        output_args=output_args,
        max_grad_norm=0.2,
        device=torch.device("cpu"),
    )

    assert_close(eager_model.weight, candidate_model.weight)
    _assert_optimizer_states_close(eager_optimizer, candidate_optimizer)
    for eager_shadow, candidate_shadow in zip(
        eager_ema.shadow_params, candidate_ema.shadow_params
    ):
        assert_close(eager_shadow, candidate_shadow)


def test_train_one_epoch_compiles_first_real_batch_once_and_reuses_wrapper(
    monkeypatch,
):
    owner = _EnergyModel()
    optimizer = torch.optim.Adam(owner.parameters())
    wrapper = object()
    compile_calls = []
    step_calls = []

    def fake_compile(model, example_batch, output_args, mode, dynamic):
        compile_calls.append((model, example_batch, output_args, mode, dynamic))
        return wrapper

    def fake_take_step(**kwargs):
        step_calls.append(kwargs)
        return torch.tensor(0.0), {"loss": 0.0, "time": 0.0}

    monkeypatch.setattr(mace_compile, "compile_training_model", fake_compile)
    monkeypatch.setattr(train_tools, "take_step", fake_take_step)
    output_args = {"forces": True, "virials": False, "stress": False}

    cached = train_tools.train_one_epoch(
        model=owner,
        loss_fn=_EnergyLoss(),
        data_loader=[_Batch(torch.tensor([1.0])), _Batch(torch.tensor([2.0]))],
        optimizer=optimizer,
        epoch=0,
        output_args=output_args,
        max_grad_norm=None,
        ema=None,
        logger=_Logger(),
        device=torch.device("cpu"),
        distributed=False,
        compile_training=True,
        compiled_training_model=None,
    )
    cached = train_tools.train_one_epoch(
        model=owner,
        loss_fn=_EnergyLoss(),
        data_loader=[_Batch(torch.tensor([3.0]))],
        optimizer=optimizer,
        epoch=1,
        output_args=output_args,
        max_grad_norm=None,
        ema=None,
        logger=_Logger(),
        device=torch.device("cpu"),
        distributed=False,
        compile_training=True,
        compiled_training_model=cached,
    )

    assert cached is wrapper
    assert len(compile_calls) == 1
    assert compile_calls[0][0] is owner
    assert_close(compile_calls[0][1]["values"], torch.tensor([1.0]))
    assert all(call["model"] is owner for call in step_calls)
    assert all(call["forward_model"] is wrapper for call in step_calls)


@pytest.mark.parametrize("unsupported", ["distributed", "lbfgs"])
def test_compile_training_rejects_unsupported_modes_before_setup(unsupported):
    args = SimpleNamespace(compile_training=True, distributed=False, lbfgs=False)
    setattr(args, unsupported, True)

    with pytest.raises(ValueError, match=unsupported):
        run_train._validate_compile_training_args(args)


@pytest.mark.parametrize("unsupported", ["distributed", "lbfgs"])
def test_public_train_rejects_unsupported_compiled_modes(unsupported):
    with pytest.raises(ValueError, match=unsupported):
        _call_public_train_with_invalid_compile_configuration(
            distributed=unsupported == "distributed",
            lbfgs=unsupported == "lbfgs",
        )


@pytest.mark.parametrize(
    "unsupported_output",
    ["dipoles", "polarizabilities", "magforces", "hessian"],
)
def test_public_train_rejects_truthy_unsupported_compiled_outputs(
    unsupported_output,
):
    with pytest.raises(ValueError, match=unsupported_output):
        _call_public_train_with_invalid_compile_configuration(
            unsupported_output=unsupported_output
        )


@pytest.mark.parametrize("compile_training", [False, True])
def test_configure_model_disables_e3nn_codegen_only_for_compiled_training(
    monkeypatch, compile_training
):
    events = []
    args = SimpleNamespace(compile_training=compile_training)

    @contextmanager
    def fake_disable_e3nn_codegen():
        events.append("enter")
        yield
        events.append("exit")

    def fake_configure_model(*args):
        events.append("configure")
        assert args == (
            expected_args,
            "loader",
            "energies",
            "foundation",
            "heads",
            "z",
            "configs",
        )
        return "model", "outputs"

    monkeypatch.setattr(
        run_train.mace_compile,
        "disable_e3nn_codegen",
        fake_disable_e3nn_codegen,
    )
    monkeypatch.setattr(run_train, "configure_model", fake_configure_model)
    expected_args = args

    result = run_train._configure_model_for_training(
        args,
        "loader",
        "energies",
        "foundation",
        "heads",
        "z",
        "configs",
    )

    assert result == ("model", "outputs")
    assert events == (
        ["enter", "configure", "exit"] if compile_training else ["configure"]
    )


def test_configure_model_restores_e3nn_codegen_when_construction_raises(monkeypatch):
    optimization_defaults = {"jit_script_fx": True}

    monkeypatch.setattr(
        run_train.mace_compile,
        "get_optimization_defaults",
        lambda: dict(optimization_defaults),
    )
    monkeypatch.setattr(
        run_train.mace_compile,
        "set_optimization_defaults",
        lambda **kwargs: optimization_defaults.update(kwargs),
    )

    def raise_from_configure_model(*args):
        del args
        raise RuntimeError("synthetic construction failure")

    monkeypatch.setattr(run_train, "configure_model", raise_from_configure_model)

    with pytest.raises(RuntimeError, match="synthetic construction failure"):
        run_train._configure_model_for_training(
            SimpleNamespace(compile_training=True), "loader"
        )

    assert optimization_defaults["jit_script_fx"] is True
