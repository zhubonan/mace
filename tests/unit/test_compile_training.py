import warnings
from configparser import ConfigParser
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ase import build
from e3nn import o3
from packaging.requirements import Requirement
from torch.testing import assert_close
from torch_ema import ExponentialMovingAverage

from mace import data, modules, tools
from mace.cli.convert_e3nn_cueq import run as convert_e3nn_to_cueq
from mace.cli.convert_e3nn_hybrid import run as convert_e3nn_to_hybrid
from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig
from mace.tools import compile as mace_compile
from mace.tools import compile_training
from mace.tools import torch_geometric
from tests.helpers import OEQ_AVAILABLE


_TABLE = tools.AtomicNumberTable([6])
_CUTOFF = 5.0


def test_project_retains_legacy_runtime_contract():
    config = ConfigParser()
    config.read(Path(__file__).resolve().parents[2] / "setup.cfg")
    requirements = tuple(
        Requirement(line.strip())
        for line in config["options"]["install_requires"].splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    e3nn_requirement = next(
        requirement
        for requirement in requirements
        if requirement.name.lower() == "e3nn"
    )
    torch_requirement = next(
        requirement
        for requirement in requirements
        if requirement.name.lower() == "torch"
    )

    assert str(e3nn_requirement.specifier) == "==0.4.4"
    assert str(torch_requirement.specifier) == ">=1.12"


def _create_model(device: str = "cpu", enable_cueq: bool = False) -> torch.nn.Module:
    torch_geometric.seed_everything(1702)
    cueq_config = None
    if enable_cueq:
        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout="ir_mul",
            group="O3_e3nn",
            optimize_all=True,
            conv_fusion=device == "cuda",
        )
    with mace_compile.disable_e3nn_codegen():
        model = modules.ScaleShiftMACE(
            r_max=_CUTOFF,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=1,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            gate=F.silu,
            atomic_energies=np.array([1.0], dtype=float),
            avg_num_neighbors=8,
            atomic_numbers=_TABLE.zs,
            correlation=2,
            radial_type="bessel",
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            cueq_config=cueq_config,
        )
    return model.to(device)


def _create_single_interaction_model(
    interaction_first: str,
) -> torch.nn.Module:
    torch_geometric.seed_everything(1702)
    with mace_compile.disable_e3nn_codegen():
        model = modules.ScaleShiftMACE(
            r_max=_CUTOFF,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=3,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticDensityResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[interaction_first],
            num_interactions=1,
            num_elements=1,
            hidden_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
            MLP_irreps=o3.Irreps("4x0e"),
            gate=F.silu,
            atomic_energies=np.array([1.0], dtype=float),
            avg_num_neighbors=8,
            atomic_numbers=_TABLE.zs,
            correlation=3,
            readout_correlation=2,
            radial_type="bessel",
            use_last_readout_only=True,
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        )
    return model


def _create_direct_single_interaction_model() -> torch.nn.Module:
    torch_geometric.seed_everything(1702)
    with mace_compile.disable_e3nn_codegen():
        model = modules.ScaleShiftMACE(
            r_max=_CUTOFF,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=1,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticDensityResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticDensityInteractionBlock"
            ],
            num_interactions=1,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            gate=F.silu,
            atomic_energies=np.array([1.0], dtype=float),
            avg_num_neighbors=8,
            atomic_numbers=_TABLE.zs,
            correlation=3,
            radial_type="bessel",
            use_last_readout_only=False,
            direct_message_hidden_readout=True,
            direct_readout_rank=2,
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        )
    return model


def _create_batch(num_graphs: int = 2) -> dict[str, torch.Tensor]:
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    configs = [data.config_from_atoms(atoms) for _ in range(num_graphs)]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=_TABLE, cutoff=_CUTOFF)
            for config in configs
        ],
        batch_size=num_graphs,
        shuffle=False,
        drop_last=False,
    )
    return next(iter(loader)).to_dict()


@pytest.mark.parametrize("lmax", range(4))
def test_legacy_traceable_spherical_harmonics_preserves_e3nn_values(lmax):
    spherical_harmonics = o3.SphericalHarmonics(
        list(range(lmax + 1)), normalize=True, normalization="component"
    )
    vectors = torch.randn(17, 3, dtype=torch.float64)
    expected = spherical_harmonics(vectors)

    with compile_training._traceable_legacy_spherical_harmonics(spherical_harmonics):
        actual = spherical_harmonics(vectors)

    assert_close(actual, expected)
    assert "forward" not in spherical_harmonics.__dict__


def test_legacy_traceable_spherical_harmonics_restores_partial_patch_on_error():
    supported = o3.SphericalHarmonics(
        list(range(4)), normalize=True, normalization="component"
    )
    unsupported = o3.SphericalHarmonics(
        list(range(5)), normalize=True, normalization="component"
    )
    model = torch.nn.ModuleList((supported, unsupported))

    with pytest.raises(RuntimeError, match="through l=3"):
        with compile_training._traceable_legacy_spherical_harmonics(model):
            pytest.fail("unsupported model should fail before tracing")

    assert "forward" not in supported.__dict__
    assert "forward" not in unsupported.__dict__


def _batch_to(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.clone().to(device) for key, value in batch.items()}


def _objective(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        outputs["energy"].square().mean()
        + 0.3 * outputs["forces"].square().mean()
        + 0.2 * outputs["virials"].square().mean()
        + 0.1 * outputs["stress"].square().mean()
    )


def _assert_parameter_gradients_close(
    eager_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    eager_parameters = dict(eager_model.named_parameters())
    candidate_parameters = dict(candidate_model.named_parameters())
    assert eager_parameters.keys() == candidate_parameters.keys()
    for name, eager_parameter in eager_parameters.items():
        candidate_parameter = candidate_parameters[name]
        assert (eager_parameter.grad is None) == (candidate_parameter.grad is None)
        if eager_parameter.grad is not None:
            try:
                assert_close(
                    eager_parameter.grad,
                    candidate_parameter.grad,
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as error:
                raise AssertionError(
                    f"parameter gradient mismatch for {name}: {error}"
                ) from error


def _run_training_fx(
    fx_model: torch.fx.GraphModule,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    output_args: dict[str, bool],
) -> dict[str, torch.Tensor]:
    prepared = compile_training._prepare_batch(batch, output_args)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    flat_inputs = [prepared[key] for key in fx_model.meta["mace_input_keys"]]
    flat_inputs.extend(
        parameters[name] for name in fx_model.meta["mace_parameter_names"]
    )
    flat_inputs.extend(buffers[name] for name in fx_model.meta["mace_buffer_names"])
    flat_outputs = fx_model(*flat_inputs)
    return dict(zip(fx_model.meta["mace_output_keys"], flat_outputs))


def test_compiled_force_stress_training_matches_eager_gradients_and_state_dict(
    tmp_path,
):
    with tools.torch_tools.default_dtype(torch.float64):
        eager_model = _create_model()
        candidate_model = _create_model()
        eager_batch = _create_batch()
        candidate_batch = {key: value.clone() for key, value in eager_batch.items()}
        candidate_batch["energy"] = torch.zeros(2)
        candidate_batch["forces"] = torch.zeros_like(candidate_batch["positions"])
        candidate_batch["weight"] = torch.ones(2)
        candidate_batch["energy_weight"] = torch.ones(2)

        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
            "dipoles": False,
            "polarizabilities": False,
            "magforces": False,
        }
        compiled_model = mace_compile.compile_training_model(
            candidate_model,
            candidate_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=_create_batch(num_graphs=3),
        )
        assert compiled_model.is_compiled
        assert compiled_model.compilation_error is None
        assert dict(compiled_model.named_modules())["base_model"] is candidate_model
        candidate_parameters = dict(candidate_model.named_parameters())
        compiled_parameters = dict(compiled_model.named_parameters())
        assert compiled_parameters.keys() == candidate_parameters.keys()
        assert all(
            compiled_parameters[name] is parameter
            for name, parameter in candidate_parameters.items()
        )
        reducer_parameters = [
            parameter
            for _, module in compiled_model.named_modules()
            for _, parameter in module.named_parameters(recurse=False)
        ]
        assert len(reducer_parameters) == len({id(p) for p in reducer_parameters})
        assert {id(p) for p in reducer_parameters} == {
            id(p) for p in candidate_parameters.values()
        }
        assert {
            "energy",
            "forces",
            "weight",
            "energy_weight",
        }.isdisjoint(compiled_model.input_keys)

        eager_outputs = eager_model(
            eager_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        compiled_outputs = compiled_model(candidate_batch)

        for key in ("energy", "forces", "virials", "stress"):
            assert_close(eager_outputs[key], compiled_outputs[key])

        _objective(eager_outputs).backward()
        _objective(compiled_outputs).backward()
        _assert_parameter_gradients_close(eager_model, candidate_model)

        assert compiled_model.state_dict().keys() == eager_model.state_dict().keys()

        model_path = tmp_path / "compiled-training-final.model"
        batch_path = tmp_path / "inference-batch.pt"
        torch.save(deepcopy(candidate_model), model_path)
        torch.save(candidate_batch, batch_path)
        legacy_python = os.environ.get("MACE_LEGACY_PYTHON", sys.executable)
        subprocess.run(
            [
                legacy_python,
                "-c",
                """
import sys

import e3nn
import torch

model = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
batch = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
assert e3nn.__version__ == "0.4.4"
assert type(model).__module__ == "mace.modules.models"
assert "CompiledTrainingModel" not in repr(model)
outputs = model(
    batch,
    training=False,
    compute_force=True,
    compute_virials=True,
    compute_stress=True,
)
for key in ("energy", "forces", "virials", "stress"):
    assert torch.isfinite(outputs[key]).all(), key
assert all(
    not key.startswith(("module.", "base_model."))
    for key in model.state_dict()
)
""",
                str(model_path),
                str(batch_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def test_compiled_training_can_disable_compilation_without_losing_base_state():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch()
        compiled_model = mace_compile.compile_training_model(
            model,
            batch,
            output_args={
                "energy": True,
                "forces": True,
                "virials": False,
                "stress": False,
            },
            validation_batch=_create_batch(num_graphs=3),
        )
        parameter_ids = [id(parameter) for parameter in model.parameters()]
        state_keys = model.state_dict().keys()

        compiled_model.disable_compilation("distributed rank fallback")

        assert not compiled_model.is_compiled
        assert compiled_model.compilation_error == "distributed rank fallback"
        assert [
            id(parameter) for parameter in compiled_model.parameters()
        ] == parameter_ids
        assert compiled_model.state_dict().keys() == state_keys


def test_compiled_training_lazily_falls_back_after_backend_failure():
    class FailingBackend:
        def __call__(self, *args):
            del args
            raise RuntimeError("synthetic lazy backend failure")

    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch()
        output_args = {
            "energy": True,
            "forces": True,
            "virials": False,
            "stress": False,
        }
        compiled_model = mace_compile.compile_training_model(
            model,
            batch,
            output_args=output_args,
            validation_batch=_create_batch(num_graphs=3),
        )
        assert compiled_model.is_compiled
        compiled_model._compiled_ref = (FailingBackend(),)  # pylint: disable=W0212

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = compiled_model(batch)
            second = compiled_model(batch)

        assert_close(first["energy"], second["energy"])
        assert not compiled_model.is_compiled
        assert compiled_model.compilation_error == (
            "RuntimeError: synthetic lazy backend failure"
        )
        assert not isinstance(compiled_model.compilation_error, BaseException)
        runtime_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 1
        assert "synthetic lazy backend failure" in str(runtime_warnings[0].message)


def test_compiled_training_materializes_backward_before_ddp_and_falls_back():
    class BackwardFailure(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            del ctx
            return value.sum().reshape(1)

        @staticmethod
        def backward(ctx, grad_output):
            del ctx, grad_output
            raise RuntimeError("synthetic lazy backward failure")

    class BackwardFailingBackend:
        def __call__(self, *args):
            parameters = [value for value in args if value.requires_grad]
            connected_value = sum(parameter.sum() for parameter in parameters)
            return (BackwardFailure.apply(connected_value),)

    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch()
        compiled_model = mace_compile.compile_training_model(
            model,
            batch,
            output_args={
                "energy": True,
                "forces": False,
                "virials": False,
                "stress": False,
            },
            validation_batch=_create_batch(num_graphs=3),
        )
        assert compiled_model.is_compiled
        compiled_model._compiled_ref = (  # pylint: disable=W0212
            BackwardFailingBackend(),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compiled_model.materialize_compiled_backward(batch)

        assert not compiled_model.is_compiled
        assert compiled_model.compilation_error == (
            "RuntimeError: synthetic lazy backward failure"
        )
        assert all(parameter.grad is None for parameter in model.parameters())
        runtime_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 1
        assert "synthetic lazy backward failure" in str(runtime_warnings[0].message)


def test_training_fx_prunes_unused_live_state_and_audits_owned_constants():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch()
        fx_model = compile_training.make_training_fx(
            model,
            batch,
            output_args={
                "energy": True,
                "forces": True,
                "virials": True,
                "stress": True,
            },
        )

        assert not dict(fx_model.named_parameters())
        assert not dict(fx_model.named_buffers())
        assert fx_model.meta["mace_architecture_constants"] == ()
        assert set(fx_model.meta["mace_parameter_names"]) <= set(
            dict(model.named_parameters())
        )
        assert set(fx_model.meta["mace_buffer_names"]) <= set(
            dict(model.named_buffers())
        )
        assert all(
            node.users for node in fx_model.graph.nodes if node.op == "placeholder"
        )


def test_dynamic_compiled_training_requires_validation_batch():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            facade = mace_compile.compile_training_model(
                model,
                _create_batch(num_graphs=2),
                output_args={"energy": True, "forces": True},
                dynamic=True,
            )

        assert not facade.is_compiled
        assert "second validation batch" in facade.compilation_error
        runtime_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("products.0.symmetric_contractions.U_matrix_3", True),
        ("node_embedding.linear.f.m.graphs.0.graph.c0", True),
        ("atomic_numbers", False),
        ("radial_embedding.bessel_fn.bessel_weights", False),
    ],
)
def test_immutable_architecture_buffer_classification(name, expected):
    assert compile_training._is_immutable_architecture_buffer(name) is expected


def _constant_graph(value: torch.Tensor) -> torch.fx.GraphModule:
    root = torch.nn.Module()
    root.register_buffer("_tensor_constant0", value)
    graph = torch.fx.Graph()
    constant = graph.get_attr("_tensor_constant0")
    graph.output(constant)
    return torch.fx.GraphModule(root, graph)


def _buffer_model(**buffers: torch.Tensor) -> torch.nn.Module:
    model = torch.nn.Module()
    block = torch.nn.Module()
    model.add_module("block", block)
    for name, value in buffers.items():
        block.register_buffer(name, value)
    return model


def _generated_constant_model(**buffers: torch.Tensor) -> torch.nn.Module:
    model = torch.nn.Module()
    block = torch.nn.Module()
    graphs = torch.nn.Module()
    graph_zero = torch.nn.Module()
    graph = torch.nn.Module()
    model.add_module("block", block)
    block.add_module("graphs", graphs)
    graphs.add_module("0", graph_zero)
    graph_zero.add_module("graph", graph)
    for name, value in buffers.items():
        graph.register_buffer(name, value)
    return model


def test_architecture_constant_classification_rejects_immutable_ambiguity():
    value = torch.arange(6.0).reshape(3, 2)
    fx_model = _constant_graph(value)
    model = _buffer_model(
        U_matrix_1=value.T.clone(),
        U_matrix_2=value.T.clone(),
    )

    with pytest.raises(RuntimeError, match="ambiguous immutable sources"):
        compile_training._classify_architecture_constants(fx_model, model)


def test_generated_constant_provenance_selects_exact_source_index():
    value = torch.tensor(1.0)
    fx_model = _constant_graph(value)
    model = _generated_constant_model(
        c0=value.clone(),
        c1=value.clone(),
    )
    trace_model = _generated_constant_model(_tensor_constant1=value)

    assert compile_training._classify_architecture_constants(
        fx_model, model, trace_model
    ) == (("_tensor_constant0", "block.graphs.0.graph.c1"),)


def test_constant_provenance_ignores_unrelated_mutable_equal_value():
    value = torch.tensor(1.0)
    fx_model = _constant_graph(value)
    model = _generated_constant_model(c0=value.clone())
    model.register_buffer("scale", value.clone())
    trace_model = _generated_constant_model(_tensor_constant0=value)

    assert compile_training._classify_architecture_constants(
        fx_model, model, trace_model
    ) == (("_tensor_constant0", "block.graphs.0.graph.c0"),)


def test_architecture_constant_classification_rejects_mutable_collision():
    value = torch.arange(6.0).reshape(3, 2)
    fx_model = _constant_graph(value)
    model = _buffer_model(
        U_matrix_1=value.T.clone(),
        mutable=value.clone(),
    )

    with pytest.raises(RuntimeError, match="mutable model buffers=.*block.mutable"):
        compile_training._classify_architecture_constants(fx_model, model)


def test_unused_trainable_parameter_placeholder_is_rejected():
    graph = torch.fx.Graph()
    input_node = graph.placeholder("input")
    parameter_node = graph.placeholder("parameter")
    parameter_node.meta["val"] = torch.ones(1, requires_grad=True)
    graph.output(input_node)
    fx_model = torch.fx.GraphModule(torch.nn.Module(), graph)

    with pytest.raises(RuntimeError, match="unused trainable parameter.*weight"):
        compile_training._prune_unused_placeholders(
            fx_model,
            input_keys=("positions",),
            parameter_names=("weight",),
            buffer_names=(),
        )


def test_unused_differentiable_buffer_placeholder_is_rejected():
    graph = torch.fx.Graph()
    input_node = graph.placeholder("input")
    buffer_node = graph.placeholder("buffer")
    buffer_node.meta["val"] = torch.ones(1, requires_grad=True)
    graph.output(input_node)
    fx_model = torch.fx.GraphModule(torch.nn.Module(), graph)

    with pytest.raises(RuntimeError, match="unused differentiable buffer.*state"):
        compile_training._prune_unused_placeholders(
            fx_model,
            input_keys=("positions",),
            parameter_names=(),
            buffer_names=("state",),
        )


def test_training_fx_preserves_noncanonical_node_attributes():
    with tools.torch_tools.default_dtype(torch.float64):
        eager_model = _create_model()
        candidate_model = _create_model()
        trace_batch = _create_batch()
        runtime_batch = {
            key: value.detach().clone() for key, value in trace_batch.items()
        }
        runtime_batch["node_attrs"].fill_(0.5)

        fx_model = compile_training.make_training_fx(
            candidate_model,
            trace_batch,
            output_args={"energy": True},
        )
        assert "node_attrs" in fx_model.meta["mace_input_keys"]

        runtime_batch["positions"].requires_grad_(True)
        parameters = dict(candidate_model.named_parameters())
        buffers = dict(candidate_model.named_buffers())
        flat_inputs = [runtime_batch[key] for key in fx_model.meta["mace_input_keys"]]
        flat_inputs.extend(
            parameters[name] for name in fx_model.meta["mace_parameter_names"]
        )
        flat_inputs.extend(buffers[name] for name in fx_model.meta["mace_buffer_names"])

        eager_outputs = eager_model(
            runtime_batch,
            training=True,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
        )
        (candidate_energy,) = fx_model(*flat_inputs)
        assert_close(eager_outputs["energy"], candidate_energy)


def test_make_training_fx_does_not_mutate_source_model():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch(num_graphs=2)
        state_before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        modules_before = {name: id(module) for name, module in model.named_modules()}

        compile_training.make_training_fx(
            model,
            batch,
            output_args={
                "energy": True,
                "forces": True,
                "virials": True,
                "stress": True,
            },
        )

        assert modules_before == {
            name: id(module) for name, module in model.named_modules()
        }
        for name, value in model.state_dict().items():
            assert_close(value, state_before[name])


@pytest.mark.parametrize(
    "interaction_first",
    (
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
    ),
)
def test_single_interaction_invariant_readout_lowers_for_training(
    interaction_first,
):
    with tools.torch_tools.default_dtype(torch.float64):
        eager_model = _create_single_interaction_model(interaction_first)
        candidate_model = _create_single_interaction_model(interaction_first)
        eager_batch = _create_batch(num_graphs=2)
        candidate_batch = {
            key: value.clone() for key, value in eager_batch.items()
        }
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }

        fx_model = compile_training.make_training_fx(
            candidate_model,
            candidate_batch,
            output_args=output_args,
            validation_batch=_create_batch(num_graphs=3),
        )
        eager_outputs = eager_model(
            eager_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        candidate_outputs = _run_training_fx(
            fx_model,
            candidate_model,
            candidate_batch,
            output_args,
        )

        assert fx_model.meta["mace_output_keys"] == (
            "energy",
            "forces",
            "virials",
            "stress",
        )
        for key in fx_model.meta["mace_output_keys"]:
            assert_close(eager_outputs[key], candidate_outputs[key])


def test_compiled_training_direct_readout_performs_optimizer_step():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_direct_single_interaction_model()
        batch = _create_batch(num_graphs=2)
        compiled_model = mace_compile.compile_training_model(
            model,
            batch,
            output_args={
                "energy": True,
                "forces": True,
                "virials": False,
                "stress": False,
            },
            dynamic=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

        optimizer.zero_grad()
        outputs = compiled_model(batch)
        loss = outputs["energy"].square().mean() + outputs["forces"].square().mean()
        loss.backward()

        assert torch.isfinite(loss)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()


def test_training_fx_accepts_two_different_symbolic_shapes():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        first = _create_batch(num_graphs=2)
        second = _create_batch(num_graphs=3)

        fx_model = compile_training.make_training_fx(
            model,
            first,
            output_args={
                "energy": True,
                "forces": True,
                "virials": True,
                "stress": True,
            },
            validation_batch=second,
        )

        assert fx_model.meta["mace_dynamic_shape_signatures"] == (
            compile_training._batch_shape_signature(first),
            compile_training._batch_shape_signature(second),
        )


def test_training_fx_rejects_identical_shape_probe():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        first = _create_batch(num_graphs=2)
        second = {key: value.clone() for key, value in first.items()}

        with pytest.raises(RuntimeError, match="different graph-derived dimensions"):
            compile_training.make_training_fx(
                model,
                first,
                output_args={"energy": True, "forces": True},
                validation_batch=second,
            )


def _expand_graph(size: int) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    expanded = graph.call_function(
        torch.ops.aten.expand.default,
        args=(value, [size]),
    )
    graph.output(expanded)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_training_fx_reports_first_fx_divergence(monkeypatch):
    first_fx = _expand_graph(16)
    second_fx = _expand_graph(24)
    lowered = iter((first_fx, second_fx))
    monkeypatch.setattr(
        compile_training,
        "_lower_training_fx",
        lambda model, batch, output_args, **kwargs: next(lowered),
    )
    first = _create_batch(num_graphs=2)
    second = _create_batch(num_graphs=3)

    with pytest.raises(RuntimeError) as exc_info:
        compile_training.make_training_fx(
            _create_model(),
            first,
            output_args={"energy": True},
            validation_batch=second,
        )

    message = str(exc_info.value)
    assert "first FX divergence" in message
    assert str(compile_training._batch_shape_signature(first)) in message
    assert str(compile_training._batch_shape_signature(second)) in message
    assert "[16]" in message
    assert "[24]" in message


def test_compiled_force_stress_training_keeps_graph_count_dynamic():
    with tools.torch_tools.default_dtype(torch.float64):
        eager_model = _create_model()
        candidate_model = _create_model()
        trace_batch = _create_batch(num_graphs=2)
        runtime_batch = _create_batch(num_graphs=3)
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }
        compiled_model = mace_compile.compile_training_model(
            candidate_model,
            trace_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=runtime_batch,
        )
        assert compiled_model.is_compiled
        assert compiled_model.dynamic_shape_signatures == (
            compile_training._batch_shape_signature(trace_batch),
            compile_training._batch_shape_signature(runtime_batch),
        )

        eager_outputs = eager_model(
            {key: value.clone() for key, value in runtime_batch.items()},
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        compiled_outputs = compiled_model(runtime_batch)

        assert compiled_model.is_compiled
        assert compiled_model.compilation_error is None
        for key in ("energy", "forces", "virials", "stress"):
            assert_close(eager_outputs[key], compiled_outputs[key])

        _objective(eager_outputs).backward()
        _objective(compiled_outputs).backward()
        _assert_parameter_gradients_close(eager_model, candidate_model)


def test_compiled_training_rejects_truthy_unsupported_output():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch()

        with pytest.raises(ValueError, match="dipoles"):
            mace_compile.compile_training_model(
                model,
                batch,
                output_args={"energy": True, "forces": True, "dipoles": True},
            )


def test_missing_runtime_graph_input_does_not_disable_compiled_variant():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch()
        compiled_model = mace_compile.compile_training_model(
            model,
            batch,
            output_args={"energy": True, "forces": True},
            mode="default",
            dynamic=True,
            validation_batch=_create_batch(num_graphs=3),
        )
        assert compiled_model.is_compiled
        incomplete_batch = dict(batch)
        del incomplete_batch["edge_index"]

        with pytest.raises(KeyError, match="edge_index"):
            compiled_model(incomplete_batch)

        assert compiled_model.is_compiled
        assert compiled_model.compilation_error is None


def test_internal_compiled_training_accepts_private_edge_derivative_mode():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        trace_batch = _create_batch()
        validation_batch = _create_batch(num_graphs=3)
        edge_compiled = compile_training.compile_training_model(
            model,
            trace_batch,
            output_args={
                "energy": True,
                "forces": True,
                "virials": True,
                "stress": True,
            },
            mode="default",
            dynamic=True,
            validation_batch=validation_batch,
            derivative_mode="edges",
        )

        assert edge_compiled.is_compiled, edge_compiled.compilation_error


def test_internal_compiled_training_rejects_unknown_derivative_mode():
    with tools.torch_tools.default_dtype(torch.float64):
        with pytest.raises(ValueError, match="derivative_mode"):
            compile_training.compile_training_model(
                _create_model(),
                _create_batch(),
                output_args={"energy": True, "forces": True},
                dynamic=False,
                derivative_mode="unknown",
            )


def test_public_compiled_training_api_does_not_expose_edge_derivative_mode():
    with pytest.raises(TypeError, match="derivative_mode"):
        mace_compile.compile_training_model(
            _create_model(),
            _create_batch(),
            output_args={"energy": True, "forces": True},
            derivative_mode="edges",
        )


def test_default_internal_compiled_training_remains_position_based():
    with tools.torch_tools.default_dtype(torch.float64):
        implicit = compile_training.make_training_fx(
            _create_model(),
            _create_batch(),
            output_args={"energy": True, "forces": True},
        )
        explicit = compile_training.make_training_fx(
            _create_model(),
            _create_batch(),
            output_args={"energy": True, "forces": True},
            derivative_mode="positions",
        )

        assert implicit.code == explicit.code


def test_edge_derivative_facade_fallback_uses_canonical_eager_model():
    with tools.torch_tools.default_dtype(torch.float64):
        model = _create_model()
        batch = _create_batch()
        facade = compile_training.compile_training_model(
            model,
            batch,
            output_args={
                "energy": True,
                "forces": True,
                "virials": True,
                "stress": True,
            },
            dynamic=True,
            validation_batch=_create_batch(num_graphs=3),
            derivative_mode="edges",
        )
        assert facade.is_compiled, facade.compilation_error
        reference = model(
            _batch_to(batch, "cpu"),
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )

        facade.disable_compilation("test fallback")
        actual = facade(_batch_to(batch, "cpu"))

        assert not facade.is_compiled
        for key in ("energy", "forces", "virials", "stress"):
            assert_close(reference[key], actual[key])
        assert facade.state_dict().keys() == model.state_dict().keys()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "enable_cueq", [False, pytest.param(True, marks=pytest.mark.cueq)]
)
def test_compiled_force_stress_training_cuda_optimizer_and_ema_parity(enable_cueq):
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")
    if enable_cueq and not CUET_AVAILABLE:
        pytest.skip("cuequivariance is not installed")

    with tools.torch_tools.default_dtype(torch.float64):
        eager_model = _create_model(device="cuda", enable_cueq=enable_cueq)
        candidate_model = _create_model(device="cuda", enable_cueq=enable_cueq)
        eager_batch = _batch_to(_create_batch(), "cuda")
        candidate_batch = _batch_to(eager_batch, "cuda")
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }
        compiled_model = mace_compile.compile_training_model(
            candidate_model,
            candidate_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=_batch_to(_create_batch(num_graphs=3), "cuda"),
        )
        assert compiled_model.is_compiled

        eager_optimizer = torch.optim.Adam(
            eager_model.parameters(), lr=1.0e-3, amsgrad=True
        )
        candidate_optimizer = torch.optim.Adam(
            candidate_model.parameters(), lr=1.0e-3, amsgrad=True
        )
        eager_ema = ExponentialMovingAverage(eager_model.parameters(), decay=0.9)
        candidate_ema = ExponentialMovingAverage(
            candidate_model.parameters(), decay=0.9
        )

        eager_outputs = eager_model(
            eager_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        compiled_outputs = compiled_model(candidate_batch)
        assert compiled_model.is_compiled
        for key in ("energy", "forces", "virials", "stress"):
            assert_close(eager_outputs[key], compiled_outputs[key])

        _objective(eager_outputs).backward()
        _objective(compiled_outputs).backward()
        for eager_parameter, candidate_parameter in zip(
            eager_model.parameters(), candidate_model.parameters()
        ):
            assert (eager_parameter.grad is None) == (candidate_parameter.grad is None)
            if eager_parameter.grad is not None:
                assert_close(eager_parameter.grad, candidate_parameter.grad)

        eager_optimizer.step()
        candidate_optimizer.step()
        eager_ema.update()
        candidate_ema.update()
        for eager_parameter, candidate_parameter in zip(
            eager_model.parameters(), candidate_model.parameters()
        ):
            assert_close(eager_parameter, candidate_parameter)
        for eager_shadow, candidate_shadow in zip(
            eager_ema.shadow_params, candidate_ema.shadow_params
        ):
            assert_close(eager_shadow, candidate_shadow)


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.oeq
def test_hybrid_oeq_runtime_tensors_are_nonpersistent_buffers_cuda():
    if not torch.cuda.is_available() or not CUET_AVAILABLE or not OEQ_AVAILABLE:
        pytest.skip("hybrid CUDA dependencies are unavailable")

    with tools.torch_tools.default_dtype(torch.float32):
        hybrid = convert_e3nn_to_hybrid(
            _create_model(), device="cuda", return_model=True
        ).cuda()

    state_keys = set(hybrid.state_dict())
    oeq_modules = [
        (name, module)
        for name, module in hybrid.named_modules()
        if module.__class__.__module__.startswith("openequivariance.")
        and module.__class__.__name__ == "TensorProductConv"
    ]
    assert oeq_modules
    for module_name, module in oeq_modules:
        apply_method = module._apply
        buffers = dict(module.named_buffers(recurse=False))
        for buffer_name in ("kernel", "workspace_buffer", "dummy_transpose_perm"):
            assert buffers[buffer_name] is getattr(module, buffer_name)
            assert buffer_name in module._non_persistent_buffers_set
            assert f"{module_name}.{buffer_name}" not in state_keys
        assert buffers["kernel"].device.type == "cpu"
        assert buffers["workspace_buffer"].device.type == "cuda"
        assert buffers["dummy_transpose_perm"].device.type == "cuda"
        module.cuda()
        assert module._apply.__func__ is apply_method.__func__
        assert module.kernel.device.type == "cpu"
        assert module.workspace_buffer.device.type == "cuda"
        assert module.dummy_transpose_perm.device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.oeq
def test_hybrid_full_force_stress_optimizer_parity_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")
    if not CUET_AVAILABLE or not OEQ_AVAILABLE:
        pytest.skip("hybrid dependencies are not installed")

    with tools.torch_tools.default_dtype(torch.float32):
        source = _create_model()
        baseline = convert_e3nn_to_cueq(
            deepcopy(source), device="cuda", conv_fusion=False
        ).cuda()
        hybrid = convert_e3nn_to_hybrid(
            deepcopy(source), device="cuda", return_model=True
        ).cuda()
        baseline_batch = _batch_to(_create_batch(), "cuda")
        hybrid_batch = _batch_to(baseline_batch, "cuda")
        baseline_optimizer = torch.optim.Adam(
            baseline.parameters(), lr=1.0e-3, amsgrad=True
        )
        hybrid_optimizer = torch.optim.Adam(
            hybrid.parameters(), lr=1.0e-3, amsgrad=True
        )

        baseline_outputs = baseline(
            baseline_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        hybrid_outputs = hybrid(
            hybrid_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        for key in ("energy", "forces", "virials", "stress"):
            assert_close(baseline_outputs[key], hybrid_outputs[key])

        _objective(baseline_outputs).backward()
        _objective(hybrid_outputs).backward()
        _assert_parameter_gradients_close(baseline, hybrid)
        for name, parameter in baseline.named_parameters():
            assert parameter.grad is not None, name
        baseline_optimizer.step()
        hybrid_optimizer.step()
        baseline_parameters = dict(baseline.named_parameters())
        hybrid_parameters = dict(hybrid.named_parameters())
        assert baseline_parameters.keys() == hybrid_parameters.keys()
        for name, baseline_parameter in baseline_parameters.items():
            hybrid_parameter = hybrid_parameters[name]
            assert_close(baseline_parameter, hybrid_parameter)
            baseline_state = baseline_optimizer.state[baseline_parameter]
            hybrid_state = hybrid_optimizer.state[hybrid_parameter]
            assert baseline_state.keys() == hybrid_state.keys()
            for state_name, baseline_value in baseline_state.items():
                hybrid_value = hybrid_state[state_name]
                if isinstance(baseline_value, torch.Tensor):
                    assert_close(baseline_value, hybrid_value)
                else:
                    assert baseline_value == hybrid_value


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.oeq
def test_hybrid_compiled_force_stress_dynamic_update_parity_cuda():
    if not torch.cuda.is_available() or not CUET_AVAILABLE or not OEQ_AVAILABLE:
        pytest.skip("hybrid CUDA dependencies are unavailable")

    with tools.torch_tools.default_dtype(torch.float32):
        source = _create_model()
        reference = convert_e3nn_to_hybrid(
            deepcopy(source), device="cuda", return_model=True
        ).cuda()
        candidate = convert_e3nn_to_hybrid(
            deepcopy(source), device="cuda", return_model=True
        ).cuda()
        trace_batch = _batch_to(_create_batch(num_graphs=2), "cuda")
        validation_batch = _batch_to(_create_batch(num_graphs=3), "cuda")
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }
        compiled = mace_compile.compile_training_model(
            candidate,
            trace_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=validation_batch,
        )
        assert compiled.is_compiled, compiled.compilation_error

        for raw_batch in (trace_batch, validation_batch):
            reference_batch = _batch_to(raw_batch, "cuda")
            candidate_batch = _batch_to(raw_batch, "cuda")
            reference_outputs = reference(
                reference_batch,
                training=True,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
            )
            candidate_outputs = compiled(candidate_batch)
            for key in ("energy", "forces", "virials", "stress"):
                assert_close(reference_outputs[key], candidate_outputs[key])
            _objective(reference_outputs).backward()
            _objective(candidate_outputs).backward()
            _assert_parameter_gradients_close(
                reference,
                candidate,
                rtol=1.0e-5,
                atol=2.0e-5,
            )
            for model in (reference, candidate):
                model.zero_grad(set_to_none=True)
            assert compiled.is_compiled, compiled.compilation_error

        graphs_after_dynamic_shapes = torch._dynamo.utils.counters["stats"][
            "unique_graphs"
        ]
        compiled(_batch_to(trace_batch, "cuda"))
        compiled(_batch_to(validation_batch, "cuda"))
        assert (
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
            == graphs_after_dynamic_shapes
        )

        reference_optimizer = torch.optim.Adam(
            reference.parameters(), lr=1.0e-3, amsgrad=True
        )
        candidate_optimizer = torch.optim.Adam(
            candidate.parameters(), lr=1.0e-3, amsgrad=True
        )
        reference_batch = _batch_to(validation_batch, "cuda")
        candidate_batch = _batch_to(validation_batch, "cuda")
        reference_outputs = reference(
            reference_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        candidate_outputs = compiled(candidate_batch)
        reference_loss = _objective(reference_outputs)
        candidate_loss = _objective(candidate_outputs)
        assert_close(reference_loss, candidate_loss)
        reference_loss.backward()
        candidate_loss.backward()
        _assert_parameter_gradients_close(
            reference,
            candidate,
            rtol=1.0e-5,
            atol=2.0e-5,
        )
        reference_optimizer.step()
        candidate_optimizer.step()

        reference_parameters = dict(reference.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        assert reference_parameters.keys() == candidate_parameters.keys()
        for name, reference_parameter in reference_parameters.items():
            candidate_parameter = candidate_parameters[name]
            assert_close(reference_parameter, candidate_parameter)
            reference_state = reference_optimizer.state[reference_parameter]
            candidate_state = candidate_optimizer.state[candidate_parameter]
            assert reference_state.keys() == candidate_state.keys()
            for state_name, reference_value in reference_state.items():
                candidate_value = candidate_state[state_name]
                if isinstance(reference_value, torch.Tensor):
                    assert_close(reference_value, candidate_value)
                else:
                    assert reference_value == candidate_value
        assert compiled.is_compiled, compiled.compilation_error


@pytest.mark.gpu
@pytest.mark.parametrize(
    "enable_cueq", [False, pytest.param(True, marks=pytest.mark.cueq)]
)
def test_edge_derivative_compiled_cuda_optimizer_and_ema_parity(enable_cueq):
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")
    if enable_cueq and not CUET_AVAILABLE:
        pytest.skip("cuequivariance is not installed")

    with tools.torch_tools.default_dtype(torch.float32):
        reference = _create_model().cuda()
        candidate = _create_model().cuda()
        if enable_cueq:
            reference = convert_e3nn_to_cueq(
                reference, device="cuda", conv_fusion=False
            ).cuda()
            candidate = convert_e3nn_to_cueq(
                candidate, device="cuda", conv_fusion=False
            ).cuda()
        trace_batch = _batch_to(_create_batch(num_graphs=2), "cuda")
        validation_batch = _batch_to(_create_batch(num_graphs=3), "cuda")
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }
        facade = compile_training.compile_training_model(
            candidate,
            trace_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=validation_batch,
            derivative_mode="edges",
        )
        assert facade.is_compiled, facade.compilation_error
        reference_optimizer = torch.optim.Adam(
            reference.parameters(), lr=1.0e-3, amsgrad=True
        )
        candidate_optimizer = torch.optim.Adam(
            candidate.parameters(), lr=1.0e-3, amsgrad=True
        )
        reference_ema = ExponentialMovingAverage(reference.parameters(), decay=0.9)
        candidate_ema = ExponentialMovingAverage(candidate.parameters(), decay=0.9)
        for raw_batch in (trace_batch, validation_batch):
            reference_batch = _batch_to(raw_batch, "cuda")
            candidate_batch = _batch_to(raw_batch, "cuda")
            reference_outputs = reference(
                reference_batch,
                training=True,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
            )
            candidate_outputs = facade(candidate_batch)
            assert facade.is_compiled, facade.compilation_error
            for key in ("energy", "forces", "virials", "stress"):
                assert_close(reference_outputs[key], candidate_outputs[key])
            reference_loss = _objective(reference_outputs)
            candidate_loss = _objective(candidate_outputs)
            assert_close(reference_loss, candidate_loss)
            reference_loss.backward()
            candidate_loss.backward()
            _assert_parameter_gradients_close(
                reference,
                candidate,
                rtol=1.0e-5,
                atol=2.0e-5,
            )
            for model in (reference, candidate):
                model.zero_grad(set_to_none=True)

        graphs_after_dynamic_shapes = torch._dynamo.utils.counters["stats"][
            "unique_graphs"
        ]
        facade(_batch_to(trace_batch, "cuda"))
        facade(_batch_to(validation_batch, "cuda"))
        assert (
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
            == graphs_after_dynamic_shapes
        )
        reference_outputs = reference(
            _batch_to(validation_batch, "cuda"),
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        candidate_outputs = facade(_batch_to(validation_batch, "cuda"))
        _objective(reference_outputs).backward()
        _objective(candidate_outputs).backward()
        _assert_parameter_gradients_close(
            reference,
            candidate,
            rtol=1.0e-5,
            atol=2.0e-5,
        )
        reference_optimizer.step()
        candidate_optimizer.step()
        reference_ema.update()
        candidate_ema.update()

        reference_parameters = dict(reference.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        assert reference_parameters.keys() == candidate_parameters.keys()
        for name, reference_parameter in reference_parameters.items():
            candidate_parameter = candidate_parameters[name]
            assert_close(reference_parameter, candidate_parameter)
            reference_state = reference_optimizer.state[reference_parameter]
            candidate_state = candidate_optimizer.state[candidate_parameter]
            assert reference_state.keys() == candidate_state.keys()
            for state_name, reference_value in reference_state.items():
                candidate_value = candidate_state[state_name]
                if isinstance(reference_value, torch.Tensor):
                    assert_close(reference_value, candidate_value)
                else:
                    assert reference_value == candidate_value
        for reference_shadow, candidate_shadow in zip(
            reference_ema.shadow_params, candidate_ema.shadow_params
        ):
            assert_close(reference_shadow, candidate_shadow)
        assert facade.state_dict().keys() == candidate.state_dict().keys()
        assert facade.is_compiled, facade.compilation_error


@pytest.mark.gpu
def test_edge_derivative_uncompiled_fx_gradient_parity_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")

    with tools.torch_tools.default_dtype(torch.float32):
        reference = _create_model().cuda()
        candidate = _create_model().cuda()
        trace_batch = _batch_to(_create_batch(num_graphs=2), "cuda")
        validation_batch = _batch_to(_create_batch(num_graphs=3), "cuda")
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }
        fx_model = compile_training.make_training_fx(
            candidate,
            trace_batch,
            output_args,
            validation_batch=validation_batch,
            derivative_mode="edges",
        )
        reference_outputs = reference(
            _batch_to(trace_batch, "cuda"),
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        candidate_outputs = _run_training_fx(
            fx_model,
            candidate,
            _batch_to(trace_batch, "cuda"),
            output_args,
        )
        for key in ("energy", "forces", "virials", "stress"):
            assert_close(reference_outputs[key], candidate_outputs[key])
        _objective(reference_outputs).backward()
        _objective(candidate_outputs).backward()
        _assert_parameter_gradients_close(reference, candidate)


@pytest.mark.gpu
def test_edge_derivative_compiled_cuda_matches_position_compiled_gradients():
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")

    with tools.torch_tools.default_dtype(torch.float32):
        position_model = _create_model().cuda()
        edge_model = _create_model().cuda()
        trace_batch = _batch_to(_create_batch(num_graphs=2), "cuda")
        validation_batch = _batch_to(_create_batch(num_graphs=3), "cuda")
        output_args = {
            "energy": True,
            "forces": True,
            "virials": True,
            "stress": True,
        }
        position_facade = compile_training.compile_training_model(
            position_model,
            trace_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=validation_batch,
            derivative_mode="positions",
        )
        edge_facade = compile_training.compile_training_model(
            edge_model,
            trace_batch,
            output_args=output_args,
            mode="default",
            dynamic=True,
            validation_batch=validation_batch,
            derivative_mode="edges",
        )
        assert position_facade.is_compiled, position_facade.compilation_error
        assert edge_facade.is_compiled, edge_facade.compilation_error
        position_outputs = position_facade(_batch_to(trace_batch, "cuda"))
        edge_outputs = edge_facade(_batch_to(trace_batch, "cuda"))
        for key in ("energy", "forces", "virials", "stress"):
            assert_close(position_outputs[key], edge_outputs[key])
        _objective(position_outputs).backward()
        _objective(edge_outputs).backward()
        _assert_parameter_gradients_close(position_model, edge_model)
