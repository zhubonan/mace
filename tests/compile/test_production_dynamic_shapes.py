import importlib.util
import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from mace import data, tools
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_PATH = (
    _REPO_ROOT / ".profiling/mace-hdf5-gpu-2026-08-13/scripts/compiled_ddp_benchmark.py"
)
_OUTPUT_ARGS = {
    "energy": True,
    "forces": True,
    "virials": False,
    "stress": True,
}


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "mace_compiled_ddp_benchmark",
        _BENCHMARK_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module from {_BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _objective(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        outputs["energy"].square().mean()
        + 0.3 * outputs["forces"].square().mean()
        + 0.1 * outputs["stress"].square().mean()
    )


def _parameter_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_benchmark_marks_the_complete_steady_state_window(monkeypatch):
    benchmark = _load_benchmark_module()
    events = []

    class FakeCudart:
        @staticmethod
        def cudaProfilerStart():
            events.append(("profiler_start", None))

        @staticmethod
        def cudaProfilerStop():
            events.append(("profiler_stop", None))

    monkeypatch.setattr(
        benchmark.torch.cuda.nvtx,
        "range_push",
        lambda name: events.append(("push", name)),
    )
    monkeypatch.setattr(
        benchmark.torch.cuda.nvtx,
        "range_pop",
        lambda: events.append(("pop", None)),
    )
    monkeypatch.setattr(benchmark.torch.cuda, "cudart", lambda: FakeCudart())

    benchmark._start_steady_state_capture("compiled", cuda_profiler_api=True)
    benchmark._stop_steady_state_capture(cuda_profiler_api=True)

    assert events == [
        ("profiler_start", None),
        ("push", "compiled_ddp_steady_state"),
        ("pop", None),
        ("profiler_stop", None),
    ]


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.slow
def test_production_force_stress_compile_stays_dynamic_across_hdf5_batches():
    dataset_path = Path(os.environ.get("MACE_OMAT_HDF5", ""))
    if not dataset_path.is_file():
        pytest.skip("set MACE_OMAT_HDF5 to the production OMat HDF5 shard")

    benchmark = _load_benchmark_module()
    device = torch.device("cuda")
    torch.set_default_dtype(torch.float32)
    atomic_numbers = list(range(1, 84)) + list(range(89, 95))
    z_table = tools.AtomicNumberTable(atomic_numbers)
    dataset = data.HDF5Dataset(
        str(dataset_path),
        r_max=6.0,
        z_table=z_table,
        heads=["Default"],
        head="Default",
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    iterator = iter(loader)
    first = next(iterator).to(device).to_dict()
    second = next(iterator).to(device).to_dict()
    assert first["edge_index"].shape[1] != second["edge_index"].shape[1]

    model = benchmark._build_model(
        atomic_numbers,
        6.0,
        device,
        "production",
        "none",
        compile_training=True,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_063_204
    facade = mace_compile.compile_training_model(
        model,
        first,
        output_args=_OUTPUT_ARGS,
        dynamic=True,
        validation_batch=second,
    )
    assert facade.is_compiled, facade.compilation_error
    assert facade._compiled_ref[0] is not None  # pylint: disable=protected-access

    for batch in (first, second):
        model.zero_grad(set_to_none=True)
        eager_outputs = model(
            batch,
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=True,
        )
        _objective(eager_outputs).backward()
        eager_gradients = _parameter_gradients(model)

        model.zero_grad(set_to_none=True)
        compiled_outputs = facade(batch)
        assert facade.is_compiled, facade.compilation_error
        _objective(compiled_outputs).backward()
        compiled_gradients = _parameter_gradients(model)

        for key in ("energy", "forces", "stress"):
            assert torch.isfinite(compiled_outputs[key]).all()
            assert_close(
                compiled_outputs[key],
                eager_outputs[key],
                rtol=1.0e-4,
                atol=1.0e-4,
            )
        assert compiled_gradients.keys() == eager_gradients.keys()
        for name, eager_gradient in eager_gradients.items():
            assert torch.isfinite(compiled_gradients[name]).all(), name
            assert_close(
                compiled_gradients[name],
                eager_gradient,
                rtol=1.0e-4,
                atol=1.0e-4,
                msg=lambda message, parameter_name=name: (
                    f"gradient mismatch for {parameter_name}: {message}"
                ),
            )
