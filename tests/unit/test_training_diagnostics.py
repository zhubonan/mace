###########################################################################################
# Tests for opt-in production training diagnostics
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mace import tools
from mace.tools.training_diagnostics import (
    TrainingDiagnostics,
    TrainingDiagnosticsConfig,
    diagnostics_config_from_args,
)

train_tools = importlib.import_module("mace.tools.train")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fake_batch(structures=4, atoms=11, edges=37):
    return SimpleNamespace(
        num_graphs=structures,
        positions=torch.zeros(atoms, 3),
        edge_index=torch.zeros(2, edges, dtype=torch.long),
    )


def summary_config(tmp_path, *, interval=2):
    return TrainingDiagnosticsConfig(
        mode="summary",
        interval=interval,
        directory=tmp_path,
        start_step=0,
        steps=0,
        backend="nvtx",
    )


def test_summary_writes_versioned_rank_records_with_actual_batch_costs(tmp_path):
    diagnostics = TrainingDiagnostics(
        summary_config(tmp_path),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        distributed=False,
    )

    diagnostics.record_step(
        epoch=2,
        loader_wait_s=0.004,
        step_period_s=0.021,
        h2d_wall_s=0.002,
        train_wall_s=0.015,
        batch=fake_batch(),
        optimizer_enabled=True,
        ema_enabled=False,
        compile_requested=True,
        compiled_active=True,
        compiler_graph_count=1,
    )
    diagnostics.close()

    records = read_jsonl(tmp_path / "rank-00000.jsonl")
    assert records[0] == {
        "record_type": "schema",
        "schema_version": 1,
        "clock": "time.perf_counter",
        "timing_semantics": "host_wall_seconds",
        "rank": 0,
        "world_size": 1,
    }
    step = records[1]
    assert step["record_type"] == "step"
    assert step["epoch"] == 2
    assert step["optimizer_step"] == 0
    assert step["structures"] == 4
    assert step["atoms"] == 11
    assert step["edges"] == 37
    assert step["loader_wait_s"] == pytest.approx(0.004)
    assert step["step_period_s"] == pytest.approx(0.021)
    assert step["h2d_submission_wall_s"] == pytest.approx(0.002)
    assert step["training_step_wall_s"] == pytest.approx(0.015)
    assert step["optimizer_enabled"] is True
    assert step["ema_enabled"] is False
    assert step["compile_requested"] is True
    assert step["compiled_active"] is True
    assert step["compiler_graph_count"] == 1
    assert step["cuda_memory_allocated_bytes"] == 0
    assert step["cuda_memory_reserved_bytes"] == 0
    diagnostics.close()


def test_summary_aggregates_fixed_size_rank_metrics_sparsely(tmp_path, monkeypatch):
    config = summary_config(tmp_path, interval=2)

    def all_gather(outputs, local):
        outputs[0].copy_(local)
        outputs[1].copy_(
            torch.tensor(
                [0.006, 0.024, 12.0, 44.0, 1200.0, 1800.0],
                dtype=local.dtype,
                device=local.device,
            )
        )

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    diagnostics = TrainingDiagnostics(
        config,
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        distributed=True,
    )
    kwargs = dict(
        epoch=1,
        loader_wait_s=0.004,
        step_period_s=0.020,
        h2d_wall_s=0.001,
        train_wall_s=0.014,
        batch=fake_batch(atoms=10, edges=40),
        optimizer_enabled=True,
        ema_enabled=True,
        compile_requested=False,
        compiled_active=False,
        compiler_graph_count=None,
    )

    diagnostics.record_step(**kwargs)
    assert not (tmp_path / "rank-00000-aggregate.jsonl").exists()
    diagnostics.record_step(**kwargs)
    diagnostics.close()

    aggregates = read_jsonl(tmp_path / "rank-00000-aggregate.jsonl")
    assert aggregates[0]["record_type"] == "schema"
    record = aggregates[1]
    assert record["record_type"] == "aggregate"
    assert record["optimizer_step"] == 1
    assert record["loader_wait_s"] == {
        "min": pytest.approx(0.004),
        "mean": pytest.approx(0.005),
        "max": pytest.approx(0.006),
        "coefficient_of_variation": pytest.approx(0.2),
        "worst_rank": 1,
    }
    assert record["edges"]["max"] == pytest.approx(44.0)
    assert record["step_period_s"]["worst_rank"] == 1
    assert record["critical_path_loader_wait_fraction"] == pytest.approx(0.25)


def test_diagnostics_configuration_is_off_by_default_and_validates_requests(tmp_path):
    default_args = SimpleNamespace(
        training_diagnostics="off",
        training_diagnostics_interval=1000,
        training_diagnostics_dir=None,
        training_diagnostics_start_step=0,
        training_diagnostics_steps=0,
        training_diagnostics_backend="nvtx",
    )
    assert (
        diagnostics_config_from_args(
            default_args, default_directory=tmp_path / "default"
        )
        is None
    )

    summary_args = SimpleNamespace(
        **{
            **vars(default_args),
            "training_diagnostics": "summary",
            "training_diagnostics_interval": 17,
        }
    )
    config = diagnostics_config_from_args(
        summary_args, default_directory=tmp_path / "default"
    )
    assert config.directory == (tmp_path / "default").resolve()
    assert config.interval == 17

    with pytest.raises(ValueError, match="interval"):
        TrainingDiagnosticsConfig(
            mode="summary",
            interval=0,
            directory=tmp_path,
            start_step=0,
            steps=0,
            backend="nvtx",
        )


def test_requested_diagnostic_output_failure_is_fatal(tmp_path):
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(OSError):
        TrainingDiagnostics(
            summary_config(output_file),
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            distributed=False,
        )


def test_training_parser_exposes_runtime_diagnostic_controls():
    args = tools.build_default_arg_parser().parse_args(
        [
            "--name",
            "diagnostic-test",
            "--training_diagnostics",
            "summary",
            "--training_diagnostics_interval",
            "25",
            "--training_diagnostics_dir",
            "/diagnostics",
        ]
    )

    assert args.training_diagnostics == "summary"
    assert args.training_diagnostics_interval == 25
    assert args.training_diagnostics_dir == "/diagnostics"


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class DelayedLoader:
    def __init__(self, clock, batches):
        self.clock = clock
        self.batches = iter(batches)

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.batches)
        self.clock.now += 0.004
        return batch


class RecordingDiagnostics:
    def __init__(self, clock):
        self.clock = clock
        self.optimizer_step = 0
        self.records = []

    def prepare_step(self):
        return None

    def range(self, _name):
        return nullcontext()

    def record_step(self, **record):
        self.records.append(record)
        self.optimizer_step += 1


def test_train_one_epoch_exposes_parent_loader_wait_only_when_requested(monkeypatch):
    clock = ManualClock()
    diagnostics = RecordingDiagnostics(clock)
    batch = fake_batch()
    logger = SimpleNamespace(log=lambda _record: None)

    def diagnostic_step(**kwargs):
        assert kwargs["diagnostics"] is diagnostics
        clock.now += 0.015
        timings = SimpleNamespace(h2d_wall_s=0.002, train_wall_s=0.013)
        return None, {"loss": np.asarray(1.0), "time": 0.015}, timings

    monkeypatch.setattr(train_tools, "take_step_with_diagnostics", diagnostic_step)
    monkeypatch.setattr(
        train_tools,
        "take_step",
        lambda **kwargs: pytest.fail("diagnostic path used ordinary take_step"),
    )

    train_tools.train_one_epoch(
        model=object(),
        loss_fn=object(),
        data_loader=DelayedLoader(clock, [batch]),
        optimizer=object(),
        epoch=3,
        output_args={},
        max_grad_norm=None,
        ema=None,
        logger=logger,
        device=torch.device("cpu"),
        distributed=False,
        training_diagnostics=diagnostics,
    )

    assert diagnostics.records[0]["loader_wait_s"] == pytest.approx(0.004)
    assert diagnostics.records[0]["step_period_s"] == pytest.approx(0.019)
    assert diagnostics.records[0]["h2d_wall_s"] == pytest.approx(0.002)
    assert diagnostics.records[0]["train_wall_s"] == pytest.approx(0.013)
    assert diagnostics.records[0]["batch"] is batch


def test_train_one_epoch_keeps_ordinary_loop_when_diagnostics_are_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_tools,
        "take_step",
        lambda **kwargs: (calls.append(kwargs["batch"]), ({}, {"loss": 1.0})[1]),
    )
    monkeypatch.setattr(
        train_tools,
        "take_step_with_diagnostics",
        lambda **kwargs: pytest.fail("disabled path constructed diagnostic timing"),
    )

    train_tools.train_one_epoch(
        model=object(),
        loss_fn=object(),
        data_loader=[fake_batch()],
        optimizer=object(),
        epoch=0,
        output_args={},
        max_grad_norm=None,
        ema=None,
        logger=SimpleNamespace(log=lambda _record: None),
        device=torch.device("cpu"),
        distributed=False,
        training_diagnostics=None,
    )

    assert len(calls) == 1


class TinyBatch:
    def __init__(self):
        self.feature = torch.tensor([2.0])
        self.target = torch.tensor([1.0])
        self.positions = torch.zeros(2, 3)
        self.edge_index = torch.zeros(2, 2, dtype=torch.long)
        self.num_graphs = 1

    def to(self, device):
        self.feature = self.feature.to(device)
        self.target = self.target.to(device)
        self.positions = self.positions.to(device)
        self.edge_index = self.edge_index.to(device)
        return self

    def to_dict(self):
        return {"feature": self.feature}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch, **_kwargs):
        return {"energy": self.weight * batch["feature"]}


class TinyLoss(torch.nn.Module):
    def forward(self, pred, ref):
        return torch.square(pred["energy"] - ref.target).sum()


class RangeRecorder:
    def __init__(self):
        self.clock = __import__("time").perf_counter
        self.labels = []

    def range(self, name):
        self.labels.append(name)
        return nullcontext()


def test_measured_step_preserves_optimizer_behavior_and_phase_boundaries():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    diagnostics = RangeRecorder()

    loss, metrics, timings = train_tools.take_step_with_diagnostics(
        model=model,
        loss_fn=TinyLoss(),
        batch=TinyBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=10.0,
        device=torch.device("cpu"),
        diagnostics=diagnostics,
    )

    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(1.0)
    assert metrics["time"] >= 0.0
    assert timings.h2d_wall_s >= 0.0
    assert timings.train_wall_s >= 0.0
    assert model.weight.item() == pytest.approx(0.6)
    assert diagnostics.labels == [
        "host_to_device",
        "forward_loss",
        "backward_ddp",
        "gradient_clipping",
        "optimizer",
    ]


def deep_config(tmp_path, *, start=1, steps=2, backend="nvtx"):
    return TrainingDiagnosticsConfig(
        mode="deep",
        interval=100,
        directory=tmp_path,
        start_step=start,
        steps=steps,
        backend=backend,
    )


def patch_cuda_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)


def record_synthetic_step(diagnostics):
    diagnostics.record_step(
        epoch=0,
        loader_wait_s=0.001,
        step_period_s=0.01,
        h2d_wall_s=0.001,
        train_wall_s=0.008,
        batch=fake_batch(),
        optimizer_enabled=True,
        ema_enabled=False,
        compile_requested=False,
        compiled_active=False,
        compiler_graph_count=None,
    )


def test_deep_configuration_requires_cuda_and_a_finite_positive_window(tmp_path):
    with pytest.raises(ValueError, match="positive --training_diagnostics_steps"):
        TrainingDiagnostics(
            deep_config(tmp_path, steps=0),
            rank=0,
            world_size=1,
            device=torch.device("cuda"),
            distributed=False,
        )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        TrainingDiagnostics(
            deep_config(tmp_path, steps=2),
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            distributed=False,
        )


def test_nvtx_ranges_are_bounded_to_declared_optimizer_steps(tmp_path, monkeypatch):
    patch_cuda_memory(monkeypatch)
    pushes = []
    pops = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", pushes.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: pops.append(True))
    diagnostics = TrainingDiagnostics(
        deep_config(tmp_path, start=1, steps=2, backend="nvtx"),
        rank=0,
        world_size=1,
        device=torch.device("cuda"),
        distributed=False,
    )

    labels = [
        "dataloader.next",
        "host_to_device",
        "forward_loss",
        "backward_ddp",
        "gradient_clipping",
        "optimizer",
        "ema",
    ]
    for _ in range(4):
        diagnostics.prepare_step()
        for label in labels:
            with diagnostics.range(label):
                pass
        record_synthetic_step(diagnostics)
    diagnostics.close()

    assert pushes == labels * 2
    assert len(pops) == len(pushes)


class FakeProfiler:
    def __init__(self):
        self.entered = 0
        self.exited = 0
        self.steps = 0
        self.exported = []
        self.collection_toggles = []

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited += 1

    def step(self):
        self.steps += 1

    def export_chrome_trace(self, path):
        self.exported.append(Path(path))
        Path(path).write_text("{}", encoding="utf-8")

    def toggle_collection_dynamic(self, enabled, activities):
        self.collection_toggles.append((enabled, tuple(activities)))


def test_torch_profiler_exports_one_bounded_chrome_trace_per_rank(
    tmp_path, monkeypatch
):
    patch_cuda_memory(monkeypatch)
    profiler = FakeProfiler()
    factories = []

    def profile_factory(**kwargs):
        factories.append(kwargs)
        return profiler

    labels = []
    monkeypatch.setattr(torch.profiler, "profile", profile_factory)
    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        lambda label: (labels.append(label), nullcontext())[1],
    )
    diagnostics = TrainingDiagnostics(
        deep_config(tmp_path, start=0, steps=2, backend="torch_profiler"),
        rank=3,
        world_size=4,
        device=torch.device("cuda"),
        distributed=False,
    )

    for _ in range(3):
        diagnostics.prepare_step()
        with diagnostics.range("forward_loss"):
            pass
        record_synthetic_step(diagnostics)
    diagnostics.close()

    assert len(factories) == 1
    assert set(factories[0]["activities"]) == {
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    }
    assert profiler.entered == profiler.exited == 1
    assert profiler.steps == 2
    assert labels == ["forward_loss", "forward_loss"]
    assert profiler.exported == [tmp_path / "rank-00003-step-00000000-00000001.json"]
    assert profiler.exported[0].is_file()


def test_torch_profiler_pauses_across_loader_exhaustion_without_splitting_trace(
    tmp_path, monkeypatch
):
    patch_cuda_memory(monkeypatch)
    profiler = FakeProfiler()
    factories = []

    def profile_factory(**kwargs):
        factories.append(kwargs)
        return profiler

    monkeypatch.setattr(torch.profiler, "profile", profile_factory)
    monkeypatch.setattr(torch.profiler, "record_function", lambda _label: nullcontext())
    diagnostics = TrainingDiagnostics(
        deep_config(tmp_path, start=1, steps=2, backend="torch_profiler"),
        rank=0,
        world_size=1,
        device=torch.device("cuda"),
        distributed=False,
    )

    diagnostics.prepare_step()
    record_synthetic_step(diagnostics)
    diagnostics.prepare_step()
    record_synthetic_step(diagnostics)

    diagnostics.prepare_step()
    diagnostics.abort_step()

    diagnostics.prepare_step()
    record_synthetic_step(diagnostics)
    diagnostics.close()

    activities = (
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    )
    assert len(factories) == 1
    assert profiler.entered == profiler.exited == 1
    assert profiler.collection_toggles == [(False, activities), (True, activities)]
    assert profiler.steps == 2
    assert profiler.exported == [tmp_path / "rank-00000-step-00000001-00000002.json"]
