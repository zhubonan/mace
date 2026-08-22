"""Opt-in low-overhead and bounded deep training diagnostics."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class TrainingDiagnosticsConfig:
    """Validated public configuration for requested diagnostics."""

    mode: str
    interval: int
    directory: Path
    start_step: int
    steps: int
    backend: str

    def __post_init__(self) -> None:
        if self.mode not in {"summary", "deep"}:
            raise ValueError("training diagnostics mode must be summary or deep")
        if self.interval < 1:
            raise ValueError("training diagnostics interval must be positive")
        if self.start_step < 0:
            raise ValueError("training diagnostics start step must be non-negative")
        if self.steps < 0:
            raise ValueError("training diagnostics steps must be non-negative")
        if self.backend not in {"nvtx", "torch_profiler"}:
            raise ValueError(
                "training diagnostics backend must be nvtx or torch_profiler"
            )
        object.__setattr__(self, "directory", Path(self.directory).resolve())


@dataclass(frozen=True)
class TrainingStepTimings:
    h2d_wall_s: float
    train_wall_s: float


def diagnostics_config_from_args(
    args, *, default_directory: Path | str
) -> TrainingDiagnosticsConfig | None:
    """Translate CLI arguments while keeping `off` allocation-free."""

    mode = getattr(args, "training_diagnostics", "off")
    if mode == "off":
        return None
    requested_directory = getattr(args, "training_diagnostics_dir", None)
    return TrainingDiagnosticsConfig(
        mode=mode,
        interval=int(getattr(args, "training_diagnostics_interval", 1000)),
        directory=(
            Path(requested_directory)
            if requested_directory is not None
            else Path(default_directory)
        ),
        start_step=int(getattr(args, "training_diagnostics_start_step", 0)),
        steps=int(getattr(args, "training_diagnostics_steps", 0)),
        backend=getattr(args, "training_diagnostics_backend", "nvtx"),
    )


class TrainingDiagnostics:
    """Buffered runtime summaries with sparse, fixed-size rank aggregation."""

    _SCHEMA_VERSION = 1
    _METRIC_NAMES = (
        "loader_wait_s",
        "step_period_s",
        "atoms",
        "edges",
        "cuda_memory_allocated_bytes",
        "cuda_memory_reserved_bytes",
    )

    def __init__(
        self,
        config: TrainingDiagnosticsConfig,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        distributed: bool,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if world_size < 1:
            raise ValueError("diagnostic world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("diagnostic rank must be in [0, world_size)")
        if config.mode == "deep" and config.steps < 1:
            raise ValueError(
                "Deep diagnostics require positive --training_diagnostics_steps"
            )
        if config.mode == "deep" and device.type != "cuda":
            raise ValueError("Deep training diagnostics requires a CUDA device")
        self.config = config
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.distributed = bool(distributed)
        self.clock = clock
        self.optimizer_step = 0
        self._closed = False
        self._rank_buffer: list[dict] = []
        self._aggregate_stream = None
        self._capture_active = False
        self._prepared_step: int | None = None
        self._profiler = None
        self._profiler_collection_enabled = False
        self._captured_steps = 0

        config.directory.mkdir(parents=True, exist_ok=True)
        rank_path = config.directory / f"rank-{self.rank:05d}.jsonl"
        self._rank_stream = rank_path.open("a", encoding="utf-8")
        self._write_line(self._rank_stream, self._schema_record())
        self._rank_stream.flush()

    def _schema_record(self) -> dict:
        return {
            "record_type": "schema",
            "schema_version": self._SCHEMA_VERSION,
            "clock": "time.perf_counter",
            "timing_semantics": "host_wall_seconds",
            "rank": self.rank,
            "world_size": self.world_size,
        }

    @staticmethod
    def _write_line(stream, record: dict) -> None:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        stream.write("\n")

    def prepare_step(self) -> None:
        """Prepare optional deep-capture state before DataLoader.next()."""

        if self.config.mode != "deep":
            return
        if self._prepared_step is not None:
            raise RuntimeError("A diagnostic step is already prepared")
        step = self.optimizer_step
        self._prepared_step = step
        capture_end = self.config.start_step + self.config.steps
        self._capture_active = self.config.start_step <= step < capture_end
        if self._capture_active and self.config.backend == "torch_profiler":
            if self._profiler is None:
                self._profiler = torch.profiler.profile(
                    activities=self._profiler_activities(),
                    record_shapes=True,
                    profile_memory=True,
                )
                self._profiler.__enter__()
                self._profiler_collection_enabled = True
            elif not self._profiler_collection_enabled:
                self._toggle_profiler_collection(True)

    def abort_step(self) -> None:
        """Cancel a prepared capture when DataLoader.next() reaches exhaustion."""

        if self.config.mode != "deep" or self._prepared_step is None:
            return
        if self._capture_active and self._profiler is not None:
            self._toggle_profiler_collection(False)
        self._capture_active = False
        self._prepared_step = None

    @staticmethod
    def _profiler_activities():
        return [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

    def _toggle_profiler_collection(self, enabled: bool) -> None:
        toggle = getattr(self._profiler, "toggle_collection_dynamic", None)
        if not callable(toggle):
            raise RuntimeError(
                "This PyTorch profiler cannot pause across DataLoader exhaustion; "
                "upgrade PyTorch or select a capture window within one epoch"
            )
        toggle(enabled, self._profiler_activities())
        self._profiler_collection_enabled = enabled

    @contextmanager
    def range(self, name: str):
        """Emit attribution only inside the requested finite deep window."""

        if not self._capture_active:
            with nullcontext():
                yield
            return
        if self.config.backend == "nvtx":
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
            return
        with torch.profiler.record_function(name):
            yield

    def _trace_path(self, end_step: int) -> Path:
        return self.config.directory / (
            f"rank-{self.rank:05d}-step-{self.config.start_step:08d}-"
            f"{end_step:08d}.json"
        )

    def _stop_profiler(self, *, export: bool) -> None:
        profiler = self._profiler
        if profiler is None:
            return
        profiler.__exit__(None, None, None)
        if export:
            end_step = self.config.start_step + self._captured_steps - 1
            profiler.export_chrome_trace(str(self._trace_path(end_step)))
        self._profiler = None
        self._profiler_collection_enabled = False

    def _finish_deep_step(self, step: int) -> None:
        if self.config.mode != "deep":
            return
        if self._prepared_step != step:
            raise RuntimeError("Diagnostic step was recorded without prepare_step")
        if self._capture_active:
            self._captured_steps += 1
            if self._profiler is not None:
                self._profiler.step()
            if self._captured_steps == self.config.steps:
                self._stop_profiler(export=True)
                self._capture_active = False
        self._prepared_step = None

    def _cuda_memory(self) -> tuple[int, int, int, int]:
        if self.device.type != "cuda":
            return 0, 0, 0, 0
        return (
            int(torch.cuda.memory_allocated(self.device)),
            int(torch.cuda.memory_reserved(self.device)),
            int(torch.cuda.max_memory_allocated(self.device)),
            int(torch.cuda.max_memory_reserved(self.device)),
        )

    @staticmethod
    def _rank_statistics(values: torch.Tensor) -> dict:
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item())
        return {
            "min": float(values.min().item()),
            "mean": mean,
            "max": float(values.max().item()),
            "coefficient_of_variation": 0.0 if mean == 0.0 else std / abs(mean),
            "worst_rank": int(values.argmax().item()),
        }

    def _aggregate(self, local_values: torch.Tensor, *, epoch: int, step: int) -> None:
        if self.distributed:
            gathered = [torch.empty_like(local_values) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered, local_values)
            rank_values = torch.stack(gathered)
        else:
            rank_values = local_values.unsqueeze(0)
        metrics = {
            name: self._rank_statistics(rank_values[:, metric_index])
            for metric_index, name in enumerate(self._METRIC_NAMES)
        }
        max_period = metrics["step_period_s"]["max"]
        max_wait = metrics["loader_wait_s"]["max"]
        record = {
            "record_type": "aggregate",
            "schema_version": self._SCHEMA_VERSION,
            "epoch": int(epoch),
            "optimizer_step": int(step),
            **metrics,
            "critical_path_loader_wait_fraction": (
                0.0 if max_period == 0.0 else max_wait / max_period
            ),
            "input_wait_dominated": bool(
                max_period > 0 and max_wait >= 0.5 * max_period
            ),
        }
        if self.rank == 0:
            if self._aggregate_stream is None:
                aggregate_path = self.config.directory / "rank-00000-aggregate.jsonl"
                self._aggregate_stream = aggregate_path.open("a", encoding="utf-8")
                self._write_line(self._aggregate_stream, self._schema_record())
            self._write_line(self._aggregate_stream, record)
            self._aggregate_stream.flush()

    def record_step(
        self,
        *,
        epoch: int,
        loader_wait_s: float,
        step_period_s: float,
        h2d_wall_s: float,
        train_wall_s: float,
        batch,
        optimizer_enabled: bool,
        ema_enabled: bool,
        compile_requested: bool,
        compiled_active: bool,
        compiler_graph_count: int | None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot record a step after diagnostics close")
        timings = (loader_wait_s, step_period_s, h2d_wall_s, train_wall_s)
        if any(not math.isfinite(value) or value < 0 for value in timings):
            raise ValueError("Diagnostic step timings must be finite and non-negative")
        allocated, reserved, peak_allocated, peak_reserved = self._cuda_memory()
        structures = int(batch.num_graphs)
        atoms = int(batch.positions.shape[0])
        edges = int(batch.edge_index.shape[1])
        step = self.optimizer_step
        record = {
            "record_type": "step",
            "schema_version": self._SCHEMA_VERSION,
            "epoch": int(epoch),
            "optimizer_step": step,
            "rank": self.rank,
            "world_size": self.world_size,
            "structures": structures,
            "atoms": atoms,
            "edges": edges,
            "loader_wait_s": float(loader_wait_s),
            "step_period_s": float(step_period_s),
            "h2d_submission_wall_s": float(h2d_wall_s),
            "training_step_wall_s": float(train_wall_s),
            "optimizer_enabled": bool(optimizer_enabled),
            "ema_enabled": bool(ema_enabled),
            "compile_requested": bool(compile_requested),
            "compiled_active": bool(compiled_active),
            "compiler_graph_count": compiler_graph_count,
            "cuda_memory_allocated_bytes": allocated,
            "cuda_memory_reserved_bytes": reserved,
            "cuda_peak_memory_allocated_bytes": peak_allocated,
            "cuda_peak_memory_reserved_bytes": peak_reserved,
        }
        self._rank_buffer.append(record)
        self.optimizer_step += 1

        if self.optimizer_step % self.config.interval == 0:
            local_values = torch.tensor(
                [
                    loader_wait_s,
                    step_period_s,
                    atoms,
                    edges,
                    allocated,
                    reserved,
                ],
                dtype=torch.float64,
                device=self.device,
            )
            self._aggregate(local_values, epoch=epoch, step=step)
            self._flush_rank()
        self._finish_deep_step(step)

    def _flush_rank(self) -> None:
        for record in self._rank_buffer:
            self._write_line(self._rank_stream, record)
        self._rank_buffer.clear()
        self._rank_stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        if self._profiler is not None:
            self._stop_profiler(export=self._captured_steps > 0)
        self._flush_rank()
        self._rank_stream.close()
        if self._aggregate_stream is not None:
            self._aggregate_stream.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
