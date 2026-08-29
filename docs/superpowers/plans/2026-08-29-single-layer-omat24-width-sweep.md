# Single-Layer OMat24 Width Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, qualify, and submit fresh 20-epoch full-OMat24 without-self runs with 96 and 64 hidden channels while holding every other scientific control fixed.

**Architecture:** Add immutable width-candidate metadata and one dedicated argument renderer, verify exact trainable counts by constructing the selected MACE topology, and launch each width through a fail-closed two-rank Slurm script. Seal tested project files into a new remote deployment and give each candidate unique run, cache, and Slurm-log roots.

**Tech Stack:** Python 3.12, MACE 0.3.17, e3nn 0.4.4, PyTorch 2.11/CUDA 13 production runtime, CuEquivariance 0.11, pytest, Bash, Slurm.

## Global Constraints

- Only widths 96 and 64 are accepted; topology is always without-self.
- Hidden irreps are `Cx0e + Cx1o + Cx2e`, `max_ell=3`, interaction correlation 3, readout correlation 2, and `MLP_irreps=16x0e`.
- Expected trainable counts are exactly 3,860,186 for width 96 and 1,828,890 for width 64.
- Both models start fresh and run epochs 0--19 in one job with `--max_num_epochs=20` and no restart option.
- Preserve full OMat24 paths, frozen validation/statistics, seed `20260729`, batch 256/rank, four workers/rank, Adam/AMSGrad, EMA, and `ReduceLROnPlateau(factor=0.5, patience=1)`.
- Each job requests two typed RTX 5090s, two tasks, eight CPUs/task, and 92,000 MB/node.
- Every two-rank `srun` uses `--gpu-bind=mask_gpu:0x3,0x3` to expose both allocated GPUs to both ranks.
- Do not modify or stop the active 128-channel jobs; never reuse a failed or partial run root.

---

### Task 1: Add immutable width candidates and renderer

**Files:**
- Modify: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer.py`
- Modify: `/home/bonan/work/train-downsample/scripts/run_single_layer_omat24.py`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer.py`

**Interfaces:**
- Produces: `SingleLayerWidthCandidate(channels: int, expected_trainable_parameters: int)`.
- Produces: `WIDTH_CANDIDATES: dict[int, SingleLayerWidthCandidate]`.
- Produces: `build_cata_width_sweep_arguments(..., channels: int) -> list[str]`.
- Produces CLI mode `cata-width-sweep` with required `--channels {64,96}`.

- [ ] **Step 1: Write renderer and CLI tests**

Add tests that render both candidates and assert:

```python
@pytest.mark.parametrize(
    ("channels", "expected"), ((96, 3_860_186), (64, 1_828_890))
)
def test_width_sweep_is_fresh_without_self_and_twenty_epochs(
    tmp_path: Path, channels: int, expected: int
) -> None:
    candidate = WIDTH_CANDIDATES[channels]
    assert candidate.expected_trainable_parameters == expected
    args = build_cata_width_sweep_arguments(
        train_root=Path("/train"),
        valid_root=Path("/valid"),
        statistics_file=Path("/stats"),
        run_root=tmp_path / str(channels),
        num_workers=4,
        batch_per_rank=256,
        channels=channels,
    )
    assert f"--hidden_irreps={channels}x0e + {channels}x1o + {channels}x2e" in args
    assert "--interaction_first=RealAgnosticDensityInteractionBlock" in args
    assert "--max_num_epochs=20" in args
    assert "--use_last_readout_only=True" in args
    assert "--readout_correlation=2" in args
    assert not any("restart" in arg for arg in args)
```

Also compare normalized 96/64 argument sets after excluding only name, hidden irreps, and output paths; reject width 128, workers other than 4, batch other than 256, non-without-self CLI variants, and missing `--channels`.

- [ ] **Step 2: Run RED**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer.py -q
```

Expected: import or assertion failures because the width candidate API and CLI mode do not exist.

- [ ] **Step 3: Implement the minimal width API**

Add:

```python
@dataclass(frozen=True)
class SingleLayerWidthCandidate:
    channels: int
    expected_trainable_parameters: int

    @property
    def hidden_irreps(self) -> str:
        return (
            f"{self.channels}x0e + {self.channels}x1o + "
            f"{self.channels}x2e"
        )


WIDTH_CANDIDATES = {
    96: SingleLayerWidthCandidate(96, 3_860_186),
    64: SingleLayerWidthCandidate(64, 1_828_890),
}
```

Extend `_base_arguments` with optional explicit model name and hidden irreps,
defaulting to the existing values so all current callers remain byte-for-byte
unchanged. Implement `build_cata_width_sweep_arguments` to require width 64 or
96, workers 4, batch 256, use `VARIANTS["without-self"]`, and set twenty
epochs. Extend the CLI payload with `channels` and target epoch 20.

- [ ] **Step 4: Run GREEN and regression**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer.py -q
```

Expected: all tests pass.

---

### Task 2: Add exact parameter-count provenance

**Files:**
- Create: `/home/bonan/work/train-downsample/src/omat_downsample/full_retrain/single_layer_width.py`
- Create: `/home/bonan/work/train-downsample/scripts/verify_single_layer_width_model.py`
- Create: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_width.py`

**Interfaces:**
- Produces: `build_width_model(channels: int) -> torch.nn.Module`.
- Produces: `width_model_provenance(channels: int) -> dict[str, object]`.
- CLI prints stable JSON and exits nonzero on topology/count mismatch.

- [ ] **Step 1: Write failing provenance tests**

Test both widths against exact counts and required topology:

```python
@pytest.mark.parametrize(
    ("channels", "expected"), ((96, 3_860_186), (64, 1_828_890))
)
def test_width_model_has_exact_topology_and_count(
    channels: int, expected: int
) -> None:
    payload = width_model_provenance(channels)
    assert payload["trainable_parameters"] == expected
    assert payload["hidden_irreps"] == (
        f"{channels}x0e + {channels}x1o + {channels}x2e"
    )
    assert payload["interaction_first"] == "RealAgnosticDensityInteractionBlock"
    assert payload["uses_self_interaction"] is False
    assert payload["num_interactions"] == 1
    assert payload["correlation"] == 3
    assert payload["readout_correlation"] == 2
```

Reject unsupported widths and verify the CLI output matches the pure function.

- [ ] **Step 2: Run RED**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer_width.py -q
```

Expected: collection fails because `single_layer_width.py` is absent.

- [ ] **Step 3: Implement exact CPU construction**

Construct `ScaleShiftMACE` with the accepted 89 atomic numbers, without-self
density interaction, one layer, candidate hidden irreps, 10 radial basis
functions, cutoff polynomial 5, `max_ell=3`, radial MLP `[64,64,64]`,
correlation 3, readout correlation 2, nonlinear `16x0e` readout, pair
repulsion, and last-readout-only mode. Count only `parameter.requires_grad`.
Compare against `WIDTH_CANDIDATES[channels].expected_trainable_parameters` and
raise `ValueError` on mismatch.

- [ ] **Step 4: Implement the thin CLI**

Accept only `--channels {64,96}` and `--output`. Print JSON to stdout and write
the same JSON atomically when `--output` is given.

- [ ] **Step 5: Run GREEN**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain/test_single_layer_width.py -q
.venv/bin/python -m py_compile \
  src/omat_downsample/full_retrain/single_layer_width.py \
  scripts/verify_single_layer_width_model.py
```

Expected: tests and compilation pass.

---

### Task 3: Add the fail-closed fresh-20 launcher

**Files:**
- Create: `/home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_width_sweep.sbatch`
- Modify: `/home/bonan/work/train-downsample/tests/full_retrain/test_single_layer_width.py`

**Interfaces:**
- Consumes environment variables `CHANNELS`, `EXPECTED_PARAMETERS`, and the
  existing project/source/runtime/data/hash roots.
- Executes CLI mode `cata-width-sweep`.

- [ ] **Step 1: Write the launcher contract test**

Require exact directives and controls:

```python
for line in (
    "#SBATCH --ntasks=2",
    "#SBATCH --gpus=5090:2",
    "#SBATCH --cpus-per-task=8",
    "#SBATCH --mem=92000M",
    '[[ "$CHANNELS" == "64" || "$CHANNELS" == "96" ]]',
    '[[ "$NUM_WORKERS" == "4" ]]',
    '[[ "$BATCH_SIZE" == "256" ]]',
    'srun --ntasks="$SLURM_NTASKS" --gpu-bind=mask_gpu:0x3,0x3',
    '"$PROJECT_ROOT/scripts/run_single_layer_omat24.py" cata-width-sweep',
    '"$PROJECT_ROOT/scripts/verify_single_layer_width_model.py"',
):
    assert line in script
```

Assert there is no restart command, the run root must be absent, runtime and
wheel hashes are checked before creating it, and the postcondition requires an
epoch-19 checkpoint named for the selected width.

- [ ] **Step 2: Run RED**

Run the Task 2 pytest command. Expected: launcher contract failure because the
script is absent.

- [ ] **Step 3: Implement the launcher**

Mirror the accepted continuation launcher admission and telemetry paths, but
require a fresh root and call the parameter verifier before mutable output
creation. Validate `EXPECTED_PARAMETERS` equals 3,860,186 for width 96 or
1,828,890 for width 64. Use the explicit GPU mask for GPU admission, runtime
admission, and training `srun` calls. Require fresh initialization, compiled
DDP, finite epoch-0 validation, and epoch-19 checkpoint after completion.

- [ ] **Step 4: Verify syntax and tests**

Run:

```bash
bash -n /home/bonan/work/train-downsample/scripts/cata/train_single_layer_omat24_width_sweep.sbatch
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest \
  tests/full_retrain/test_single_layer.py \
  tests/full_retrain/test_single_layer_width.py -q
```

Expected: Bash and tests pass.

---

### Task 4: Run complete regression and seal deployment

**Files:**
- Create remotely: `/scratch/bonan/train-downsample-single-layer/e3d8580-6509ea2-width-sweep-c96-c64-v1/`
- Append: `/home/bonan/work/train-downsample/docs/project/experiments.md`

**Interfaces:**
- Produces immutable project archive, manifest, two dry renders, two parameter
  provenance JSON files, and isolated candidate roots.

- [ ] **Step 1: Run the full regression**

Run:

```bash
cd /home/bonan/work/train-downsample
.venv/bin/python -m pytest tests/full_retrain -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Build a metadata-free project archive**

Archive the tested `src/omat_downsample`, `scripts/run_single_layer_omat24.py`,
`scripts/verify_single_layer_width_model.py`, and required `scripts/cata`
helpers. Exclude `__pycache__`, bytecode, editable metadata, tests, runs, and
caches. Record SHA-256 locally and remotely.

- [ ] **Step 3: Qualify the remote deployment**

Using the proven pinned runtime and MACE source, run `py_compile`, launcher
`bash -n`, both parameter-verifier invocations, and both dry renders. Require
the exact counts, widths, full-data paths, fresh 20 epochs, no restart flag,
batch 256, workers 4, and only width/name/path differences between renders.

- [ ] **Step 4: Record deployment evidence**

Append the deployment root, archive hash, source/runtime hashes, exact dry-run
diff, and parameter JSON hashes to `docs/project/experiments.md` without
overwriting prior results.

---

### Task 5: Submit and verify both production jobs

**Files:**
- Create remotely: distinct `runs/c96`, `runs/c64`, `cache/c96`, `cache/c64`,
  and Slurm logs under the sealed deployment.

**Interfaces:**
- Produces two active Slurm job IDs and live health evidence.

- [ ] **Step 1: Inspect live Cata capacity**

Query `scontrol show node gn04`, active GPU jobs, and current 128-channel jobs.
Do not cancel or alter unrelated work.

- [ ] **Step 2: Submit width 96**

Submit the sealed launcher with width 96, expected count 3,860,186, unique run
and cache roots, two typed RTX 5090s, two tasks, 16 CPUs, and 92,000 MB.

- [ ] **Step 3: Submit width 64**

Submit the same sealed launcher with width 64, expected count 1,828,890, and
separate roots. Allow Slurm to queue it if four GPUs are not simultaneously
available.

- [ ] **Step 4: Verify resolved requests**

For each job require `gres/gpu=2`, `JOB_GRES=gpu:5090:2`, two tasks, 16 CPUs,
92,000 MB, the sealed command path, zero restarts, and unique stdout/stderr.
Cancel and replace only a newly submitted job whose resolved request is wrong.

- [ ] **Step 5: Verify live training health**

After allocation, require exact parameter provenance, both GPUs visible to
both ranks, fresh initialization, one compiled graph, finite optimizer steps,
bounded RSS, low loader wait, and no fatal signature. If pending, report the
validated pending reason without changing the protocol.

- [ ] **Step 6: Record and report**

Append job IDs and verified allocation/startup evidence to the experiment log.
Report candidate parameter reductions and current state; do not claim final
accuracy until epoch-19 validation exists.
