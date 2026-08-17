#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 {eager|compiled} WORLD_SIZE" >&2
    exit 2
fi

mode=$1
world_size=$2
if [[ "$mode" != "eager" && "$mode" != "compiled" ]]; then
    echo "mode must be eager or compiled" >&2
    exit 2
fi
if ! [[ "$world_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORLD_SIZE must be a positive integer" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
profiling_dir=$(cd "$script_dir/.." && pwd)
repo_root=$(cd "$profiling_dir/../.." && pwd)
dataset=${MACE_DDP_DATASET:-"$repo_root/.profiling/pr1641-eager/data/valid-4.h5"}
output_dir=${MACE_DDP_OUTPUT_DIR:-"$profiling_dir/compiled-ddp/${mode}-${world_size}"}

if [[ -n "${MACE_DDP_PYTHON:-}" ]]; then
    python_bin=$MACE_DDP_PYTHON
elif [[ -x "$profiling_dir/runtime/pytorch-211-cu128/bin/python" ]]; then
    python_bin="$profiling_dir/runtime/pytorch-211-cu128/bin/python"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
else
    echo "set MACE_DDP_PYTHON to the PyTorch 2.11 CUDA environment" >&2
    exit 2
fi
torchrun_bin=$(dirname "$python_bin")/torchrun
if [[ ! -x "$torchrun_bin" ]]; then
    echo "torchrun not found next to $python_bin" >&2
    exit 2
fi
if [[ ! -r "$dataset" ]]; then
    echo "dataset is not readable: $dataset" >&2
    exit 2
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -c \
    'import sys, torch; requested=int(sys.argv[1]); visible=torch.cuda.device_count(); print(f"visible_gpus={visible}"); raise SystemExit(0 if visible >= requested else 3)' \
    "$world_size"

mkdir -p "$output_dir"
extra_args=()
if [[ "${MACE_DDP_DUMP_FX_ONLY:-0}" == "1" ]]; then
    extra_args+=(--dump-fx-only)
fi
if [[ "${MACE_DDP_CUDA_PROFILER_CAPTURE:-0}" == "1" ]]; then
    extra_args+=(--cuda-profiler-capture)
fi
exec "$torchrun_bin" \
    --standalone \
    --nproc-per-node="$world_size" \
    "$script_dir/compiled_ddp_benchmark.py" \
    --mode="$mode" \
    --backend="${MACE_DDP_BACKEND:-cueq}" \
    --dataset="$dataset" \
    --output-dir="$output_dir" \
    --model-preset="${MACE_DDP_MODEL_PRESET:-small}" \
    --production-ablation="${MACE_DDP_PRODUCTION_ABLATION:-none}" \
    --derivative-mode="${MACE_DDP_DERIVATIVE_MODE:-positions}" \
    --batch-size="${MACE_DDP_BATCH_SIZE:-64}" \
    --num-workers="${MACE_DDP_NUM_WORKERS:-0}" \
    --warmup-steps="${MACE_DDP_WARMUP_STEPS:-2}" \
    --timed-steps="${MACE_DDP_TIMED_STEPS:-16}" \
    "${extra_args[@]}"
