#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
phase_dir=$(cd "$script_dir/.." && pwd)
profiling_dir=$(cd "$phase_dir/.." && pwd)
repo_root=$(cd "$profiling_dir/../.." && pwd)
launcher="$profiling_dir/scripts/run_compiled_ddp.sh"
adapter="$script_dir/run_training_input_backend.py"
python_bin="$profiling_dir/runtime/pytorch-211-cu128/bin/python"
hdf5=${MACE_PHASE44_HDF5:-/mnt/hdd2/bonan/omat-training/full-omat24-native-hdf5-pilot-1000/archives/rattled-1000-subsampled/shards/omat-000000000-000001000.h5}
lmdb_root=${MACE_PHASE44_LMDB_ROOT:-/mnt/hdd2/bonan/datasets/omat24-aselmdb/rattled-1000-subsampled}
output_root=${MACE_PHASE44_GPU_OUTPUT:-"$phase_dir/gpu-matrix"}
cache_dir=${MACE_PHASE44_INDUCTOR_CACHE:-/tmp/mace-phase44-inductor-cache}
physical_gpu=${MACE_PHASE44_GPU:-0}
source_revision=$(git -C "$repo_root" rev-parse HEAD)

if [[ ! -r "$hdf5" ]]; then
    echo "HDF5 pilot is not readable: $hdf5" >&2
    exit 2
fi
if [[ ! -d "$lmdb_root" ]]; then
    echo "LMDB pilot root is not readable: $lmdb_root" >&2
    exit 2
fi
mkdir -p "$output_root" "$cache_dir"

stop_telemetry() {
    local telemetry_pid=$1
    if kill -0 "$telemetry_pid" 2>/dev/null; then
        kill "$telemetry_pid" 2>/dev/null || true
    fi
    wait "$telemetry_pid" 2>/dev/null || true
}

run_case() {
    local repetition=$1
    local policy=$2
    local workers=$3
    local lmdb_backend batch_sampling
    case "$policy" in
        hdf5_random)
            lmdb_backend=hdf5
            batch_sampling=random
            ;;
        native_random)
            lmdb_backend=omat_direct
            batch_sampling=random
            ;;
        native_balanced)
            lmdb_backend=omat_direct
            batch_sampling=shard_window_atom_balanced
            ;;
        *)
            echo "Unknown policy: $policy" >&2
            return 2
            ;;
    esac

    local output_dir="$output_root/r${repetition}-${policy}-w${workers}"
    if [[ -s "$output_dir/rank-0.json" ]]; then
        echo "Skipping completed $output_dir" >&2
        return 0
    fi
    mkdir -p "$output_dir"
    nvidia-smi -i "$physical_gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits > "$output_dir/gpu-info.csv"
    nvidia-smi dmon -i "$physical_gpu" -s pucvmet -d 1 -o DT \
        > "$output_dir/dmon.csv" 2> "$output_dir/dmon.err" &
    local telemetry_pid=$!
    local status=0
    env \
        "CUDA_VISIBLE_DEVICES=$physical_gpu" \
        "PYTHONPATH=/tmp/e3nn044-overlay:$repo_root:$script_dir" \
        "TORCHINDUCTOR_CACHE_DIR=$cache_dir" \
        "MACE_DDP_PYTHON=$python_bin" \
        "MACE_DDP_SCRIPT=$adapter" \
        "MACE_DDP_DATASET=$hdf5" \
        "MACE_DDP_OUTPUT_DIR=$output_dir" \
        "MACE_DDP_BACKEND=hybrid" \
        "MACE_DDP_MODEL_PRESET=production" \
        "MACE_DDP_DERIVATIVE_MODE=positions" \
        "MACE_DDP_MATMUL_PRECISION=highest" \
        "MACE_DDP_BATCH_SIZE=128" \
        "MACE_DDP_NUM_WORKERS=$workers" \
        "MACE_DDP_WARMUP_STEPS=4" \
        "MACE_DDP_TIMED_STEPS=256" \
        "MACE_PHASE44_LMDB_BACKEND=$lmdb_backend" \
        "MACE_PHASE44_BATCH_SAMPLING=$batch_sampling" \
        "MACE_PHASE44_LMDB_ROOT=$lmdb_root" \
        "MACE_PHASE44_WINDOW_SHARDS=8" \
        "MACE_PHASE44_MAX_OPEN_SHARDS=8" \
        "MACE_PHASE44_RECORD_LIMIT=1000" \
        "MACE_PHASE44_REPETITION=$repetition" \
        "MACE_PHASE44_SOURCE_REVISION=$source_revision" \
        "$launcher" compiled 1 > "$output_dir/run.log" 2>&1 || status=$?
    stop_telemetry "$telemetry_pid"
    if (( status != 0 )); then
        echo "Training failed for $output_dir (status $status)" >&2
        return "$status"
    fi
}

orders=(
    "hdf5_random:8 native_random:8 native_balanced:8 hdf5_random:16 native_random:16 native_balanced:16"
    "native_balanced:16 native_random:16 hdf5_random:16 native_balanced:8 native_random:8 hdf5_random:8"
    "native_random:8 hdf5_random:16 native_balanced:8 native_random:16 hdf5_random:8 native_balanced:16"
)

for repetition in 0 1 2; do
    read -r -a cases <<< "${orders[$repetition]}"
    for entry in "${cases[@]}"; do
        policy=${entry%%:*}
        workers=${entry##*:}
        run_case "$repetition" "$policy" "$workers"
    done
done

"$python_bin" "$script_dir/verify_phase44.py" "$output_root" \
    --source-revision "$source_revision" --world-size 1 --repetitions 3 \
    --output "$output_root/summary.json"
