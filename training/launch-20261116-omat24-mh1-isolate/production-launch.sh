#!/usr/bin/env bash
set -euo pipefail

: "${MACE_CONFIG:?MACE_CONFIG is required}"
: "${MACE_RUN_ROOT:?MACE_RUN_ROOT is required}"
: "${MACE_CACHE_ROOT:?MACE_CACHE_ROOT is required}"

ROOT=/scratch/bonan/omat24-mh1-isolate-20261116
RUNTIME=/scratch/bonan/train-downsample-qualifications/529d986/runtime-cu128

test -f "${MACE_CONFIG}"
test -f "${ROOT}/scripts/production-rank.sh"
test -d "${ROOT}/code/mace"
test -d "${ROOT}/python-overrides/matscipy"
test -f "${ROOT}/references/derived/mace-mh-1-omat_pbe.model"
shard_count="$(find "${ROOT}/data/omat-aselmdb-flat" -maxdepth 1 -type l -name '*.aselmdb' | wc -l)"
if (( shard_count < 2000 )); then
    echo "Expected the flat OMat24 shard directory, found only ${shard_count} links." >&2
    exit 2
fi

if [[ "${MACE_DRY_RUN:-0}" != "1" && "${MACE_RESTART_LATEST:-0}" != "1" ]]; then
    if find "${MACE_RUN_ROOT}/checkpoints" -type f -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing fresh launch: checkpoints already exist in ${MACE_RUN_ROOT}." >&2
        exit 3
    fi
fi

export PATH="${RUNTIME}/bin:${PATH}"
export PYTHONPATH="${ROOT}/code:${ROOT}/python-overrides${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${RUNTIME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export NVIDIA_TF32_OVERRIDE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export MACE_CUEQ_CONV_FUSION_NODE_BUCKET=256
export MACE_RUNTIME="${RUNTIME}"
export MACE_MASTER_ADDR="${MACE_MASTER_ADDR:-$(hostname -s)}"
export MACE_MASTER_PORT="${MACE_MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export MPLCONFIGDIR="${MACE_CACHE_ROOT}/matplotlib"

mkdir -p "${MACE_RUN_ROOT}/worker-logs" "${MACE_CACHE_ROOT}/cueq" "${MACE_CACHE_ROOT}/matplotlib"

echo "Starting upstream-main + compiled-training + global-random isolation at $(date -u +%FT%TZ)"
echo "node=$(hostname -s) job=${SLURM_JOB_ID} config=${MACE_CONFIG}"
echo "run_root=${MACE_RUN_ROOT} restart=${MACE_RESTART_LATEST:-0} dry_run=${MACE_DRY_RUN:-0}"
echo "base=public/main+PR1801 sampler=global_random excluded=custom-loss-warmup-guards-native-reader"

MATSCIPY_VERSION="$(PYTHONPATH="${ROOT}/code:${ROOT}/python-overrides" "${RUNTIME}/bin/python" -c 'import matscipy; print(matscipy.__version__)')"
if [[ "${MATSCIPY_VERSION}" != "1.3.1" ]]; then
    echo "Expected project Matscipy 1.3.1, got ${MATSCIPY_VERSION}." >&2
    exit 4
fi
echo "matscipy=${MATSCIPY_VERSION} shards=${shard_count}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec srun --kill-on-bad-exit=1 --ntasks=4 --cpu-bind=cores --gpu-bind=none \
    "${ROOT}/scripts/production-rank.sh"
