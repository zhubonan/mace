#!/usr/bin/env bash
set -euo pipefail

ulimit -n 16384

export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK=0
export MASTER_ADDR="${MACE_MASTER_ADDR}"
export MASTER_PORT="${MACE_MASTER_PORT}"
export CUDA_VISIBLE_DEVICES="${SLURM_PROCID}"

export TORCHINDUCTOR_CACHE_DIR="${MACE_CACHE_ROOT}/inductor/mh1-isolate-${SLURM_JOB_ID}-rank${SLURM_PROCID}"
export TRITON_CACHE_DIR="${MACE_CACHE_ROOT}/triton/mh1-isolate-${SLURM_JOB_ID}-rank${SLURM_PROCID}"
export CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR="${MACE_CACHE_ROOT}/cueq/rank${SLURM_PROCID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR}"
echo "rank=${SLURM_PROCID} inductor_cache=${TORCHINDUCTOR_CACHE_DIR} triton_cache=${TRITON_CACHE_DIR}"

extra_args=()
if [[ "${MACE_DRY_RUN:-0}" == "1" ]]; then
    extra_args+=(--dry_run)
fi
if [[ "${MACE_RESTART_LATEST:-0}" == "1" ]]; then
    extra_args+=(--restart_latest)
fi

exec "${MACE_RUNTIME}/bin/python" -m mace.cli.run_train \
    --config="${MACE_CONFIG}" \
    --work_dir="${MACE_RUN_ROOT}" \
    "${extra_args[@]}"
