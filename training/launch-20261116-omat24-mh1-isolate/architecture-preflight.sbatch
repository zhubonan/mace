#!/usr/bin/env bash
#SBATCH --job-name=mh1-iso-preflight
#SBATCH --partition=gpu
#SBATCH --reservation=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gn05
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bonan/omat24-mh1-isolate-20261116/architecture-preflight-%j.out

set -euo pipefail

ROOT=/scratch/bonan/omat24-mh1-isolate-20261116
RUNTIME=/scratch/bonan/train-downsample-qualifications/529d986/runtime-cu128
RUN_ROOT="${ROOT}/preflight/architecture-${SLURM_JOB_ID}"

export PATH="${RUNTIME}/bin:${PATH}"
export PYTHONPATH="${ROOT}/code:${ROOT}/python-overrides${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${RUNTIME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export NVIDIA_TF32_OVERRIDE=0
export MACE_CUEQ_CONV_FUSION_NODE_BUCKET=256
export MACE_RUNTIME="${RUNTIME}"
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="${ROOT}/cache/architecture-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${ROOT}/cache/triton-architecture-${SLURM_JOB_ID}"
export CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR="${ROOT}/cache/cueq-architecture-${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR}"

exec "${RUNTIME}/bin/python" -m mace.cli.run_train \
    --config="${ROOT}/configs/architecture-preflight.yaml" \
    --work_dir="${RUN_ROOT}" \
    --dry_run
