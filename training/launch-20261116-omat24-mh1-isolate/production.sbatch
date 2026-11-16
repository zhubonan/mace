#!/usr/bin/env bash
#SBATCH --job-name=mh1-iso-main-cg
#SBATCH --partition=gpu
#SBATCH --reservation=train
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gn05
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/bonan/omat24-mh1-isolate-20261116/production-slurm-%j.out

set -euo pipefail

export MACE_CONFIG="${MACE_CONFIG:-/scratch/bonan/omat24-mh1-isolate-20261116/configs/stage-a-mh1-isolate.yaml}"
export MACE_RUN_ROOT="${MACE_RUN_ROOT:-/home/bonan/scratch/omat24-mh1-isolate-20261116/production/omat24-mh1-isolate-main-compile-global-random-512-128}"
export MACE_CACHE_ROOT="${MACE_CACHE_ROOT:-/scratch/bonan/omat24-mh1-isolate-20261116/cache/production}"
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    export MACE_RESTART_LATEST=1
fi

exec bash /scratch/bonan/omat24-mh1-isolate-20261116/scripts/production-launch.sh
