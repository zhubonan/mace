#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/bonan/omat24-mh1-isolate-20261116
SBATCH="${ROOT}/scripts/production.sbatch"
RUN_ROOT=/home/bonan/scratch/omat24-mh1-isolate-20261116/production/omat24-mh1-isolate-main-compile-global-random-512-128

mode="${1:-dry-run}"
case "${mode}" in
    preflight)
        exec sbatch "${ROOT}/scripts/architecture-preflight.sbatch"
        ;;
    dry-run)
        stamp="$(date +%Y%m%d-%H%M%S)"
        exec sbatch --export="ALL,MACE_DRY_RUN=1,MACE_RUN_ROOT=${ROOT}/preflight/production-dry-run-${stamp}" "${SBATCH}"
        ;;
    submit)
        exec sbatch --export="ALL,MACE_DRY_RUN=0,MACE_RESTART_LATEST=0,MACE_RUN_ROOT=${RUN_ROOT}" "${SBATCH}"
        ;;
    restart)
        exec sbatch --export="ALL,MACE_DRY_RUN=0,MACE_RESTART_LATEST=1,MACE_RUN_ROOT=${RUN_ROOT}" "${SBATCH}"
        ;;
    *)
        echo "Usage: $0 {preflight|dry-run|submit|restart}" >&2
        exit 2
        ;;
esac
