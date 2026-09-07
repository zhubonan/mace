#!/usr/bin/env bash
# Train single-v2 from the OMat-0 first layer with a fresh optimizer/readout.
# The caller must provide an immutable source deployment and a new RUN_ROOT.
#SBATCH --job-name=omat0-single-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=92000M
#SBATCH --qos=urgent
#SBATCH --time=3-00:00:00

set -euo pipefail

: "${MACE_SOURCE_ROOT:?set the deployed source tree}"
: "${MACE_COMMIT:?set the expected source commit}"
: "${RUNTIME_ROOT:?set the CUDA 13 MACE runtime}"
: "${FOUNDATION_MODEL:?set the staged OMat-0 medium checkpoint}"
: "${FOUNDATION_SHA256:?set the checkpoint SHA-256}"
: "${TRAIN_ROOT:?set the OMat24 LMDB root}"
: "${VALID_ROOT:?set the validation root}"
: "${STATISTICS_FILE:?set the frozen statistics JSON}"
: "${RUN_ROOT:?set a new empty run root}"
: "${CACHE_ROOT:?set a writable compiler-cache root}"

[[ "$SLURM_NTASKS" == "2" ]]
[[ "${SLURM_NTASKS_PER_NODE:-}" == "2" ]]
[[ "$SLURM_CPUS_PER_TASK" == "8" ]]
test -r "$MACE_SOURCE_ROOT/mace/__init__.py"
test "$(tr -d '\n' < "$MACE_SOURCE_ROOT/.source-commit")" = "$MACE_COMMIT"
test -x "$RUNTIME_ROOT/bin/python"

readonly REQUIRED_MATSCIPY_VERSION="1.2.0+omatfix1"
readonly REQUIRED_MATSCIPY_WHEEL_SHA256="3c8fe808619ed6faa2c6b55add052bf4ed09f6ed9fb753a6838e12ee9090426d"
readonly MATSCIPY_WHEEL_SHA256_FILE="$RUNTIME_ROOT/environment/matscipy-wheel.sha256"
runtime_matscipy_version="$($RUNTIME_ROOT/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("matscipy"))')"
test "$runtime_matscipy_version" = "$REQUIRED_MATSCIPY_VERSION"
test -r "$MATSCIPY_WHEEL_SHA256_FILE"
test "$(awk 'NR==1 {print $1}' "$MATSCIPY_WHEEL_SHA256_FILE")" = "$REQUIRED_MATSCIPY_WHEEL_SHA256"
test -r "$FOUNDATION_MODEL"
test "$(sha256sum "$FOUNDATION_MODEL" | cut -d ' ' -f1)" = "$FOUNDATION_SHA256"
test -d "$TRAIN_ROOT"
test -d "$VALID_ROOT"
test -r "$STATISTICS_FILE"
if [[ -e "$RUN_ROOT" ]]; then
    echo "RUN_ROOT must be new: $RUN_ROOT" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"/{checkpoints,logs,models,results,training-diagnostics}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$MACE_SOURCE_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MASTER_ADDR="$(hostname -s)"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton-$SLURM_JOB_ID"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

srun --ntasks="$SLURM_NTASKS" --gpu-bind=mask_gpu:0x3,0x3 --kill-on-bad-exit=1 \
    "$RUNTIME_ROOT/bin/python" -c \
    'import torch; assert torch.cuda.device_count() == 2; assert all("RTX 5090" in torch.cuda.get_device_name(i) for i in range(2))'

srun --ntasks="$SLURM_NTASKS" --gpu-bind=mask_gpu:0x3,0x3 --kill-on-bad-exit=1 \
    "$RUNTIME_ROOT/bin/python" -m mace.cli.run_train \
    --name=omat0-single-v2 \
    --model=MACE \
    --foundation_model="$FOUNDATION_MODEL" \
    --foundation_model_elements=true \
    --omat0_first_layer_readout=true \
    --E0s=foundation \
    --r_max=6.0 \
    --distance_transform=Agnesi \
    --pair_repulsion=true \
    --use_last_readout_only=true \
    --loss=stress \
    --energy_weight=1 \
    --forces_weight=10 \
    --stress_weight=10 \
    --default_dtype=float32 \
    --batch_size=256 \
    --valid_batch_size=64 \
    --num_workers=4 \
    --pin_memory=true \
    --lmdb_backend=omat_direct \
    --batch_sampling=shard_window_atom_balanced \
    --lmdb_shard_window=8 \
    --lmdb_max_open_shards=8 \
    --train_file="$TRAIN_ROOT" \
    --valid_file="$VALID_ROOT" \
    --statistics_file="$STATISTICS_FILE" \
    --max_num_epochs=10 \
    --seed=20260729 \
    --optimizer=adam \
    --amsgrad \
    --beta=0.9 \
    --lr=0.01 \
    --weight_decay=1e-8 \
    --scheduler=ReduceLROnPlateau \
    --lr_factor=0.5 \
    --scheduler_patience=1 \
    --ema \
    --ema_decay=0.999 \
    --clip_grad=10 \
    --keep_checkpoints \
    --save_all_checkpoints \
    --save_cpu \
    --compile_training \
    --enable_cueq=true \
    --distributed \
    --launcher=slurm \
    --device=cuda \
    --work_dir="$RUN_ROOT" \
    --checkpoints_dir="$RUN_ROOT/checkpoints" \
    --log_dir="$RUN_ROOT/logs" \
    --model_dir="$RUN_ROOT/models" \
    --results_dir="$RUN_ROOT/results"
