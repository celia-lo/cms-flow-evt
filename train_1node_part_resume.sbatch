#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -J my-train-part-resume
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module load conda
conda activate /global/common/software/m3246/ylo/conda/cmsflow

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=hsn
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
export PL_TORCH_DISTRIBUTED_TIMEOUT=7200

WORKDIR=/pscratch/sd/y/ylo/myexp_49377070
EXPERIMENT=rcfm_cms_part_20260225-T201233
CONFIG_PATH="$WORKDIR/saved_models/$EXPERIMENT/part.yaml"
CKPT_PATH="$WORKDIR/saved_models/$EXPERIMENT/ckpts/last.ckpt"

if [[ ! -d "$WORKDIR" ]]; then
  echo "Workdir $WORKDIR not found" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config $CONFIG_PATH not found" >&2
  exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint $CKPT_PATH not found" >&2
  exit 1
fi

cd "$WORKDIR"

master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master
export MASTER_PORT=23456

srun --ntasks=1 \
  python train.py \
    -c "$CONFIG_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --gpus 0,1,2,3 

if [[ -d "$WORKDIR/saved_models" ]]; then
  mkdir -p "$SLURM_SUBMIT_DIR/saved_models"
  rsync -av "$WORKDIR/saved_models/" "$SLURM_SUBMIT_DIR/saved_models/"
fi
