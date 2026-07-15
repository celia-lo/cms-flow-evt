#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular               # use 'debug' (30 min limit) only for quick tests
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16         # Perlmutter has 64 CPU cores; 16 per GPU is a good start
#SBATCH -J my-train-part
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Optional: #SBATCH --exclusive

set -euo pipefail

# ---- Env & conda ----
module load conda
conda activate /global/common/software/m3246/ylo/conda/cmsflow

# ---- Performance env (safe defaults) ----
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=hsn           # Perlmutter's HPE Slingshot host fabric
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PL_TORCH_DISTRIBUTED_TIMEOUT=7200

# ---- I/O: use /pscratch for speed ----
# export WORKDIR=/pscratch/sd/$USER/myexp_${SLURM_JOB_ID}
export WORKDIR=$SCRATCH/myexp_${SLURM_JOB_ID}
mkdir -p "$WORKDIR" logs
cp -r configs/evt.yaml "$WORKDIR"/
cp -r train.py "$WORKDIR"/
rsync -a --exclude logs --exclude lightning_logs "$SLURM_SUBMIT_DIR"/ "$WORKDIR"/
cd "$WORKDIR"

master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master
export MASTER_PORT=23456
# export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))

# evt train

# srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
#   torchrun \
#     --nnodes="$SLURM_JOB_NUM_NODES" \
#     --nproc_per_node="$SLURM_GPUS_ON_NODE" \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
#     --node_rank="$SLURM_PROCID" \
#     train.py -c configs/evt.yaml --gpus 0,1,2,3 
    # --ckpt_path saved_models/fm_cms_pow_evt_net_20251016-T182013/ckpts/last.ckpt

# part train

# srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
#   torchrun \
#     --nnodes="$SLURM_JOB_NUM_NODES" \
#     --nproc_per_node="$SLURM_GPUS_ON_NODE" \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
#     --node_rank="$SLURM_PROCID" \
#     train.py -c configs/part.yaml --gpus 0,1,2,3

# srun --ntasks=1 \
#   python train.py -c /pscratch/sd/y/ylo/myexp_48863281/saved_models/rcfm_cms_part_20260213-T223118/part.yaml \
#   --ckpt_path /pscratch/sd/y/ylo/myexp_48863281/saved_models/rcfm_cms_part_20260213-T223118/ckpts/last.ckpt \
#   --gpus 0,1,2,3

srun --ntasks=1 \
  python train.py -c configs/part.yaml \
  --gpus 0,1,2,3


# ---- Launch (PyTorch/Lightning, true DDP) ----
# Option A: Torch Distributed (works everywhere)
# torchrun --nproc_per_node=4 train.py -c configs/evt.yaml --gpus 0,1,2,3
# torchrun \
#   --nnodes="$SLURM_JOB_NUM_NODES" \
#   --nproc_per_node="$SLURM_GPUS_ON_NODE" \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
#   train.py -c configs/evt.yaml --ckpt_path saved_models/fm_cms_pow_evt_net_20251016-T152929/ckpts/last.ckpt --gpus 0,1,2,3

if [ -d "$WORKDIR/saved_models" ]; then
  mkdir -p "$SLURM_SUBMIT_DIR/saved_models"
  rsync -av "$WORKDIR/saved_models/" "$SLURM_SUBMIT_DIR/saved_models/"
fi

# Option B: Let srun spawn ranks (Lightning detects SLURM automatically)
# srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores \
#   python -u train.py -c <path_to_config_file> \
#   --devices 4 --accelerator gpu --strategy ddp --precision 16-true

