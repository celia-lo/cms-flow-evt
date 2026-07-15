#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -J train-evt-chain
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

# ===========================================================================
# CONFIGURE THESE TWO LINES BEFORE FIRST SUBMISSION
# ===========================================================================
CHAIN_WORKDIR="$SCRATCH/cms_flow_evt_chain_60PU"   # fixed path — known before job runs
TARGET_EPOCHS=400                              # must match num_epochs in evt.yaml
# ===========================================================================

CONFIG_NAME="rcfm_cms_pow_evt"   # matches `name:` in configs/evt.yaml

mkdir -p logs

# ---- Env & conda ----
module load conda
conda activate /global/common/software/m3246/ylo/conda/cmsflow

# ---- Performance env ----
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

master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master
export MASTER_PORT=23456

# ---- Requeue function ----
function requeue() {
    if [[ -f "$CHAIN_WORKDIR/training_complete.flag" ]]; then
        echo "[chain] Training complete — not requeuing."
        return
    fi
    restarts=$(scontrol show jobid "$SLURM_JOB_ID" 2>/dev/null \
               | grep -o 'Restarts=[0-9]*' | cut -d= -f2 || echo 0)
    restarts=${restarts:-0}
    if (( restarts >= 99 )); then
        echo "[chain] Max restarts reached — not requeuing." >&2
    else
        echo "[chain] Requeuing (restart $restarts)..."
        scontrol requeue "$SLURM_JOB_ID"
    fi
}

# ---- Trap SIGTERM (sent 120s before time limit) to requeue gracefully ----
function handle_timeout() {
    echo "[chain] Time limit approaching — requeuing job $SLURM_JOB_ID"
    kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
    requeue
    exit 0
}
trap 'handle_timeout' SIGTERM

# ---- Detect first run vs resume ----
LATEST_CKPT=$(find "$CHAIN_WORKDIR/saved_models" -name "last.ckpt" \
                   -printf "%T@ %p\n" 2>/dev/null \
              | sort -rn | awk 'NR==1{print $2}') || true

if [[ -z "$LATEST_CKPT" ]]; then
  # ---- FIRST RUN: copy repo to fixed workdir and train fresh ----
  echo "[chain] First run — setting up $CHAIN_WORKDIR"
  mkdir -p "$CHAIN_WORKDIR"
  rsync -a --exclude logs --exclude lightning_logs --exclude saved_models \
    --exclude my_sample --exclude evals \
    "$SLURM_SUBMIT_DIR"/ "$CHAIN_WORKDIR"/
  cd "$CHAIN_WORKDIR"

  set +e
  srun --ntasks=1 python train.py -c configs/evt.yaml --gpus 0,1,2,3 &
  TRAIN_PID=$!
  wait "$TRAIN_PID"
  TRAIN_EXIT=$?
  set -e

else
  # ---- RESUME: use the newest last.ckpt found in saved_models ----
  CONFIG_PATH="$(dirname "$(dirname "$LATEST_CKPT")")/${CONFIG_NAME#*_}.yaml"
  [[ -f "$CONFIG_PATH" ]] || CONFIG_PATH="$(dirname "$(dirname "$LATEST_CKPT")")/evt.yaml"

  echo "[chain] Resuming from $LATEST_CKPT"
  echo "[chain] Config: $CONFIG_PATH"
  cd "$CHAIN_WORKDIR"

  set +e
  srun --ntasks=1 python train.py \
    -c "$CONFIG_PATH" \
    --ckpt_path "$LATEST_CKPT" \
    --gpus 0,1,2,3 &
  TRAIN_PID=$!
  wait "$TRAIN_PID"
  TRAIN_EXIT=$?
  set -e
fi

# ---- Check epoch and requeue if target not reached ----
FINAL_CKPT=$(find "$CHAIN_WORKDIR/saved_models" -name "last.ckpt" \
                  -printf "%T@ %p\n" 2>/dev/null \
             | sort -rn | awk 'NR==1{print $2}') || true

if [[ -z "$FINAL_CKPT" ]]; then
  echo "[chain] ERROR: no checkpoint found after training" >&2
  requeue
  exit 1
fi

CURRENT_EPOCH=$(python - "$FINAL_CKPT" <<'PYEOF'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(ckpt.get("epoch", -1))
PYEOF
)

echo "[chain] Finished at epoch $CURRENT_EPOCH (target: $((TARGET_EPOCHS - 1)))"

# Lightning epoch is 0-indexed: epoch 399 means 400 epochs are complete
if (( CURRENT_EPOCH < TARGET_EPOCHS - 1 )); then
  requeue
else
  touch "$CHAIN_WORKDIR/training_complete.flag"
  echo "[chain] Target reached — chain complete."
fi
