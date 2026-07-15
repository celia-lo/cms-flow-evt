#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J eval-minbias-noPU-800k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module load conda

cd /global/cfs/cdirs/m3246/ylo/parnassus-hep/cms-flow-evt
mkdir -p evals logs

conda activate /global/common/software/m3246/ylo/conda/cmsflow
python eval.py \
  -c trained_models_2024_minbias_noPU/part.yaml \
  -p trained_models_2024_minbias_noPU/cms_part_epoch=24.ckpt \
  -ce trained_models_2024_minbias_noPU/evt.yaml \
  -pe trained_models_2024_minbias_noPU/cms_evt_epoch=229.ckpt \
  --test_path my_sample/minbias_2024_noPU_noPtcut/minbias_nopu_800k.root \
  -ne 999999 \
  -bs 500 \
  --prefix full_2024_minbias_noPU_noPtCut_800k_

conda activate /global/cfs/cdirs/m3246/ylo/conda/cmsflow_fj
python postprocess_eval.py \
  -i evals/full_2024_minbias_noPU_noPtCut_800k_rcfm_cms_part_24_40.root \
  -d my_sample/minbias_2024_noPU_noPtcut/minbias_nopu_800k.root \
  -o 2024_minbias_noPU_noPtCut_800k_postprocess.root \
  -n 999999 \
  -m
