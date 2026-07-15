# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account m3246
module load conda
# conda activate /global/common/software/m3246/ylo/conda/herwig
conda activate /global/common/software/m3246/ylo/conda/cmsflow
conda activate /global/cfs/cdirs/m3246/ylo/conda/cmsflow_fj