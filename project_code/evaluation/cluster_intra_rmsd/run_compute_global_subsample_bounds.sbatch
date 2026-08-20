#!/bin/bash
#SBATCH --job-name=global_sub_rmsd
#SBATCH --partition=standard-opig-gpu
#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/cluster_intra_rmsd/script_outputs/global_subsample_bounds_%j.out

set -euo pipefail

source /etc/profile.d/modules.sh
module purge || true
module use /apps/modulefiles/Core
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate calculate_metrics

export PYTHONPATH=/opig-shared/users/lina4783/ABB4

BASE=/opig-shared/users/lina4783/abb4_experiments/evaluation/cluster_intra_rmsd
META=/opig-shared/users/lina4783/structures_final/metadata_no_nano.csv
OUT=${BASE}/outputs/bounds

python ${BASE}/compute_global_subsample_bounds.py \
    --meta_path ${META} \
    --output_dir ${OUT} \
    --n_samples 100 \
    --n_replicates 100 \
    --n_jobs 10
