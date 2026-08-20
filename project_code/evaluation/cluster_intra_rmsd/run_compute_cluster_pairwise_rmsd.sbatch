#!/bin/bash
#SBATCH --job-name=cluster_rmsds
#SBATCH --partition=standard-opig-gpu
#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/cluster_intra_rmsd/script_outputs/cluster_intra_rmsd_%j.out

set -euo pipefail

source /etc/profile.d/modules.sh
module purge || true
module use /apps/modulefiles/Core
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate calculate_metrics

export PYTHONPATH=/opig-shared/users/lina4783/ABB4

BASE=/opig-shared/users/lina4783/abb4_experiments/evaluation/cluster_intra_rmsd
META_GT=/opig-shared/users/lina4783/structures_final/train_meta.csv
META_PRED=/opig-shared/users/lina4783/structures_final/test_meta.csv
OUT=${BASE}/outputs/ckpt_5139

#python ${BASE}/compute_cluster_pairwise_rmsd.py \
#    --mode ground_truth \
#    --meta_path ${META_GT} \
#    --output_dir ${OUT} \
#    --save_pairs \
#    --n_jobs 10

python ${BASE}/compute_cluster_pairwise_rmsd.py \
    --mode prediction \
    --meta_path ${META_PRED} \
    --pred_path '/opig-shared/users/lina4783/abb4_experiments/evaluation/predictions_ckpt_5139_imgt' \
    --summary_path '/opig-shared/users/lina4783/abb4_experiments/evaluation/predictions_ckpt_5139_imgt/struc_pred_metrics_test_summary.csv' \
    --output_dir ${OUT} \
    --save_pairs \
    --n_jobs 10
