#!/bin/bash
#SBATCH --job-name=same_len_rmsd
#SBATCH --partition=standard-opig-gpu
#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/cluster_intra_rmsd/script_outputs/same_length_aggregate_%j.out

set -euo pipefail

source /etc/profile.d/modules.sh
module purge || true
module use /apps/modulefiles/Core
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate calculate_metrics

export PYTHONPATH=/opig-shared/users/lina4783/ABB4

BASE=/opig-shared/users/lina4783/abb4_experiments/evaluation/cluster_intra_rmsd
META=/opig-shared/users/lina4783/structures_final/test_meta.csv
OUT=${BASE}/outputs

python ${BASE}/aggregate_same_length_pairwise_rmsd.py \
    --pairs_path ${OUT}/cluster_gt_pairwise_pairs.csv \
    --meta_path ${META} \
    --cluster_rmsd_path ${OUT}/cluster_gt_pairwise_rmsd.csv \
    --output_path ${OUT}/cluster_gt_pairwise_rmsd_same_len.csv

python ${BASE}/aggregate_same_length_pairwise_rmsd.py \
    --pairs_path ${OUT}/ckpt_5139/cluster_pred_pairwise_pairs.csv \
    --meta_path ${META} \
    --cluster_rmsd_path ${OUT}/ckpt_5139/cluster_pred_pairwise_rmsd.csv \
    --output_path ${OUT}/ckpt_5139/cluster_pred_pairwise_rmsd_same_len.csv
