#!/bin/bash
#SBATCH --job-name=abb4_metrics
#S BATCH --account=opig                        # Project Account  

#SBATCH --partition=standard-opig-gpu
# SBATCH --nodelist=nagagpu09.cpu.stats.ox.ac.uk

#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
# SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/script_outputs/metrics%j.out

set -euo pipefail

# Load environment

# Load modules
source /etc/profile.d/modules.sh

# Avoid failing if the login shell has a stale/broken conda module loaded
module purge || true

module use /apps/modulefiles/Core
module load conda

# Initialise conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate calculate_metrics

#python -m pip uninstall -y abb4
#python -m pip install -e /opig-shared/users/lina4783/ABB4



#run script
python /opig-shared/users/lina4783/ABB4/abb4/analysis/postproc/calculate_struc_pred_metrics.py \
    --pred_path /opig-shared/users/lina4783/abb4_experiments/evaluation/predictions_ckpt_5139_imgt \
    --meta_path /opig-shared/users/lina4783/structures_final/test_meta.csv \
    --dset test \
    --all_samples \
    --n_jobs 10