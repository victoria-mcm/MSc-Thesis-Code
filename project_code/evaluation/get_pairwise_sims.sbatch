#!/bin/bash
#SBATCH --job-name=pairwise_sims

#SBATCH --partition=standard-opig-gpu
# SBATCH --nodelist=nagagpu06.cpu.stats.ox.ac.uk

#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/script_outputs/pairwise_sims%j.out

set -euo pipefail

# Load environment
source /etc/profile.d/modules.sh
module purge
module use /apps/modulefiles/Core
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate antibody_similarity


#run script
python /opig-shared/users/lina4783/abb4_experiments/evaluation/get_train_test_similarities.py \
  --train /opig-shared/users/lina4783/structures_final/train_nano.csv \
  --test /opig-shared/users/lina4783/structures_final/nano_test.csv\
  --seq-col concat_CDR \
  --id-col pdb_name \
  --outdir cdrh3_similarity_out/nano_concat_CDR