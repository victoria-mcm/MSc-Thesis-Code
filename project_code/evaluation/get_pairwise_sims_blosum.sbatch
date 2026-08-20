#!/bin/bash
#SBATCH --job-name=pairwise_sims_blosum

#SBATCH --partition=standard-opig-gpu
# SBATCH --nodelist=nagagpu06.cpu.stats.ox.ac.uk

#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/script_outputs/pairwise_sims_blosum%j.out

set -euo pipefail

# Use the project conda env directly; `conda activate` can fail on compute nodes
# and silently fall back to system python (which lacks numpy).
ENV_PYTHON="${HOME}/b34d5526-ba19-4cf5-9b10-b36e0ce4fe0c/.conda/envs/antibody_similarity/bin/python"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  ENV_PYTHON="/ceph/volumes/homes_rstudent/lina4783/b34d5526-ba19-4cf5-9b10-b36e0ce4fe0c/.conda/envs/antibody_similarity/bin/python"
fi
if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Could not find antibody_similarity python at:" >&2
  echo "  ${HOME}/b34d5526-ba19-4cf5-9b10-b36e0ce4fe0c/.conda/envs/antibody_similarity/bin/python" >&2
  echo "  /ceph/volumes/homes_rstudent/lina4783/b34d5526-ba19-4cf5-9b10-b36e0ce4fe0c/.conda/envs/antibody_similarity/bin/python" >&2
  exit 1
fi

"${ENV_PYTHON}" -c "import numpy, pandas; from Bio.Align import PairwiseAligner"

"${ENV_PYTHON}" /opig-shared/users/lina4783/abb4_experiments/evaluation/get_train_test_similarities.py \
  --train /opig-shared/users/lina4783/structures_final/train_meta.csv \
  --test /opig-shared/users/lina4783/structures_final/test_meta.csv \
  --seq-col cdr3_h \
  --id-col pdb_name \
  --similarity-type blosum \
  --outdir cdrh3_similarity_out/train_test_blosum
