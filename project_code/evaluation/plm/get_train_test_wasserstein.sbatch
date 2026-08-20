#!/bin/bash
#SBATCH --job-name=wasserstein_train_test

#SBATCH --partition=standard-opig-gpu
# SBATCH --nodelist=nagagpu06.cpu.stats.ox.ac.uk

#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/plm/script_outputs/wasserstein_train_test%j.out
#SBATCH --error=/opig-shared/users/lina4783/abb4_experiments/plm/script_outputs/wasserstein_train_test%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1

ENV_PYTHON="/scratch/fast/lina4783/conda_envs/esmc/bin/python"

"${ENV_PYTHON}" -c "import ot, torch, numpy, pandas; print('Dependencies OK')"

"${ENV_PYTHON}" /opig-shared/users/lina4783/abb4_experiments/plm/get_train_test_wasserstein.py \
  --train /opig-shared/users/lina4783/structures_final/train_meta.csv \
  --test /opig-shared/users/lina4783/structures_final/test_meta.csv \
  --id-col pdb_name \
  --seq-col concat_CDR \
  --manifest /opig-shared/users/lina4783/abb4_experiments/plm/esmc_600m_vhvl.csv \
  --embedding-mode vhvl_cdr \
  --outdir /opig-shared/users/lina4783/abb4_experiments/plm/wasserstein_out/vhvl_wasserstein_cdrs_600m \
  --similarity-type wasserstein \
  --n-jobs 10 \
  --resume
