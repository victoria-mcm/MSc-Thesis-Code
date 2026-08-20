#!/bin/bash
#SBATCH --job-name=w2_well_ill_q
#SBATCH --partition=standard-opig-gpu
#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/plm/script_outputs/wasserstein_well_ill_quartiles_%j.out
#SBATCH --error=/opig-shared/users/lina4783/abb4_experiments/plm/script_outputs/wasserstein_well_ill_quartiles_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1

ENV_PYTHON="/scratch/fast/lina4783/conda_envs/esmc/bin/python"
SCRIPT="/opig-shared/users/lina4783/abb4_experiments/plm/get_train_test_wasserstein.py"
PLM="/opig-shared/users/lina4783/abb4_experiments/plm"
MANIFEST="${PLM}/esmc_300m_concat_cdr.csv"
WELL="${PLM}/test_meta_well_predicted.csv"
ILL="${PLM}/test_meta_ill_predicted.csv"
OUT="${PLM}/wasserstein_out"

COMMON=(
  --id-col pdb_name
  --seq-col concat_CDR
  --manifest "${MANIFEST}"
  --embedding-mode cdr
  --similarity-type wasserstein
  --n-jobs 10
  --resume
)

"${ENV_PYTHON}" -c "import ot, torch, numpy, pandas; print('Dependencies OK')"

echo "=== within well (train=test=well quartile) ==="
"${ENV_PYTHON}" "${SCRIPT}" \
  --train "${WELL}" \
  --test "${WELL}" \
  --outdir "${OUT}/well_well_concat_cdr_300m" \
  "${COMMON[@]}"

echo "=== within ill (train=test=ill quartile) ==="
"${ENV_PYTHON}" "${SCRIPT}" \
  --train "${ILL}" \
  --test "${ILL}" \
  --outdir "${OUT}/ill_ill_concat_cdr_300m" \
  "${COMMON[@]}"

echo "=== between well (train) vs ill (test) ==="
"${ENV_PYTHON}" "${SCRIPT}" \
  --train "${WELL}" \
  --test "${ILL}" \
  --outdir "${OUT}/well_train_ill_test_concat_cdr_300m" \
  "${COMMON[@]}"

echo "All three Wasserstein jobs finished."
