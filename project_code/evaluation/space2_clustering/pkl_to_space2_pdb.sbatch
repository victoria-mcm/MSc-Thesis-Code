#!/bin/bash
#SBATCH --job-name=space2_pkl_export
#SBATCH --partition=standard-opig-gpu
#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/space2_clustering/logs/pkl_export_%j.out

set -euo pipefail

export PYTHONPATH="/opig-shared/users/lina4783/ABB4:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ENV_PYTHON="/scratch/fast/lina4783/conda_envs/space2_clustering/bin/python"
SCRIPT="/opig-shared/users/lina4783/abb4_experiments/evaluation/space2_clustering/pkl_to_space2_pdb.py"

# Override for full run:
RUN_NAME=full
METADATA=/opig-shared/users/lina4783/structures_final/metadata.csv

#RUN_NAME="${RUN_NAME:-baby_test}"
#if [[ "${RUN_NAME}" == "full" ]]; then
#  METADATA="/opig-shared/users/lina4783/structures_final/metadata.csv"
#else
#  METADATA="/opig-shared/users/lina4783/structures_final/baby_test_meta.csv"
#fi

OUT_BASE="/opig-shared/users/lina4783/abb4_experiments/evaluation/space2_clustering/outputs/${RUN_NAME}"
PDB_DIR="${OUT_BASE}/pdbs"
MANIFEST="${OUT_BASE}/export_manifest.csv"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Missing env at ${ENV_PYTHON}. Run setup_space2_env.sh first."
  exit 1
fi

"${ENV_PYTHON}" "${SCRIPT}" \
  --metadata "${METADATA}" \
  --pdb-dir "${PDB_DIR}" \
  --manifest "${MANIFEST}"
