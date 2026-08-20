#!/bin/bash
#SBATCH --job-name=space2_cluster
#SBATCH --partition=standard-opig-gpu
#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/space2_clustering/logs/space2_cluster_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1

ENV_PYTHON="/scratch/fast/lina4783/conda_envs/space2_clustering/bin/python"
SCRIPT="/opig-shared/users/lina4783/abb4_experiments/evaluation/space2_clustering/run_space2_clustering.py"
MERGE_SCRIPT="/opig-shared/users/lina4783/abb4_experiments/evaluation/space2_clustering/merge_space2_results.py"

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
MANIFEST="${OUT_BASE}/export_manifest.csv"
CLUSTERS="${OUT_BASE}/space2_clusters.csv"
MERGED="${OUT_BASE}/metadata_with_space2.csv"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Missing env at ${ENV_PYTHON}. Run setup_space2_env.sh first."
  exit 1
fi

"${ENV_PYTHON}" "${SCRIPT}" \
  --manifest "${MANIFEST}" \
  --output "${CLUSTERS}" \
  --summary "${OUT_BASE}/space2_cluster_summary.json" \
  --cutoff 1.25 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-12}"

"${ENV_PYTHON}" "${MERGE_SCRIPT}" \
  --metadata "${METADATA}" \
  --clusters "${CLUSTERS}" \
  --manifest "${MANIFEST}" \
  --output "${MERGED}"
