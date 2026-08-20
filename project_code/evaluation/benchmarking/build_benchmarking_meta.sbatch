#!/bin/bash
#SBATCH --job-name=test_set_filtering
#SBATCH --partition=standard-opig-gpu
#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/benchmarking/logs/test_set_filtering_%j.out

set -euo pipefail

source /etc/profile.d/modules.sh
module purge || true
module use /apps/modulefiles/Core
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate calculate_metrics

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="${HOME}/.conda/envs/calculate_metrics/bin/python"
fi

"${PYTHON}" /opig-shared/users/lina4783/abb4_experiments/evaluation/benchmarking/build_benchmarking_meta.py
