#!/bin/bash
#SBATCH --job-name=esmc_test

#SBATCH --partition=standard-opig-gpu
# SBATCH --nodelist=nagagpu06.cpu.stats.ox.ac.uk

#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/plm/script_outputs/esmc_test%j.out

set -euo pipefail

# Keep Hugging Face / XDG caches off the full home quota; model is already cached here.
export SCRATCH_BASE=/scratch/fast/lina4783
export XDG_CACHE_HOME="${SCRATCH_BASE}/xdg_cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

# Load environment
source /etc/profile.d/modules.sh
module purge
module use /apps/modulefiles/Core
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /scratch/fast/lina4783/conda_envs/esmc

echo "Host: $(hostname)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi || true
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

#run script
python /opig-shared/users/lina4783/abb4_experiments/plm/test_esmc.py