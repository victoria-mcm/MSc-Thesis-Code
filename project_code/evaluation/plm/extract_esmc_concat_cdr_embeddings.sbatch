#!/bin/bash
#SBATCH --job-name=esmc_concat_cdr

#SBATCH --partition=standard-opig-gpu
# SBATCH --nodelist=nagagpu06.cpu.stats.ox.ac.uk

#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/plm/script_outputs/esmc_concat_cdr%j.out
#SBATCH --error=/opig-shared/users/lina4783/abb4_experiments/plm/script_outputs/esmc_concat_cdr%j.out

set -euo pipefail

# Keep Hugging Face / XDG caches off the full home quota; model is already cached here.
export SCRATCH_BASE=/scratch/fast/lina4783
export XDG_CACHE_HOME="${SCRATCH_BASE}/xdg_cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

# Use scratch env python directly; conda activate can fail silently on compute nodes.
ENV_PYTHON="/scratch/fast/lina4783/conda_envs/esmc/bin/python"

"${ENV_PYTHON}" -c "import torch; print('PyTorch:', torch.__version__)"
nvidia-smi || true
"${ENV_PYTHON}" -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run script
"${ENV_PYTHON}" /opig-shared/users/lina4783/abb4_experiments/plm/extract_esmc_concat_cdr_embeddings.py \
--seq-col full_seq_concat \
--outdir /opig-shared/users/lina4783/abb4_experiments/plm/embeddings/esmc_600m_vhvl \
--manifest /opig-shared/users/lina4783/abb4_experiments/plm/esmc_600m_vhvl.csv \
--model biohub/ESMC-600M
