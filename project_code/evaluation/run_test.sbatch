#!/bin/bash
#SBATCH --job-name=abb_test
#S BATCH --account=opig                        # Project Account  

#SBATCH --partition=standard-opig-gpu
# SBATCH --nodelist=nagagpu06.cpu.stats.ox.ac.uk

#SBATCH --mail-user=victoria.mcmillan@linacre.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --clusters=swan
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=14-00:00:0
#SBATCH --output=/opig-shared/users/lina4783/abb4_experiments/evaluation/script_outputs/model_test%j.out

set -euo pipefail

# Load environment
source /etc/profile.d/modules.sh
module purge
module use /apps/modulefiles/Core
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate abb4_env

#python -m pip uninstall -y abb4
#python -m pip install -e /opig-shared/users/lina4783/ABB4

python -c "import abb4; print(abb4.__file__)"


#run script
python -W ignore -m torch.distributed.run --nproc_per_node=2 --master_port=12355 \
  /opig-shared/users/lina4783/ABB4/abb4/experiments/testing.py -cn testing