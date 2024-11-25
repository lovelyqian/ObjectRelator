#!/bin/bash
#SBATCH --job-name=mask
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1             # Number of tasks (total)
#SBATCH --cpus-per-task=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-gpu=42G
#SBATCH --time=04:00:00         # Job timeout
#SBATCH --gpus-per-node=l4-24g:1
#SBATCH --output=mask.log      # Redirect stdout to a log file



cd /home/yuqian_fu/Projects/PSALM/datasets

srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f /home/yuqian_fu/environment_xmem.yml -- \
sh -c "
    python build_DAVIS_test.py
    "





