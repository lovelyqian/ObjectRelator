#!/bin/bash
#SBATCH --job-name=json
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=16       # Number of CPU cores (threads) per task
#SBATCH --mem-per-gpu=24G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=24:00:00         # Job timeout
#SBATCH --output=json_exo.log      # Redirect stdout to a log file
#SBATCH --gpus-per-node=l4-24g:1



cd /home/yuqian_fu/Projects/PSALM

srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f psalm.yml -- \
sh -c "
    pip install natsort && \
    python ./datasets/build_exo_framelevel.py
"





