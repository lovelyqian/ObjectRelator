#!/bin/bash
#SBATCH --job-name=json
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=16     # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=4G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=04:00:00         # Job timeout
#SBATCH --output=create_json.log      # Redirect stdout to a log file
#SBATCH --gpus-per-node=l4-24g:1


cd /home/yuqian_fu/Projects/PSALM/datasets

srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f /home/yuqian_fu/Projects/PSALM/psalm.yml -- \
sh -c "
    python build_ego.py
"





