#!/bin/bash
#SBATCH --job-name=envir
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=4G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=04:00:00         # Job timeout
#SBATCH --output=envir.log      # Redirect stdout to a log file
#SBATCH --gpus-per-task=l4-24g:1


srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f /home/yuqian_fu/environment_psalm.yml -- \
sh -c "
echo love
    "





