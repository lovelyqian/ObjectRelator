#!/bin/bash
#SBATCH --job-name=mask
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1             # Number of tasks (total)
#SBATCH --cpus-per-task=16       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00         # Job timeout
#SBATCH --output=vis.log      # Redirect stdout to a log file



cd /home/yuqian_fu/Projects/PSALM/scripts

srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f /home/yuqian_fu/environment.yml -- \
sh -c "
    python /home/yuqian_fu/Projects/PSALM/scripts/get_gtmask_fuse_differentcolor.py

    "


#/home/yuqian_fu/Projects/PSALM/scripts/get_gtmask_fuse_differentcolor.py
#/home/yuqian_fu/Projects/PSALM/scripts/vis_psalm_differentcolor.py

