#!/bin/bash
#SBATCH --job-name=download_v3
#SBATCH --output=download_new2.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G 
#SBATCH --gpus-per-node=l4-24g:1

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/work-gcp-europe-west4-a/yuqian_fu/Ego/huggingface
cd /home/yuqian_fu/Projects/PSALM

srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f /home/yuqian_fu/final.yml -- \
sh -c "
    python download.py
    "

