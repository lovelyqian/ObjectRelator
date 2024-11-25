#!/bin/bash
#SBATCH --job-name=psalm_retrain_FullJson
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100-40g:8
#SBATCH --cpus-per-gpu=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-gpu=40G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=96:00:00         # Job timeout
#SBATCH --output=EgoQuery_241022_psalm_retrain_withPretrained_onFullTrainJson.log      # Redirect stdout to a log file
#SBATCH --nodelist=gcp-us-3

head_node=$(hostname)
rdzv_port=$((30000+SLURM_JOB_ID%30000))

cd /home/yuqian_fu/Projects/PSALM
export NCCL_SOCKET_IFNAME=eth
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f psalm.yml -- \
sh -c "
    bash ./scripts/train.sh
    "





