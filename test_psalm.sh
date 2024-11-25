#!/bin/bash
#SBATCH --job-name=psalm
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-gpu=40G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=24:00:00         # Job timeout
#SBATCH --output=psalm_1.log      # Redirect stdout to a log file
#SBATCH --gpus-per-node=a100-40g:1
#SBATCH --nodelist=gcp-us-0


cd /home/yuqian_fu/Projects/PSALM

srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f psalm.yml -- \
sh -c "
    python psalm/eval/panoptic_segmentation.py --image_folder /data/work-gcp-europe-west4-a/yuqian_fu/datasets/coco/val2017 --model_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/huggingface/hub/PSALM --json_path /data/work-gcp-europe-west4-a/yuqian_fu/datasets/coco
"





