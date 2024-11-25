#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100-40g:3
#SBATCH --cpus-per-gpu=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-gpu=40G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=48:00:00         # Job timeout
#SBATCH --output=debug_1112.log     # Redirect stdout to a log file
#SBATCH --nodelist=gcp-us-3


cd /home/yuqian_fu/Projects/PSALM


srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \
mkenv -f psalm.yml -- \
sh -c "

python psalm/eval/eval_ego4d.py --image_folder /data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap --model_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/ExoQuery_241026_psalm_retrain_withPretrained_onsmallTrainJson_ep4_correctdata/checkpoint-3424 --json_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/exoquery_val_framelevel.json




"

#for muliticondition
#python psalm/eval/eval_ego4d_MultiCondition.py --image_folder /data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap --model_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/OurFullModel-exp4-oldMultiConditionStage1-SSLAfterMultiCondition-eculidean-k1-1104/checkpoint-3056 --json_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/egoexo_val_framelevel_newprompt_all_instruction.json
#python psalm/eval/eval_ego4d.py --image_folder /data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap --model_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/ExoQuery_241026_psalm_retrain_withPretrained_onsmallTrainJson_ep4_correctdata/checkpoint-3424 --json_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/exoquery_val_framelevel.json

#for davis
#python psalm/eval/eval_davis_sslmodel.py --image_folder /data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS  --model_path /data/work-gcp-europe-west4-a/yuqian_fu/Ego/DAVIS-PSALMModel-from-PSALMPretrained-Epoch4-1111/checkpoint-255 --json_path /data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval_val_psalm_val.json
