#!/bin/bash
#SBATCH --job-name=psalm_retrain_FullJson
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100-40g:4
#SBATCH --cpus-per-gpu=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-gpu=40G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=96:00:00         # Job timeout
#SBATCH --output=ExoQuery_241026_psalm_retrain_withPretrained_onfullTrainJson_ep4.log      # Redirect stdout to a log file
#SBATCH --nodelist=gcp-us-3


head_node=$(hostname)
rdzv_port=$((30000+SLURM_JOB_ID%30000))
# 设置网络接口（根据你的环境选择 eth 或其他接口）
export NCCL_SOCKET_IFNAME=eth  # 检查是否需要修改 eth0 为正确的网络接口

# 获取 GPU UUID 列表并转换为 GPU 索引
gpu_uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader)

# 显示 GPU UUID 和对应的索引
echo "GPU UUIDs and corresponding indices:"
nvidia-smi --query-gpu=index,uuid --format=csv

# 创建 UUID -> index 映射并动态设置 CUDA_VISIBLE_DEVICES
index=0
visible_devices=""
for uuid in $gpu_uuids; do
    if [ -z "$visible_devices" ]; then
        visible_devices="$index"
    else
        visible_devices="$visible_devices,$index"
    fi
    index=$((index + 1))
done

# 设置 CUDA_VISIBLE_DEVICES 环境变量为 GPU 索引
export CUDA_VISIBLE_DEVICES=$visible_devices

# 检查是否正确设置
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"


srun --nodes "$SLURM_NNODES" --ntasks-per-node 1 -- \ 
mkenv -f psalm.yml -- \
sh -c "
    bash ./scripts/train.sh
    "




