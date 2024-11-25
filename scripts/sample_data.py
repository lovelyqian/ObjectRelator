import json
import random

json_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/ExoQuery_FullTrain_newprompt_all_instruction.json"
output_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/ExoQuery_SmallTrain_newprompt_instruction.json"

random.seed(42)
# 读取原始的完整数据
with open(json_path, 'r') as f:
    data_full = json.load(f)

# 计算三分之一的数据量
third_size = len(data_full) // 3


# 随机选择三分之一的数据
# data_smallsize = random.sample(data_full, third_size)

# 按照原顺序，每间隔三个数据存储一个
data_smallsize = data_full[::3]



# 将缩小后的数据保存到新的json文件中
with open(output_path, 'w') as f:
    json.dump(data_smallsize, f)

print(f'Saved {len(data_smallsize)} samples to smallsize_version')