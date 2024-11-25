import json
from pycocotools.coco import COCO
from tqdm import tqdm
import string

def extract_object_name(text):
    parts = text.split("is")
    if len(parts) > 1:
        return parts[1].strip()
    return None

text_pth  = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/ExoQuery_val_newprompt_all_instruction.json"
save_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval_val_psalm_instruction_train.jsonss"

new_data = []
sent_id = 0

with open(text_pth, "r") as fp:
   datas = json.load(fp)



# data是一帧帧图片
for data in datas:
    instruct_list = []
    for anno in data["first_frame_anns"]:
        text = anno["text"]
        # 提取is之后的句子
        raw = extract_object_name(text)
        #将raw变小写
        raw_lower = raw.lower()
        # 删除 "green" 并去掉多余的空格
        result = raw_lower.replace("green", "").strip()  
        # 删除所有标点符号
        sent = result.translate(str.maketrans('', '', string.punctuation))
        tokens = sent.split()
        sample = {
            "tokens": tokens,
            "raw": raw,
            "sent_id": sent_id,
            "sent": sent
        }
        sent_id += 1
        instruct_list.append(sample)
        # del anno["text"]  #debug
    data["instruction"] = instruct_list
    new_data.append(data)
print(sent_id)
# with open(save_path, "w") as fp:
#     json.dump(new_data, fp)
