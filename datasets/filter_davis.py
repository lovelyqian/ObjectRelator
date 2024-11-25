import json
import os
from PIL import Image
import numpy as np
from pycocotools.mask import encode, decode, frPyObjects
from tqdm import tqdm
import copy

if __name__ == '__main__':
    data_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval_val_psalm_train_new_augument_n4_instruction.json"
    save_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval_val_psalm_train_new_augument_n4_instruction_1118.json"
    k = 0
    sent_id = 0
    with open(data_path, "r") as fp:
        datas = json.load(fp)
    new_datalist = []
    for data in datas:
        if data['anns'] == []:
            continue
        data['new_img_id'] = k
        k += 1
        instruct_list = data["instruction"]
        instruct_list_new = []
        for sample in instruct_list:
             sample["sent_id"] = sent_id
             sent_id += 1
             instruct_list_new.append(sample)
        data["instruction"] = instruct_list_new
        new_datalist.append(data)
    
    print(len(new_datalist))
    print("sent_id:", sent_id)
   
    with open(save_path, 'w') as json_file:
        json.dump(new_datalist, json_file)