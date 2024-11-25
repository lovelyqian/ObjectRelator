import argparse
import json
import tqdm
import cv2
import os
import numpy as np
from pycocotools import mask as mask_utils
import random
from PIL import Image
import copy
import string


def extract_object_name(text):
    parts = text.split("is")
    if len(parts) > 1:
        return parts[1].strip()
    return None


if __name__ == "__main__":
    #每一帧随机增强数量
    augu_num = 4
    original_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval_val_psalm_instruction_train_correct_new.json"
    with open(original_path, "r") as fp:
        datas_origianl = json.load(fp)
    #用来存放新数据 最后的数据是原始数据和新数据的拼接
    new_data = []
    json_path = '/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval_val_psalm_withtext_train_new_targetframe.json'
    save_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval_val_psalm_train_new_augument_n4_instruction.json"
    with open(json_path, "r") as fp:
        datas = json.load(fp)
    

    #用来计数
    total_num = len(datas)
    k = 0
    sent_id = 9444

    #统计video_name
    set_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval/ImageSets/2017/train.txt"
    video_names = []
    with open(set_path, 'r') as f:
        for line in f:
            video_names.append(line.strip())


    for video in video_names:
        #在同一个video内做增强
        data_list = []
        for data in datas:
            if data["video_name"] == video:
                data_list.append(data)
        for data in data_list:
            sample_unique_instances = []
            for ann in data['anns']:
                sample_unique_instances.append(ann['category_id'])

            data_sample = random.sample(data_list, augu_num)
            for sample in data_sample:
                #对随机挑选出的k个样本做筛选
                if data['new_img_id'] == sample['new_img_id']:
                    continue
                #当前帧物体的数量要小于参考帧的数量，筛选
                if len(data['anns']) > len(sample['anns']):
                    continue

                unique_instances = []
                #参考帧的物体类别一定要包含当前帧的
                for ann in sample['anns']:
                    unique_instances.append(ann['category_id'])
                
                skip = False
                for id in sample_unique_instances:
                    if id not in unique_instances:
                        skip = True
                        break
                if skip:
                    continue

                first_frame_anns = copy.deepcopy(sample['anns'])
                if len(data['anns']) < len(first_frame_anns):
                    first_frame_anns = [ann for ann in first_frame_anns if ann['category_id'] in sample_unique_instances]
                # print(len(data['anns']), len(first_frame_anns))
                # print("unique_instances", unique_instances)
                # print("sample_unique_instances", sample_unique_instances) #debug
                assert len(data['anns']) == len(first_frame_anns)

                skip_text = False #debug
                instruct_list = []
                for anno in first_frame_anns:
                    text = anno["text"]
                    # 提取is之后的句子
                    raw = extract_object_name(text)
                    #将raw变小写
                    if raw == None:  #debug
                        skip_text = True
                        print(sample['image'])
                        break
                    raw_lower = raw.lower()
                    # 删除 "green" 并去掉多余的空格
                    result = raw_lower.replace("green", "").strip()  
                    # 删除所有标点符号
                    sent = result.translate(str.maketrans('', '', string.punctuation))
                    tokens = sent.split()
                    sample_text = {
                        "tokens": tokens,
                        "raw": raw,
                        "sent_id": sent_id,
                        "sent": sent
                    }
                    sent_id += 1
                    instruct_list.append(sample_text)
                if skip_text:
                    continue


                data_new = {
                'image': data['image'],
                'image_info': data['image_info'],
                'anns': data['anns'],
                'first_frame_image':sample['image'],
                'first_frame_anns': first_frame_anns,
                'new_img_id': total_num+k,
                'video_name': data['video_name'],
                "instruction": instruct_list
                }
                new_data.append(data_new)
                k += 1


    data_all = datas_origianl + new_data
    with open(save_path, 'w') as f:
        json.dump(data_all, f)
    print(f'Save at {save_path}. Total sample: {len(data_all)}')