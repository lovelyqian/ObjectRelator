import argparse
import json
import tqdm
import cv2
import os
import numpy as np
import random
from pycocotools.mask import encode, decode, frPyObjects

EVALMODE = "test"


def fuse_mask(mask_list):
    fused_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        fused_mask[mask == 1] = 1
    return fused_mask


def blend_mask(input_img, binary_mask, alpha=0.5, color="g"):
    if input_img.ndim == 2:
        return input_img
    mask_image = np.zeros(input_img.shape, np.uint8)
    if color == "r":
        mask_image[:, :, 0] = 255
    if color == "g":
        mask_image[:, :, 1] = 255
    if color == "b":
        mask_image[:, :, 2] = 255
    if color == "o":
        mask_image[:, :, 0] = 255
        mask_image[:, :, 1] = 165
        mask_image[:, :, 2] = 0
    if color == "c":
        mask_image[:, :, 0] = 0
        mask_image[:, :, 1] = 255
        mask_image[:, :, 2] = 255
    if color == "p":
        mask_image[:, :, 0] = 128
        mask_image[:, :, 1] = 0
        mask_image[:, :, 2] = 128
    if color == "l":
        mask_image[:, :, 0] = 128
        mask_image[:, :, 1] = 128
        mask_image[:, :, 2] = 0
    if color == "m":
        mask_image[:, :, 0] = 128
        mask_image[:, :, 1] = 128
        mask_image[:, :, 2] = 128
    if color == "q":
        mask_image[:, :, 0] = 165
        mask_image[:, :, 1] = 80
        mask_image[:, :, 2] = 30


    mask_image = mask_image * np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    blend_image = input_img[:, :, :].copy()
    pos_idx = binary_mask > 0
    for ind in range(input_img.ndim):
        ch_img1 = input_img[:, :, ind]
        ch_img2 = mask_image[:, :, ind]
        ch_img3 = blend_image[:, :, ind]
        ch_img3[pos_idx] = alpha * ch_img1[pos_idx] + (1 - alpha) * ch_img2[pos_idx]
        blend_image[:, :, ind] = ch_img3
    return blend_image


def upsample_mask(mask, frame):
    H, W = frame.shape[:2]
    mH, mW = mask.shape[:2]

    if W > H:
        ratio = mW / W
        h = H * ratio
        diff = int((mH - h) // 2)
        if diff == 0:
            mask = mask
        else:
            mask = mask[diff:-diff]
    else:
        ratio = mH / H
        w = W * ratio
        diff = int((mW - w) // 2)
        if diff == 0:
            mask = mask
        else:
            mask = mask[:, diff:-diff]

    mask = cv2.resize(mask, (W, H))
    return mask


def downsample(mask, frame):
    H, W = frame.shape[:2]
    mH, mW = mask.shape[:2]

    mask = cv2.resize(mask, (W, H))
    return mask


#datapath /datasegswap
#inference_path /inference_xmem_ego_last/coco
#output /data/work2-gcp-europe-west4-a/yuqian_fu/Ego/vis_piano
#--show_gt要加上
#视角的切换主要依赖于inference_path是什么
if __name__ == "__main__":
   
    # test_ids = os.listdir(args.datapath)
    #修改
    color = ['g', 'r', 'b', 'o', 'c', 'p', 'l', 'm', 'q']
    filter_byname_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/filter_takes_byname.json" 
    split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
    data_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
    #json_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/egoexo_val_framelevel_all.json" #debug
    json_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/exoquery_val_framelevel.json"
    output_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/vis_gt_predictions_colorful"
    #setting = "ego2exo"
    setting = "exo2ego"

    with open(split_path, "r") as fp:
        raw_takes = json.load(fp)
    with open(json_path, "r") as fp:
        datas = json.load(fp)
    with open(filter_byname_path, "r") as fp:
        take_names = json.load(fp)
    

    #random.seed(0)
    #random.shuffle(test_ids)
    # takes_ids = raw_takes['val']
    #takes_ids = random.sample(takes_ids, 1)
    # takes_ids = take_names["music"]
    # takes_ids = random.sample(takes_ids, 10)
    takes_ids = ['d300ece0-41a7-4707-a08b-2f48aedeb75f', '9feb55a1-b244-4323-b4d9-fa1af2893864', '6e5211e1-72d8-4032-ba56-b4095c0f2b36', 'b74169ba-77b0-447d-aeff-cb7934ff6711', '58e5920e-e71c-4184-a0ae-a5f1f6ac3294', '0fe5b647-cdd0-43e9-8710-b33a2e0f83ef', 'b5f8232e-5686-43ba-8e7a-ffeadb232444', 'ae2207b2-9b33-4f6a-aeb5-e2718a253c3b', 'd8db054f-b5ee-41c3-ba66-cb84ed2d286e']
    #takes_ids = ['5c12c6e3-d34e-4e9a-bd7f-74aed58427a7', 'c4c0e97d-135b-4dec-b736-e184442bd42e', 'b511dfed-58f4-4c91-bf0a-f8ce9d47aea9', 'cc8bfc86-777e-4537-b207-9c599e9a1f40', 'b60db7b1-c623-4f04-8c63-695c24b4648b', 'eafee432-77a1-4f9c-949e-aea614671b1c', 'ede7b9c1-c501-40b3-bc6d-205ea142366c', 'd4c27ee9-443b-4ebf-a3ef-57829e24d991', '5bd830a8-5752-4aab-bd01-e9fb518864fe', 'e09f97c8-0493-4d0a-98a6-e678d4c7dc00']
    #takes_ids = ['76ec56ea-a516-4988-a516-40d04a5b21de', '32dfc6fe-c801-4ce5-bb39-8841fca7a075']
    #takes_ids = ["0fe5b647-cdd0-43e9-8710-b33a2e0f83ef", "7c341853-eea3-4243-ab99-e11919aefa4c", "27369696-d356-4d1b-8a22-be8bd4442150", "f291d174-596d-471f-836b-993315197824", "d300ece0-41a7-4707-a08b-2f48aedeb75f", "2fe390a8-1506-4420-9008-74199f92797b"]
    for take_id in tqdm.tqdm(takes_ids):
        data_list = []
        for data in datas:
            if data["video_name"] == take_id:
                data_list.append(data)
        #获取每一个take下的摄像机
        data_tmp = data_list[0]
        target_cam = data_tmp["image"].split("/")[-2]
        query_cam = data_tmp["first_frame_image"].split("/")[-2]

        #开始按帧保存fuse-mask
        for data in data_list:
            name = data["image"].split("/")[-1]
            frame_idx = name.split(".")[0]
            #target gt
            frame_target = cv2.imread(
                            f"{data_path}/{data['image']}"
                        )
            
            #debug 提高分辨率
            # h,w = frame_target.shape[:2]
            # frame_target = cv2.resize(frame_target, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

            for i,ann in enumerate(data["anns"]):
                mask = decode(ann["segmentation"])
                mask = downsample(mask, frame_target)
                #mask = upsample_mask(mask, frame_target) #debug 提高分辨率
                frame_target = blend_mask(frame_target, mask, color=color[i])
        
            os.makedirs(
                            f"{output_path}/{setting}/music/{take_id}/gt/{target_cam}", #debug
                            exist_ok=True,
                        )
            cv2.imwrite(
                            f"{output_path}/{setting}/music/{take_id}/gt/{target_cam}/{frame_idx}.jpg",  #debug
                            frame_target,
                        )
            
            #query gt
            frame_query = cv2.imread(
                            f"{data_path}/{data['first_frame_image']}"
                        )
            
            # h2,w2 = frame_query.shape[:2]  #debug: 提高分辨率
            # frame_query = cv2.resize(frame_query, (w2 * 2, h2 * 2), interpolation=cv2.INTER_CUBIC)


            for i,ann in enumerate(data["first_frame_anns"]):
                mask = decode(ann["segmentation"])
                mask = downsample(mask, frame_query)
                #mask = upsample_mask(mask, frame_query)   #debug 提高分辨率
                frame_query = blend_mask(frame_query, mask, color=color[i])

            os.makedirs(
                            f"{output_path}/{setting}/music/{take_id}/gt/{query_cam}",  #debug
                            exist_ok=True,
                        )
            cv2.imwrite(
                            f"{output_path}/{setting}/music/{take_id}/gt/{query_cam}/{frame_idx}.jpg",  #debug
                            frame_query,
                        )