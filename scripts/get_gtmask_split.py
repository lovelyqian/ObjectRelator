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
    color = ['g', 'r', 'b', 'o', 'c', 'p']
    filter_byname_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/filter_takes_byname.json" 
    split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
    data_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
    json_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/egoexo_val_framelevel_all.json" #debug
    #json_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/exoquery_val_framelevel.json"
    output_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/vis_gt_predictions_split_1113"
    setting = "ego2exo"
    #setting = "exo2ego"

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
    # takes_ids = take_names["soccer"]
    # takes_ids = random.sample(takes_ids, 10)

    takes_ids = ["7785da94-b19b-491c-94dc-89106e095f79"]

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
                out = blend_mask(frame_target, mask, color=color[0])
                os.makedirs(
                            f"{output_path}/{setting}/bike/{take_id}/gt/obj_{i}/{target_cam}", #debug
                            exist_ok=True,
                        )
                cv2.imwrite(
                            f"{output_path}/{setting}/bike/{take_id}/gt/obj_{i}/{target_cam}/{frame_idx}.jpg",  #debug
                            out,
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
                out = blend_mask(frame_query, mask, color=color[0])
                os.makedirs(
                            f"{output_path}/{setting}/bike/{take_id}/gt/obj_{i}/{query_cam}",  #debug
                            exist_ok=True,
                        )
                cv2.imwrite(
                            f"{output_path}/{setting}/bike/{take_id}/gt/obj_{i}/{query_cam}/{frame_idx}.jpg",  #debug
                            out,
                        )





            