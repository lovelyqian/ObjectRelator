import argparse
import json
import tqdm
import cv2
import os
import numpy as np
from pycocotools import mask as mask_utils
import random
from PIL import Image
from natsort import natsorted

EVALMODE = "test"


def blend_mask(input_img, binary_mask, alpha=0.5):
    if input_img.ndim == 2:
        return input_img
    mask_image = np.zeros(input_img.shape, np.uint8)
    mask_image[:, :, 1] = 255
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
#output /vis_piano
#--show_gt要加上
if __name__ == "__main__":


    set_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval/ImageSets/2017/train.txt"
    data_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/trainval/JPEGImages/480p"
    output_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/vis_davis"
    mask_path = "/data/work-gcp-europe-west4-a/yuqian_fu/datasets/DAVIS/2017/predictions_psalmpretrained_finetune_ep3/480p"
    model_name = mask_path.split("/")[-2]

    video_names = []
    with open(set_path, 'r') as f:
        for line in f:
            video_names.append(line.strip())
    video_names = ["bike-packing"]


    for name in tqdm.tqdm(video_names):
        #实验需改动
        prediction_path = os.path.join(mask_path, name)
        if not os.path.exists(prediction_path):
            print(name)
            continue
        

        file_names = natsorted(os.listdir(prediction_path))
        idxs = [f.split(".")[0] for f in file_names]

        out_path = f"{output_path}/{name}/predictions_{model_name}"
        os.makedirs(
            out_path, exist_ok=True
        )



        #为了节省内存 实际上可以idx[:60]来可视化部分帧
        for id in idxs:
            frame_idx = id
            frame = cv2.imread(
                f"{data_path}/{name}/{frame_idx}.jpg"
            )
            mask = Image.open(f"{prediction_path}/{frame_idx}.png")
            mask = np.array(mask)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            try:
                mask = upsample_mask(mask, frame)
                out = blend_mask(frame, mask)
            except:
                breakpoint()

            cv2.imwrite(
                f"{out_path}/{frame_idx}.jpg",
                out,
            )

