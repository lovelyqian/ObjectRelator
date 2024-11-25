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


    split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
    data_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
    output_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/vis_gt_predictions"
    setting = "ego2exo"
    #setting = "exo2ego"
    mask_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/mask_predictions/egofullmodel_smalljson"
    model_name = mask_path.split("/")[-1]


    with open(split_path, "r") as fp:
        raw_takes = json.load(fp)
    takes_ids = raw_takes['val']
    takes_ids = ["6449cb24-e14c-4238-9d57-2e0efc4794ba", "b511dfed-58f4-4c91-bf0a-f8ce9d47aea9"]


    for take_id in tqdm.tqdm(takes_ids):
        #实验需改动
        prediction_path = os.path.join(mask_path, take_id)
        if not os.path.exists(prediction_path):
            print(take_id)
            continue
        target_cam = os.listdir(prediction_path)[0]
        prediction_path = os.path.join(prediction_path, target_cam)

        file_names = natsorted(os.listdir(prediction_path))
        idxs = [int(f.split(".")[0]) for f in file_names]

        out_path = f"{output_path}/{setting}/{take_id}/predictions_{model_name}/{target_cam}"
        os.makedirs(
            out_path, exist_ok=True
        )



        #为了节省内存 实际上可以idx[:60]来可视化部分帧
        for id in idxs:
            frame_idx = str(id)
            frame = cv2.imread(
                f"{data_path}/{take_id}/{target_cam}/{frame_idx}.jpg"
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

