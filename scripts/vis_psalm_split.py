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
#output /vis_piano
#--show_gt要加上
if __name__ == "__main__":

    color = ['g', 'r', 'b', 'o', 'c', 'p']
    filter_byname_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/filter_takes_byname.json" 
    split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
    data_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
    output_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/vis_gt_predictions_split_1113"
    setting = "ego2exo"  #debug
    #setting = "exo2ego"
    #mask_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/mask_predictions/psalm_original"
    mask_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/mask_predictions/egofullmodel_smalljson" #debug
    #mask_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/mask_predictions/exofullmodel_smalljson"
    #mask_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/mask_predictions/psalmfinetune_ego_exo_smalljson"
    model_name = mask_path.split("/")[-1]


    with open(split_path, "r") as fp:
        raw_takes = json.load(fp)
    with open(filter_byname_path, "r") as fp:
        take_names = json.load(fp)
    # takes_ids = raw_takes['val']
    # takes_ids = take_names["soccer"]
    # takes_ids = random.sample(takes_ids, 10)
    #print(takes_ids)

    takes_ids = ["7785da94-b19b-491c-94dc-89106e095f79"]
    

    #takes_ids = ["0fe5b647-cdd0-43e9-8710-b33a2e0f83ef", "7c341853-eea3-4243-ab99-e11919aefa4c", "27369696-d356-4d1b-8a22-be8bd4442150", "f291d174-596d-471f-836b-993315197824", "d300ece0-41a7-4707-a08b-2f48aedeb75f", "2fe390a8-1506-4420-9008-74199f92797b"]

    #takes_ids_bike_exo2ego = ['c935b13b-462a-4ced-a041-f06592dff0e5', '99052842-9369-415d-b87b-1bf756ab13dd', '6fbe4821-85ec-4834-80bb-0a214fa80893', '2dc49d13-5347-4d4b-8506-5510f2ae0b84', '85b4c3c3-6486-43bd-b5b6-7cae6ffd4c9f', '349a435d-c2ad-4c64-9067-8733e7cd5ec9', '71b8a896-84d6-4da6-97ab-78cfee6cd5da', '1bc1334f-2546-4175-a9f4-b4338f46a246', 'cda93e67-c411-48cd-83f0-d7aafc6e4b3f', '812bb3bb-217a-454c-995c-d3e995576b81']
    #takes_ids_cooking_exo2ego = ['e420250d-b67f-4c46-bc40-31b8c9003d7e', '645a45ce-c993-4f67-b136-fe8c3bad9b6b', '3ca3a186-a650-4f8b-a9d0-f0bdb19bd860', '8caf94c0-835e-48cd-9e7c-fa6bfddc0d5c', '4d799166-0362-4daa-9f26-72e27a4feb31', 'ac5bc1ee-b7d5-4948-90aa-8f418ca4c0fa', 'aa6257e0-001a-422b-9935-2f9c2b5973e6', '99a39dee-cd6d-4a27-b32e-bc2dee856c02', 'a56ce926-8c85-4883-9ff5-dd7779d71b64']
    #takes_ids_basketball_exo2ego = ['936e7e6e-aa06-4aca-b877-8d5f3c59ecc1', '0669b09c-fda3-4fbb-9c41-b9b1fd3ff31e', '2ec7440e-5e29-410b-a5f6-40e3a2e45a31', '9e08e1e7-179a-4dac-bd0e-6e98e47407d2', '1247a29c-9fda-47ac-8b9c-78b1e76e977e', '9ebe4ac3-1472-4094-9e73-6014e78e0539', '6c4e2422-e83b-4ae5-8b9f-7bfdf3f2d3f7', '2e00eb80-4fd0-4ba5-bcbb-5e671d7f3627', '6449cb24-e14c-4238-9d57-2e0efc4794ba', '5d25cc61-f04d-47e3-9f0e-dbbf7707f0a0']

    #takes_ids_music_ego2exo = ['b5f8232e-5686-43ba-8e7a-ffeadb232444', '9feb55a1-b244-4323-b4d9-fa1af2893864', 'd8db054f-b5ee-41c3-ba66-cb84ed2d286e', 'ae2207b2-9b33-4f6a-aeb5-e2718a253c3b', '7a0757f4-8ada-424a-b4fb-81fac89c7259']
    #takes_ids_bike_ego2exo = ['6fbe4821-85ec-4834-80bb-0a214fa80893', '725b6b84-0a79-4053-b581-828a5da77753', '2bb5152e-cab4-46dc-9aa6-e73007f7df8b', '2da6c2ce-a88f-4790-9642-8dfa810ae91b', 'dfe05c45-851b-4a94-9c22-05aca881fda6', '7785da94-b19b-491c-94dc-89106e095f79', 'c692c40e-f2ca-4338-bb9e-1c779a7288a2', '85b4c3c3-6486-43bd-b5b6-7cae6ffd4c9f', '349a435d-c2ad-4c64-9067-8733e7cd5ec9', 'cda93e67-c411-48cd-83f0-d7aafc6e4b3f']
    #takes_ids_basketball_ego2exo = ['053fcb5e-1f33-493f-88c2-f61df147b4e5', '1247a29c-9fda-47ac-8b9c-78b1e76e977e', '47850a65-9719-4770-92e4-e428953addb3', '2e00eb80-4fd0-4ba5-bcbb-5e671d7f3627', '6c4e2422-e83b-4ae5-8b9f-7bfdf3f2d3f7', 'f2934ab9-bcfd-42fa-a329-7d0fc28818aa', '3a1b3ec6-13fd-43f4-8af6-f943953e01e4', '67e74067-407f-4aa9-b8d9-dbac6ea2464f', '6ca51642-c089-4989-b5a3-07977ec927d7', '09417ca4-3572-4ba1-a1db-7eaf3bd0b1c8']


    for take_id in tqdm.tqdm(takes_ids):
        #实验需改动
        prediction_path = os.path.join(mask_path, take_id)
        if not os.path.exists(prediction_path):
            print(take_id)
            continue

        #debug:检查是否存在
        # test_path = f"{output_path}/{setting}/music/{take_id}"
        # if os.path.exists(test_path):
        #     continue
        # print(take_id)


        #here is our model
        target_cam = os.listdir(prediction_path)[0] #debug

        #here is psalm_original model #debug
        # cams = natsorted(os.listdir(prediction_path))
        # if setting == "ego2exo":
        #     target_cam = cams[1]
        # elif setting == "exo2ego":
        #     target_cam = cams[0]
            

        prediction_path = os.path.join(prediction_path, target_cam)

        file_names = natsorted(os.listdir(prediction_path))
        idxs = [int(f.split(".")[0]) for f in file_names]

        


        #为了节省内存 实际上可以idx[:60]来可视化部分帧
        for id in idxs:
            frame_idx = str(id)
            frame = cv2.imread(
                f"{data_path}/{take_id}/{target_cam}/{frame_idx}.jpg"
            )
            mask = Image.open(f"{prediction_path}/{frame_idx}.png")
            mask = np.array(mask)
            unique_instances = np.unique(mask)
            unique_instances = unique_instances[unique_instances != 0]
            for i,instance_value in enumerate(unique_instances):
                binary_mask = (mask == instance_value).astype(np.uint8)
                binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
                binary_mask = upsample_mask(binary_mask, frame)
                out = blend_mask(frame, binary_mask, color=color[0])
                os.makedirs(
                            f"{output_path}/{setting}/bike/{take_id}/predictions_{model_name}/obj_{i}/{target_cam}",  #debug
                            exist_ok=True,
                        )
                cv2.imwrite(
                    f"{output_path}/{setting}/bike/{take_id}/predictions_{model_name}/obj_{i}/{target_cam}/{frame_idx}.jpg",
                    out,
                )


         

            

