import json
import argparse
from pycocotools import mask as mask_utils
import numpy as np
import tqdm
from sklearn.metrics import balanced_accuracy_score

import utils
import cv2
import os
from PIL import Image
from pycocotools.mask import encode, decode, frPyObjects
from natsort import natsorted

pred_root = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/predictions/ego_query_finalnew"
split_path = "/home/yuqian_fu/Projects/ego-exo4d-relation/correspondence/SegSwap/data/split.json"
data_path = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
val_set = os.listdir(pred_root)
# val_set.remove("066cccd7-d7ca-4ce3-a80e-90ce9013c1ab")
# val_set = ["725b6b84-0a79-4053-b581-828a5da77753"]


def evaluate_take(take_id):
    num_frame = 0
    pred_path = os.path.join(pred_root, take_id)
    cams = os.listdir(pred_path)
    exo = cams[0]
    pred_path = os.path.join(pred_path, exo)

    gt_path = f"{data_path}/{take_id}/annotation.json"
    with open(gt_path, 'r') as fp:
        gt = json.load(fp)
    # objs = natsorted(list(gt["masks"].keys()))
    objs = list(gt["masks"].keys())
    # print(objs)
    # 创建逆字典
    coco_id_to_cont_id = {cont_id + 1: coco_id for cont_id, coco_id in enumerate(objs)}
    id_range = list(coco_id_to_cont_id.keys())
    # print("id_range", id_range)
    # print("coco_id_to_cont_id", coco_id_to_cont_id)

    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []

    frames = os.listdir(pred_path)
    idx = [f.split(".")[0] for f in frames]


    obj_exo = []
    for obj in objs:
        if exo in gt["masks"][obj].keys():
            obj_exo.append(obj)

    for id in idx:
        obj_range = []
        for obj in obj_exo:
            if id in gt["masks"][obj][exo].keys():
                obj_range.append(obj)

        pred_mask = Image.open(f"{pred_path}/{id}.png")
        # print(f"{pred_path}/{id}.png")
        pred_mask = np.array(pred_mask)
        unique_instances = np.unique(pred_mask)
        unique_instances = unique_instances[unique_instances != 0]
        unique_instances = [x for x in unique_instances if x in id_range]
        print(unique_instances)
        if len(unique_instances) == 0:
            continue

        num_frame += 1
        for instance_value in unique_instances:
            binary_mask = (pred_mask == instance_value).astype(np.uint8)
            h,w = binary_mask.shape
            obj_name = coco_id_to_cont_id[instance_value]
            if obj_name not in obj_range:
                continue
            gt_mask = decode(gt["masks"][obj_name][exo][id])
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            iou, shape_acc = utils.eval_mask(gt_mask, binary_mask)
            ex_acc = utils.existence_accuracy(gt_mask, binary_mask)
            location_score = utils.location_score(gt_mask, binary_mask, size=(h, w))
            IoUs.append(iou)
            ShapeAcc.append(shape_acc)
            ExistenceAcc.append(ex_acc)
            LocationScores.append(location_score)

    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)

    print(np.mean(IoUs))
    return IoUs.tolist(), ShapeAcc.tolist(), ExistenceAcc.tolist(), LocationScores.tolist(), num_frame


def main():
    total_iou = []
    total_shape_acc = []
    total_existence_acc = []
    total_location_scores = []
    num_total = 0
    for take_id in val_set:
        ious, shape_accs, existence_accs, location_scores, num_frame = evaluate_take(take_id)
        total_iou += ious
        total_shape_acc += shape_accs
        total_existence_acc += existence_accs
        total_location_scores += location_scores
        num_total += num_frame

    print('TOTAL IOU: ', np.mean(total_iou))
    print('TOTAL LOCATION SCORE: ', np.mean(total_location_scores))
    print('TOTAL SHAPE ACC: ', np.mean(total_shape_acc))
    print("total frames:", num_total)


if __name__ == '__main__':
    main()