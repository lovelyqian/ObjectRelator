
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
data_path = "/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
val_set = os.listdir(pred_root)
# val_set = ["1d0f3c10-ed0a-4f60-b0d2-a516690ff1cf"]



# with open(split_path, "r") as fp:
#     data_split = json.load(fp)
# val_set = ["val"]


def fuse_davis_mask(mask_list):
    fused_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        fused_mask[mask == 1] = 1
    return fused_mask

# not_regular_size = []
def evaluate_take(take_id):

    pred_path = os.path.join(pred_root, take_id)
    cams = os.listdir(pred_path)
    exo = cams[0]
    pred_path = os.path.join(pred_path, exo)


    gt_path = f"{data_path}/{take_id}/annotation.json"
    with open(gt_path, 'r') as fp:
        gt = json.load(fp)

    objs = list(gt['masks'].keys())
    total_cam = []
    for obj in objs:
        total_cam += list(gt['masks'][obj].keys())
    total_cam = set(total_cam)
    ego_cams = [x for x in total_cam if 'aria' in x]
    if len(ego_cams)==0:
        print(take_id)
    ego = ego_cams[0]
    

    objs_both_have = []
    for obj in objs:
        if ego in gt["masks"][obj].keys() and exo in gt["masks"][obj].keys():
            objs_both_have.append(obj)

    obj_ref = objs_both_have[0]
    for obj in objs_both_have:
        if len(list(gt["masks"][obj_ref][ego].keys())) < len(list(gt["masks"][obj][ego].keys())):
            obj_ref = obj


    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []
    
    frames = os.listdir(pred_path)
    idx = [f.split(".")[0] for f in frames]


    #TODO first_anno_key出错了 对于exo的预测从第一帧来说,下面的代码是对的
    # first_anno_key = idx[0]
    all_ref_keys = np.asarray(
        natsorted(gt["masks"][obj_ref][ego])
    ).astype(np.int64)
    first_anno_key = str(all_ref_keys[0])


    # pred_mask_tmp = Image.open(f"{pred_path}/{first_anno_key}.png")
    # pred_mask_tmp = np.array(pred_mask_tmp)
    #统计h为960的exo takes
    # h_tmp,w_tmp = pred_mask_tmp.shape
    # if h_tmp != 540:
    #     not_regular_size.append(take_id)



    obj_list_ego = []
    for obj in objs_both_have:
        if first_anno_key in gt["masks"][obj][ego].keys():
            obj_list_ego.append(obj)

    for id in idx:

        obj_list_exo = []
        for obj in obj_list_ego:
            if id in gt["masks"][obj][exo].keys():
                obj_list_exo.append(obj)

        gt_mask_list = []
        #获取所有的gtmask
        for obj in obj_list_exo:
            gt_mask = gt["masks"][obj][exo][id]
            gt_mask = decode(gt_mask)
            gt_mask_list.append(gt_mask)

        # pred_mask_list = [tensor_.astype(np.uint8) for tensor_ in pred_mask_list]
        if len(gt_mask_list) == 0:
            continue

        pred_mask = Image.open(f"{pred_path}/{id}.png")
        pred_mask = np.array(pred_mask)
        pred_mask[pred_mask != 0] = 1
        h, w = pred_mask.shape

        fused_gt_mask = fuse_davis_mask(gt_mask_list)

        #修改，将解码后gt_mask调整大小为pred_mask的大小
        gt_mask = cv2.resize(fused_gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)





        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
        ex_acc = utils.existence_accuracy(gt_mask, pred_mask)
        location_score = utils.location_score(gt_mask, pred_mask, size=(h, w))
        IoUs.append(iou)
        ShapeAcc.append(shape_acc)
        ExistenceAcc.append(ex_acc)
        LocationScores.append(location_score)

    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)

    print(np.mean(IoUs))
    return IoUs.tolist(), ShapeAcc.tolist(), ExistenceAcc.tolist(), LocationScores.tolist()

def main():
    total_iou = []
    total_shape_acc = []
    total_existence_acc = []
    total_location_scores = []
    for take_id in val_set:
        ious, shape_accs, existence_accs, location_scores = evaluate_take(take_id)
        total_iou += ious
        total_shape_acc += shape_accs
        total_existence_acc += existence_accs
        total_location_scores += location_scores

    print('TOTAL IOU: ', np.mean(total_iou))
    print('TOTAL LOCATION SCORE: ', np.mean(total_location_scores))
    print('TOTAL SHAPE ACC: ', np.mean(total_shape_acc))
    # print(not_regular_size)

if __name__ == '__main__':
    main()