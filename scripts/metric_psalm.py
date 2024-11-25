
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



def evaluate_take():
    
    #实验需改动
    pred_path = "/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/predictions/ego_query5/92b2221b-ae92-44f0-bb31-e2d27cb736d6/gp02"
    root_path = "/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
    tmp = pred_path.split("/")
    take_id = tmp[-2]
    exo = tmp[-1]
    gt_path = f"{root_path}/{take_id}/annotation.json"
    with open(gt_path, 'r') as fp:
        gt = json.load(fp)
    #实验需改动
    ego_cams = [x for x in gt['masks']["piano"].keys() if 'aria' in x]
    ego = ego_cams[0]
    
    
    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []
    
    frames = os.listdir(pred_path)
    idx = [f.split(".")[0] for f in frames]

    for id in idx:
        #实验需改动
        gt_mask = gt["masks"]["piano"][exo][id]
        gt_mask = mask_utils.decode(gt_mask)

        #修改，将解码后gt_mask调整大小为pred_mask的大小
        gt_mask = cv2.resize(gt_mask, (960, 540), interpolation=cv2.INTER_NEAREST)
        pred_mask = Image.open(f"{pred_path}/{id}.png")
        pred_mask = np.array(pred_mask)
        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
        ex_acc = utils.existence_accuracy(gt_mask, pred_mask)
        location_score = utils.location_score(gt_mask, pred_mask, size=(540, 960))
        IoUs.append(iou)
        ShapeAcc.append(shape_acc)
        ExistenceAcc.append(ex_acc)
        LocationScores.append(location_score)

    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)

    print("iou:", np.mean(IoUs))
    print("LocationScores:", np.mean(LocationScores))
    print("ShapeAcc:", np.mean(ShapeAcc))




if __name__ == '__main__':
    evaluate_take()
    