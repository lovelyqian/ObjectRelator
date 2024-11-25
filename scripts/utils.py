import numpy as np
import cv2

import metrics

def eval_mask(gt_masks: np.ndarray, fake_masks: np.ndarray):
    """TODO: Docstring for eval_mask.

    Args:
        gt_masks (np.ndarray): The
        fake_masks (np.ndarray): TODO

    Returns: TODO

    """
    iou = metrics.db_eval_iou(gt_masks, fake_masks)
    boundary = metrics.db_eval_boundary(gt_masks, fake_masks)
    return iou, boundary

def existence_accuracy(gt_mask: np.ndarray, pred_mask: np.ndarray):
    gt_zeros = np.all(gt_mask == 0)
    pred_zeros = np.all(pred_mask == 0)

    return gt_zeros == pred_zeros

def location_score(gt_mask, pred_mask, size=(480, 480)):
    H, W = size
    (gt_size, pred_size), (centroid_gt, centroid_pred), (gt_compact_mask, pred_compact_mask) = metrics.crop_mask(gt_mask, pred_mask)
    centroid_distance = np.sqrt((centroid_gt[0] - centroid_pred[0])**2 + (centroid_gt[1] - centroid_pred[1])**2)
    lscore = centroid_distance / np.sqrt(H**2 + W**2)
    return lscore