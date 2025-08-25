'''
utils for vis
'''
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
from pycocotools.mask import encode, decode, frPyObjects


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