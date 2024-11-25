import argparse
import json
#import tqdm
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


from PIL import Image, ImageDraw
import numpy as np
import cv2


def scale_mask_object(img, mask, scale_factor):
    """
    Scales the object in the mask by a given factor and applies the scaled mask onto the image.
    
    Parameters:
    img (PIL.Image or numpy array): The original image.
    mask (PIL.Image or numpy array): The COCO mask, where non-zero regions represent the object.
    scale_factor (float): The scaling factor (e.g., 2.0 for doubling, 0.5 for half).
    
    Returns:
    new_img (PIL.Image): The modified image with the scaled object.
    new_mask (PIL.Image): The modified mask with the scaled object.
    """
    # Convert PIL images to numpy arrays if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Get bounding box of the object in the mask
    y, x = np.where(mask > 0)  # Find all non-zero points
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No object found in the mask.")

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    object_crop = mask[ymin:ymax+1, xmin:xmax+1]  # Crop the object from the mask

    # Scale the cropped object mask
    obj_height, obj_width = object_crop.shape[:2]
    new_obj_height = int(obj_height * scale_factor)
    new_obj_width = int(obj_width * scale_factor)
    scaled_object_crop = cv2.resize(object_crop, (new_obj_width, new_obj_height), interpolation=cv2.INTER_NEAREST)

    # Scale the cropped object from the original image
    img_object_crop = img[ymin:ymax+1, xmin:xmax+1]
    scaled_img_object_crop = cv2.resize(img_object_crop, (new_obj_width, new_obj_height), interpolation=cv2.INTER_LINEAR)

    # Calculate new positions
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    new_xmin = max(center_x - new_obj_width // 2, 0)
    new_ymin = max(center_y - new_obj_height // 2, 0)
    new_xmax = min(new_xmin + new_obj_width, img.shape[1])
    new_ymax = min(new_ymin + new_obj_height, img.shape[0])

    # Create new mask and image with the scaled object
    new_mask = np.zeros_like(mask)
    new_mask[new_ymin:new_ymax, new_xmin:new_xmax] = scaled_object_crop[:new_ymax-new_ymin, :new_xmax-new_xmin]

    new_img = img.copy()
    new_img[new_ymin:new_ymax, new_xmin:new_xmax] = scaled_img_object_crop[:new_ymax-new_ymin, :new_xmax-new_xmin]

    # Convert back to PIL images if needed
    #new_img = Image.fromarray(new_img)
    #new_mask = Image.fromarray(new_mask)

    return new_img, new_mask


from PIL import Image
import numpy as np
import cv2

def scale_mask_object_with_background(img, mask, scale_factor, padding=0.25):
    """
    Scales the object in the mask by a given factor and adjusts the background region accordingly.
    
    Parameters:
    img (PIL.Image or numpy array): The original image.
    mask (PIL.Image or numpy array): The binary mask image where non-zero regions represent the object.
    scale_factor (float): Scaling factor (e.g., 2.0 for double, 0.5 for half).
    padding (float): Fractional padding to include around the object during scaling. For example, 0.25 adds 25% padding.
    
    Returns:
    new_img (PIL.Image): The modified image with the scaled object and adjusted background.
    new_mask (PIL.Image): The modified mask with the scaled object and adjusted background.
    """
    # Convert PIL images to numpy arrays if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Get bounding box of the object in the mask
    y, x = np.where(mask > 0)  # Find all non-zero points
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No object found in the mask.")

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Determine padding size based on object dimensions
    height, width = ymax - ymin, xmax - xmin
    pad_x = int(width * padding)
    pad_y = int(height * padding)

    # Crop a region around the object with padding
    crop_xmin = max(xmin - pad_x, 0)
    crop_ymin = max(ymin - pad_y, 0)
    crop_xmax = min(xmax + pad_x, img.shape[1])
    crop_ymax = min(ymax + pad_y, img.shape[0])

    # Crop the object and its surrounding background from the mask and image
    object_crop_mask = mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    object_crop_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Scale the cropped region (including background and object)
    new_height = int(object_crop_mask.shape[0] * scale_factor)
    new_width = int(object_crop_mask.shape[1] * scale_factor)
    scaled_object_crop_mask = cv2.resize(object_crop_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    scaled_object_crop_img = cv2.resize(object_crop_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate position to center the scaled object in the new mask
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    new_xmin = max(center_x - new_width // 2, 0)
    new_ymin = max(center_y - new_height // 2, 0)
    new_xmax = min(new_xmin + new_width, img.shape[1])
    new_ymax = min(new_ymin + new_height, img.shape[0])

    # Create new mask and image with the scaled object and adjusted background
    new_mask = np.zeros_like(mask)
    new_img = img.copy()

    new_mask[new_ymin:new_ymax, new_xmin:new_xmax] = scaled_object_crop_mask[:new_ymax-new_ymin, :new_xmax-new_xmin]
    new_img[new_ymin:new_ymax, new_xmin:new_xmax] = scaled_object_crop_img[:new_ymax-new_ymin, :new_xmax-new_xmin]

    # Convert back to PIL images if needed
    #new_img = Image.fromarray(new_img)
    #new_mask = Image.fromarray(new_mask)

    return new_img, new_mask


from PIL import Image
import numpy as np
import cv2

from PIL import Image
import numpy as np
import cv2

def scale_image_and_keep_mask_centered(img, mask, scale_factor):
    """
    Scales the entire image and mask, ensuring that the mask's object remains within the view.
    
    Parameters:
    img (PIL.Image or numpy array): The original image.
    mask (PIL.Image or numpy array): The binary mask image where non-zero regions represent the object.
    scale_factor (float): Scaling factor (e.g., 2.0 for double size, 0.5 for half size).
    
    Returns:
    new_img (PIL.Image): The modified image with the scaled region.
    new_mask (PIL.Image): The modified mask with the scaled region.
    """
    # Convert PIL images to numpy arrays if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Get bounding box of the object in the mask
    y, x = np.where(mask > 0)  # Find all non-zero points
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No object found in the mask.")

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Calculate the center of the mask object
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    # Scale the entire image and mask
    original_height, original_width = img.shape[:2]
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    
    # Resize the image and mask
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Calculate the offset to keep the mask centered in the view
    offset_x = max(center_x * scale_factor - original_width // 2, 0)
    offset_y = max(center_y * scale_factor - original_height // 2, 0)

    # Crop to original size to keep mask centered in the output
    crop_xmin = int(offset_x)
    crop_ymin = int(offset_y)
    crop_xmax = min(crop_xmin + original_width, new_width)
    crop_ymax = min(crop_ymin + original_height, new_height)

    cropped_img = scaled_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    cropped_mask = scaled_mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Convert back to PIL images if needed
    #new_img = Image.fromarray(cropped_img)
    #new_mask = Image.fromarray(cropped_mask)

    return cropped_img, cropped_mask


def scale_image_with_mask(img, mask, scale_factor, padding=0.25):
    """
    Scales a region of the image (including background and mask) around the object in the mask by a given factor.
    
    Parameters:
    img (PIL.Image or numpy array): The original image.
    mask (PIL.Image or numpy array): The binary mask image where non-zero regions represent the object.
    scale_factor (float): Scaling factor (e.g., 2.0 for double, 0.5 for half).
    padding (float): Fractional padding to include around the object during scaling. For example, 0.25 adds 25% padding.
    
    Returns:
    new_img (PIL.Image): The modified image with the scaled region.
    new_mask (PIL.Image): The modified mask with the scaled region.
    """
    # Convert PIL images to numpy arrays if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Get bounding box of the object in the mask
    y, x = np.where(mask > 0)  # Find all non-zero points
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No object found in the mask.")

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Determine padding size based on object dimensions
    height, width = ymax - ymin, xmax - xmin
    pad_x = int(width * padding)
    pad_y = int(height * padding)

    # Crop a region around the object with padding
    crop_xmin = max(xmin - pad_x, 0)
    crop_ymin = max(ymin - pad_y, 0)
    crop_xmax = min(xmax + pad_x, img.shape[1])
    crop_ymax = min(ymax + pad_y, img.shape[0])

    # Crop the region containing the object and background from the mask and image
    region_crop_mask = mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    region_crop_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Scale the cropped region (both mask and image)
    new_height = int(region_crop_mask.shape[0] * scale_factor)
    new_width = int(region_crop_mask.shape[1] * scale_factor)
    scaled_region_crop_mask = cv2.resize(region_crop_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    scaled_region_crop_img = cv2.resize(region_crop_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate position to center the scaled region in the new mask and image
    center_x = (crop_xmin + crop_xmax) // 2
    center_y = (crop_ymin + crop_ymax) // 2
    new_xmin = max(center_x - new_width // 2, 0)
    new_ymin = max(center_y - new_height // 2, 0)
    new_xmax = min(new_xmin + new_width, img.shape[1])
    new_ymax = min(new_ymin + new_height, img.shape[0])

    # Create new mask and image with the scaled region placed in the correct position
    new_mask = np.zeros_like(mask)
    new_img = img.copy()

    new_mask[new_ymin:new_ymax, new_xmin:new_xmax] = scaled_region_crop_mask[:new_ymax-new_ymin, :new_xmax-new_xmin]
    new_img[new_ymin:new_ymax, new_xmin:new_xmax] = scaled_region_crop_img[:new_ymax-new_ymin, :new_xmax-new_xmin]

    # Convert back to PIL images if needed
    #new_img = Image.fromarray(new_img)
    #new_mask = Image.fromarray(new_mask)

    return new_img, new_mask

#datapath /datasegswap
#inference_path /inference_xmem_ego_last/coco
#output /vis_piano
#--show_gt要加上
if __name__ == "__main__":

    #实验需改动
    root_path = "/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/predictions/exo_query_test/92b2221b-ae92-44f0-bb31-e2d27cb736d6/aria01_214-1"
    file_names = natsorted(os.listdir(root_path))
    idxs = [int(f.split(".")[0]) for f in file_names]

    tmp = root_path.split("/")
    datapath = "/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap"
    take_id = tmp[-2]
    target_cam = tmp[-1]
    out_path = f"/data/work2-gcp-europe-west4-a/yuqian_fu/Ego/data_segswap/vis_psalm/exo_query_test/{take_id}/{target_cam}"
    os.makedirs(
        out_path, exist_ok=True
    )
    print(take_id, target_cam)

    #为了节省内存 实际上可以idx[:60]来可视化部分帧
    idxs = idxs[:2]
    for id in idxs:
        frame_idx = str(id)
        frame = cv2.imread(
            f"{datapath}/{take_id}/{target_cam}/{frame_idx}.jpg"
        )
        mask = Image.open(f"{root_path}/{frame_idx}.png")
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


        #scale img: 2
        print('frame:', frame.shape, 'mask:', mask.shape)
        #img_new, mask_new = scale_mask_object(frame, mask, 0.5)
        #img_new, mask_new = scale_mask_object_with_background(frame, mask, 0.5)
        #img_new, mask_new = scale_image_with_mask(frame, mask, 0.5)
        img_new, mask_new = scale_image_and_keep_mask_centered(frame, mask, 0.25)
        print('img_new:', img_new.shape, 'mask_new:', mask_new.shape)
        out_new = blend_mask(img_new, mask_new)
        print('img saved at:', f"{out_path}/{frame_idx}_new.jpg")
        cv2.imwrite(
            f"{out_path}/{frame_idx}_new.jpg",
            out_new,
        )

