import json
import os
from PIL import Image
import numpy as np
from pycocotools.mask import encode, decode, frPyObjects
from tqdm import tqdm
import copy
from natsort import natsorted
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, required=True, help="Root path of the dataset.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the output JSON file.")
parser.add_argument("--split", type=str, choices=["train", "test"], default="train", help="Dataset split to build.")
args = parser.parse_args()

if __name__ == '__main__':
    root_path = args.root_path
    save_path = args.save_path
    # to store 
    handal_dataset = []
    new_img_id = 0
    obj_name = os.listdir(root_path)[:1]
    for obj in tqdm(obj_name):
        full_path = os.path.join(root_path, obj)
        if not os.path.isdir(full_path):
            continue
        data_path = os.path.join(full_path, args.split)
        val_set = os.listdir(data_path)
        for val_name in val_set:
            vid_path = os.path.join(data_path, val_name)
            img_path = os.path.join(vid_path, "rgb")
            anno_path = os.path.join(vid_path, "mask")
            frame_idx = natsorted(os.listdir(img_path))
            frame_idx  = [f.split(".")[0] for f in frame_idx]
            video_len = len(frame_idx)
            for i,idx in enumerate(frame_idx):
                if i+100 > video_len-1:
                    break
                target_idx = frame_idx[i+100]

                first_frame_annotation_path = os.path.join(anno_path, idx+"_000000.png")
                first_frame_annotation_relpath = os.path.relpath(first_frame_annotation_path, root_path)

                first_frame_img_path = os.path.join(img_path, idx+".jpg")
                first_frame_img_relpath = os.path.relpath(first_frame_img_path, root_path)

                first_frame_annotation_img = Image.open(first_frame_annotation_path)
                first_frame_annotation = np.array(first_frame_annotation_img)
                height, width = first_frame_annotation.shape
                unique_instances = np.unique(first_frame_annotation)
                unique_instances = unique_instances[unique_instances != 0]
                coco_format_annotations = []
                for instance_value in unique_instances:
                    binary_mask = (first_frame_annotation == instance_value).astype(np.uint8)
                    segmentation = encode(np.asfortranarray(binary_mask))
                    segmentation = {
                        'counts': segmentation['counts'].decode('ascii'),
                        'size': segmentation['size'],
                    }
                    area = binary_mask.sum().astype(float)
                    coco_format_annotations.append(
                        {
                            'segmentation': segmentation,
                            'area': area,
                            'category_id': instance_value.astype(float),
                        }
                    )

                sample_img_path = os.path.join(img_path, target_idx+".jpg")
                sample_img_relpath = os.path.relpath(sample_img_path, root_path)
                image_info = {
                    'file_name': sample_img_relpath,
                    'height': height,
                    'width': width,
                }
                sample_annotation_path = os.path.join(anno_path, target_idx+"_000000.png")
                sample_annotation = np.array(Image.open(sample_annotation_path))

                sample_unique_instances = np.unique(sample_annotation)
                sample_unique_instances = sample_unique_instances[sample_unique_instances != 0]
                anns = []
                for instance_value in sample_unique_instances:
                    assert instance_value in unique_instances, 'Found new target not in the first frame'
                    binary_mask = (sample_annotation == instance_value).astype(np.uint8)
                    segmentation = encode(np.asfortranarray(binary_mask))
                    segmentation = {
                        'counts': segmentation['counts'].decode('ascii'),
                        'size': segmentation['size'],
                    }
                    area = binary_mask.sum().astype(float)
                    anns.append(
                        {
                            'segmentation': segmentation,
                            'area': area,
                            'category_id': instance_value.astype(float),
                        }
                    )
                first_frame_anns = copy.deepcopy(coco_format_annotations)
                if len(anns) < len(first_frame_anns):
                    first_frame_anns = [ann for ann in first_frame_anns if ann['category_id'] in sample_unique_instances]
                assert len(anns) == len(first_frame_anns)
                sample = {
                    'image': sample_img_relpath,
                    'image_info': image_info,
                    'anns': anns,
                    'first_frame_image': first_frame_img_relpath,
                    'first_frame_anns': first_frame_anns,
                    'new_img_id': new_img_id,
                    'video_name': sample_img_relpath.split("/")[0],
                }
                handal_dataset.append(sample)
                new_img_id += 1
    
   
    with open(save_path, 'w') as f:
        json.dump(handal_dataset, f)
    print(f'Save at {save_path}. Total sample: {len(handal_dataset)}')