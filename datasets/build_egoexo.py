import json
import os
from PIL import Image
import numpy as np
from pycocotools.mask import encode, decode, frPyObjects
from tqdm import tqdm
import copy
from natsort import natsorted
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', required=True,
                    help='Root path of the dataset')
parser.add_argument('--save_path', type=str, default='', required=True,
                    help='Path to save the json file')
parser.add_argument('--split_path', type=str, default='', required=True,
                    help='Path to the split file')
parser.add_argument("--split", type=str, default="val", help="Split to use (train/val/test)")
parser.add_argument('--task', type=str, default='ego2exo', help='Task type (ego2exo/exo2ego)')
args = parser.parse_args()


if __name__ == '__main__':
    # Set relevant paths
    root_path = args.root_path
    save_path = args.save_path
    split_path = args.split_path

    # Read takes_id
    with open(split_path, "r") as fp:
        data_split = json.load(fp)
    data_set = data_split[args.split]

    # Read missing files
    with open("datasets/missing_takes.txt", "r") as fp:
        missing_files = [line.strip() for line in fp.readlines()]
   
    # to count
    new_img_id = 0
    # to store data
    egoexo_dataset = []

    for vid_name in tqdm(data_set):
        if vid_name in missing_files:
            continue

        # Read the annotation file under this take
        vid_root_path = os.path.join(root_path, vid_name)
        anno_path = os.path.join(vid_root_path, "annotation.json")
        with open(anno_path, 'r') as fp:
            annotations = json.load(fp)

        # Extract all objects from this take
        objs = natsorted(list(annotations["masks"].keys()))
        coco_id_to_cont_id = {coco_id: cont_id + 1 for cont_id, coco_id in enumerate(objs)}

        # Extract ego and exo cameras
        valid_cams = os.listdir(vid_root_path)
        valid_cams.remove("annotation.json")
        valid_cams = natsorted(valid_cams)
        ego_cams = []
        exo_cams = []
        for vc in valid_cams:
            if 'aria' in vc:
                ego_cams.append(vc)
            else:
                exo_cams.append(vc)
        ego = ego_cams[0]
        exo = exo_cams[0]
        vid_ego_path = os.path.join(vid_root_path, ego)
        ego_frames = natsorted(os.listdir(vid_ego_path))
        ego_frames = [f.split(".")[0] for f in ego_frames]
        objs_both_have = []
        for obj in objs:
            if ego in annotations["masks"][obj].keys() and exo in annotations["masks"][obj].keys():
                objs_both_have.append(obj)
        # If the number of exo cameras is greater than 1, take the exo camera with the largest number of shared objects
        if len(exo_cams) > 1:
            for cam in exo_cams[1:]:
                objs_both_have_tmp = []
                for obj in objs:
                    if ego in annotations["masks"][obj].keys() and cam in annotations["masks"][obj].keys():
                        objs_both_have_tmp.append(obj)
                if len(objs_both_have_tmp) > len(objs_both_have):
                    exo = cam
                    objs_both_have = objs_both_have_tmp
        if len(objs_both_have) == 0:
            continue
        vid_exo_path = os.path.join(vid_root_path, exo)
        exo_frames = natsorted(os.listdir(vid_exo_path))
        exo_frames = [f.split(".")[0] for f in exo_frames]

        # Set the query/target cameras based on the task type
        if args.task == 'ego2exo':
            query_cam = ego
            target_cam = exo
            target_cam_anno_frames = exo_frames
            vid_target_path = vid_exo_path
            vid_query_path = vid_ego_path
        elif args.task == 'exo2ego':
            query_cam = exo
            target_cam = ego
            target_cam_anno_frames = ego_frames
            vid_target_path = vid_ego_path
            vid_query_path = vid_exo_path
        else:
            raise ValueError("Task must be either 'ego2exo' or 'exo2ego'.")
        
        # Use all annotation frames of the longest-appearing object from query_cam as reference frames
        obj_ref = objs_both_have[0]
        for obj in objs_both_have:
            if len(list(annotations["masks"][obj_ref][query_cam].keys())) < len(list(annotations["masks"][obj][query_cam].keys())):
                obj_ref = obj
        query_cam_anno_frames = natsorted(list(annotations["masks"][obj_ref][query_cam].keys()))
        frames = natsorted(np.intersect1d(query_cam_anno_frames, target_cam_anno_frames))

        for idx in frames:
            coco_format_annotations = []
            filename = f"{idx}.jpg"
            
            sample_img_path = os.path.join(vid_target_path, filename)
            sample_img_relpath = os.path.relpath(sample_img_path, root_path)
            first_frame_img_path = os.path.join(vid_query_path, filename)
            first_frame_img_relpath = os.path.relpath(first_frame_img_path, root_path)

            # Identify visible objects in the query image
            obj_list_query = []
            for obj in objs_both_have:
                if idx in annotations["masks"][obj][query_cam].keys():
                    mask_query = decode(annotations["masks"][obj][query_cam][idx])
                    area_new = mask_query.sum().astype(float)
                    if area_new != 0:
                        obj_list_query.append(obj)
            if len(obj_list_query) == 0:
                continue
            obj_list_query_new = []
            for obj in obj_list_query:
                segmentation_tmp = annotations["masks"][obj][query_cam][idx]
                binary_mask = decode(segmentation_tmp)
                h, w = binary_mask.shape
                if args.task == 'ego2exo':
                    binary_mask = cv2.resize(binary_mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                elif args.task == 'exo2ego':
                    binary_mask = cv2.resize(binary_mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                area = binary_mask.sum().astype(float)
                if area == 0:
                    continue
                segmentation = encode(np.asfortranarray(binary_mask))
                segmentation = {
                    'counts': segmentation['counts'].decode('ascii'),
                    'size': segmentation["size"],
                }
                obj_list_query_new.append(obj)
                coco_format_annotations.append(
                    {
                        'segmentation': segmentation,
                        'area': area,
                        'category_id': float(coco_id_to_cont_id[obj]),
                    }
                )
            if len(obj_list_query_new) == 0:
                continue

            # Identify visible objects in the target image
            obj_list_target = []
            for obj in obj_list_query_new:
                if idx in annotations["masks"][obj][target_cam].keys():
                    mask_target = decode(annotations["masks"][obj][target_cam][idx])
                    area_target = mask_target.sum().astype(float)
                    if area_target != 0:
                        obj_list_target.append(obj)
            if len(obj_list_target) == 0:
                continue
            height, width = annotations["masks"][obj_list_target[0]][target_cam][idx]["size"]
            if args.task == 'ego2exo':
                image_info = {
                    'file_name': sample_img_relpath,
                    'height': height // 4,
                    'width': width // 4,
                }
            elif args.task == 'exo2ego':
                image_info = {
                    'file_name': sample_img_relpath,
                    'height': height // 2,
                    'width': width // 2,
                }
            anns = []
            obj_list_target_new = []
            for obj in obj_list_target:
                assert obj in obj_list_query_new, 'Found new target not in the first frame'
                segmentation_tmp = annotations["masks"][obj][target_cam][idx]
                binary_mask = decode(segmentation_tmp)
                h, w = binary_mask.shape
                if args.task == 'ego2exo':
                    binary_mask = cv2.resize(binary_mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                elif args.task == 'exo2ego':
                    binary_mask = cv2.resize(binary_mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                area = binary_mask.sum().astype(float)
                if area == 0:
                    continue
                segmentation = encode(np.asfortranarray(binary_mask))
                segmentation = {
                    'counts': segmentation['counts'].decode('ascii'),
                    'size': segmentation['size'],
                }
                obj_list_target_new.append(obj)
                anns.append(
                    {
                        'segmentation': segmentation,
                        'area': area,
                        'category_id': float(coco_id_to_cont_id[obj]),
                    }
                )
            if len(obj_list_target_new) == 0:
                continue

            sample_unique_instances = [float(coco_id_to_cont_id[obj]) for obj in obj_list_target_new]
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
                'video_name': vid_name,
            }
            egoexo_dataset.append(sample)
            new_img_id += 1

    with open(save_path, 'w') as f:
        json.dump(egoexo_dataset, f)
    print(f'Save at {save_path}. Total sample: {len(egoexo_dataset)}')
