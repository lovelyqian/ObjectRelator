import copy
import logging

import numpy as np
import torch
import random
import cv2

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks
from pycocotools import mask as coco_mask
from pycocotools.mask import encode, decode, frPyObjects


def draw_circle(mask, center, radius):
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask[distance <= radius] = 1


def enhance_with_circles(binary_mask, radius=5):
    if not isinstance(binary_mask, np.ndarray):
        binary_mask = np.array(binary_mask)

    binary_mask = binary_mask.astype(np.uint8)

    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    points = np.argwhere(binary_mask == 1)
    for point in points:
        draw_circle(output_mask, (point[0], point[1]), radius)
    return output_mask


def is_mask_non_empty(rle_mask):
    if rle_mask is None:
        return False
    binary_mask = decode(rle_mask)
    return binary_mask.sum() > 0


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        # T.ResizeScale(
        #     min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        # ),
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation


class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = build_transform_gen(cfg)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def preprocess(self, dataset_dict, region_mask_type=None, mask_format='polygon'):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if isinstance(dataset_dict["file_name"],str):
            image = utils.read_image(dataset_dict["file_name"], format='RGB')
        else:
            image = np.array(dataset_dict["file_name"])
        # print(dataset_dict)
        # print(image)
        utils.check_image_size(dataset_dict, image)
        utils.check_image_size(dataset_dict, image)

        #为了适配eval_ego脚本增加
        gt_masks_list = []
        for ann in dataset_dict["annotations"]:
            mask_tmp = decode(ann["segmentation"])
            gt_masks_list.append(mask_tmp)
        dataset_dict["gt_mask_list"] = gt_masks_list
        # dataset_dict["region_masks"] = gt_masks_list
        dataset_dict["vp_file_path"] = dataset_dict["vp_image"]

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        #transforms,将对exo图像的变换记录了下来，这里的对图像的变换是resize、crop
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w
        # print("exo_image_shape:", image_shape)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = (image - self.pixel_mean) / self.pixel_std
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict['transforms'] = transforms
        region_masks = []

        if 'vp_image' in dataset_dict:
            if isinstance(dataset_dict["vp_image"], str):
                vp_image = utils.read_image(dataset_dict["vp_image"], format='RGB')
            else:
                vp_image = np.array(dataset_dict["vp_image"])

            # TODO: get padding mask
            # by feeding a "segmentation mask" to the same transforms
            vp_padding_mask = np.ones(vp_image.shape[:2])

            #变换到1024
            vp_image, vp_transforms = T.apply_transform_gens(self.tfm_gens, vp_image)
            # the crop transformation has default padding value 0 for segmentation
            # print("vp_image final_shape:", vp_image.shape)
            vp_padding_mask = vp_transforms.apply_segmentation(vp_padding_mask)
            # print(vp_padding_mask.shape)
            vp_padding_mask = ~ vp_padding_mask.astype(bool)

            #1024x1024
            vp_image_shape = vp_image.shape[:2]  # h, w
            # print("vp_image_shape:", vp_image_shape)

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            vp_image = torch.as_tensor(np.ascontiguousarray(vp_image.transpose(2, 0, 1)))
            dataset_dict["vp_image"] = (vp_image - self.pixel_mean) / self.pixel_std
            dataset_dict["vp_padding_mask"] = torch.as_tensor(np.ascontiguousarray(vp_padding_mask))
            dataset_dict['vp_transforms'] = vp_transforms
            vp_region_masks = []
            vp_fill_number = []
            # print(f"vp_image_shape:{vp_image_shape}")
            # print(dataset_dict.pop("vp_annotations")[0])
            #这里的obj是exo每一帧中的mask
            #对该帧下每个物体的mask进行与vp-image相同的变换


            #这里的vp_image_shape是变换后的目标大小，所以应该是1024x1024
            #vp_annos存储的是经过变换后的参考帧的所有物体mask
            vp_annos = [
                utils.transform_instance_annotations(obj, vp_transforms, vp_image_shape)
                for obj in dataset_dict.pop("vp_annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            if len(vp_annos) == 0:
                print('error')
            else:
                for vp_anno in vp_annos:
                    vp_region_mask = vp_anno['segmentation']
                    vp_fill_number.append(int(vp_anno['category_id']))
                    # vp_scale_region_mask = transforms.apply_segmentation(vp_region_mask)
                    vp_region_masks.append(vp_region_mask)
            #vp_region_masks存储的是参考帧里的所有RLE格式的coco mask


        if "annotations" in dataset_dict:
            #print("annotations in dataset_dict") # YES
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annotations = dataset_dict['annotations']

            # USER: Implement additional transformations if you have other types of data
            #annos存储的是target帧中所有经过变换的物体mask
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            if len(annos) ==0:
                print('error')
                #print(dataset_dict["file_name"]) #debug

            filter_annos = []

            #到这里只处理了anno['segmentation'],anno['mask_visual_prompt_mask']还是RLE格式的
            # print("annos:", annos[0])
            # if 'point_visual_prompt_mask' in annos[0]:
            '''
            确定anno中是哪种形式的mask。这里需要根据交互式任务的不同到getitem中对anno={"segmentation","area","class_id"}进行修改，
            把原生的segmentation替换为{"mask_visual_prompt_mask","point_visual_prompt_mask",..."area","class_id"}的形式
            '''

            if 'mask_visual_prompt_mask' in annos[0]:
                if region_mask_type is None:
                    # region_mask_type = ['point_visual_prompt_mask', 'mask_visual_prompt_mask', 'box_visual_prompt_mask',
                    #                     'scribble_visual_prompt_mask']
                    #根据任务的不同进行替换，前提是anno中必须有这个键，要不然会报错
                    region_mask_type = ['mask_visual_prompt_mask']


                #这里的意思是同一个物体可能有许多不同格式的mask，把同一个物体所有不同格式的mask类型都取出来放在non_empty_masks中
                for anno in annos:
                    non_empty_masks = []
                    for mask_type in region_mask_type:
                        if is_mask_non_empty(anno[mask_type]):
                            non_empty_masks.append(mask_type)
                    # assert non_empty_masks, 'No visual prompt found in {}'.format(dataset_dict['file_name'])
                    if len(non_empty_masks) == 0:
                        continue
                    #对于每个物体，每次随机地选择一种mask类型
                    #region_masks里存储的是解码，且经过变换后的物体mask
                    used_mask_type = random.choice(non_empty_masks)
                    region_mask = decode(anno[used_mask_type])
                    if used_mask_type in ['point_visual_prompt_mask', 'scribble_visual_prompt_mask']:
                        radius = 10 if used_mask_type == 'point_visual_prompt_mask' else 5
                        region_mask = enhance_with_circles(region_mask, radius)
                    scale_region_mask = transforms.apply_segmentation(region_mask)
                    region_masks.append(scale_region_mask)
                    filter_annos.append(anno)
            if len(filter_annos) == 0:
                filter_annos = annos
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            # instances = utils.annotations_to_instances(annos, image_shape)
            instances = utils.annotations_to_instances(filter_annos, image_shape, mask_format=mask_format)
            if 'lvis_category_id' in filter_annos[0]:
                lvis_classes = [int(obj["lvis_category_id"]) for obj in annos]
                lvis_classes = torch.tensor(lvis_classes, dtype=torch.int64)
                instances.lvis_classes = lvis_classes
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            # non_empty_instance_mask = [len(obj.get('segmentation', [])) > 0 for obj in annos]
            non_empty_instance_mask = [len(obj.get('segmentation', [])) > 0 for obj in filter_annos]

            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                if hasattr(gt_masks,'polygons'):
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = gt_masks.tensor.to(dtype=torch.uint8)
                instances.gt_masks = gt_masks

            if region_masks:
                region_masks = [m for m, keep in zip(region_masks, non_empty_instance_mask) if keep]
                assert len(region_masks) == len(instances), 'The number of region masks must match the number of instances'
                region_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in region_masks])
                )
                instances.region_masks = region_masks

            if 'vp_image' in dataset_dict:
                vp_region_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in vp_region_masks])
                )
                instances.vp_region_masks = vp_region_masks
                instances.vp_fill_number = torch.tensor(vp_fill_number, dtype=torch.int64)

            # print("instances:", instances)
            # coco mapper中的instances实际上就是一个帧中的所有物体
            dataset_dict["instances"] = instances
            #print ('instances:', instances)
            #print("dataset_dict:", dataset_dict.keys())
        return dataset_dict


def build_transform_gen_for_eval(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation


class COCOInstanceNewBaselineDatasetMapperForEval(COCOInstanceNewBaselineDatasetMapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tfm_gens = build_transform_gen_for_eval(cfg)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
