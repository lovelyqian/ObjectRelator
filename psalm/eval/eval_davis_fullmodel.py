import torch
import os
from enum import Enum
from tqdm import tqdm
import numpy as np
from detectron2.structures import BitMasks
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import cv2
from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from psalm.train.train_datasets_eval import COCO_interactive_dataset
from psalm.eval.eval_davis_evaonly import Multicondition_Dataset

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
from pathlib import Path
from segmentation_evaluation import openseg_classes
COLOR_MAP = openseg_classes.ADE20K_150_CATEGORIES
from detectron2.data import detection_utils as utils
import pickle
import math
@dataclass
class DataCollatorForCOCODatasetV2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if len(instances[0]) == 0:
            return {}
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'vp_image' in instances[0]:
            vp_images = [instance['vp_image'] for instance in instances]
            if all(x is not None and x.shape == vp_images[0].shape for x in vp_images):
                batch['vp_images'] = torch.stack(vp_images)
            else:
                batch['vp_images'] = vp_images
        for instance in instances:
            for key in ['input_ids', 'labels', 'image']:
                del instance[key]
        batch['seg_info'] = [instance for instance in instances]

        if 'dataset_type' in instances[0]:
            batch['dataset_type'] = [instance['dataset_type'] for instance in instances]

        if 'class_name_ids' in instances[0]:
            class_name_ids = [instance['class_name_ids'] for instance in instances]
            if any(x.shape != class_name_ids[0].shape for x in class_name_ids):
                batch['class_name_ids'] = torch.nn.utils.rnn.pad_sequence(
                    class_name_ids,
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                batch['class_name_ids'] = torch.stack(class_name_ids, dim=0)
        if 'token_refer_id' in instances[0]:
            token_refer_id = [instance['token_refer_id'] for instance in instances]
            batch['token_refer_id'] = token_refer_id
        if 'cls_indices' in instances[0]:
            cls_indices = [instance['cls_indices'] for instance in instances]
            if any(x.shape != cls_indices[0].shape for x in cls_indices):
                batch['cls_indices'] = torch.nn.utils.rnn.pad_sequence(
                    cls_indices,
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                batch['cls_indices'] = torch.stack(cls_indices, dim=0)
        if 'random_idx' in instances[0]:
            random_idxs = [instance['random_idx'] for instance in instances]
            batch['random_idx'] = torch.stack(random_idxs, dim=0)
        if 'class_name_embedding_indices' in instances[0]:
            class_name_embedding_indices = [instance['class_name_embedding_indices'] for instance in instances]
            class_name_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                class_name_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['class_name_embedding_indices'] = class_name_embedding_indices
        if 'refer_embedding_indices' in instances[0]:
            refer_embedding_indices = [instance['refer_embedding_indices'] for instance in instances]
            refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                refer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['refer_embedding_indices'] = refer_embedding_indices

        # print("batch:", batch.keys())

        return batch

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    only_two_class: bool = False
    old_two_class: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/home/emzhang/data/segmentation/refer_seg/images/mscoco/images/train2014')
    # mask_config: Optional[str] = field(default="./llava/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    region_mask_type: Optional[str] = field(default=None)
    # json_path: str = '/home/emzhang/code/LLaVA/datasets/refcoco/refcoco_train_sampled.json'
    json_path: str = '/home/emzhang/code/LLaVA/datasets/refcoco/refcoco_val.json'
    model_path: str = '/home/emzhang/code/llava_zem/checkpoints/SEG_class_refcoco_after_fixbug'
    model_map_name: str = 'psalm_video'
    version: str = 'opt-iml-1.3b'
    SEG_norm: bool = field(default=False)
    SEG_proj: bool = field(default=True)
    criterion_type: Optional[str] = field(default="concat_seg")
    matcher_type: Optional[str] = field(default="wo_class")
    llm_pos: Optional[str] = field(default="none")
    ln_2048: bool = field(default=False)
    seg_idx_back: bool = field(default=False)
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    thr: float = 0.5
    topk: int=1
    fuse_score: bool = field(default=False)
    seg_task: Optional[str] = field(default="region")
    seg_last: bool = field(default=True)
    num_chunks: int=1
    chunk_idx: int=0

def parse_outputs(outputs,gt_mask):
    res_list = []
    for output in outputs:
        # gt = output['gt'].cpu().numpy().astype(np.uint8)

        pred_mask = output['instances'].pred_masks
        pred_mask = pred_mask.cpu().numpy()
        scores = output['instances'].scores.transpose(1,0).cpu().numpy()
        gt_mask = output['gt'].cpu().numpy().astype(np.uint8)
        try:
            pred_cls = output['instances'].pred_classes.cpu().numpy()
        except:
            pred_cls = None
        assert scores.shape[0] == gt_mask.shape[0]
        for i in range(gt_mask.shape[0]):
            res = {
                'pred':pred_mask,
                'gt': gt_mask[i],
                'scores':scores[i],
                'pred_cls':pred_cls
            }
            res_list.append(res)
    return res_list

def get_center(mask,h,w):
    y_coords, x_coords = np.where(mask == 1)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return 0.5, 0.5
    
    centroid_y = int(np.mean(y_coords))
    centroid_x = int(np.mean(x_coords))
    # centroid_x, centroid_y = np.median(mask.nonzero(), axis=1)[::-1]
    centroid_y = centroid_y / h
    centroid_x = centroid_x / w
    return centroid_y, centroid_x

def get_distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def iou(mask1,mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_metric(le_meter,intersection_meter,union_meter,acc_iou_meter,results_list,thr=0.5,topk=3,vis=False):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    tot = 0
    cor = 0
    for results in results_list:
        gt = results['gt']
        preds = results['pred']
        scores = results['scores']
        # import ipdb;ipdb.set_trace()
        preds = preds.astype(np.uint8)
        _,idx = torch.topk(torch.tensor(scores),topk)
        idx = idx.cpu().numpy()
        topk_preds = preds[idx,:]
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0
        for i,pred_ in enumerate(topk_preds):
            h,w = pred_.shape[:2]
            pred_y, pred_x = get_center(pred_,h,w)
            gt_y, gt_x = get_center(gt,h,w)
            dist = get_distance(pred_x,pred_y,gt_x,gt_y)
            le_meter.update(dist)
            intersection, union, _ = intersectionAndUnionGPU(
                torch.tensor(pred_).int().cuda().contiguous().clone(), torch.tensor(gt).int().cuda().contiguous(), 2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # no-object target
            fore_acc_iou = acc_iou[1]
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                max_i = i
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        pred_list.append(topk_preds[max_i])
        gt_list.append(gt)

        fg_iou = acc_iou[1]
        if fg_iou > 0.5:
            cor += 1
            tot += 1
        else:
            tot += 1

    return pred_list,gt_list, cor, tot

def resize_decoded_mask(decoded_mask,resized_h, resized_w):
    segm = mask.decode(decoded_mask).astype(np.uint8)
    new_mask = cv2.resize(segm,(resized_w,resized_h))
    new_mask[new_mask > 0] = 1
    new_mask = new_mask.astype(np.uint8)
    resized_mask = mask.encode(np.asfortranarray(new_mask))
    return resized_mask

def decode_mask(decoded_mask):
    segm = mask.decode(decoded_mask).astype(np.uint8)
    return segm

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# class EGO4D_Dataset(COCO_interactive_dataset):
#     def __init__(self, json_path, tokenizer, data_args):
#         super(EGO4D_Dataset).__init__()
#         if isinstance(json_path, list):
#             data = []
#             for path in json_path:
#                 with open(path) as f:
#                     cur_data = json.load(f)
#                 data.extend(cur_data)
#         else:
#             with open(json_path) as f:
#                 data = json.load(f)
#         self.data = get_chunk(data,data_args.num_chunks,data_args.chunk_idx)
#         # self.data = data
#         self.tokenizer = tokenizer
#         self.data_args = data_args
#         coco_class_ids = [
#             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
#             18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
#             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
#             50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
#             64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
#             82, 84, 85, 86, 87, 88, 89, 90
#         ]
#         coco_class_name = [
#             'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#             'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#             'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#             'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#             'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
#             'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#             'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle',
#             'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#             'banana', 'apple', 'sandwich', 'orange', 'broccoli',
#             'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#             'couch', 'potted plant', 'bed', 'dining table', 'toilet',
#             'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
#             'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
#         ]
#         self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
#         self.coco_class_name = coco_class_name + ['background']
#     def __getitem__(self, idx):
#         ego_query = False
#         data = self.data[idx]
#         image_root = os.path.join(self.data_args.image_folder,data['root_dir'],'frame_aligned_videos/downscaled/448')
#         image_frame = str(data['take_frame']) + '.jpg'
#         image_path = None
#         vp_image_path = None
#         vp_mask_list = []
#         gt_mask_list = []
#         object_pool = []
#         object_list = data['object_list']
#         for object in object_list:
#             for key, value in data[object].items():
#                 cur_object_info = (os.path.join(image_root,key,image_frame),value)
#                 object_pool.append(cur_object_info)
#
#         if ego_query:
#             for obj in object_pool:
#                 name, mask = obj
#                 if 'ari' not in name:
#                     if image_path is None:
#                         image_path = name
#                     else:
#                         assert image_path == name, f'exsit name is {image_path}, while also a name {name}'
#                     gt_mask_list.append(mask)
#                 else:
#                     if vp_image_path is None:
#                         vp_image_path = name
#                     else:
#                         assert vp_image_path == name, f'exsit vp name is {image_path}, while also a name {name}'
#                     vp_mask_list.append(mask)
#         else:
#             for obj in object_pool:
#                 name, mask = obj
#                 if 'ari' in name:
#                     if image_path is None:
#                         image_path = name
#                     else:
#                         assert image_path == name, f'exsit name is {image_path}, while also a name {name}'
#                     gt_mask_list.append(mask)
#
#                 else:
#                     if vp_image_path is None:
#                         vp_image_path = name
#                     else:
#                         assert vp_image_path == name, f'exsit vp name is {image_path}, while also a name {name}'
#                     vp_mask_list.append(mask)
#
#         # resize mask
#         if not os.path.exists(image_path) or not os.path.exists(vp_image_path):
#             print(f'cannot find {image_path}')
#             return {}
#         # image = cv2.imread(image_path)
#         # h,w = image.shape[:2]
#         # vp_image = cv2.imread(vp_image_path)
#         # vp_h,vp_w = vp_image.shape[:2]
#         gt_mask_h,gt_mask_w = decode_mask(gt_mask_list[0]).shape[:2]
#         vp_mask_h,vp_mask_w = decode_mask(vp_mask_list[0]).shape[:2]
#         img = utils.read_image(image_path, format='RGB')
#         vp_img = utils.read_image(vp_image_path, format='RGB')
#         img = cv2.resize(img,(gt_mask_w,gt_mask_h))
#         vp_img = cv2.resize(vp_img,(vp_mask_w,vp_mask_h))
#         # gt_mask_list = [decode_mask(mask_) for mask_ in gt_mask_list]
#         # vp_mask_list = [decode_mask(mask_) for mask_ in vp_mask_list]
#         data_dict = {}
#         data_dict['file_path'] = image_path
#         data_dict['gt_mask_list'] = copy.deepcopy(gt_mask_list)
#         data_dict['vp_file_path'] = vp_image_path
#         data_dict['file_name'] = img
#         data_dict['vp_image'] = vp_img
#         data_dict['height'] = gt_mask_h
#         data_dict['width'] = gt_mask_w
#         data_dict['image_id'] = idx
#         data_dict['annotations'] = []
#         data_dict['vp_annotations'] = []
#         for mask in gt_mask_list:
#             anno = {}
#             anno['bbox_mode'] = BoxMode.XYXY_ABS
#             anno['bbox'] = [0, 0, 0, 0]
#             anno['image_id'] = idx
#             anno['category_id'] = 1
#             anno['segmentation'] = mask
#             data_dict['annotations'].append(anno)
#         for mask in vp_mask_list:
#             anno = {}
#             anno['bbox_mode'] = BoxMode.XYXY_ABS
#             anno['bbox'] = [0, 0, 0, 0]
#             anno['image_id'] = idx
#             anno['category_id'] = 1
#             anno['segmentation'] = mask
#             data_dict['vp_annotations'].append(anno)
#         try:
#             if isinstance(self.data_args.image_processor,dict):
#                 processor = self.data_args.image_processor['instance']
#             else:
#                 processor = self.data_args.image_processor
#             region_mask_type = getattr(self.data_args,'region_mask_type',None)
#             if region_mask_type is not None:
#                 region_mask_type = region_mask_type.split('||')
#             data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')
#         except:
#             print('load data wrong')
#             return {}
#
#         num_target = len(data_dict['instances'])
#         prefix_inst = 'This is an image <image>, Please segment by given regions'
#         # prompt_inst = 'Iteractive segmentation: using provided region as a reference'
#         regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
#         sources_value = f'\nThis is all regions: {regions_inst}\n'
#
#         if self.data_args.seg_last:
#             sources = [
#                 [{'from': 'human', 'value': prefix_inst + sources_value},
#                  {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
#         else:
#             sources = [
#                 [{'from': 'human', 'value': prefix_inst + sources_value},
#                  {'from': 'gpt', 'value': '\n[SEG]'}]]
#         # sources = self.preprocess_multimodal(copy.deepcopy(sources))
#
#         text_dict = self.preprocess_llama2(sources, self.tokenizer)
#         input_ids = text_dict['input_ids'][0]
#         labels = text_dict['labels'][0]
#         data_dict['input_ids'] = input_ids
#         data_dict['labels'] = labels
#         data_dict['dataset_type'] = 'region_coco'
#
#         return data_dict


class DAVIS_Dataset(COCO_interactive_dataset):

    #注意，这里所有的处理逻辑针对的都是一帧图像
    def __getitem__(self, idx):
        data = self.data[idx]

        #图片的相对路径名称，like2017/trainval/JPEGImages/480p/bike-packing/00001.jpg
        image_file = data['image']
        #image_folder是data_root根路径 在这里是data_segswap
        image_folder = self.data_args.image_folder


        data_dict = {}
        #file_name是图片的完整路径名称，like /data/Davis/2017/trainval/JPEGImages/480p/bike-packing/00001.jpg
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        #image_id可以理解为计数器，编号
        data_dict['image_id'] = data['new_img_id']
        #annotations，本帧对应的注释，coco格式的分割mask，一张图片可能包含多个实例的mask
        data_dict['annotations'] = data['anns']
        #vp_annotations，每段视频中第一帧的注释
        data_dict['vp_annotations'] = data['first_frame_anns']
        #vp_image，每段视频中第一帧的完整路径，like /data/Davis/2017/trainval/JPEGImages/480p/bike-packing/00000.jpg
        data_dict['vp_image'] = os.path.join(image_folder,data['first_frame_image'])
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            #边界框左上角和右下角的坐标都为原点，意思是将边界框置为空框
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']
        for annotation in data_dict['vp_annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']

        #初始化processor，应该是个图像预处理器，再送进visual encoder之前，总体来说下面的一小段代码是对输入图像和mask的预处理
        # print("self.data_args.image_processor", self.data_args.image_processor)
        if isinstance(self.data_args.image_processor,dict):
            #根据是否是对齐ego exo的size进行切换，图像预处理器
            processor = self.data_args.image_processor['instance']
            # processor = self.data_args.image_processor['instance_resize']
        else:
            processor = self.data_args.image_processor
        #print('processor:', processor)
        #尝试从命令行参数中获取region_mask_type
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
     
        #根据region_mask_type和mask_format（这里是0、1掩码），对原始的data_dict进行预处理，将Detectron2格式的dataset dict转化为MaskFormer格式的
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')


        #num_target，本帧图像中有多少个对象
        #下面的一小段代码，主要是利用llama处理输入的文本，生成对应的token
        num_target = len(data_dict['instances'])
        #<image> 是一个特殊的占位符，表示图像的输入
        prefix_inst = 'This is an image <image>, Please segment by given regions'
        #<region> 占位符来表示每个需要分割的区域，用逗号分隔，最后一个 <region> 以句号结束，例如，如果有 3 个区域，结果是 ' <region>, <region>, <region>.'
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'

        #sources构建了一个人类和模型交互的对话格式，定义了来自人类的输入和来自模型的输出
        sources = [
            [{'from': 'human', 'value': prefix_inst + sources_value},
             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        #input_ids是模型的实际输入，是由分词器将文本 sources 转换为的一系列数字标识（token IDs）
        input_ids = text_dict['input_ids'][0]
        #labels是模型训练时的token的真实标签，与input_ids对应
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'region_coco'

        return data_dict

class Ego_Train_Dataset(COCO_interactive_dataset):

    #注意，这里所有的处理逻辑针对的都是一帧图像
    def __getitem__(self, idx):
        data = self.data[idx]

        #图片的相对路径名称，like2017/trainval/JPEGImages/480p/bike-packing/00001.jpg
        image_file = data['image']
        #image_folder是data_root根路径 在这里是data_segswap
        image_folder = self.data_args.image_folder


        data_dict = {}
        #file_name是图片的完整路径名称，like /data/Davis/2017/trainval/JPEGImages/480p/bike-packing/00001.jpg
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        #image_id可以理解为计数器，编号
        data_dict['image_id'] = data['new_img_id']
        #annotations，本帧对应的注释，coco格式的分割mask，一张图片可能包含多个实例的mask
        data_dict['annotations'] = data['anns']
        #vp_annotations，每段视频中第一帧的注释
        data_dict['vp_annotations'] = data['first_frame_anns']
        #vp_image，每段视频中第一帧的完整路径，like /data/Davis/2017/trainval/JPEGImages/480p/bike-packing/00000.jpg
        data_dict['vp_image'] = os.path.join(image_folder,data['first_frame_image'])
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            #边界框左上角和右下角的坐标都为原点，意思是将边界框置为空框
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']
            #为了训练的时候instance能有region_mask属性而增设
            # annotation['mask_visual_prompt_mask'] = annotation['segmentation']
        for annotation in data_dict['vp_annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']

        # 初始化processor，应该是个图像预处理器，再送进visual encoder之前，总体来说下面的一小段代码是对输入图像和mask的预处理
        # print("self.data_args.image_processor", self.data_args.image_processor)
        if isinstance(self.data_args.image_processor,dict):
            #根据是否是对齐ego exo的size进行切换，图像预处理器
            processor = self.data_args.image_processor['instance']
            # processor = self.data_args.image_processor['instance_resize']
        else:
            processor = self.data_args.image_processor
        #尝试从命令行参数中获取region_mask_type
        #print("processor:", processor)    #coco_instance_mapper
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        #print('region_mask_type:', region_mask_type) # None
        # print("region_mask_type:", region_mask_type)
        #根据region_mask_type和mask_format（这里是0、1掩码），对原始的data_dict进行预处理，将Detectron2格式的dataset dict转化为MaskFormer格式的
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')


        #num_target，本帧图像中有多少个对象
        #下面的一小段代码，主要是利用llama处理输入的文本，生成对应的token
        num_target = len(data_dict['instances'])
        #<image> 是一个特殊的占位符，表示图像的输入
        prefix_inst = 'This is an image <image>, Please segment by given regions'
        #<region> 占位符来表示每个需要分割的区域，用逗号分隔，最后一个 <region> 以句号结束，例如，如果有 3 个区域，结果是 ' <region>, <region>, <region>.'
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'

        #sources构建了一个人类和模型交互的对话格式，定义了来自人类的输入和来自模型的输出
        sources = [
            [{'from': 'human', 'value': prefix_inst + sources_value},
             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        #input_ids是模型的实际输入，是由分词器将文本 sources 转换为的一系列数字标识（token IDs）
        input_ids = text_dict['input_ids'][0]
        #labels是模型训练时的token的真实标签，与input_ids对应
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'region_coco'

        return data_dict



def fuse_davis_mask(mask_list,fill_number_list):
    fused_mask = np.zeros_like(mask_list[0])
    for mask, fill_number in zip(mask_list,fill_number_list):
        fill_number = int(fill_number)
        fused_mask[mask == 1] = fill_number
    return fused_mask


import os
import re

def get_latest_checkpoint_path(model_path):
    # 正则表达式用于匹配 checkpoint 文件夹名称格式：checkpoint-<iter>
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    
    # 检查是否已经是具体的 checkpoint 路径
    if os.path.basename(model_path).startswith("checkpoint-") and checkpoint_pattern.match(os.path.basename(model_path)):
        return model_path  # 已经是具体的 checkpoint，直接返回
    
    # 如果是目录路径，查找其中的最新 checkpoint
    elif os.path.isdir(model_path):
        checkpoints = [d for d in os.listdir(model_path) if checkpoint_pattern.match(d)]
        
        if not checkpoints:
            raise ValueError("No checkpoints found in the specified directory.")
        
        # 根据迭代次数找到最新的 checkpoint
        max_checkpoint = max(checkpoints, key=lambda x: int(checkpoint_pattern.match(x).group(1)))
        model_path = os.path.join(model_path, max_checkpoint)
    
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified path '{model_path}' does not exist.")
    
    return model_path


def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_path = get_latest_checkpoint_path(model_path)     #xiugai: to adapt only input model path without sepcify the ckp path
    print('------------------------TESTING----------------- ckp:', model_path)
    model_name = get_model_name_from_path(model_path)
    print(f'current model is {model_path}')
    print('save model name:', model_name)
    model_name = 'psalm_SSL_MultiCondition'
    print('now changed the model name to:', model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')
    # ckpt = torch.load(os.path.join(model_path,'pytorch_model.bin'))
    # model.load_state_dict(ckpt,strict=True)

    data_args.image_processor = image_processor
    #print('image_processor:', image_processor)

    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    # eval_dataset = EGO4D_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    # eval_dataset = DAVIS_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    #eval_dataset = Ego_Train_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    eval_dataset = Multicondition_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)

    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])

    def load_ref_dataset():
        return RefCOCO_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)

    DatasetCatalog.register('refcoco_dataset', load_ref_dataset)
    MetadataCatalog.get('refcoco_dataset').set(stuff_classes=['object'],)
    gt_json_path = data_args.json_path
    save_dir = os.path.dirname(gt_json_path)
    save_dir = os.path.join(save_dir,'mask_predictions')

    # evaluator = my_refcoco_evaluator('refcoco_dataset', output_dir='./output/instruction_segmentation', distributed=False)
    # evaluator.reset()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    model.to(device=device,dtype=torch.float).eval()
    # inference_on_dataset(model, eval_dataloader, evaluator)
    save_list = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    le_meter = AverageMeter("LE", ":6.3f", Summary.SUM)
    cor = 0
    tot = 0
    prev_image = None
    prev_mask_list = None
    prev_fill_number_list = None
    prev_video = None
    prev_transformer = None


    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            if len(inputs) == 0:
                print('no data load')
                continue

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]

            try:

                if 'instance' in data_args.model_map_name:
                    outputs = model.eval_video(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        images=inputs['images'].float(),
                        vp_images=inputs['vp_images'].float(),
                        seg_info=inputs['seg_info'],
                        class_name_embedding_indices=inputs['class_name_embedding_indices'],
                        class_name_ids=inputs['class_name_ids'],
                        cls_indices=inputs['cls_indices'],
                        labels=inputs['labels']
                    )
                else:
                    #print('comes else!') # YES
                    '''
                    outputs = model.eval_video(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        images=inputs['images'].float(),
                        vp_images=inputs['vp_images'].float(),
                        seg_info=inputs['seg_info'],
                        labels=inputs['labels']
                    )
                    '''
                    #print('EVAL INPUT:', 'token_refer_id:', inputs['token_refer_id'], 'refer_embedding_indices:', inputs['refer_embedding_indices']) #Yes
                    outputs = model.eval_video(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        images=inputs['images'].float(),
                        vp_images=inputs['vp_images'].float(),
                        seg_info=inputs['seg_info'],
                        token_refer_id = inputs['token_refer_id'],
                        refer_embedding_indices=inputs['refer_embedding_indices'],
                        labels=inputs['labels']
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except:
                print('something wrong when infer')
                continue
            
            output = outputs[0]
            pred_mask = output['instances'].pred_masks
            pred_mask = pred_mask.cpu().numpy()
            scores = output['instances'].scores.transpose(1, 0).cpu().numpy()
            gt_mask = output['gt'].cpu().numpy().astype(np.uint8)
            assert len(scores) == len(inputs['seg_info'][0]['instances'].vp_fill_number)
            pred_mask_list = []
            pred_score_list = []
            fill_number_list = []
            prev_idx = []
            for i in range(len(scores)):
                cur_scores = scores[i]
                cur_fill_number = inputs['seg_info'][0]['instances'].vp_fill_number[i]
                max_score, idx = torch.topk(torch.tensor(cur_scores), 10, largest=True, sorted=True)
                idx = idx.cpu().numpy()
                for i in range(10):
                    if idx[i] not in prev_idx:
                        prev_idx.append(idx[i])
                        pick_idx = idx[i]
                        pick_score = max_score[i]
                        break
                #TODO这里curpred是单个物体的mask，可以在这里看看能不能提取种类id信息
                cur_pred = pred_mask[pick_idx, :]
                pred_score_list.append(pick_score)
                pred_mask_list.append(cur_pred)
                fill_number_list.append(cur_fill_number)
            pred_mask_list = [tensor_.astype(np.uint8) for tensor_ in pred_mask_list]

            fused_pred_mask = fuse_davis_mask(pred_mask_list,fill_number_list)

            #保存分割mask以及可视化的彩色图像
            save_name = inputs['seg_info'][0]['file_name']
            save_name = "egofullmodel_smalljson_new/" + save_name.split('/data_segswap/')[1]
            #save_name = '480p/' + save_name.split('/480p/')[1] #debug
            save_path = os.path.join(save_dir,save_name).split('.')[0] + '.png'
            Path(os.path.dirname(save_path)).mkdir(exist_ok=True,parents=True)
            cv2.imwrite(save_path,fused_pred_mask)
    print(f'==>finish eval DAVIS, save in {save_dir}')



def evaluate_with_json():
    import pickle
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    le_meter = AverageMeter("LE", ":6.3f", Summary.SUM)
    name_number = 0
    good_data = []
    with open("/data/work-gcp-europe-west4-a/yuqian_fu/Ego/huggingface/hub/PSALM/pred_pkl/pred_gt_1_1_0.pkl",'rb') as f:
        data = pickle.load(f)
    for data_ in tqdm(data):
        pred_ = data_['pred'][0]

        pred_ = mask.decode(pred_)
        gt = data_['gt'][0]
        gt = mask.decode(gt)
        h,w = pred_.shape[:2]
        pred_y, pred_x = get_center(pred_,h,w)
        gt_y, gt_x = get_center(gt,h,w)
        dist = get_distance(pred_x,pred_y,gt_x,gt_y)
        le_meter.update(dist)
        intersection, union, _ = intersectionAndUnionGPU(
            torch.tensor(pred_).int().cuda().contiguous().clone(), torch.tensor(gt).int().cuda().contiguous(), 2, ignore_index=255
        )
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = intersection / (union + 1e-5)
        acc_iou[union == 0] = 1.0  # no-object target
        fore_acc_iou = acc_iou[1]
        if fore_acc_iou > 0.5:
            good_data.append(data_)
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)
    print(f'total {len(good_data)} good data, save')
    with open("/data/work-gcp-europe-west4-a/yuqian_fu/Ego/huggingface/hub/PSALM/pred_pkl/good_sample_egoquery,pkl", 'wb') as f:
        pickle.dump(good_data, f)


if __name__ == '__main__':
    # 144 takes  64606 frames
    evaluation()
    
