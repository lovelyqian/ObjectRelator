import torch
import os
from enum import Enum
from tqdm import tqdm
import numpy as np
from detectron2.structures import BitMasks
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX
#debug: 新的builder方便能使用llava_phi_condition中的模型
from psalm.model.builder_condition import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import cv2
from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib

# debug
# from psalm.train.train_datasets import COCO_interactive_dataset
from psalm.train.train_datasets_eval import COCO_interactive_dataset

import json
import re
from pycocotools import mask
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
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, DEFAULT_CLS_TOKEN, CLS_TOKEN_INDEX, DEFAULT_REGION_TOKEN, \
    REGION_TOKEN_INDEX, REFER_TOKEN_INDEX

from psalm.model.language_model.llava_phi_condition import PSALMForDAVISEval, PSALM


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
    #debug: model_map_name设置为psalm_video，默认使用的就是PSALMForDAVISE
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

# debug: vp和ref解析输出的函数不太一样
# vp的结果解析
def parse_outputs_vp(outputs,gt_mask):
    res_list = []
    # outputs指的是多帧的结果
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
        # 判断预测物体的数量和gt中物体的数量是否一致
        assert scores.shape[0] == gt_mask.shape[0]
        #debug：区别在于vp中每一帧存储多个物体的预测结果和gt信息，而ref每一帧仅存储一个物体的信息
        for i in range(gt_mask.shape[0]):
            res = {
                'pred':pred_mask,
                'gt': gt_mask[i],
                'scores':scores[i],
                'pred_cls':pred_cls
            }
            res_list.append(res)
    return res_list

# ref的结果解析
def parse_outputs_ref(outputs,gt_mask):
    res_list = []
    for output in outputs:
        # gt = output['gt'].cpu().numpy().astype(np.uint8)

        pred_mask = output['instances_ref'].pred_masks
        pred_mask = pred_mask.cpu().numpy()
        scores = output['instances_ref'].scores.cpu().numpy()
        try:
            pred_cls = output['instances_ref'].pred_classes.cpu().numpy()
        except:
            pred_cls = None
        res = {
            'pred':pred_mask,
            'gt': gt_mask,
            'scores':scores,
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

# vp的指标计算函数
def compute_metric_vp(le_meter,intersection_meter,union_meter,acc_iou_meter,results_list,thr=0.5,topk=3,vis=False):
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

# ref的指标计算函数
def compute_metric_ref(intersection_meter,union_meter,acc_iou_meter, gt_cls, results_list):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    for results in results_list:
        gt = results['gt']
        preds = results['pred']
        scores = results['scores']
        preds = preds.astype(np.uint8)
        # pick mask with maximum score
        topk_scores,idx = torch.topk(torch.tensor(scores),1)
        idx = idx.cpu().numpy()
        topk_preds = preds[idx,:]
        if results['pred_cls'] is not None:
            topk_pred_cls = results['pred_cls'][idx]
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0
        # here topk=1, len(topk_preds)=1
        for i,pred_ in enumerate(topk_preds):
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

    return pred_list,gt_list

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

class Multicondition_Dataset(COCO_interactive_dataset):

    #将ref instruction转化为整数tokens序列，并在末尾加上代表整个句子全部含义的[SEG]token
    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        tokenized = tokenized + [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    
    # 相较于interatitive类，新增加了<ref>
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX,refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index, '<refer>':refer_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>|<refer>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

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
        #debug：这里没有把refdataset里的category_id处理搬过来，不知道有影响吗
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

       #初始化processor，应该是个图像预处理器，再送进visual encoder之前，总体来说下面的一小段代码是对输入图像和mask的预处理
        # print("self.data_args.image_processor", self.data_args.image_processor)
        if isinstance(self.data_args.image_processor,dict):
            #根据是否是对齐ego exo的size进行切换，图像预处理器
            processor = self.data_args.image_processor['instance']
            # processor = self.data_args.image_processor['instance_resize']
        else:
            processor = self.data_args.image_processor
        #尝试从命令行参数中获取region_mask_type
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        # print("region_mask_type:", region_mask_type)
        #根据region_mask_type和mask_format（这里是0、1掩码），对原始的data_dict进行预处理，将Detectron2格式的dataset dict转化为MaskFormer格式的
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')

        #debug: 目前为止和egodataset完全一样，除了上面增加的两个函数
        sentences = data['instruction']

        #num_target，本帧图像中有多少个对象
        #下面的一小段代码，主要是利用llama处理输入的文本，生成对应的token
        num_target = len(data_dict['instances'])
        #<image> 是一个特殊的占位符，表示图像的输入

        #debug: 这里有个问题，使用哪种前缀提示词
        # prefix_inst = 'This is an image <image>, Please segment by given regions'
        # prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        #debug:自己创造的前缀词
        prefix_inst = 'This is an image <image>, Please segment by given regions and instruction'

        #debug: 提取一帧图像中所有的物体描述并拼接在一起
        # instruction="a bag.a cup.a pencil"
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])

        #debug: 这些特殊的站位符号本质上还是字符串
        #<region> 占位符来表示每个需要分割的区域，用逗号分隔，最后一个 <region> 以句号结束，例如，如果有 3 个区域，结果是 ' <region>, <region>, <region>.'
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'

        #sources构建了一个人类和模型交互的对话格式，定义了来自人类的输入和来自模型的输出
        
        #debug: vp_seg的对话形式
        # sources = [
        #     [{'from': 'human', 'value': prefix_inst + sources_value},
        #      {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
      
        #debug: refseg的对话形式，看看怎么把两种任务的形式结合在一起
        #[SEG]指的是代表整个句子的token，<seg>指的是代表mask token
        # sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
        #             {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        
        #debug: 自己创造的对话形式，这里需要解决的是gpt返回的value是什么SEG]<seg> or <seg>
        sources = [[{'from': 'human', 'value': prefix_inst + sources_value + "and this is the instruction: " + '<refer>\n'},
                    {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        
      
      
        #debug：sources的作用主要是输出text_dict
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        #input_ids是模型的实际输入，是由分词器将文本 sources 转换为的一系列数字标识（token IDs）
        input_ids = text_dict['input_ids'][0]
        #labels是模型训练时的token的真实标签，与input_ids对应
        labels = text_dict['labels'][0]

       
        #debug: 这里为针对ref新增加的
        # instruction在这里才用上
        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        # refer_embedding_indices[input_ids == 50256] = 1 #debug
        
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'referring_coco'
        #debug: 看看这里的dataset_type的设置有影响吗
        # data_dict['dataset_type'] = 'region_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict





def fuse_davis_mask(mask_list,fill_number_list):
    fused_mask = np.zeros_like(mask_list[0])
    for mask, fill_number in zip(mask_list,fill_number_list):
        fill_number = int(fill_number)
        fused_mask[mask == 1] = fill_number
    return fused_mask


def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f'current model is {model_path}')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')
    # ckpt = torch.load(os.path.join(model_path,'pytorch_model.bin'))
    # model.load_state_dict(ckpt,strict=True)

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    # debug: 不知道data_args.refcoco_image_folder这个有用吗
    data_args.refcoco_image_folder = data_args.image_folder
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
    with open(gt_json_path) as f:
        gt_data = json.load(f)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device=device,dtype=torch.float).eval()
    save_list = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    # debug: eval_ego多了一个le_meter指标
    le_meter = AverageMeter("LE", ":6.3f", Summary.SUM)
    cor = 0
    tot = 0
   

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            if len(inputs) == 0:
                print('no data load')
                continue
            
            #debug: 从json文件中读取target帧的mask作为gt
            gt = gt_data[idx]['anns']
            h, w = gt_data[idx]['image_info']['height'], gt_data[idx]['image_info']['width']
            masks = []
            for annotation in gt:
                if isinstance(annotation['segmentation'], list):
                    segm = np.zeros((h, w), dtype=np.uint8)
                    for poly in annotation['segmentation']:
                        poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(segm, [poly], 1)
                    masks.append(segm.astype(np.bool_))
                else:
                    if isinstance(annotation['segmentation']['counts'], list):
                        rle = mask.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
                        segm = mask.decode(rle)
                    else:
                        segm = mask.decode(annotation['segmentation'])
                    masks.append(segm.astype(np.bool_))
            # assert len(masks) == 1  #debug：一帧中可以有很多物体，这里不知道为什么要强行判断为1
            gt_mask = masks[0].astype(np.uint8)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
            
            # 基于vp_mask的推理
            # 其实模型只要推理都会输出两种任务的结果，outputs_vp也是包含两种任务的结果，分开推理没意义了
            # outputs_vp = model.eval_video(
            #     input_ids=inputs['input_ids'],
            #     attention_mask=inputs['attention_mask'],
            #     images=inputs['images'].float(),
            #     vp_images=inputs['vp_images'].float(),
            #     seg_info=inputs['seg_info'],
            #     labels=inputs['labels'],
            #     token_refer_id = inputs['token_refer_id'],
            #     refer_embedding_indices=inputs['refer_embedding_indices']
            #         )
            # print("outputs_vp",outputs_vp) #debug

            # 基于ref的推理
            # outputs_ref = model.eval_seg(
            #     input_ids=inputs['input_ids'],
            #     attention_mask=inputs['attention_mask'],
            #     images=inputs['images'].float(),
            #     vp_images=inputs['vp_images'].float(),
            #     seg_info=inputs['seg_info'],
            #     token_refer_id = inputs['token_refer_id'],
            #     refer_embedding_indices=inputs['refer_embedding_indices'],
            #     labels=inputs['labels']
            # )
            # print("outputs_ref", outputs_ref) #debug


            #v2 实际上因为self.region_on和self.reffering_on都打开了，可以用一次前向推理输出两种任务的结果
            # 即原来一帧的字典中只有{"instances":...} 现在变为了{"instances_ref":..., "instances:"...}
            outputs_total = model.eval_video(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                vp_images=inputs['vp_images'].float(),
                seg_info=inputs['seg_info'],
                token_refer_id = inputs['token_refer_id'],
                refer_embedding_indices=inputs['refer_embedding_indices'],
                labels=inputs['labels']
            )
            print("outputs_total", outputs_total) #debug







            #debug: ref的推理将每个物体的类别取了出来
            gt_cls = inputs['seg_info'][0]['instances'].gt_classes
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            #debug：ref的output解析传入了gt_mask
            cur_res = parse_outputs_ref(outputs_ref,gt_mask)
            # vp的output解析
            cur_res = parse_outputs_vp(outputs_vp, None)

            #debug：指标的计算不太一样
            # ref的计算
            pred,gt_mask = compute_metric_ref(intersection_meter,union_meter,acc_iou_meter, gt_cls, cur_res)
            # vp的计算
            pred,gt_mask,cur_cor, cur_tot = compute_metric_vp(le_meter,intersection_meter,union_meter,acc_iou_meter,cur_res,topk=data_args.topk)
            cor += cur_cor
            tot += cur_tot

            # print("inputs['seg_info']",inputs['seg_info'][0])
            # save_info = {'gt':inputs['seg_info'][0]['gt_mask_list'],
            #              'name':inputs['seg_info'][0]['file_name'],
            #              'vp_name':inputs['seg_info'][0]['vp_file_path']}
            # save_list.append(save_info)
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    #debug： 下面的指标是eval_vp里独有的
    le = le_meter.avg
    bg_giou = acc_iou_meter.avg[0]
    miou = (giou + bg_giou) / 2
    acc = cor / tot
    msg = "benchmark: {}: top {}, giou: {:.4f}, ciou: {:.4f}, miou: {:.4f}, acc: {:.4f}, LE: {:.4f}".format('ego4d',
                                                                                                data_args.topk,
                                                                                                giou, ciou, miou,
                                                                                acc, le)
    print(msg)
    
    
    
    # save_path = os.path.join(data_args.model_path, 'pred_pkl')
    # Path(save_path).mkdir(parents=True, exist_ok=True)
    # with open(os.path.join(save_path, f'pred_gt_{data_args.topk}_{data_args.num_chunks}_{data_args.chunk_idx}.pkl'), 'wb') as f:
    #     pickle.dump(save_list, f)
    # with open(os.path.join(save_path, f'pred_ego_{data_args.topk}_{data_args.num_chunks}_{data_args.chunk_idx}.txt'), 'w') as f:
    #     f.write(msg)




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
    