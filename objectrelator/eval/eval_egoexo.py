import torch
import os
from enum import Enum
from tqdm import tqdm
import numpy as np
from detectron2.structures import BitMasks
from objectrelator.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX
from objectrelator.model.builder import load_pretrained_model
from objectrelator.utils import disable_torch_init
from objectrelator.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from objectrelator.mask_config.data_args import DataArguments
import cv2
from torch.utils.data import Dataset, DataLoader
from objectrelator import conversation as conversation_lib
from datasets.egoexo_dataset import EgoExo_Dataset_eval
from pycocotools.mask import encode, decode, frPyObjects
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
import json
import utils_metric
import os 
import re
from natsort import natsorted


# collection func
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

        return batch
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

# fuse mask
def fuse_mask(mask_list,fill_number_list):
    fused_mask = np.zeros_like(mask_list[0])
    for mask, fill_number in zip(mask_list,fill_number_list):
        fill_number = int(fill_number)
        fused_mask[mask != 0] = fill_number 
    return fused_mask

# metric calculation
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

def get_center(mask,h,w):
    y_coords, x_coords = np.where(mask == 1)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return 0.5, 0.5
    
    centroid_y = int(np.mean(y_coords))
    centroid_x = int(np.mean(x_coords))
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

def parse_outputs(outputs,gt_mask):
    res_list = []
    for output in outputs:
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

# latest checkpoint path
def get_latest_checkpoint_path(model_path):
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    if os.path.basename(model_path).startswith("checkpoint-") and checkpoint_pattern.match(os.path.basename(model_path)):
        return model_path  
    elif os.path.isdir(model_path):
        checkpoints = [d for d in os.listdir(model_path) if checkpoint_pattern.match(d)]
        if not checkpoints:
            raise ValueError("No checkpoints found in the specified directory.")
        max_checkpoint = max(checkpoints, key=lambda x: int(checkpoint_pattern.match(x).group(1)))
        model_path = os.path.join(model_path, max_checkpoint)
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified path '{model_path}' does not exist.")
    return model_path


# hyperparameters
parser = transformers.HfArgumentParser(DataArguments)
data_args = parser.parse_args_into_dataclasses()[0]

# load json_file
with open(data_args.json_path, 'r') as f:
    datas = json.load(f)
with open(data_args.split_path, 'r') as f:
    takes_all = json.load(f)
eval_takes = takes_all[data_args.split]

# load model
disable_torch_init()
model_path = os.path.expanduser(data_args.model_path)
model_path = get_latest_checkpoint_path(model_path)
print(f'current model is {model_path}')
model_name = 'ObjectRelator'
print('Loading model:', model_name)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')
print('Model loaded successfully!')
data_args.image_processor = image_processor
data_args.is_multimodal = True
conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version_val]


def evaluation(take_id):
    # Construct the dataset for the current take
    data_list = []
    for data in datas:
        if data['video_name'] == take_id:
            data_list.append(data)
    eval_dataset = EgoExo_Dataset_eval(data_list=data_list, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers_val,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device=device,dtype=torch.float).eval()

    # Get the target camera and object bounds under the current take
    cam_target = data_list[0]['image'].split('/')[-2]
    gt_path = f"{data_args.image_folder}/{take_id}/annotation.json"
    with open(gt_path, 'r') as fp:
        gt = json.load(fp)
    objs = natsorted(list(gt["masks"].keys()))
    coco_id_to_cont_id = {cont_id + 1: coco_id for cont_id, coco_id in enumerate(objs)}
    id_range = list(coco_id_to_cont_id.keys())
    obj_target = []
    for obj in objs:
        if cam_target in gt["masks"][obj].keys():
            obj_target.append(obj)
    
    # Initialize metrics
    num_frame = 0
    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    le_meter = AverageMeter("LE", ":6.3f", Summary.SUM)

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            if len(inputs) == 0:
                print('no data load')
                continue
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
            frame_id = inputs['seg_info'][0]['file_name'].split('/')[-1].split('.')[0]

           # forward pass
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
            cur_res = parse_outputs(outputs, None)
            _,_,_,_ = compute_metric(le_meter,intersection_meter,union_meter,acc_iou_meter,cur_res,topk=data_args.topk)
            
            # Parse the results and compute metrics
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
                cur_pred = pred_mask[pick_idx, :]
                pred_score_list.append(pick_score)
                pred_mask_list.append(cur_pred)
                fill_number_list.append(cur_fill_number)
            pred_mask_list = [tensor_.astype(np.uint8) for tensor_ in pred_mask_list]
            fused_pred_mask = fuse_mask(pred_mask_list,fill_number_list)
            
            obj_range = []
            for obj in obj_target:
                if frame_id in gt["masks"][obj][cam_target].keys():
                    obj_range.append(obj)
            pred_mask = fused_pred_mask
            unique_instances = np.unique(pred_mask)
            unique_instances = unique_instances[unique_instances != 0]
            unique_instances = [x for x in unique_instances if x in id_range]
            if len(unique_instances) == 0:
                continue

            num_frame += 1
            for instance_value in unique_instances:
                binary_mask = (pred_mask == instance_value).astype(np.uint8)
                h,w = binary_mask.shape
                obj_name = coco_id_to_cont_id[instance_value]
                if obj_name not in obj_range:
                    continue
                gt_mask = decode(gt["masks"][obj_name][cam_target][frame_id])
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                _, shape_acc = utils_metric.eval_mask(gt_mask, binary_mask) 
                ex_acc = utils_metric.existence_accuracy(gt_mask, binary_mask)
                location_score = utils_metric.location_score(gt_mask, binary_mask, size=(h, w))
                ShapeAcc.append(shape_acc)
                ExistenceAcc.append(ex_acc)
                LocationScores.append(location_score)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou = iou_class[1]
    IoUs.append(iou)

    
    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)

    return IoUs.tolist(), ShapeAcc.tolist(), ExistenceAcc.tolist(), LocationScores.tolist(), num_frame



if __name__ == '__main__':
    
    total_iou = []
    total_shape_acc = []
    total_existence_acc = []
    total_location_scores = []
    num_total = 0
   
    for take_id in eval_takes:
        ious, shape_accs, existence_accs, location_scores, num_frame = evaluation(take_id)
        total_iou += ious
        total_shape_acc += shape_accs
        total_existence_acc += existence_accs
        total_location_scores += location_scores
        num_total += num_frame

    print('TOTAL IOU: ', np.mean(total_iou))
    print('TOTAL LOCATION SCORE: ', np.mean(total_location_scores))
    print('TOTAL SHAPE ACC: ', np.mean(total_shape_acc))
    print('TOTAL EXISTENCE ACC: ', np.mean(total_existence_acc))
    print("total frames:", num_total)
    