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
# from psalm.train.train_datasets_eval import COCO_interactive_dataset #debug
from psalm.train.train_datasets import COCO_interactive_dataset


from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
from pathlib import Path
# from segmentation_evaluation import openseg_classes
from psalm.eval.segmentation_evaluation import openseg_classes
COLOR_MAP = openseg_classes.ADE20K_150_CATEGORIES


import re
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX, REGION_TOKEN_INDEX, REFER_TOKEN_INDEX

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




#从eval_davis中的DataCollatorForCOCODatasetV2类中，可以看出DAVIS_Dataset类每一帧对应的字典有哪些键
@dataclass
class DataCollatorForCOCODatasetV2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    #sequence表示列表、元组等有序对象，instances的类型表示为字典组成的有序列表，其中一个字典表示一帧图像
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
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
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/path/to/val2017')
    model_path: Optional[str] = field(default="/path/to/model")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/path/to/coco'
    model_map_name: str = 'psalm_video'
    version: str = 'llava_phi'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="region")
    region_mask_type: Optional[str] = field(default=None)
    with_memory: bool = False

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


def compute_metric(intersection_meter,union_meter,acc_iou_meter, results_list):
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
        #尝试从命令行参数中获取region_mask_type
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        print("region_mask_type:", region_mask_type)
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

#将多个实例的分割掩码融合成一个完整的掩码
def fuse_davis_mask(mask_list,fill_number_list):
    fused_mask = np.zeros_like(mask_list[0])
    for mask, fill_number in zip(mask_list,fill_number_list):
        fill_number = int(fill_number)
        #TODO这里的fill_number感觉像是obj_id
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

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    eval_dataset = DAVIS_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)

    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])

    def load_ref_dataset():
        return DAVIS_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)

    #注册load_ref_dataset函数，方便快速获取数据集
    DatasetCatalog.register('refcoco_dataset', load_ref_dataset)
    MetadataCatalog.get('refcoco_dataset').set(stuff_classes=['object'],)
    gt_json_path = data_args.json_path
    #save_dir /data/..../data_segswap
    save_dir = os.path.dirname(gt_json_path)
    save_dir = os.path.join(save_dir,'predictions')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device=device,dtype=torch.float).eval()
    
    #prev是preservation的缩写，即历史信息
    prev_image = None
    prev_mask_list = None
    prev_fill_number_list = None
    prev_video = None
    prev_transformer = None
    # area_total = 0


    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            #seg_info等信息是经过batch处理后才有的，这里是通过每个batch中第一帧图片的路径来获取视频的名字
            video_name = inputs['seg_info'][0]['file_name'].split('/')[-2]
            #TODO这里利用takes的名字就可以读取annotation文件，然后就可获取所有物体，创建obj->id字典
            #如果是一个take一个take的处理，这种是可行的

            if data_args.with_memory:
                #reset memory list
                if prev_video is None or prev_video != video_name:
                    print(f'old video: {prev_video} -> current video: {video_name}')
                    """
                    prev_mask_list用于存储之前视频帧的分割mask，prev_fill_number_list用于存储分割mask对应的编号id
                    每当处理新的一帧时，代码会根据视频的名称判断是否是新视频的帧，如果是新视频，则会清空这两个列表
                    在处理当前帧时，代码会参考这个列表中的历史分割 mask，将这些历史信息作为视觉提示输入到模型中（inputs['vp_images'] 和 vp_region_masks）
                    这样模型在处理新的一帧时，可以利用前几帧的信息，帮助改进分割效果。
                    """
                    prev_mask_list = []
                    prev_fill_number_list = []
                    prev_video = video_name

                # update memory list
                #判断当前帧中的对象数量和之前帧中的对象数量是否一致
                # area_total = 0
                # for mask in prev_mask_list:
                #     area_total += mask.sum().astype(float)
                if len(prev_mask_list) != 0 and len(inputs['seg_info'][0]['instances'].vp_fill_number) == len(
                        prev_fill_number_list):
                    #把上一帧的图像及推理得到的mask进行存放
                    inputs['vp_images'] = prev_image
                    vp_region_masks = []

                    for mask_ in prev_mask_list:
                        scale_mask = prev_transformer.apply_segmentation(mask_)
                        vp_region_masks.append(scale_mask)

                    vp_region_masks = BitMasks(
                        torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in vp_region_masks])
                    )
                    inputs['seg_info'][0]['instances'].vp_region_masks = vp_region_masks
                    inputs['seg_info'][0]['instances'].vp_fill_number = torch.tensor(prev_fill_number_list,
                                                                                     dtype=torch.int64)

                if len(prev_mask_list) != 0 and len(inputs['seg_info'][0]['instances'].vp_fill_number) != len(
                        prev_fill_number_list):
                    print('some object missing, using original visual prompts')

            #这里可以很好的看出模型推理的时候需要哪些信息，及一帧图片对应的数据字典中需要有哪些键
            outputs = model.eval_video(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                vp_images=inputs['vp_images'].float(),
                seg_info=inputs['seg_info'],
                labels=inputs['labels']
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()


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


            #memory机制，保存前几帧的分割 mask 和位置信息，当视频切换时，会重置记忆
            # update memory list

            #完善memory机制，有可能出现预测出的所有物体mask都是0的情况，这时候也不能更新memory
            # area_total = 0
            # for mask in pred_mask_list:
            #     area_total += mask.sum().astype(float)

            if data_args.with_memory:
                memory_correct_flag = True
                for i in range(len(pred_mask_list)):
                    for j in range(len(pred_mask_list)):
                        if i != j:
                            intersection = np.logical_and(pred_mask_list[i], pred_mask_list[j])
                            union = np.logical_or(pred_mask_list[i], pred_mask_list[j])
                            iou = np.sum(intersection) / np.sum(union)
                            #新增加判断条件
                            if iou > 0.4:
                                # memory is wrong, using origin visual prompt
                                memory_correct_flag = False
                if memory_correct_flag:
                    prev_mask_list = pred_mask_list
                    prev_fill_number_list = fill_number_list
                    #把当前帧作为上一帧，便于处理后续帧
                    prev_image = inputs['images'].float()
                    prev_transformer = inputs['seg_info'][0]['transforms']
                else:
                    print('memory is wrong, using origin visual prompt')


            #保存分割mask以及可视化的彩色图像
            save_name = inputs['seg_info'][0]['file_name']
            print("save_name:", save_name)
            save_name = "exo_query_test/" + save_name.split('/data_segswap/')[1]
            # save_name = '480p/' + save_name.split('/480p/')[1]
            save_path = os.path.join(save_dir,save_name).split('.')[0] + '.png'
            Path(os.path.dirname(save_path)).mkdir(exist_ok=True,parents=True)
            save_color_path = os.path.join(save_dir,save_name).split('.')[0] + '_color.jpg'
            save_color_path = save_color_path.replace('predictions','visualization')
            Path(os.path.dirname(save_color_path)).mkdir(exist_ok=True, parents=True)

            #创建彩色图像，并给彩色图像上色
            # color_image = np.zeros((fused_pred_mask.shape[0],fused_pred_mask.shape[1],3), dtype=np.uint8)
            #
            # for fill_number in fill_number_list:
            #     fill_number = int(fill_number)
            #     color_image[fused_pred_mask == fill_number] = COLOR_MAP[fill_number]['color']

            cv2.imwrite(save_path,fused_pred_mask)
            # cv2.imwrite(save_color_path,color_image)

    print(f'==>finish eval DAVIS, save in {save_dir}')

if __name__ == '__main__':
    evaluation()
