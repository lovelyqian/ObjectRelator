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
import re
from psalm import conversation as conversation_lib
from psalm.train.train_datasets_eval import COCO_interactive_dataset


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

from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, DEFAULT_CLS_TOKEN, CLS_TOKEN_INDEX, DEFAULT_REGION_TOKEN, \
    REGION_TOKEN_INDEX, REFER_TOKEN_INDEX

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
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        
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
        
        # sources = [
        #     [{'from': 'human', 'value': prefix_inst + sources_value},
        #      {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
      
        #debug: refseg的对话形式，看看怎么把两种任务的形式结合在一起
        #[SEG]指的是代表整个句子的token，<seg>指的是代表mask token
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        
        
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


