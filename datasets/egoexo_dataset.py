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
import cv2
from torch.utils.data import Dataset, DataLoader
from objectrelator import conversation as conversation_lib
from objectrelator.train.train_datasets import COCO_interactive_dataset_eval, COCO_interactive_dataset_train, COCO_interactive_dataset 
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
from pathlib import Path
from objectrelator.eval.segmentation_evaluation import openseg_classes
COLOR_MAP = openseg_classes.ADE20K_150_CATEGORIES
import re
from objectrelator.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX, REGION_TOKEN_INDEX, REFER_TOKEN_INDEX

'''
Create custom dataset classes for EgoExo and Handal datasets
'''


class EgoExo_Dataset_eval(COCO_interactive_dataset_eval):

    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        tokenized = tokenized + [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id

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

    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image']
        image_folder = self.data_args.image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        data_dict['vp_annotations'] = data['first_frame_anns']
        data_dict['vp_image'] = os.path.join(image_folder,data['first_frame_image'])
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']
        for annotation in data_dict['vp_annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')
        sentences = data['instruction']
        num_target = len(data_dict['instances'])
      
        prefix_inst = 'This is an image <image>, Please segment by given regions and instruction'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'
        sources = [[{'from': 'human', 'value': prefix_inst + sources_value + "and this is the instruction: " + '<refer>\n'},
                    {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
      
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'referring_coco'
        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict
    

class EgoExo_Dataset_train(COCO_interactive_dataset_train):

    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        tokenized = tokenized + [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    
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

    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image']
        image_folder = self.data_args.image_folder


        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        data_dict['vp_annotations'] = data['first_frame_anns']
        data_dict['vp_image'] = os.path.join(image_folder,data['first_frame_image'])
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']
        for annotation in data_dict['vp_annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')
        sentences = data['instruction']
        num_target = len(data_dict['instances'])
        prefix_inst = 'This is an image <image>, Please segment by given regions and instruction'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'
        sources = [[{'from': 'human', 'value': prefix_inst + sources_value + "and this is the instruction: " + '<refer>\n'},
                    {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'referring_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict

class Handal_Dataset_eval(COCO_interactive_dataset):
    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        tokenized = tokenized + [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    
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

    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image']
        image_folder = self.data_args.image_folder


        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        data_dict['vp_annotations'] = data['first_frame_anns']
        data_dict['vp_image'] = os.path.join(image_folder,data['first_frame_image'])
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']
        for annotation in data_dict['vp_annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')
        sentences = data['instruction']
        num_target = len(data_dict['instances'])
        prefix_inst = 'This is an image <image>, Please segment by given regions and instruction'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'
        sources = [[{'from': 'human', 'value': prefix_inst + sources_value + "and this is the instruction: " + '<refer>\n'},
                    {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'referring_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict
    
class Handal_Dataset_train(COCO_interactive_dataset_train):
    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        tokenized = tokenized + [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    
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

    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image']
        image_folder = self.data_args.image_folder


        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        data_dict['vp_annotations'] = data['first_frame_anns']
        data_dict['vp_image'] = os.path.join(image_folder,data['first_frame_image'])
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']
        for annotation in data_dict['vp_annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            annotation['bbox'] = [0,0,0,0]
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type,mask_format='bitmask')
        sentences = data['instruction']
        num_target = len(data_dict['instances'])
        prefix_inst = 'This is an image <image>, Please segment by given regions and instruction'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'
        sources = [[{'from': 'human', 'value': prefix_inst + sources_value + "and this is the instruction: " + '<refer>\n'},
                    {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'referring_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict