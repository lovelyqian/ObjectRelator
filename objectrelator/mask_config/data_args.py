from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    only_two_class: bool = False
    old_two_class: bool = False
    is_multimodal: bool = False
    # image path
    image_folder: Optional[str] = field(default='/home/emzhang/data/segmentation/refer_seg/images/mscoco/images/train2014')
    mask_config: Optional[str] = field(default="./objectrelator/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    region_mask_type: Optional[str] = field(default=None)
    # json path from building script
    json_path: str = '/home/emzhang/code/LLaVA/datasets/refcoco/refcoco_val.json'
    # json file to split takes
    split_path: str = ''
    split: str = 'val'
    model_path: str = '/home/emzhang/code/llava_zem/checkpoints/SEG_class_refcoco_after_fixbug'
    model_map_name: str = 'ObjectRelator'  
    SEG_norm: bool = field(default=False)
    SEG_proj: bool = field(default=True)
    criterion_type: Optional[str] = field(default="concat_seg")
    matcher_type: Optional[str] = field(default="wo_class")
    llm_pos: Optional[str] = field(default="none")
    ln_2048: bool = field(default=False)
    version_val: str = 'opt-iml-1.3b'
    seg_idx_back: bool = field(default=False)
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers_val: int = 4
    thr: float = 0.5
    topk: int=1
    fuse_score: bool = field(default=False)
    seg_task: Optional[str] = field(default="region")
    seg_last: bool = field(default=True)
    num_chunks: int=1
    chunk_idx: int=0
    # multi-condition/single-condition
    condition: str = 'multi-condition'

    # for training
    refcoco_image_folder: Optional[str] = "/path/to/refer_seg/images/mscoco/images/train2014"
    image_first: bool = field(default=True)
    instruction_version: str = 'v1'
    instance_json_path: str = '/path/to/instruction_segmentation_train.json'
    lvis_json_path: str = '/path/to/lvis_instance_train.json'
    lvis_categories_path: str = '/path/to/lvis_instance_categories.json'
    # json path from building script
    region_json_path: str = '/path/to/visual_prompt_segmentation_train.json'
    panoptic_json_path: str = "/path/to/coco"
    ref_coco_path: str = '/path/to/refcoco/refcoco_train.json'
    ref_coco_plus_path: str = '/path/to/refcoco+/refcoco+_train.json'
    ref_coco_g_path: str = '/path/to/refcocog/refcocog_train.json'
    mmconv_path: str = '/path/to/llava_1_5'
    data_ratio: str = '1||1||1||1'
    fix_dataset_len: int = 0
    # json paths for joint training
    joint_json_ego2exo: str = '/path/to/joint_ego_exo.json'
    joint_json_exo2ego: str = '/path/to/joint_exo_ego.json'
  

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    train_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    with_norm: bool = field(default=True)
    with_layernorm: bool = field(default=False)
    skip_init_vision: bool = field(default=False)
    with_sam: bool = field(default=False)
    with_swin: bool = field(default=False)
    with_teacher: bool = field(default=False)
    swin_type: Optional[str] = field(default="base")
    projector_outdim: Optional[int] = field(default=2048)
    mm_projector_type: Optional[str] = field(default="swin_conv")
    model_version: Optional[str] = field(default="v1")
    load_mask2former: bool = field(default=True)
    dino_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False  
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    dataloader_drop_last: bool = True

    # set to True if you want to use handal dataset
    is_handal: bool = False
    # set to True if you want to use joint training 
    joint_training: bool = False
    # set to True if you want to use the first stage of training
    first_stage: bool = False
    # pretrained model path
    pretrained_model_path: str = "/path/to/pretrained_model"
    output_dir: str = "/path/to/output_dir"