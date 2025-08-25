# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from objectrelator.train.train_datasets import *
from datasets.egoexo_dataset import EgoExo_Dataset_train, Handal_Dataset_train 
from objectrelator.mask_config.config import Config

from objectrelator.model.language_model.llava_phi import ObjectRelator 
from objectrelator.train.llava_trainer_SSL import LLaVATrainerSSL

from objectrelator.mask_config.data_args import DataArguments, TrainingArguments, ModelArguments
from fvcore.common.config import CfgNode
import warnings


warnings.filterwarnings('ignore')
local_rank = None

def print_trainable_parm(model,prefix):
    for name, module in model.named_modules():
        print_flag = False
        for p in module.parameters():
            if p.requires_grad == True:
                print(f'{prefix}:  {name}')
                print_flag = True
                break

def get_mask_config(config='./objectrelator/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'):
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg

def print_dtype(model,prefix,dtype):
    for name,p in model.named_parameters():
        if p.dtype != dtype:
            print(f'{prefix}: {name}')
            print(p.dtype)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def make_unify_datamodule(tokenizer, data_args, training_args):
    data_ratio = data_args.data_ratio
    data_ratio = data_ratio.split('||')
    data_ratio = [int(data_) for data_ in data_ratio]
    
    if training_args.is_handal:
        egoexo_dataset = Handal_Dataset_train(json_path=data_args.region_json_path, tokenizer=tokenizer,data_args=data_args) 
    else:
        egoexo_dataset = EgoExo_Dataset_train(json_path=data_args.region_json_path, tokenizer=tokenizer,data_args=data_args) 
    datasets = [egoexo_dataset]
   

    # you can change 16 to your frequency sets, it represents how many samples to change tasks
    train_dataset = UnifyDatasetSingleDatasetForBatch(datasets,data_ratio,16,fix_dataset_len=data_args.fix_dataset_len)
    print(f'total unify dataset number is {len(train_dataset)}')
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_unify_datamodule_joint(tokenizer, data_args, training_args):
    data_ratio = data_args.data_ratio
    data_ratio = data_ratio.split('||')
    data_ratio = [int(data_) for data_ in data_ratio]
    
    egoexo_dataset = EgoExo_Dataset_train(json_path=data_args.joint_json_ego2exo, tokenizer=tokenizer,data_args=data_args)
    exoego_dataset = EgoExo_Dataset_train(json_path=data_args.joint_json_exo2ego, tokenizer=tokenizer,data_args=data_args)
    datasets = [egoexo_dataset + exoego_dataset]
   

    # you can change 16 to your frequency sets, it represents how many samples to change tasks
    train_dataset = UnifyDatasetSingleDatasetForBatch(datasets,data_ratio,16,fix_dataset_len=data_args.fix_dataset_len)
    print(f'total unify dataset number is {len(train_dataset)}')
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    mask_cfg = get_mask_config(config=data_args.mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = data_args.seg_task
    bnb_model_from_pretrained_args = {}
    
    model = ObjectRelator.from_pretrained(
        training_args.pretrained_model_path,
        mask_decoder_cfg=mask_cfg,
        add_cross_attn=True,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
                )
    
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32), device=training_args.device)
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if not model_args.train_backbone:
            model.model.vision_tower.requires_grad_(False)


        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False


        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    model.get_special_token(SEG=tokenizer("[SEG]", return_tensors='pt', add_special_tokens=False)['input_ids'], EOS=tokenizer.eos_token_id)
    
    if training_args.joint_training:
        if training_args.is_handal:
            raise ValueError("Joint training is not supported for HANDAL dataset")
        else:
            data_module =  make_unify_datamodule_joint(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    else:    
        data_module = make_unify_datamodule(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    training_args.dataloader_drop_last = True

    # For Stage1 Training 
    if training_args.first_stage:
        for name, param in model.named_parameters():  
            if "fuse_model" in name:
                param.requires_grad = True
                print(name)
            else:
                param.requires_grad = False

    trainer = LLaVATrainerSSL(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
