# Model Zoo

------

We release our ObjRelator model below,  you can use it for quick inference on cross-view tasks:

| Version       | Checkpoint |
| ------------- | ---------- |
| Ego2Exo-Small | [link](https://huggingface.co/wangzeze/ObjectRelator-Ego2Exo-Small)       |
| Ego2Exo-Full  | [link](https://huggingface.co/wangzeze/ObjectRelator-Ego2Exo-Full)        |
| Exo2Ego-Small | [link](https://huggingface.co/wangzeze/ObjectRelator-Exo2Ego-Small)       |
| Exo2Ego-Full  | [link](https://huggingface.co/wangzeze/ObjectRelator-Exo2Ego-Full)        |
| Joint         | [link](https://huggingface.co/wangzeze/ObjectRelator-Joint)               |

If you want to train from scratch, please download the checkpoint files related to PSALM:

​	Download Siwn-B Mask2former from [here](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl).

​	Download Phi-1.5 based on huggingface from [here](https://huggingface.co/susnato/phi-1_5_dev).

​	LLava pretrained projector can be downloaded [here](https://huggingface.co/EnmingZhang/PSALM_stage1).

​	Download pre-trained PSALM [here](https://huggingface.co/EnmingZhang/PSALM).

# Quick Start

------

Run the following command to experience quick inference with ObjRelator:

```
# set data_args.condition in objectrelator/mask_config/data_args.py to control the number of prompt modalities used

python objectrelator/eval/eval_egoexo.py --image_folder /path/to/ego-exo4d/data_root --model_path /path/to/pretrained_model --json_path /path/to/save/ego2exo_val_visual_text.json --split_path /path/to/ego-exo4d/data_root/split.json --split val
```





