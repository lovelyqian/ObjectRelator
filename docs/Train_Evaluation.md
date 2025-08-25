# Train 

------

#### Training on Ego-Exo4D

```
'''
1. change model paths and dataset paths to the exact ego-exo related paths in train_ObjectRelator.sh
2. You can adjust the training behavior by modifying the configuration parameters in objectrelator/mask_config/data_args.py
3. data_args.condition controls the number of prompt modalities used
4. training_args.joint_training determines whether joint training is enabled
5. training_args.first_stage determines whether to use the first stage of training
'''

# stage-1 training: set training_args.first_stage=True
bash scripts/train_ObjectRelator.sh 

# stage-2 training: set training_args.first_stage=False, training_args.pretrained_model_path=/path/to/stage-1
bash scripts/train_ObjectRelator.sh 
```

#### Training on HANDAL

```
# change model paths and dataset paths to the exact handal related paths in train_ObjectRelator.sh
# set training_args.is_handal=True in data_args.py
# The remaining training procedure is identical to that of Ego-Exo.

bash scripts/train_ObjectRelator.sh 
```

# Evaluation

------

#### Eval on Ego-Exo4D

```
# set data_args.condition in objectrelator/mask_config/data_args.py to control the number of prompt modalities used

python objectrelator/eval/eval_egoexo.py --image_folder /path/to/ego-exo4d/data_root --model_path /path/to/pretrained_model --json_path /path/to/save/ego2exo_val_visual_text.json --split_path /path/to/ego-exo4d/data_root/split.json --split val
```

#### Eval on HANDAL

```
# set data_args.condition in objectrelator/mask_config/data_args.py to control the number of prompt modalities used

python objectrelator/eval/eval_handal.py --image_folder /path/to/handal/data_root --model_path /path/to/pretrained_model --json_path /path/to/save/handal_val_visual_text.json
```

## 
