# Prepare Datasets for ObjectRelator

#### Ego-Exo4D

------

Follow [SegSwap](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/SegSwap) to download Ego-Exo4D videos and pre-process the data into images. After processing, you will obtain image folders structured as follows:

```
data_root
└── take_id_01/
    ├── ego_cam/
    		├── 0.jpg
    		├── ...
    		└── n.jpg
    ├── exo_cam/
        ├── 0.jpg
        ├── ...
        └── n.jpg
    └── annotation.json
├── ...
├── take_id_n
└── split.json
```

Next, we use the images and annotations to generate a JSON file for training and evaluating ObjectRelator (w/o text prompt):

```
python datasets/build_egoexo.py --root_path /path/to/ego-exo4d/data_root --save_path /path/to/save/ego2exo_train_visual.json --split_path /path/to/ego-exo4d/data_root/split.json --split train --task ego2exo
```

This gives us a JSON file without text prompts. We then use LLaVA to generate textual descriptions for the objects in the images:

```
cd LLaVA
conda activate llava
python gen_text.py --image_path /path/to/ego-exo4d/data_root --json_path /path/to/save/ego2exo_train.json --save_path /path/to/save/ego2exo_train_visual_text_tmp.json
```

In the final step, we process the LLaVA-generated text to extract object names and convert them into tokenized form, producing a complete JSON file that includes both visual and textual prompts.

```
python datasets/build_text.py --text_path /path/to/save/ego2exo_train_visual_text_tmp.json --save_path /path/to/save/ego2exo_train_visual_text.json
```

For convenience, we release our pre-generated JSON file here. (TODO: google Drive)

#### HANDAL

------

Download all ZIP files in [HANDAL](https://drive.google.com/drive/folders/10mDNZnYrg55ZiP9GV4upKWnxlxay1OwM). You can use `gdown` in the command line as follows:

```
gdown "https://drive.google.com/file/d/1bYP3qevtmjiG3clRiP93mwVBTxyiDQFq/view?usp=share_link" --fuzzy
```

Once unzipped, the dataset will be organized into image folders as shown below:

```
data_root
└── handal_dataset_{obj_name}/
    ├── dynamic/
    ├── models/
    ├── models_parts/
    ├── test/
    └── train/
├── ...
└── handal_dataset_{obj_name}
```

Next, we use the images and masks to generate a JSON file for training and evaluating ObjectRelator (w/o text prompt):

```
python datasets/build_handal.py --root_path /path/to/handal/data_root --save_path /path/to/save/handal_train_visual.json --split train
```

The following text prompt generation steps are the same as those for Ego-Exo4D. Refer to the instructions above.

For convenience, we release our pre-generated JSON file here. (TODO: google Drive)
