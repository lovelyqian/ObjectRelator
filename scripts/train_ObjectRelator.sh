export DISABLE_ADDMM_CUDA_LT=1
deepspeed --master_port=29526 objectrelator/train/train_ObjectRelator.py \
    --pretrained_model_path "/path/to/pretrained_model" \
    --image_folder "/path/to/image/data_root" \
    --region_json_path "/path/to/ego2exo_train_visual_text.json" \
    --joint_json_ego2exo "/path/to/ego2exo_train_visual_text.json" \
    --joint_json_exo2ego "/path/to/exo2ego_train_visual_text.json" \
    --output_dir "/path/to/output_dir" \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "/path/to/huggingface/hub/phi-1_5_dev" \
    --version "llava_phi" \
    --panoptic_json_path "/path/to/coco" \
    --ref_coco_path "/path/to/refcoco/refcoco_train.json" \
    --ref_coco_plus_path "/path/to/refcoco+/refcoco+_train.json" \
    --ref_coco_g_path "/path/to/refcocog/refcocog_train.json" \
    --refcoco_image_folder "/path/to/coco/train2014" \
    --mmconv_path "/path/to/llava_1_5" \
    --vision_tower "/path/to/huggingface/hub/Mask2former/model_final_54b88a.pkl" \
    --pretrain_mm_mlp_adapter "/path/to/huggingface/hub/PSALM_stage1/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 6e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --seg_task 'region' \






