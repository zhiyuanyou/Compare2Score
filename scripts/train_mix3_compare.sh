#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=:./$PYTHONPATH
LOAD='/opt/data/private/142/ModelZoo/VLM/mplug-owl/mplug-owl2-llama2-7b'

DATA_FILE=/opt/data/private/142/DataDepictQA/Compare2Score/train_koniq_spaq_kadid_compare_96k.json
deepspeed --master_port 25801 q_align/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder /opt/data/private/142/DataDepictQA \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./compare2score_mix3_4times \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
