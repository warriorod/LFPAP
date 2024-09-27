#!/bin/bash

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path ../model/qwen2-7b-it \
    --dataset ft_dataset_100 \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --output_dir ../model/saves/qwen7b/lora/ft_model_100 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --warmup_steps 20 \
    --save_steps 10 \
    --eval_steps 5 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --max_samples 544 \
    --val_size 0.1 \
    --plot_loss \
    --fp16

# llamafactory-cli train \
#     --stage sft \
#     --do_train \
#     --model_name_or_path ../model/qwen2-7b-it \
#     --dataset ft_dataset_75 \
#     --dataset_dir ./data \
#     --template qwen \
#     --finetuning_type lora \
#     --output_dir ../model/saves/qwen7b/lora/ft_model_75 \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --warmup_steps 20 \
#     --save_steps 10 \
#     --eval_steps 5 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --max_samples 408 \
#     --val_size 0.1 \
#     --plot_loss \
#     --fp16

# llamafactory-cli train \
#     --stage sft \
#     --do_train \
#     --model_name_or_path ../model/qwen2-7b-it \
#     --dataset ft_dataset_50 \
#     --dataset_dir ./data \
#     --template qwen \
#     --finetuning_type lora \
#     --output_dir ../model/saves/qwen7b/lora/ft_model_50 \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --warmup_steps 20 \
#     --save_steps 10 \
#     --eval_steps 5 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --max_samples 272 \
#     --val_size 0.1 \
#     --plot_loss \
#     --fp16

# llamafactory-cli train \
#     --stage sft \
#     --do_train \
#     --model_name_or_path ../model/qwen2-7b-it \
#     --dataset ft_dataset_25 \
#     --dataset_dir ./data \
#     --template qwen \
#     --finetuning_type lora \
#     --output_dir ../model/saves/qwen7b/lora/ft_model_25 \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --warmup_steps 20 \
#     --save_steps 10 \
#     --eval_steps 5 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --max_samples 136 \
#     --val_size 0.1 \
#     --plot_loss \
#     --fp16

