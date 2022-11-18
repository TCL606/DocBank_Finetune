#!/bin/bash

layoutlmft="/root/bqqi/changli/layoutlmft"
output_dir="/root/bqqi/changli/layoutlmft/output/funsd_layoutlmv2"

model=microsoft/layoutlmv2-base-uncased
train_epochs=100
lr=5e-5
bs_per_deivce=16
max_steps=-1
wandb=funsd_layoutlmv2
gpu=1

cd $layoutlmft
python3 -m torch.distributed.launch --nproc_per_node=$gpu examples/run_funsd.py \
        --model_name_or_path $model \
        --output_dir $output_dir \
        --do_train \
        --do_predict \
        --max_steps $max_steps \
        --num_train_epochs $train_epochs \
        --learning_rate $lr \
        --per_device_train_batch_size $bs_per_deivce \
        --warmup_ratio 0.1 \
        --fp16 \
        --report_to wandb \
        --run_name $wandb