#!/bin/bash

model=microsoft/layoutlmv2-base-uncased
wandb_name=$model-docbank

epoch=1
bs_per_gpu=4
lr=5e-5
eval_step=10000
save_steps=25000

cd /root/bqqi/changli/layoutlmft/docbank_ft

python3 run_docbank.py --model_name_or_path $model \
                --output_dir /root/bqqi/changli/layoutlmft/output/$wandb_name \
                --require_image \
                --do_train \
                --do_eval \
                --eval_step $eval_step \
                --evaluation_strategy steps \
                --do_predict \
                --num_train_epochs $epoch \
                --per_device_train_batch_size $bs_per_gpu \
                --per_device_eval_batch_size $bs_per_gpu \
                --learning_rate $lr \
                --max_steps -1 \
                --warmup_ratio 0.1 \
                --dataloader_num_workers 4 \
                --save_steps $save_steps \
                --fp16 \
                --report_to wandb \
                --run_name $wandb_name
