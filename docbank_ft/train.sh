#!/bin/bash

model=microsoft/layoutlm-base-uncased
wandb_name=$model-docbank

epoch=1
bs_per_gpu=16
lr=5e-5
eval_step=1000

cd /root/bqqi/changli/layoutlmft/docbank_ft

python3 run_docbank.py --model_name_or_path $model \
                --output_dir /root/bqqi/changli/layoutlmft/output/Debug \
                --do_train \
                --do_eval \
                --eval_step $eval_step \
                --do_predict \
                --num_train_epochs $epoch \
                --per_device_train_batch_size $bs_per_gpu \
                --learning_rate $lr \
                --max_steps -1 \
                --warmup_ratio 0.1 \
                --fp16 \
                --report_to wandb \
                --run_name $wandb_name
