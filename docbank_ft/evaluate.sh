#!/bin/bash

model=allenai/ivila-block-layoutlm-finetuned-docbank
wandb_name=predict/layoutlm_ft_new_metrics

cd /root/bqqi/changli/layoutlmft/docbank_ft

python3 run_docbank.py --model_name_or_path $model \
                --output_dir /root/bqqi/changli/layoutlmft/output/$wandb_name \
                --do_predict \
                --per_device_eval_batch_size 4 \
                --dataloader_num_workers 4 \
                --report_to wandb \
                --run_name $wandb_name \
                --fp16