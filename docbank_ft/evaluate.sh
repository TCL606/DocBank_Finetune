#!/bin/bash

model=allenai/hvila-block-layoutlm-finetuned-docbank

cd /root/bqqi/changli/layoutlmft/docbank_ft

python3 run_docbank.py --model_name_or_path $model \
                --output_dir /root/bqqi/changli/layoutlmft/output/Debug \
                --do_predict \
                --fp16