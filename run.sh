#!/bin/bash

nvidia-smi
nvcc -V

ori_path=$PWD
echo ori_path
which python
which pip

base_dir=/Users/chenliu/Desktop/projects/LV55_INC_OUT_CLS
cd $base_dir


python ${base_dir}/main.py \
--save_dir ${base_dir}/checkpoints \
--bert_dir ${base_dir}/model_hub/chinese-bert-wwm \
--data_dir ${base_dir}/data \
--log_dir ${base_dir}/logs \
--max_seq_len 18 \
--train \
--dev \
--batch_size 4


# 捕获返回
if [[ $? -ne 0 ]]; then
    exit 1
fi

#tar -zcvf LV55_INC_OUT_CLS.tar.gz --exclude=LV55_INC_OUT_CLS/.ipynb_checkpoints --exclude=LV55_INC_OUT_CLS/.git --exclude=LV55_INC_OUT_CLS/checkpoints --exclude=LV55_INC_OUT_CLS/model_hub --exclude=LV55_INC_OUT_CLS/venv LV55_INC_OUT_CLS
