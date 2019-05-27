#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=/usr/bin/python3
dataset=stage_two
model=transformer
exp=block_selection_new

$py train.py --data_dir=../AugData --dataset_name=$dataset --model=$model --batch_size=2 --num_windows=0 --nepoch=500 --nepoch_decay=500 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=5000 \
    --val_epoch_freq=0 \
    --pad_batches \
    --feature_name=mel \
    --feature_size=100 \
    --tgt_vocab_size=2003 \
    --label_smoothing \
    --max_token_seq_len=512 \
    --gpu_ids=1 \
    --level_diff=Expert \
    --workers=8 \
    --reduced_state \
    --src_vector_input \
    --d_src=$((100+2)) \
    --continue_train \
    --load_iter=60000 \
    # --tgt_vector_input \
    # --d_tgt=$((2003+2)) \
    # --d_src=$((100+0)) \
