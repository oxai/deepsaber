#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=/usr/bin/python3
dataset=stage_two
model=transformer
exp=block_selection_new

$py train.py --data_dir=../../data/extracted_data --dataset_name=$dataset --model=$model --batch_size=2 --num_windows=0 --nepoch=500 --nepoch_decay=500 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=50000 \
    --val_epoch_freq=0 \
    --pad_batches \
    --feature_name=mel \
    --feature_size=100 \
    --tgt_vocab_size=2003 \
    --label_smoothing \
    --max_token_seq_len=512 \
    --gpu_ids=0,1,2,3,4,5,6,7 \
    --level_diff=Expert,ExpertPlus \
    --workers=16 \
    --reduced_state \
    --src_vector_input \
    --d_src=$((100+2)) \
    --continue_train \
    --load_iter=1600000 \
    #--max_token_seq_len=2048 \
    # --tgt_vector_input \
    # --d_tgt=$((2003+2)) \
    # --d_src=$((100+0)) \
