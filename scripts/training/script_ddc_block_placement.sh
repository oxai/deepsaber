#!/bin/bash

#export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

py=/usr/bin/python3
#py=/media/usr/bin/python3
dataset=general_beat_saber
model=ddc
layers=7
blocks=3
exp=block_placement_ddc
num_windows=10

$py train.py --data_dir=../../data/extracted_data --dataset_name=$dataset --model=$model --batch_size=1 \
    --num_windows=$num_windows --nepoch=500 --nepoch_decay=500 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=10000 --time_offset=0\
    --time_shifts=15\
    --output_length=100 \
    --val_epoch_freq=0 \
    --feature_name=multi_mel \
    --feature_size=80 \
    --num_classes=$((1+4)) \
    --workers=32 \
    --level_diff=Expert \
    --reduced_state \
    --binarized \
     --gpu_ids=0,1,2,3,4,5,6,7 \
    # --continue_train \
    # --load_iter=930000 \
    #--gpu_ids=0 \
    # --input_channels=$((80+4)) \
