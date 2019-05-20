#!/bin/bash

#export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

py=python3
dataset=general_beat_saber
model=wavenet
layers=7
blocks=3
exp=test
num_windows=10

$py train.py --data_dir=../AugDataTest --dataset_name=$dataset --model=$model --batch_size=5 --output_length=128 --num_windows=$num_windows --nepoch=500 --nepoch_decay=500 --layers=$layers --blocks=$blocks \
    --print_freq=1 --experiment_name=$exp --save_by_iter --save_latest_freq=1000 \
    --time_shifts=16 \
    --input_channels=$((24*16)) \
    --num_classes=$((1+3)) \
    --extra_output \
    --entropy_loss_coeff=0.0 \
    --workers=0 \
    --level_diff=Expert \
    --reduced_state \
    --binarized \
    --gpu_ids=0
    # --dilation_channels=512 \
    # --residual_channels=256 \
    # --skip_channels=256 \
    # --end_channels=512 \
    #--concat_outputs \
    # --gpu_ids=0,1,2,3,4,5,6,7 \
    # --load \
    # --load_iter=141000 \
