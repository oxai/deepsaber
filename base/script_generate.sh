#!/bin/bash

song_path=$1
#exp1=block_placement_dropout
exp1=block_placement_new_nohumreg
cpt1=1220000
#cpt1=755000
exp2=block_selection_new
cpt2=1600000
#ddc_file=/home/guillefix/ddc_infer/57257860-f345-4e5c-ba69-36f57b561118/57257860-f345-4e5c-ba69-36f57b561118.sm
ddc_file=$2

python3 generate.py --song_path $song_path --experiment_name $exp1 --checkpoint $cpt1 --experiment_name2 $exp2 --checkpoint2 $cpt2 --two_stage \
    --peak_threshold 0.007 \
    --temperature 1.00 \
    --open_in_browser \
    --ddc_file $ddc_file \
    --ddc_diff 3 \
    --bpm 128 \
    --use_beam_search \
    #--use_ddc \
    #--generate_full_song \
    #--peak_threshold 0.008 \
