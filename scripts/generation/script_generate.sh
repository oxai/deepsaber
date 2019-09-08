#!/bin/bash

song_path=$1
#exp1=block_placement_dropout
exp1=block_placement_new_nohumreg
cpt1=1220000
#cpt1=755000
exp2=block_selection_new
cpt2=1600000
#ddc_file=/home/guillefix/ddc_infer/57257860-f345-4e5c-ba69-36f57b561118/57257860-f345-4e5c-ba69-36f57b561118.sm

py=python3

# $py generate_end2end.py --song_path $song_path --experiment_name $exp1 --checkpoint $cpt1 \
#     --temperature 1.00 \
#     --open_in_browser \
#     --bpm 128 \

# ddc_file=$2
# $py generate_stage1_ddc.py --song_path $song_path --bpm 128 \
#   --ddc_file $ddc_file \
#   --temperature 1.00 \
#   --ddc_diff 3 \

$py generate_stage1.py --song_path $song_path --experiment_name $exp1 --checkpoint $cpt1 --bpm 128 \
  --peak_threshold 0.007 \
  --temperature 1.00 | tail -1 | ( read json_file;
  $py generate_stage2.py --song_path $song_path --json_file $json_file --experiment_name $exp2 --checkpoint $cpt2 --bpm 128 --temperature 1.00 \
    --use_beam_search \
    #--generate_full_song \
  )
