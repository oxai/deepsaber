param (
  [Parameter(Mandatory=$true)][string]$song_path
)

$type="deepsaber"

# exp1=block_placement_new_nohumreg
# cpt1=1220000
#exp1=test_ddc_block_placment
#cpt1=$2
#exp1=block_placement_ddc
$exp1="block_placement_ddc2"
#cpt1=1070000
$cpt1=130000

$exp2="block_selection_new2"
$cpt2=2150000
#cpt2=1200000
#cpt2=1450000
#exp2=test_block_selection
#cpt2=$3
#ddc_file=/home/guillefix/ddc_infer/57257860-f345-4e5c-ba69-36f57b561118/57257860-f345-4e5c-ba69-36f57b561118.sm

mkdir generated -ea 0

if ( $type -eq "end2end" ) {
  python generate_end2end.py --song_path $song_path --experiment_name $exp1 --checkpoint $cpt1 `
      --temperature 1.00 `
      --open_in_browser `
      --bpm 128
}

if ( "$type" -eq "ddc" ) {
  $ddc_file=$2
  #ddc_file=$3
  python generate_stage1_ddc.py --song_path $song_path --bpm 128 `
    --ddc_file $ddc_file `
    --temperature 1.00 `
    --ddc_diff 3

  $basename=$song_path.split('\.')[-2]
  $json_file="generated\${basename}_${exp1}_${cpt1}_0.33_1.0.dat"
  python generate_stage2.py --song_path $song_path --json_file $json_file --experiment_name $exp2 --checkpoint $cpt2 --bpm 128 --temperature 1.00 `
    --use_beam_search `
    --open_in_browser `
    #--generate_full_song `
}

if ( $type -eq "deepsaber" ) {
  python generate_stage1.py --cuda --song_path $song_path --experiment_name $exp1 --checkpoint $cpt1 --bpm 128 `
    --peak_threshold 0.33 `
    --temperature 1.0
  $basename=$song_path.split('\.')[-2]
  $json_file="generated\${basename}_${exp1}_${cpt1}_0.33_1.0.dat"
  python generate_stage2.py --cuda --song_path $song_path --json_file $json_file --experiment_name $exp2 --checkpoint $cpt2 --bpm 128 `
  --temperature 1.00 `
  --use_beam_search `
  --open_in_browser `
  #--generate_full_song `
    

  # python generate_stage1.py --song_path $song_path --experiment_name $exp1 --checkpoint $cpt1 --bpm 128 \
  # --peak_threshold 0.007 \
  # --temperature 1.00
}
