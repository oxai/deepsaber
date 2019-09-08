cloud.google.com
log in to labs account
menu on top right > compute engine > VM instances

for fast data processing (lots of CPU no GPU), just create an instance with 96cpu and 360Gb memory. Use latest deepsaber snapshot for boot disk

can ssh from web app, and also copy gcloud command for sshing in terminal (need to have gcloud installed and set up, and you need to run gcloud auth login first time; it tells you did, it's very self-guiding:)).

https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning > launch
configure. Choose us-central1 as we have quotas there.


###########

#Example of feature extraction
_for stage1 ddc_
mpiexec -n 96 /usr/bin/python3 process_songs.py --feature_name multi_mel --feature_size 80 ../../data/extracted_data Expert,ExpertPlus
_for stage2 transformer_
mpiexec -n 96 /usr/bin/python3 process_songs.py --feature_name mel --feature_size 100 ../../data/extracted_data Expert,ExpertPlus

#Getting the standard sorted_states pickle
mega-get 'https://mega.nz/#!BAR1gCKb!RT3X-klMy6lz9WzFs6a_638Uq7PooJIGHFoQVgUOFnY'
