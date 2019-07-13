#!/bin/bash

feature="chroma_24"
folder_name="chroma_24_"$(date +%F)
#url="https://www.dropbox.com/s/at947aofuuep68j/AugData_chroma_features.tar.gz?dl=1"
url=$1

wget -O ./AugData_${folder_name}_features.tar.gz ${url}

tar xzvf AugData_${folder_name}_features.tar.gz

#find AugData_${folder_name}_features/ -name "*_"${feature}".npy" -printf '%P\0' | parallel -0 'cp AugData_'${folder_name}'_features/{} ./AugData/{}'
find AugData_${folder_name}_features/ -name "*_"${feature}".npy" -printf '%P\0' | xargs -0 -i cp AugData_${folder_name}_features/{} ./AugData/{}


## to download from google cloud
# rsync -avm --include='*_mel_100.npy' --include='*/' --exclude='*' instance-1.us-central1-a.skillful-eon-241416:~/code/beatsaber/AugData ./AugData/
