#!/bin/bash

feature="mel_100"
folder_name="mel_100"
#url="https://www.dropbox.com/s/at947aofuuep68j/AugData_chroma_features.tar.gz?dl=1"

#find AugData_${folder_name}_features/ -name "*_"${feature}".npy" -printf '%P\0' | parallel -0 'cp AugData_'${folder_name}'_features/{} ./AugData/{}'
find AugData_${folder_name}_features/ -name "*_"${feature}".npy" -printf '%P\0' | xargs -0 -i mv AugData_${folder_name}_features/{} ./AugData/{}


