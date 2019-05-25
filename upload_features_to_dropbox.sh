#!/bin/bash

day=today
#day=yesterday

#feature="chroma_24"
feature="mel_100"
#folder_name="chroma_24_"$(date --date="${day}" +%F)
folder_name="mel_100"

##find AugData -type d | sed -n 's|^AugData/||p' | parallel 'mkdir -p AugData_'${feature}'_features/{}'

#find AugData/ -type d -printf '%P\0' | parallel -0 'mkdir -p AugData_'${folder_name}'_features/{}'

##find AugData/ -name "*_"${feature}".npy" | sed -n 's|^AugData/||p' | parallel 'cp AugData/{} ./AugData_'${feature}'_features/{}'
##find AugData/ -type f -name "*_"${feature}".npy" -newermt $(date --date="${day}" +%F) -printf '%P\0' | parallel -0 'cp AugData/{} ./AugData_'${folder_name}'_features/{}'

#find AugData/ -type f -name "*_"${feature}".npy" -printf '%P\0' | parallel -0 'cp AugData/{} ./AugData_'${folder_name}'_features/{}'

#tar --use-compress-program="pigz --best --recursive | pv" -cf AugData_${folder_name}_features.tar.gz AugData_${folder_name}_features/
tar --use-compress-program="pigz --best --recursive | pv" -cf /media/guillefix/Maelstrom/AugData_${folder_name}_features.tar.gz AugData_${folder_name}_features/

#../dropbox_uploader.sh upload AugData_${folder_name}_features.tar.gz /
../dropbox_uploader.sh upload /media/guillefix/Maelstrom/AugData_${folder_name}_features.tar.gz /
