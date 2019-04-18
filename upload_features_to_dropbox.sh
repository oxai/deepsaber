#!/bin/bash

feature="chroma_24"

find AugData -type d | sed -n 's|^AugData/||p' | parallel 'mkdir -p AugData_'${feature}'_features/{}'

find AugData/ -name "*_"${feature}".npy" | sed -n 's|^AugData/||p' | parallel 'cp AugData/{} ./AugData_'${feature}'_features/{}'

tar --use-compress-program="pigz --best --recursive | pv" -cf AugData_${feature}_features.tar.gz AugData_${feature}_features/

../dropbox_uploader.sh upload AugData_${feature}_features.tar.gz /
