#!/bin/bash

feature="chroma_24"
url="https://www.dropbox.com/s/at947aofuuep68j/AugData_chroma_features.tar.gz?dl=1"

wget -O ./AugData_${feature}_features.tar.gz ${url}

tar xzvf AugData_${feature}_features.tar.gz

find AugData_chroma_features/ -name "*_"${feature}".npy" | sed -n 's|^AugData_'${feature}'_features/||p' | parallel 'cp AugData_'${feature}'_features/{} ./AugData/{}'
find AugData_chroma_features/ -name "*_"${feature}".npy" -print0 | sed -n 's|^AugData_'${feature}'_features/||p' | xargs -0 -I {} cp AugData_${feature}_features/{} ./AugData/{}
