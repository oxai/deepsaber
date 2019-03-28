#!/bin/bash

#mkdir AugData

for i in 30 60 90; do
    #mkdir AugData/Dataup${i}; rsync -a --exclude='*.npy' DataE/* AugData/Dataup${i};
    #find AugData/Dataup${i}/ -name *.ogg | split -l $L - AugData/filesup${i}
    find AugData/Dataup${i}/ -name *.ogg > AugData/filesup${i}
done
for i in 30 60 90; do
    #mkdir AugData/Datadown${i}; rsync -a --exclude='*.npy' DataE/* AugData/Datadown${i};
    #find AugData/Datadown${i}/ -name *.ogg | split -l $L - AugData/filesdown${i}
    find AugData/Datadown${i}/ -name *.ogg > AugData/filesdown${i}
done

for i in 30 60 90; do cat AugData/filesup${i} | parallel "sox {} {}-down${i}.ogg pitch +${i}"; done
for i in 30 60 90; do cat AugData/filesdown${i} | parallel "sox {} {}-up${i}.ogg pitch -${i}"; done
