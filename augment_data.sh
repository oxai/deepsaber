#!/bin/bash

#variables for parallelizing. Divide files into 10 chunks of size 100 for parallel processing with 10 cores
L=100
N=10

for i in 30 60 90; do
    mkdir AugData/Dataup${i}; rsync -av --exclude='*.npy' DataE/* AugData/Dataup${i};
    find AugData/Dataup${i}/ -name *.ogg | split -l $L - AugData/filesup${i}
done
for i in 30 60 90; do
    mkdir AugData/Datadown${i}; rsync -av --exclude='*.npy' DataE/* AugData/Datadown${i};
    find AugData/Datadown${i}/ -name *.ogg | split -l $L - AugData/filesdown${i}
done

for i in 30 60 90; do
    for f in $(ls AugData/filesup*); do
        ./pitch_shift_data_up $f $i &
    done
    sleep 1
done

for i in 30 60 90; do
    for f in $(ls AugData/filesdown*); do
        ./pitch_shift_data_down $f $i &
    done
    sleep 1
done

