#!/bin/bash

#mkdir AugData

#find AugData/DataE -name *.ogg > AugData/files

for i in 30 60 90; do cat AugData/files  | parallel "echo shifting up by $(($i*10)) {}; sox {} {}-up${i}.ogg pitch +$(($i*10))"; done
for i in 30 60 90; do cat AugData/files  | parallel "echo shifting down by $(($i*10)) {}; sox {} {}-down${i}.ogg pitch -$(($i*10))"; done
