#!/bin/bash

#mkdir AugData

find ../../data/extracted_data -name *.egg > ../../data/files

for i in 30 60 90; do cat ../../data/files  | parallel "echo shifting up by $(($i*10)) {}; sox {} {}-up${i}.ogg pitch +$(($i*10))"; done
for i in 30 60 90; do cat ../../data/files  | parallel "echo shifting down by $(($i*10)) {}; sox {} {}-down${i}.ogg pitch -$(($i*10))"; done

for i in 30 60 90; do cat ../../data/files  | parallel "mv {}-up${i}.ogg {}.egg"; done
for i in 30 60 90; do cat ../../data/files  | parallel "mv {}-down${i}.ogg {}.egg"; done
