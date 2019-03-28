#!/bin/bash

while read f; do
 echo shifting $f
 sox "${f}" "${f}"-down${2}.ogg pitch -${2}
done <$1
