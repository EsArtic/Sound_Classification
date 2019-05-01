#!/bin/bash

cd ./raw
rm -rf augmentation

mkdir augmentation
cd ./augmentation
for i in $(seq 1 10)
do
    mkdir fold$i
done

cd ../../scripts
python augmentation.py