#!/bin/bash

cd ./raw

if [ -d "./augmentation" ]; then
    rm -rf augmentation
fi

mkdir augmentation
cd ./augmentation
for i in $(seq 1 10)
do
    mkdir fold$i
done

cd ../../scripts
python augmentation.py