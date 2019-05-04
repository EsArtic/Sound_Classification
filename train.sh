#!/bin/bash

if [ ! -d "./models" ]; then
    mkdir models
fi

cd ./scripts
python train.py -s ../data/extracted_spectrogram_data.hdf5 -e 50 -a