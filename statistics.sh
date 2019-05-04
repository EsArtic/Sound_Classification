#!/bin/bash

if [ ! -d "./log" ]; then
    mkdir log
fi

cd ./scripts
python statistics.py -s ../data/extracted_spectrogram_data.hdf5 -e 50