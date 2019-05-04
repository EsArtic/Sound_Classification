#!/bin/bash

if [ ! -d "./data" ]; then
    mkdir data
fi

cd ./scripts
python extract_data.py -t mel -a