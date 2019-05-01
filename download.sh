#!/bin/bash

cd ./raw
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar -xzvf UrbanSound8K.tar.gz
mv UrbanSound8K/audio ./
mv UrbanSound8K/metadata ./
rm -rf UrbanSound8K