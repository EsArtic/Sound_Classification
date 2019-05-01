# Sound_Classification
Machine Learning Course Project

## Directory Structure
* data     -- The extracted audio features
* log      -- The training log and statisitcs
* models   -- The snapshots of trained models
* notebook -- The jupyter notebooks
* raw      -- The audio files and metadata
* scripts  -- The source codes
* temp     -- The temporary data

## Dependencies
This program is implemented by python 3.6. Before running the program, please make sure that the following packages are installed correctly:
* Data processing: librosa, pandas, h5py
* Model training: numpy, tensorflow, keras
* Display: matplotlib, seaborn, tqdm

## Running
The procedure of running this program:
1) Run download.sh  The UrbanSound8K dataset would be downloaded an decompress.

2) Run augmentation.sh  The augmentation data set would be generated.

3) Run extract_feature.sh
    The Melspectrogram features would be generated. If you want to extract other features, you can use the following command to get the usage of extact_data.py:
    * python extract_data.py --help

4) Run train.sh
    Perform training using the default model. Similarly, you can use the following command to get the usage of train.py:
    * python train.py --help  To generate the training log, you can run train.py scripts using pipeline and redirection like:
    * python train.py 2>&1 | tee train.log


5) Run statistics.sh
    Two statistics files would be generated (./log/precision.csv and ./log/recall.csv). Similarly, you can use the following command to get the usage of statistics.py:
    * python statistics.py --help