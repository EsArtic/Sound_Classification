from tqdm import tqdm
import os
import librosa
import numpy as np
import pandas as pd
import h5py

import argparse

AUDIO_SOURCE = '../raw/audio'
CSV_SOURCE = '../raw/metadata/UrbanSound8K.csv'
AUGUMENT_SOURCE = '../raw/augmentation'
CSV_AUGMENT = '../raw/metadata/augmentation.csv'
DIST = '../data'
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

folders = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5',
           'fold6', 'fold7', 'fold8', 'fold9', 'fold10']

def get_features(path, mode):
    if mode == 'time_sequence':
        x, sample_rate = librosa.load(path, sr = 8000)
        return x
    elif mode == 'mel':
        x, sample_rate = librosa.load(path, duration = 2.97)
        feature = librosa.feature.melspectrogram(
            y = x, # audio time-series
            sr = sample_rate, # sample rate normaly 22050
            n_fft = N_FFT, # length of the FFT window
            hop_length = HOP_LENGTH, # number of samples between successive frames
            n_mels = N_MELS, # number of Mel bands to generate
            norm = 1
        )
        return feature
    elif mode == 'mfcc':
        x, sample_rate = librosa.load(path, duration = 2.97)
        feature = librosa.feature.mfcc(y = x, sr = sample_rate)
        return feature

def do_padding(data, expect_shape):
    ret = data
    if data.shape != expect_shape:
        diff = 128 - data.shape[1]
        pattern = [0.0] * expect_shape[0]
        paddings = []
        for i in range(diff):
            paddings.append(pattern)
        paddings = np.array(paddings)
        ret = np.concatenate((data, paddings.T), axis = 1)
    return ret

def time_sequence_padding(data, expect_len):
    ret = data
    if data.shape[0] < expect_len:
        diff = expect_len - data.shape[0]
        paddings = [0.0] * diff
        ret = np.concatenate((data, paddings))
    elif data.shape[0] > expect_len:
        ret = data[: expect_len]
    return ret

def extract(source, csv, mode):
    index = pd.read_csv(csv)

    labels = {}
    data = {}

    print('Extracting audio...')
    for row in tqdm(index.itertuples()):
        original = get_features(
            os.path.join(source, os.path.join(folders[row.fold - 1], row.slice_file_name)),
            mode
        )
        curr_data = original
        if mode == 'mel':
            curr_data = do_padding(curr_data, (128, 128))
        elif mode == 'mfcc':
            curr_data = do_padding(curr_data, (20, 128))
        elif mode == 'time_sequence':
            expect_len = 32000
            curr_data = time_sequence_padding(curr_data, expect_len)
            curr_data = curr_data.reshape((expect_len, 1))

        if row.fold not in labels.keys():
            labels[row.fold] = []
            data[row.fold] = []

        labels[row.fold].append(row.classID)
        data[row.fold].append(curr_data)

    return data, labels

def assign(data, labels, h5_out, complement):
    for i in labels.keys():
        label = np.array(labels[i])
        curr_data = np.array(data[i])
        h5_out.create_dataset(folders[i - 1] + complement + '_label', data = label)
        h5_out.create_dataset(folders[i - 1] + complement + '_data', data = curr_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type = str, required = False, default = 'mel',
                        help = 'Type of the extracted features: {mel, mfcc, time_sequence}.')
    parser.add_argument('-a', '--augment', action = 'store_true',
                        help = 'Use augment data.')
    args = parser.parse_args()

    if not (args.type == 'mel' or args.type == 'mfcc' or args.type == 'time_sequence'):
        print('Incorrect feature type.')
        return

    dist = os.path.join(DIST, 'extracted_spectrogram_data.hdf5')
    if args.type == 'mfcc':
        dist = os.path.join(DIST, 'extracted_mfcc_data.hdf5')
    elif args.type == 'time_sequence':
        dist = os.path.join(DIST, 'extracted_time_sequence_data.hdf5')

    h5_out = h5py.File(dist, 'w')

    data, labels = extract(AUDIO_SOURCE, CSV_SOURCE, args.type)
    assign(data, labels, h5_out, '')
    if args.augment:
        aug_data, aug_labels = extract(AUGUMENT_SOURCE, CSV_AUGMENT, args.type)
        assign(aug_data, aug_labels, h5_out, '_aug')

    h5_out.close()

if __name__ == '__main__':
    main()
