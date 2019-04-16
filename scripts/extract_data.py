from tqdm import tqdm
import os
import librosa
import numpy as np
import pandas as pd
import h5py

AUDIO_SOURCE = '../raw/audio'
CSV_SOURCE = '../raw/metadata/UrbanSound8K.csv'
AUGUMENT_SOURCE = '../raw/augmentation'
CSV_AUGMENT = '../raw/metadata/augmentation.csv'
DIST = '../data'
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

expect_shape = (128, N_MELS)

folders = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5',
           'fold6', 'fold7', 'fold8', 'fold9', 'fold10']

upper = 8.100903208161608
lower = -27.631021115928547

INITIALIZED = True

def get_original_featres(path):
    x, sample_rate = librosa.load(path, duration = 2.97)
    ps = librosa.feature.melspectrogram(
        y = x, # audio time-series
        sr = sample_rate, # sample rate normaly 22050
        n_fft = N_FFT, # length of the FFT window
        hop_length = HOP_LENGTH, # number of samples between successive frames
        n_mels = N_MELS, # number of Mel bands to generate
    )
    return ps

def do_norm(data):
    ret = (data - lower) / (upper - lower)
    return ret

def do_padding(data):
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

def extract(source, csv):
    index = pd.read_csv(csv)

    labels = {}
    data = {}

    global upper
    global lower

    print('Extracting audio...')
    for row in tqdm(index.itertuples()):
        original = get_original_featres(
            os.path.join(source, os.path.join(folders[row.fold - 1], row.slice_file_name))
        )

        log_scale = np.log(original + 1.0e-12)
        curr_data = log_scale

        if not INITIALIZED:
            upper = max(np.max(log_scale), upper)
            lower = min(np.min(log_scale), lower)
        else:
            normed = do_norm(log_scale)
            curr_data = do_padding(normed)

        if row.fold not in labels.keys():
            labels[row.fold] = []
            data[row.fold] = []

        labels[row.fold].append(row.classID)
        data[row.fold].append(curr_data)

    return data, labels

def norm_padding(data):
    new_data = {}
    for key in data.keys():
        new_data[key] = []
        for i in range(len(data[key])):
            curr_data = data[key][i]
            normed = do_norm(curr_data)
            reshaped = do_padding(normed)
            new_data[key].append(reshaped)
    return new_data

def assign(data, labels, h5_out, complement):
    for i in labels.keys():
        label = np.array(labels[i])
        curr_data = np.array(data[i])
        h5_out.create_dataset(folders[i - 1] + complement + '_label', data = label)
        h5_out.create_dataset(folders[i - 1] + complement + '_data', data = curr_data)

def main():
    dist = os.path.join(DIST, 'extracted_data_norm_correction.hdf5')
    h5_out = h5py.File(dist, 'w')

    data, labels = extract(AUDIO_SOURCE, CSV_SOURCE)
    aug_data, aug_labels = extract(AUGUMENT_SOURCE, CSV_AUGMENT)

    print('Global upper bound:', upper)
    print('Global lower bound:', lower)

    if not INITIALIZED:
        data = norm_padding(data)
        aug_data = norm_padding(data)

    assign(data, labels, h5_out, '')
    assign(aug_data, aug_labels, h5_out, '_aug')

    h5_out.close()

if __name__ == '__main__':
    main()