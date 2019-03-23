from tqdm import tnrange, tqdm
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import h5py

AUDIO_SOURCE = '../raw/audio/'
CSV_SOURCE = '../raw/metadata/UrbanSound8K.csv'
DIST = '../data/extracted_data.hdf5'

folders = ['fold1/', 'fold2/', 'fold3/', 'fold4/', 'fold5/',
           'fold6/', 'fold7/', 'fold8/', 'fold9/', 'fold10/']

def extract(source, csv, distination):
    h5_out = h5py.File(distination, 'w')
    index = pd.read_csv(csv)

    labels = {}
    data = {}
    print('Extracting audio...')
    for row in tqdm(index.itertuples()):
        x, sample_rate = librosa.load(source + folders[row.fold - 1] + row.slice_file_name, duration = 2.97)
        ps = librosa.feature.melspectrogram(y = x, sr = sample_rate)

        padding = 0
        if ps.shape != (128, 128):
            padding = 1
            diff = 128 - ps.shape[1]
            pattern = [0.0] * 128
            paddings = []
            for i in range(diff):
                paddings.append(pattern)
            paddings = np.array(paddings)
            ps = np.concatenate((ps, paddings.T), axis = 1)

        if row.fold not in labels.keys():
            labels[row.fold] = []
            data[row.fold] = []

        labels[row.fold].append(row.classID)
        data[row.fold].append(ps)

    for i in labels.keys():
        label = np.array(labels[i])
        curr_data = np.array(data[i])
        h5_out.create_dataset(folders[i - 1][: -1] + '_label', data = label)
        h5_out.create_dataset(folders[i - 1][: -1] + '_data', data = curr_data)

    '''
    for key in h5_out.keys():
        print(key)
        print(h5_out[key].name)
        print(h5_out[key].shape)
        print(h5_out[key].value)

    print(h5_out['train_data'].shape)
    print(h5_out['train_data'].value)
    '''

def main():
    extract(AUDIO_SOURCE, CSV_SOURCE, DIST)

if __name__ == '__main__':
    main()