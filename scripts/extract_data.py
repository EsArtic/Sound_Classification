from tqdm import tnrange, tqdm
import librosa
import librosa.display
import numpy as np
import pandas as pd
import h5py

RAW_TRAIN = '../raw/Train/'
CSV_TRAIN = '../raw/train.csv'
DIST = '../data/extracted_data.hdf5'

RAW_TEST = '../raw/Test/'
CSV_TEST = '../raw/modified_test.csv'

classes = {'siren': 0,
           'street_music': 1,
           'drilling': 2,
           'dog_bark': 3,
           'children_playing': 4,
           'gun_shot': 5,
           'engine_idling': 6,
           'air_conditioner': 7,
           'jackhammer': 8,
           'car_horn': 9}

def extract(train_source, train_csv, test_source, test_csv, distination):
    h5_out = h5py.File(distination, 'w')
    train_index = pd.read_csv(train_csv)
    test_index = pd.read_csv(test_csv)

    labels = []
    datas = []
    print('Extracting train set audio...')
    for row in tqdm(train_index.itertuples()):
        x, sample_rate = librosa.load(train_source + str(row.ID) + '.wav', duration = 2.97)
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

        labels.append(classes[row.Class])
        datas.append(ps)

    labels = np.array(labels)
    datas = np.array(datas)

    h5_out.create_dataset('train_label', data = labels)
    h5_out.create_dataset('train_data', data = datas)

    labels = []
    datas = []
    print('Extracting test set audio...')
    for row in tqdm(test_index.itertuples()):
        x, sample_rate = librosa.load(test_source + str(row.ID) + '.wav', duration = 2.97)
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

        labels.append(classes[row.Class])
        datas.append(ps)

    labels = np.array(labels)
    datas = np.array(datas)

    h5_out.create_dataset('test_label', data = labels)
    h5_out.create_dataset('test_data', data = datas)
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
    extract(RAW_TRAIN, CSV_TRAIN, RAW_TEST, CSV_TEST, DIST)

if __name__ == '__main__':
    main()