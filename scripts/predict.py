import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
import librosa
from tqdm import tnrange, tqdm
import numpy as np

AUDIO_SOURCE = '../raw/audio/'
FOLD = 'fold1/'
AUDIO = '101415-3-0-2.wav'

mapping = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

def load_audio(path):
    y1, sr1 = librosa.load(path, duration = 2.97)
    ps = librosa.feature.melspectrogram(y = y1, sr = sr1)

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

    return ps

def main():
    source = AUDIO_SOURCE + FOLD + AUDIO

    gt = int(AUDIO.split('-')[1])
    data = load_audio(source)

    model = load_model('../models/Sample_CNN_Epoch_30_Fold_1.h5')

    data = data.reshape((1, 128, 128, 1))

    predict_vec = model.predict(data)
    label = ''
    max_prob = -1
    for i in range(predict_vec.shape[1]):
        if predict_vec[0, i] > max_prob:
            label = mapping[i]
            max_prob = predict_vec[0, i]

    print('\nPrediction for', source)
    print('The label is:', mapping[gt])
    print('The prediction is:', label)

if __name__ == '__main__':
    main()