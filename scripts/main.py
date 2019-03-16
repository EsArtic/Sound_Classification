import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from tqdm import tnrange, tqdm
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random

RAW_TRAIN = './datasets/UrbanSound/Train/'
CSV_TRAIN = './datasets/UrbanSound/train.csv'
DIST_TRAIN = './datasets/UrbanSound/Extracted_Train/extracted_train'

RAW_TEST = './datasets/UrbanSound/TEST/'
CSV_TEST = './datasets/UrbanSound/test.csv'
DIST_TEST = './datasets/UrbanSound/Extracted_Test/extracted_test'

def extract(source, distination, csv, labeled):
    input_index = pd.read_csv(csv)
    classes = None
    if labeled:
        classes = {}

    curr_out = open(distination, 'w')
    for row in tqdm(input_index.itertuples()):

        if labeled:
            if row.Class not in classes.keys():
                classes[row.Class] = len(classes.keys())
        x, sample_rate = librosa.load(source + str(row.ID) + '.wav', duration = 2.97)
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

        curr_out.write(str(row.ID) + ' ')

        if labeled:
            curr_out.write(str(classes[row.Class]) + ' ')

        curr_out.write(str(padding) + ' ')

        N, M = ps.shape
        for i in range(N):
            for j in range(M):
                curr_out.write(str(ps[i, j]) + ' ')

        curr_out.write('\n')

    curr_out.close()
    print(classes)

def load_data(path):
    S = []
    curr_input = open(path)
    for line in tqdm(curr_input):
        items = line.strip().split(' ')
        label = int(items[1])

        items = items[3:]
        features = []
        row = []
        for i in range(len(items)):
            row.append(float(items[i
                ]))
            if i % 128 == 127:
                features.append(row)
                row = []
        features = np.array(features)
        S.append((features, label))

    curr_input.close()
    return S

def main():
    # extract(RAW_TRAIN, DIST_TRAIN, CSV_TRAIN, True)
    extract(RAW_TEST, DIST_TEST, CSV_TEST, False)

    '''
    S = load_data(DIST_TRAIN)
    random.shuffle(S)

    X_train, Y_train = zip(*S)
    X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])

    Y_train = np.array(keras.utils.to_categorical(Y_train, 10))

    model = Sequential()
    input_shape=(128, 128, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy'])

    model.fit(
        x = X_train, 
        y = Y_train,
        epochs = 20,
        batch_size = 128,
        # validation_data= (X_test, y_test)
    )

    model.save('./my_model.h5')
    # model = load_model('./mymodel.h5')
    # print(model.predict(input))

    score = model.evaluate(
        x=X_test,
        y=Y_test
    )

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''

if __name__ == '__main__':
    main()