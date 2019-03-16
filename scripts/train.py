import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from tqdm import tnrange, tqdm
import numpy as np
import random
import h5py

DIST_TRAIN = './datasets/UrbanSound/Extracted_Train/extracted_train.hdf5'

def load_data(path):
    S = []
    h5_in = h5py.File(path, 'r')
    data = h5_in['train_data']
    labels = h5_in['train_label']
    print(labels.shape)
    for i in range(labels.shape[0]):
        S.append((data[i], labels[i]))
    return S

def main():
    S = load_data(DIST_TRAIN)
    random.shuffle(S)

    train = S[: 5000]
    val = S[5000: ]

    X_train, Y_train = zip(*train)
    X_val, Y_val = zip(*val)
    X_train = np.array([x.reshape((128, 128, 1)) for x in X_train])
    X_val = np.array([x.reshape((128, 128, 1)) for x in X_val])

    Y_train = np.array(keras.utils.to_categorical(Y_train, 10))
    Y_val = np.array(keras.utils.to_categorical(Y_val, 10))

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
        metrics=['accuracy']
    )

    model.fit(
        x = X_train, 
        y = Y_train,
        epochs = 10,
        batch_size = 128,
        validation_data= (X_val, Y_val)
    )

    score = model.evaluate(
        x=X_val,
        y=Y_val
    )

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()