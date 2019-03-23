import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from tqdm import tnrange, tqdm
import numpy as np
import random
import h5py

DIST_TRAIN = '../data/extracted_data.hdf5'
EPOCH = 30

def load_data(path):
    data = []
    labels = []
    h5_in = h5py.File(path, 'r')
    for i in range(1, 11):
        key = 'fold' + str(i)
        curr_data = h5_in[key + '_data']
        label = h5_in[key + '_label']
        print(key, 'shape:', curr_data.shape, label.shape)
        data.append(curr_data)
        labels.append(label)
    return data, labels

def split_train_test(data, labels, test_id):
    train = []
    val = []
    for i in range(10):
        if i == test_id:
            for j in range(labels[i].shape[0]):
                val.append((data[i][j], labels[i][j]))
        else:
            for j in range(labels[i].shape[0]):
                train.append((data[i][j], labels[i][j]))

    return train, val

def create_model():
    model = Sequential()
    input_shape=(128, 128, 1)

    model.add(Conv2D(24, (5, 5), strides = (1, 1), input_shape = input_shape))
    model.add(MaxPooling2D((4, 2), strides = (4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding = 'valid'))
    model.add(MaxPooling2D((4, 2), strides = (4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding = 'valid'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate = 0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate = 0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(
        optimizer = 'Adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model

def main():
    data, labels = load_data(DIST_TRAIN)

    cumulate_loss = 0
    cumulate_accuracy = 0
    for i in range(10):
        train, val = split_train_test(data, labels, i)

        X_train, Y_train = zip(*train)
        X_val, Y_val = zip(*val)
        X_train = np.array([x.reshape((128, 128, 1)) for x in X_train])
        X_val = np.array([x.reshape((128, 128, 1)) for x in X_val])

        Y_train = np.array(keras.utils.to_categorical(Y_train, 10))
        Y_val = np.array(keras.utils.to_categorical(Y_val, 10))

        model = create_model()

        model.fit(
            x = X_train,
            y = Y_train,
            epochs = EPOCH,
            batch_size = 128,
            validation_data = (X_val, Y_val)
        )

        score = model.evaluate(
            x = X_val,
            y = Y_val
        )

        cumulate_loss += score[0]
        cumulate_accuracy += score[1]

        model.save('../models/Sample_CNN_Epoch_' + str(EPOCH) +'_Fold_' + str(i + 1) + '.h5')

        print('Cross Validation Fold', i + 1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    print('Summary:')
    print('Average loss:', cumulate_loss / 10)
    print('Average accuracy:', cumulate_accuracy / 10)

if __name__ == '__main__':
    main()