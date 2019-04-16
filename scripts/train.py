import keras
from model_generator import CNN_generator
from tqdm import tqdm
import numpy as np
import random
import h5py

DIST_TRAIN = '../data/extracted_data_spectrogram.hdf5'
EPOCH = 50

def load_data(path):
    data = []
    aug_data = []
    labels = []
    aug_labels = []
    h5_in = h5py.File(path, 'r')
    for i in range(1, 11):
        key = 'fold' + str(i)
        curr_data = h5_in[key + '_data']
        curr_aug_data = h5_in[key + '_aug_data']
        label = h5_in[key + '_label']
        aug_label = h5_in[key + '_aug_label']
        print(key, 'raw_shape:', curr_data.shape, label.shape,
            'aug_shape:', curr_aug_data.shape, aug_label.shape)
        data.append(curr_data)
        aug_data.append(curr_aug_data)
        labels.append(label)
        aug_labels.append(aug_label)
    return data, aug_data, labels, aug_labels

def split_train_test(data, aug_data, labels, aug_labels, test_id):
    train = []
    val = []
    for i in range(10):
        if i == test_id:
            for j in range(labels[i].shape[0]):
                val.append((data[i][j], labels[i][j]))
        else:
            for j in range(labels[i].shape[0]):
                train.append((data[i][j], labels[i][j]))
            for j in range(aug_labels[i].shape[0]):
                train.append((aug_data[i][j], aug_labels[i][j]))

    return train, val

def main():
    data, aug_data, labels, aug_labels = load_data(DIST_TRAIN)

    cumulate_loss = 0
    cumulate_accuracy = 0
    for i in range(10):
        train, val = split_train_test(data, aug_data, labels, aug_labels, i)

        X_train, Y_train = zip(*train)
        X_val, Y_val = zip(*val)

        input_shape = (X_train[0].shape[0], X_train[0].shape[1], 1)
        X_train = np.array([x.reshape(input_shape) for x in X_train])
        X_val = np.array([x.reshape(input_shape) for x in X_val])

        Y_train = np.array(keras.utils.to_categorical(Y_train, 10))
        Y_val = np.array(keras.utils.to_categorical(Y_val, 10))

        model = CNN_generator.three_layers(input_shape)

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

        model.save('../models/CNN_Classifier_Epoch_' + str(EPOCH) +'_Fold_' + str(i + 1) + '.h5')

        print('Cross Validation Fold', i + 1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    print('Summary:')
    print('Average loss:', cumulate_loss / 10)
    print('Average accuracy:', cumulate_accuracy / 10)

if __name__ == '__main__':
    main()