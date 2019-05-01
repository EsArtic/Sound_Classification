import keras
from model_generator import CNN_generator
from model_generator import FNN_generator
from model_generator import CRNN_generator
from tqdm import tqdm
import numpy as np
import random
import h5py

import argparse

def load_data(path, aug):
    data = []
    aug_data = []
    labels = []
    aug_labels = []
    h5_in = h5py.File(path, 'r')
    for i in range(1, 11):
        key = 'fold' + str(i)
        curr_data = h5_in[key + '_data']
        label = h5_in[key + '_label']
        data.append(curr_data)
        labels.append(label)
        if aug:
            curr_aug_data = h5_in[key + '_aug_data']
            aug_label = h5_in[key + '_aug_label']
            aug_data.append(curr_aug_data)
            aug_labels.append(aug_label)
    return data, aug_data, labels, aug_labels

def split_train_test(data, aug_data, labels, aug_labels, test_id, aug):
    train = []
    val = []
    for i in range(10):
        if i == test_id:
            for j in range(labels[i].shape[0]):
                val.append((data[i][j], labels[i][j]))
        else:
            for j in range(labels[i].shape[0]):
                train.append((data[i][j], labels[i][j]))
            if aug:
                for j in range(aug_labels[i].shape[0]):
                    train.append((aug_data[i][j], aug_labels[i][j]))

    return train, val

def get_model(model, param, input_shape):
    if model == 'crnn':
        return CRNN_generator.CRNN5(input_shape)
    elif model == 'cnn':
        if param == 2:
            return CNN_generator.CNN2(input_shape)
        elif param == 3:
            return CNN_generator.CNN3(input_shape)
        elif param == 4:
            return CNN_generator.CNN4(input_shape)
    elif model == 'fnn':
        if param == 2:
            return FNN_generator.FNN2(input_shape)
        elif param == 3:
            return FNN_generator.FNN3(input_shape)
        elif param == 4:
            return FNN_generator.FNN4(input_shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type = str, required = True,
                        help = 'The path of the extracted data features.')
    parser.add_argument('-e', '--epoch', type = int, required = True,
                        help = 'The number of epoches for training.')
    parser.add_argument('-m', '--model', type = str, required = False, default = 'cnn',
                        help = 'The model used for training: {cnn, fnn, crnn}.')
    parser.add_argument('-p', '--param', type = int, required = False, default = 3,
                        help = 'The param for cnn, fnn model: {2, 3, 4}.')
    parser.add_argument('-a', '--augment', action = 'store_true',
                        help = 'Use augment data.')
    args = parser.parse_args()

    if not (args.model == 'cnn' or args.model == 'fnn' or args.model == 'crnn'):
        print('Incorrect model selection.')
        return

    if not (args.param == 2 or args.param == 3 or args.param == 4):
        print('Incorrect model param.')
        return

    data, aug_data, labels, aug_labels = load_data(args.source, args.augment)
    model = None

    cumulate_loss = 0
    cumulate_accuracy = 0
    for i in range(10):
        train, val = split_train_test(data, aug_data, labels, aug_labels, i, args.augment)

        X_train, Y_train = zip(*train)
        X_val, Y_val = zip(*val)

        input_shape = (X_train[0].shape[0], X_train[0].shape[1], 1)
        if args.model == 'crnn':
            input_shape = (X_train[0].shape[0], 1)
        X_train = np.array([x.reshape(input_shape) for x in X_train])
        X_val = np.array([x.reshape(input_shape) for x in X_val])

        Y_train = np.array(keras.utils.to_categorical(Y_train, 10))
        Y_val = np.array(keras.utils.to_categorical(Y_val, 10))

        model = get_model(args.model, args.param, input_shape)

        model.fit(
            x = X_train,
            y = Y_train,
            epochs = args.epoch,
            batch_size = 128,
            validation_data = (X_val, Y_val)
        )

        score = model.evaluate(
            x = X_val,
            y = Y_val
        )

        cumulate_loss += score[0]
        cumulate_accuracy += score[1]

        if args.model == 'cnn':
            model.save('../models/CNN_Classifier_Layer_' + str(args.param) + '_Epoch_' + str(args.epoch) +'_Fold_' + str(i + 1) + '.h5')
        elif args.model == 'fnn':
            model.save('../models/FNN_Classifier_Layer_' + str(args.param) + '_Epoch_' + str(args.epoch) +'_Fold_' + str(i + 1) + '.h5')
        else:
            model.save('../models/CRNN_Classifier_Epoch_' + str(args.epoch) +'_Fold_' + str(i + 1) + '.h5')

        print('Cross Validation Fold', i + 1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    print('Summary:')
    print('Average loss:', cumulate_loss / 10)
    print('Average accuracy:', cumulate_accuracy / 10)

if __name__ == '__main__':
    main()