import keras
from tqdm import tqdm
from keras.models import load_model
import pandas as pd
import numpy as np
import h5py

import argparse

CATEGORIES = 10
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

def load_data(path):
    data = []
    labels = []
    h5_in = h5py.File(path, 'r')
    for i in range(1, 11):
        key = 'fold' + str(i)
        curr_data = h5_in[key + '_data']
        label = h5_in[key + '_label']
        data.append(curr_data)
        labels.append(label)
    return data, labels

def get_label(y):
    return np.argmax(y)

def init_map(category_num):
    ret = {}
    for i in range(category_num):
        ret[i] = 0
    return ret

def get_statistics(predict, y):
    n = y.shape[0]
    count = 0
    positive_counts = init_map(CATEGORIES)
    true_positive_counts = init_map(CATEGORIES)
    predict_positive_counts = init_map(CATEGORIES)
    for i in range(n):
        if predict[i] == y[i]:
            true_positive_counts[y[i]] += 1
            count += 1

        predict_positive_counts[predict[i]] += 1
        positive_counts[y[i]] += 1

    total_acc = count / n
    precision = {}
    recall = {}
    for key in positive_counts.keys():
        precision[key] = true_positive_counts[key] / predict_positive_counts[key]
        recall[key] = true_positive_counts[key] / positive_counts[key]
    return total_acc, precision, recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type = str, required = True,
                        help = 'The path of the extracted data features.')
    parser.add_argument('-e', '--epoch', type = int, required = True,
                        help = 'The number of epoches for training.')
    parser.add_argument('-m', '--model', type = str, required = False, default = 'cnn',
                        help = 'The model used for training: {cnn, fnn}.')
    parser.add_argument('-p', '--param', type = int, required = False, default = 3,
                        help = 'The param for cnn, fnn model: {2, 3, 4}.')
    args = parser.parse_args()

    if not (args.model == 'cnn' or args.model == 'fnn'):
        print('Incorrect model selection.')
        return

    if not (args.param == 2 or args.param == 3 or args.param == 4):
        print('Incorrect model param.')
        return

    prefix = '../models/'
    if args.model == 'cnn':
        prefix += 'CNN_Classifier_Layer_' + str(args.param) + '_Epoch_' + str(args.epoch) + '_Fold_'
    elif args.model == 'fnn':
        prefix += 'FNN_Classifier_Layer_' + str(args.param) + '_Epoch_' + str(args.epoch) + '_Fold_'

    data, labels = load_data(args.source)

    accuracys = []
    acc_count = 0
    precisions = {}
    precision_sum = [0.0] * CATEGORIES
    recalls = {}
    recall_sum = [0.0] * CATEGORIES
    for i in range(CATEGORIES):
        precisions[i] = []
        recalls[i] = []

    for i in tqdm(range(CATEGORIES)):
        model = load_model(prefix + str(i + 1) + '.h5')

        X = data[i]
        input_shape = (X[0].shape[0], X[0].shape[1], 1)
        X = np.array([x.reshape(input_shape) for x in X])
        Y = labels[i]

        predict_vec = model.predict(X)

        predict_labels = []
        for j in range(predict_vec.shape[0]):
            predict_labels.append(get_label(predict_vec[j, :]))

        acc, precision, recall = get_statistics(predict_labels, Y)
        accuracys.append(acc)
        acc_count += acc
        for key in precision.keys():
            precisions[key].append(precision[key])
            precision_sum[key] += precision[key]
            recalls[key].append(recall[key])
            recall_sum[key] += recall[key]

    accuracys.append(acc_count / CATEGORIES)
    idx = list(range(1, CATEGORIES + 1))
    idx.append('average')

    pdf = {}
    rdf = {}
    pdf['fold'] = idx
    rdf['fold'] = idx
    rdf['total_acc'] = accuracys
    for key in precisions.keys():
        precisions[key].append(precision_sum[key] / CATEGORIES)
        recalls[key].append(recall_sum[key] / CATEGORIES)
        pdf[mapping[key]] = precisions[key]
        rdf[mapping[key]] = recalls[key]
    pdf = pd.DataFrame(data = pdf)
    rdf = pd.DataFrame(data = rdf)

    pdf.to_csv('../log/precision.csv', index = False)
    rdf.to_csv('../log/recall.csv', index = False)

if __name__ == '__main__':
    main()