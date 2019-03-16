import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
from tqdm import tnrange, tqdm
import numpy as np

RAW_TEST = './datasets/UrbanSound/TEST/'
CSV_TEST = './datasets/UrbanSound/test.csv'
DIST_TEST = './datasets/UrbanSound/Extracted_Test/extracted_test'

mapping = {0: 'siren', 1: 'street_music', 2: 'drilling', 3: 'dog_bark',
           4: 'children_playing', 5: 'gun_shot', 6: 'engine_idling', 7: 'air_conditioner', 
           8: 'jackhammer', 9: 'car_horn'}

def load_data(path):
    S = []
    curr_input = open(path)
    for line in tqdm(curr_input):
        items = line.strip().split(' ')
        aid = int(items[0])

        items = items[2:]
        features = []
        row = []
        for i in range(len(items)):
            row.append(float(items[i
                ]))
            if i % 128 == 127:
                features.append(row)
                row = []
        features = np.array(features)
        S.append((aid, features))

    curr_input.close()
    return S

def main():
    T = load_data(DIST_TEST)

    out = open('./test_prediction.csv', 'w')
    out.write('ID,Class\n')

    model = load_model('./log/my_model_100.h5')

    for ID, X in tqdm(T):
        X = X.reshape((1, 128, 128, 1))
        predict_vec = model.predict(X)
        label = ''
        max_prob = -1
        for i in range(predict_vec.shape[1]):
            if predict_vec[0, i] > max_prob:
                label = mapping[i]
                max_prob = predict_vec[0, i]
        out.write(str(ID) + ',' + label + '\n')

    out.close()

if __name__ == '__main__':
    main()