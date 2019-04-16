from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
import random
import os

AUDIO_SOURCE = '../raw/audio'
CSV_SOURCE = '../raw/metadata/UrbanSound8K.csv'
AUDIO_DIST = '../raw/augmentation'
CSV_DIST = '../raw/metadata/augmentation.csv'

folders = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5',
           'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
augment_types = ['ts', 'ps']
augment_params = [[0.81, 0.93, 1.07, 1.23], [-2, -1, 1, 2]]

def get_new_wav(source, sample_rate, type_idx, param_idx):
    ret = source
    if type == 0:
        ret = librosa.effects.time_stretch(source, augment_params[type_idx][param_idx])
    elif type == 1:
        ret = librosa.effects.pitch_shift(source, sample_rate, n_steps = augment_params[type_idx][param_idx])

    return ret

def do_generation(wav, sr, source_name, source_label, augment_idx, dist_path, csv_out, fold):
    param_idx = random.randint(0, 3)
    new_wav = get_new_wav(wav, sr, augment_idx, param_idx)
    wav_name = source_name[:-4] + '-' + augment_types[augment_idx]
    wav_name += '-' + str(augment_params[augment_idx][param_idx]) + '.wav'
    librosa.output.write_wav(os.path.join(dist_path, wav_name), new_wav, sr)
    csv_out.write(wav_name + ',' + str(fold) + ',' + str(source_label) + '\n')

def augmentation(source, csv, destination, csv_destination):
    index = pd.read_csv(csv)
    csv_out = open(csv_destination, 'w')
    csv_out.write('slice_file_name,fold,classID\n')

    for row in tqdm(index.itertuples()):
        x, sr = librosa.load(os.path.join(os.path.join(source, folders[row.fold - 1]), row.slice_file_name))
        dist_path = os.path.join(destination, folders[row.fold - 1])

        augment_num = random.randint(0, 2)
        if augment_num == 1:
            augment_idx = random.randint(0, 1)
            do_generation(x, sr, row.slice_file_name, row.classID, augment_idx, dist_path, csv_out, row.fold)
        elif augment_num == 2:
            do_generation(x, sr, row.slice_file_name, row.classID, 0, dist_path, csv_out, row.fold)
            do_generation(x, sr, row.slice_file_name, row.classID, 1, dist_path, csv_out, row.fold)

    csv_out.close()

def main():
    augmentation(AUDIO_SOURCE, CSV_SOURCE, AUDIO_DIST, CSV_DIST)

if __name__ == '__main__':
    main()