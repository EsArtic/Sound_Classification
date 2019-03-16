import librosa
import librosa.display

def main():
    y1, sr1 = librosa.load('../raw/Train/1021.wav', duration = 2.97)
    # y1, sr1 = librosa.load('../raw/Test/102.wav', duration = 2.97)
    ps = librosa.feature.melspectrogram(y = y1, sr = sr1)

    out = open('../temp/1021.txt', 'w')
    N, M = ps.shape
    for i in range(N):
        for j in range(M):
            out.write(str(ps[i, j]) + ' ')
        out.write('\n')

if __name__ == '__main__':
    main()