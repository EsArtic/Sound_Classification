from keras import optimizers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.models import Sequential

class CNN_generator:
    def three_layers(input_shape):
        model = Sequential()

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

    def two_layers(input_shape):
        model = Sequential()

        model.add(Conv2D(80, (57, 6), strides = (1, 1), input_shape = input_shape))
        model.add(MaxPooling2D((4, 3), strides = (1, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(80, (1, 3), strides = (1, 1), padding = 'valid'))
        model.add(MaxPooling2D((1, 3), strides = (1, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dropout(rate = 0.5))

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(rate = 0.5))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(
            optimizer = 'Adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model

class CRNN_generator:
    def CRNN5(input_shape):
        model = Sequential()

        model.add(Conv1D(64, 80, strides = 4, input_shape = input_shape))
        model.add(MaxPooling1D(4, strides = 1))
        model.add(Activation('relu'))

        model.add(Conv1D(64, 3, strides = 1))
        model.add(MaxPooling1D(4, strides = 1))
        model.add(Activation('relu'))

        '''
        model.add(Conv1D(128, 3, strides = 1))
        model.add(MaxPooling1D(4, strides = 1))
        model.add(Activation('relu'))

        model.add(Conv1D(256, 3, strides = 1))
        model.add(MaxPooling1D(4, strides = 1))
        model.add(Activation('relu'))

        model.add(Conv1D(512, 3, strides = 1))
        model.add(MaxPooling1D(4, strides = 1))
        model.add(Activation('relu'))
        '''

        model.add(LSTM(64, input_shape = (64, 1), activation = 'tanh'))
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