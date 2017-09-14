import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from constants import *


def m3(num_classes=10):
    print('Using Model M3')
    m = Sequential()
    m.add(Conv1D(256,
                 input_shape=[AUDIO_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Lambda(lambda x: K.mean(x, axis=1)))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def m5(num_classes=10):
    print('Using Model M5')
    m = Sequential()
    m.add(Conv1D(128,
                 input_shape=[AUDIO_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(512,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Lambda(lambda x: K.mean(x, axis=1)))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def m11(num_classes=10):
    print('Using Model M11')
    m = Sequential()
    m.add(Conv1D(64,
                 input_shape=[AUDIO_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(64,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(128,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(3):
        m.add(Conv1D(256,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))

    m.add(Lambda(lambda x: K.mean(x, axis=1)))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def m_rec(num_classes=10):
    from keras.layers.recurrent import LSTM
    print('Using Model LSTM 1')
    m = Sequential()
    m.add(Conv1D(64,
                 input_shape=[AUDIO_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(LSTM(32,
               kernel_regularizer=regularizers.l2(l=0.0001),
               return_sequences=True,
               dropout=0.2))
    m.add(LSTM(32,
               kernel_regularizer=regularizers.l2(l=0.0001),
               return_sequences=False,
               dropout=0.2))
    m.add(Dense(32))
    m.add(Dense(num_classes, activation='softmax'))
    return m


def m18(num_classes=10):
    print('Using Model M18')
    m = Sequential()
    m.add(Conv1D(64,
                 input_shape=[AUDIO_LENGTH, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(64,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(128,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(256,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))

    m.add(Lambda(lambda x: K.mean(x, axis=1)))
    m.add(Dense(num_classes, activation='softmax'))
    return m
