import os

DATA_AUDIO_DIR = '/home/kay/UrbanSound8K/audio'
OUTPUT_DIR = '/home/kay/raw-waveforms'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')

TARGET_SR = 8000
AUDIO_LENGTH = 32000


def print_delimiter():
    print('-' * 80)


def print_total_trainable_parameters_count():
    import tensorflow as tf
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
