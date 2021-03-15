import numpy as np
import tensorflow as tf


def load_mnist():
    # the data, shuffled and split between train and test sets
    savepath = '/home/kexin/data/mnisttf'
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x/255.
    print('MNIST:', x.shape)
    return x, y

if __name__ == "__main__":
    from time import time

    load_mnist()

