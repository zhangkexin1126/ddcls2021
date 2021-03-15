'''
Convolutional Autoencoder
'''

import sys
sys.path.append('/home/kexin/phdwork/ddcls2021/code')
from time import time
import numpy as np

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger
from sklearn.cluster import KMeans

from metrics import clusteringmetric
from dataprepare import mnist

def CAE(input_shape=(28, 28, 2), filters=[32, 64, 128, 20]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    # build encoder
    model.add(Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='conv1',
                     input_shape=input_shape, data_format='channels_last'))
    model.add(Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='conv2'))
    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
    # Embedding
    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8),activation='relu'))
    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    # build decoder
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    #model.summary()
    return model

def CAEencoder(model):
    return Model(inputs=model.get_layer(name='conv1').input,
                 outputs=model.get_layer(name='embedding').output)

if __name__ == "__main__":
    from time import time
    print('start cae')
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='isdb', choices=['mnist', 'isdb'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='/home/kexin/phdwork/ddcls2021/code/saves', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('save_dir', args.save_dir)
    # load dataset
    x, y = mnist.load_mnist()
    # prepare model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
    plot_model(model, to_file=args.save_dir + '/%s-cae-model.png' % args.dataset,
               show_shapes=True, show_dtype=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    csv_logger = CSVLogger(args.save_dir + '/%s-train-log.csv' % args.dataset)

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-train-model-%d.h5' % (args.dataset, args.epochs))

    # extract features
    feature_model = Model(inputs=model.get_layer(name='conv1').input, outputs=model.get_layer(name='embedding').output)
    feature_model.summary()
    features = feature_model.predict(x)
    print('feature shape=', features.shape)
    # use features for clustering
    km = KMeans(n_clusters=args.n_clusters)
    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    print('acc=', clusteringmetric.acc(y, pred),
          'nmi=', clusteringmetric.nmi(y, pred),
          'ari=', clusteringmetric.ari(y, pred))

    print('==============', model.input)
    print('==============', model.get_layer(name='embedding'))
    print('==============', model.output)