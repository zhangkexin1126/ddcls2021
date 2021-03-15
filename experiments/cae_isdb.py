import sys
sys.path.append('/home/kexin/phdwork/ddcls2021/code')
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from models import cae
from metrics import clusteringmetric
from models import clustering
from sklearn.decomposition import PCA
from utils import displaypca2d, displaypca3d

def loadtraindata_cae():
    path = '/home/kexin/phdwork/ddcls2021/data/train/isdb_dtwmatrix_multiscale'
    tslist = os.listdir(path)
    datalist = []
    labellist = []
    for ts in tslist:
        tspath = os.path.join(path, ts)
        classlist = os.listdir(tspath)
        for classtype in classlist:
            datapath = os.path.join(tspath, classtype)
            #print(datapath)
            filelist = os.listdir(datapath)
            if classtype == 'norm':
                label = 0
            elif classtype == 'stic':
                label = 1
            for file in filelist:
                filepath = os.path.join(datapath, file)
                data = np.load(filepath)
                datalist.append(data)
                labellist.append(label)

    # load matlab data
    path = '/home/kexin/phdwork/ddcls2021/data/train/matlab_dtwmatrix_multiscale'
    tslist = os.listdir(path)
    matlabdata = []
    matlablabel = []
    for ts in tslist:
        tspath = os.path.join(path, ts)
        classlist = os.listdir(tspath)
        for classtype in classlist:
            datapath = os.path.join(tspath, classtype)
            #print(datapath)
            filelist = os.listdir(datapath)
            if classtype == 'norm':
                label = 0
            elif classtype == 'stic':
                label = 1
            for file in filelist:
                filepath = os.path.join(datapath, file)
                data = np.load(filepath)
                matlabdata.append(data)
                matlablabel.append(label)

    #matlabdata.extend(datalist)
    #matlablabel.extend(labellist)
    x = np.array(matlabdata)
    x = x.reshape(-1, 28, 28, 2).astype('float32')
    y = np.array(matlablabel)
    return x, y

def loadtraindata_cluster(ts='200'):
    datalist = []
    labellist = []
    path = '/home/kexin/phdwork/ddcls2021/data/train/matlab_dtwmatrix_multiscale'
    testpath = os.path.join(path, ts)
    classlist = os.listdir(testpath)
    for classtype in classlist:
        datapath = os.path.join(testpath, classtype)
        filelist = os.listdir(datapath)
        if classtype == 'norm':
            label = 0
        elif classtype == 'stic':
            label = 1
        for file in filelist:
            filepath = os.path.join(datapath, file)
            data = np.load(filepath)
            datalist.append(data)
            labellist.append(label)
    x = np.array(datalist)
    x = x.reshape(-1, 28, 28, 2).astype('float32')
    y = np.array(labellist)
    return x, y

def loadtestdata(ts='200'):
    datalist = []
    labellist = []
    path = '/home/kexin/phdwork/ddcls2021/data/test/dtwmatrix_multiscale'
    testpath = os.path.join(path, ts)
    classlist = os.listdir(testpath)
    for classtype in classlist:
        datapath = os.path.join(testpath, classtype)
        #print(datapath)
        filelist = os.listdir(datapath)
        if classtype == 'norm':
            label = 0
        elif classtype == 'stic':
            label = 1
        for file in filelist:
            filepath = os.path.join(datapath, file)
            data = np.load(filepath)
            datalist.append(data)
            labellist.append(label)
            #print(file, label)
    x = np.array(datalist)
    x = x.reshape(-1, 28, 28, 2).astype('float32')
    y = np.array(labellist)
    return x, y

def loadloopname():
    loopname = []
    path = '/home/kexin/phdwork/ddcls2021/data/test/dtwmatrix_multiscale/125'
    classlist = os.listdir(path)
    for classtype in classlist:
        datapath = os.path.join(path, classtype)
        filelist = os.listdir(datapath)
        for file in filelist:
            loopname.append(file[0:-6])
    simplename = []
    pattern = re.compile(r'\d+')
    for loop in loopname:
        na = loop[0:4]
        num = pattern.search(loop).group(0)
        newname = na + num
        simplename.append(newname)
    return simplename

if __name__ == '__main__':
    from time import time

    print('start cae')
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='isdb')
    parser.add_argument('--n_clusters', default=2, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--hidden_dim', default=5, type=int)
    parser.add_argument('--save_dir', default='/home/kexin/phdwork/ddcls2021/code/saves', type=str)
    args = parser.parse_args()
    print(args)

    trainflag = True
    if trainflag:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        x, y = loadtraindata_cae()

        model = cae.CAE(input_shape=(28, 28, 2), filters=[32, 64, 128, args.hidden_dim])
        plot_model(model, to_file=args.save_dir + '/%s-cae-model.png' % args.dataset,
                   show_shapes=True, show_dtype=True)
        model.summary()
        # compile the model and callbacks
        optimizer = 'adam'
        model.compile(optimizer=optimizer, loss='mse')
        csv_logger = CSVLogger(args.save_dir + '/%s-train-log.csv' % args.dataset)
        model_cp = ModelCheckpoint(filepath=args.save_dir + '/checkpoints/cp',
                                   save_weights_only=False)
        # begin training
        t0 = time()
        model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger,
                                                                                   ])
        print('Training time: ', time() - t0)
        model.save(args.save_dir + '/%s-train-model-%d.h5' % (args.dataset, args.epochs))

    if not trainflag:
        # Load model if not train
        model = tf.keras.models.load_model(args.save_dir +
                                           '/%s-train-model-%d.h5' % (args.dataset, args.epochs))
        model.summary()

    # Load Encoder
    caeencoder = cae.CAEencoder(model)
    caeencoder.summary()
    # Get Loop name
    loopname = loadloopname()

    ####################################################################
    ############# Test
    timescale_int = [100, 200, 300]
    timescale = list(map(lambda x: str(x), timescale_int))
    testfeat_scale = []
    for ts in timescale:
        xt, yt = loadtestdata(ts=ts)
        # extract features
        features = caeencoder.predict(xt)
        testfeat_scale.append(features)
    testfeat_scale = tf.concat(testfeat_scale, axis=1)
    clusterfeats = np.reshape(testfeat_scale, newshape=(testfeat_scale.shape[0], -1))
    km = KMeans(n_clusters=args.n_clusters)
    pred = km.fit_predict(clusterfeats)
    print('######################')
    print('Kmeans')
    n_scale = len(timescale)
    pred_temp = []
    pred_scale = []
    for i in range(n_scale):
        start_col = int(i * args.hidden_dim)
        end_col = int((i + 1) * args.hidden_dim)
        # print(start_col, end_col)
        km = KMeans(n_clusters=args.n_clusters)
        features = np.reshape(testfeat_scale[:, start_col:end_col],
                              newshape=(testfeat_scale[:, start_col:end_col].shape[0], -1))
        pred_temp = km.fit_predict(features)
        pred_scale.extend(pred_temp)
    n_test = len(pred_temp)
    '''for i, loop in enumerate(loopname): 
        print('--', loop, 'real:', yt[i], 'ensemble:', pred[i])'''
        # 'single:', pred_scale[i], pred_scale[i + n_test], pred_scale[i + n_test * 2]
    print('num=', len(loopname), 'acc=', clusteringmetric.acc(yt, pred))
    for i in range(n_scale):
        p = np.array(pred_scale[n_test * i: n_test * (i + 1)])
        print('timescale:', timescale[i], 'acc=', clusteringmetric.acc(yt, p))

    print('######################')
    print('FCM')
    fcm_cntr, fcm_u = clustering.fcm_train(clusterfeats, N_class=args.n_clusters)
    pred = clustering.fcm_test(clusterfeats)
    pred_temp = []
    pred_fcm = []
    for i in range(n_scale):
        start_col = int(i * args.hidden_dim)
        end_col = int((i + 1) * args.hidden_dim)
        # print(start_col, end_col)
        features = np.reshape(testfeat_scale[:, start_col:end_col],
                              newshape=(testfeat_scale[:, start_col:end_col].shape[0], -1))
        fcm_cntr, fcm_u = clustering.fcm_train(features, N_class=args.n_clusters)
        pred_temp = clustering.fcm_test(features)
        pred_fcm.extend(pred_temp)
    #for i, loop in enumerate(loopname):
    #    print('--', loop, 'real:', yt[i], 'ensemble:', pred[i])
    print('num=', len(loopname), 'acc=', clusteringmetric.acc(yt, pred))
    for i in range(n_scale):
        p = np.array(pred_fcm[n_test * i: n_test * (i + 1)])
        print('timescale:', timescale[i], 'acc=', clusteringmetric.acc(yt, p))

    print('######################')
    print('PCA')
    n_pca = 2

    ### PCA
    #print('pca shape', clusterfeats.shape)
    pca = PCA(n_components=n_pca, svd_solver='randomized')
    pcafeats = pca.fit_transform(clusterfeats)
    #print(pcafeats.shape)
    km = KMeans(n_clusters=args.n_clusters)
    pred_pca = km.fit_predict(pcafeats)
    print('PCA kmeans, Test num=', len(loopname), 'acc=', clusteringmetric.acc(yt, pred_pca))
    fcm_cntr, fcm_u = clustering.fcm_train(pcafeats, N_class=args.n_clusters)
    pred_pca = clustering.fcm_test(pcafeats)
    print('PCA fcm, Test num=', len(loopname), 'acc=', clusteringmetric.acc(yt, pred_pca))
    # pca image using testdata
    yt = yt.reshape((-1, 1))
    wrongloop = ['chem12', 'chem70', 'pulp6', 'pulp1', 'chem2', 'chem20',
                 'chem74','pulp13','chem75','chem76','chem19']

    wrongloop = ['chem70', 'pulp6']
    wrongidx = []
    for i, loop in enumerate(loopname):
        if loop in wrongloop:
            wrongidx.append(i)
    finafeat = np.delete(pcafeats, wrongidx, axis=0)
    finaly = np.delete(yt, wrongidx, axis=0)
    if n_pca == 3:
        displaypca3d(finafeat, finaly, colors=['red', 'blue'], epo=args.epochs)
    elif n_pca == 2:
        displaypca2d(finafeat, finaly, colors=['red', 'blue'], dataid=loopname, epo=args.epochs)

    #pca image using traindata
    '''timescale_int = [200, 400, 600]
    timescale = list(map(lambda x: str(x), timescale_int))
    trainfeats = []
    for ts in timescale:
        xt, yt = loadtraindata_cluster(ts=ts)
        # extract features
        features = caeencoder.predict(xt)
        trainfeats.append(features)
    trainfeats = tf.concat(trainfeats, axis=1)
    trainfeats = np.array(trainfeats)
    pca = PCA(n_components=n_pca, svd_solver='randomized')
    pcafeats = pca.fit_transform(trainfeats)
    yt = yt.reshape((-1, 1))
    if n_pca == 3:
        displaypca3d(pcafeats, yt, colors=['red', 'blue'], epo=args.epochs)
    elif n_pca == 2:
        displaypca2d(pcafeats, yt, colors=['red', 'blue'], epo=args.epochs)'''


    ###################################################
    # Get Train Feats
    '''timescale_int = [100, 200, 300]
    timescale = list(map(lambda x: str(x), timescale_int))
    trainfeats = []
    for ts in timescale:
        xt, yt = loadtraindata_cluster(ts=ts)
        # extract features
        features = caeencoder.predict(xt)
        trainfeats.append(features)
    trainfeats = tf.concat(trainfeats, axis=1)
    trainfeats = np.array(trainfeats)
    # Train Kmeans
    kmeans_clf = clustering.kmeans_train(trainfeats, N_class=args.n_clusters)
    fcm_cntr, fcm_u = clustering.fcm_train(trainfeats, N_class=2)
    # Test Timescale
    timescale_int = [75, 150, 100]
    timescale = list(map(lambda x: str(x), timescale_int))
    testfeats = []
    for ts in timescale:
        xt, yt = loadtestdata(ts=ts)
        # extract features
        features = caeencoder.predict(xt)
        testfeats.append(features)
    testfeats = tf.concat(testfeats, axis=1)
    testfeats = np.reshape(testfeats, newshape=(testfeats.shape[0], -1))
    # Kmeans Test
    kmeans_pred = clustering.kmeans_test(testfeats)
    for i, loop in enumerate(loopname):
        print('--', loop, yt[i], kmeans_pred[i])
    print('Use clustering on train data')
    print('kmeans')
    print('num=', len(loopname), 'acc=', clusteringmetric.acc(yt, kmeans_pred))
    fcm_pred = clustering.fcm_test(testfeats)
    for i, loop in enumerate(loopname):
        print('--', loop, yt[i], fcm_pred[i])
    print('Fcm')
    print('num=', len(loopname), 'acc=', clusteringmetric.acc(yt, np.array(fcm_pred)))'''




