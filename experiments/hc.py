from sklearn.cluster import AgglomerativeClustering
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from metrics import clusteringmetric

def loadtraindata(ts='400'):
    path = r'/home/kexin/phdwork/ddcls2021/data/train/matlab_dtwmatrix_multiscale'
    trainpath = os.path.join(path, ts)
    classlist = os.listdir(trainpath)
    matlabdata = []
    matlablabel = []
    for classtype in classlist:
        datapath = os.path.join(trainpath, classtype)
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
    x = np.array(matlabdata)
    x = x.reshape(x.shape[0], -1)
    y = np.array(matlablabel)
    return x, y

def loadtestdata(ts='100'):
    path = r'/home/kexin/phdwork/ddcls2021/data/test/dtwmatrix_multiscale'
    trainpath = os.path.join(path, ts)
    classlist = os.listdir(trainpath)
    matlabdata = []
    matlablabel = []
    for classtype in classlist:
        datapath = os.path.join(trainpath, classtype)
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
    x = np.array(matlabdata)
    x = x.reshape(x.shape[0], -1)
    y = np.array(matlablabel)
    return x, y

if __name__ == '__main__':
    traindata, trainlabel = loadtraindata(ts='400')
    print(traindata.shape)
    testdata, testlabel = loadtestdata(ts='100')
    print(testdata.shape)

    # Training model
    km = AgglomerativeClustering(n_clusters=2)
    pred_label = km.fit_predict(testdata)

    # Metrics
    print('####################################')
    print('############# Metrics ##############')
    # 1. accuracy: how many samples are correct
    acc = clusteringmetric.acc(testlabel, pred_label)
    print('accuracy classification score:', acc)
    '''# 2. classification_report: main classification metrics
    target_names = ['non-stiction', 'stiction']
    print('classification_report:')
    print(classification_report(testlabel, pred_label, target_names=target_names))
    # 3. confusion_matrix
    tn, fp, fn, tp = confusion_matrix(testlabel, pred_label).ravel()
    print('confusion_matrix:')
    print('TN:', tn, 'TP:', tp, 'FN:', fn, 'TP:', tp)'''