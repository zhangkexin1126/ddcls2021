from sklearn import svm
import os
import numpy as np
from sklearn.metrics import accuracy_score


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

    clf = svm.SVC(C=3, kernel='rbf', degree=3, gamma='scale', tol=1e-3, max_iter=-1).fit(traindata, trainlabel)
    # Testing model
    pred_label = clf.predict(testdata)
    print('####################################')
    print('############# Metrics ##############')
    # 1. accuracy: how many samples are correct
    acc = accuracy_score(testlabel, pred_label)
    print('accuracy classification score:', acc)