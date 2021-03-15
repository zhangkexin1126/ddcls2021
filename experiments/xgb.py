import xgboost as xgb
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

    # Xgboost Data Interface
    dtrain = xgb.DMatrix(traindata, label=trainlabel)
    dtest = xgb.DMatrix(testdata, label=testlabel)
    # Model Parameters
    param = {'booster': 'dart',
             'max_depth': 2,
             'eta': 1,
             'gamma': 6.4,
             'objective': 'binary:logistic'}
    # Training
    num_round = 4
    bst = xgb.train(param, dtrain, num_round)
    bst.save_model('/home/kexin/phdwork/ddcls2021/code/saves/xgboost/xg.model')
    bst_reload = xgb.Booster(param)
    bst_reload.load_model('/home/kexin/phdwork/ddcls2021/code/saves/xgboost/xg.model')
    ypred = bst_reload.predict(dtest)
    pred_label = []
    for i in range(len(ypred)):
        if ypred[i] >= 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)
    print('####################################')
    print('############# Metrics ##############')
    # 1. accuracy: how many samples are correct
    acc = accuracy_score(testlabel, pred_label)
    print('accuracy classification score:', acc)