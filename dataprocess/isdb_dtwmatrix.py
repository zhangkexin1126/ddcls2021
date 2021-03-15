import os
import time
import shutil
import numpy as np
from dataprepare import isdb
from dataprocess import dtwmatrix_conversion
from sklearn import preprocessing
from scipy import signal


sticloop = ['chemicals_loop19', 'power_loop4', 'chemicals_loop2', 'chemicals_loop8',
            'chemicals_loop9', 'pulpPapers_loop5', 'chemicals_loop18', 'chemicals_loop30',
            'chemicals_loop28', 'chemicals_loop12', 'chemicals_loop20', 'chemicals_loop32',
            'chemicals_loop1', 'chemicals_loop26', 'chemicals_loop11', 'chemicals_loop24',
            'chemicals_loop7', 'chemicals_loop23', 'chemicals_loop35', 'power_loop2',
            'buildings_loop7', 'pulpPapers_loop3', 'pulpPapers_loop2', 'chemicals_loop10',
            'power_loop1', 'chemicals_loop29', 'chemicals_loop5', 'chemicals_loop6',
            'buildings_loop6', 'pulpPapers_loop13', 'mining_loop1', 'chemicals_loop22',
            'pulpPapers_loop1', 'pulpPapers_loop12', 'pulpPapers_loop11']
normloop = ['pulpPapers_loop10', 'chemicals_loop55', 'chemicals_loop45', 'chemicals_loop64',
            'chemicals_loop48', 'buildings_loop1', 'chemicals_loop49', 'pulpPapers_loop6',
            'chemicals_loop73', 'chemicals_loop31', 'chemicals_loop51', 'pulpPapers_loop8',
            'metals_loop3', 'power_loop5', 'chemicals_loop63', 'chemicals_loop50', 'chemicals_loop61',
            'chemicals_loop46', 'buildings_loop8', 'chemicals_loop53', 'chemicals_loop57',
            'chemicals_loop52', 'chemicals_loop72', 'chemicals_loop75', 'chemicals_loop74',
            'chemicals_loop60', 'chemicals_loop54', 'chemicals_loop40', 'power_loop3', 'chemicals_loop70',
            'chemicals_loop59', 'chemicals_loop43', 'chemicals_loop76', 'buildings_loop2',
            'chemicals_loop47', 'pulpPapers_loop9', 'chemicals_loop42', 'chemicals_loop71',
            'chemicals_loop58', 'chemicals_loop56', 'chemicals_loop62']

#badsticloop = ['chemicals_loop7', 'chemicals_loop24', 'pulpPapers_loop11']
#badnormloop = ['buildings_loop1', 'power_loop5', 'buildings_loop8', 'chemicals_loop57','buildings_loop2']

badsticloop = ['chemicals_loop9','chemicals_loop7', 'pulpPapers_loop11',
               'chemicals_loop18', 'chemicals_loop8']
badnormloop = ['pulpPapers_loop9']

selectnum = 30

def get_loop_datanew(data, maxl=400, scaler = 'minmax'):
    length = len(data)
    if length > maxl:
        da = data[0:400, 0:2]
    else:
        da = data[:, 0:2]
    if scaler == 'minmax':
        datascaler = preprocessing.MinMaxScaler()
    elif scaler == 'standard':
        datascaler = preprocessing.StandardScaler()
    elif scaler == 'norm':
        datascaler = preprocessing.Normalizer()
    datanew = datascaler.fit_transform(da)
    b, a = signal.butter(3, 0.15, 'lowpass')
    datanewf = signal.filtfilt(b, a, datanew, axis=0)
    return datanewf

def matlab_conversion_train(dtwsize = 28, epochs = 5, maxl=400, dtwtype='fixed'):
    stdata, normdata = isdb.matlab_prepare()
    trainpath = r'/home/kexin/phdwork/ddcls2021/data/train/matlab_dtwmatrix_multiscale'
    ts = str(maxl)
    tspath = os.path.join(trainpath, ts)
    if not os.path.exists(tspath):
        os.makedirs(tspath)
    shutil.rmtree(tspath)
    os.makedirs(tspath)
    normsavepath = os.path.join(trainpath, ts, 'norm')
    sticsavepath = os.path.join(trainpath, ts, 'stic')
    os.makedirs(normsavepath)
    os.makedirs(sticsavepath)
    seedlist = np.random.randint(0, 1000, epochs)
    for ep in range(epochs):
        print('-------------epo:', ep)
        epo = str(ep)

        for i in range(len(stdata)):
            k = str(i)
            tailname = ts + '_' + epo + '_' + k
            op = stdata.iloc[i, 0:maxl].values.reshape(maxl, -1)
            pv = stdata.iloc[i, 1000:1000+maxl].values.reshape(maxl, -1)
            oppv = np.concatenate((op, pv), axis=1)
            M_dtw = dtwmatrix_conversion.multivariate_dtwmatrix_fast(oppv, mdtwsize=dtwsize,
                                                                     dtwtpye=dtwtype,
                                                                     mdtwseed=seedlist[ep])
            savename = os.path.join(sticsavepath, tailname)
            np.save(savename, M_dtw)

        for i in range(len(normdata)):
            k = str(i)
            tailname = ts + '_' + epo + '_' + k
            op = normdata.iloc[i, 0:maxl].values.reshape(maxl, -1)
            pv = normdata.iloc[i, 1000:1000+maxl].values.reshape(maxl, -1)
            oppv = np.concatenate((op, pv), axis=1)
            M_dtw = dtwmatrix_conversion.multivariate_dtwmatrix_fast(oppv, mdtwsize=dtwsize,
                                                                     dtwtpye=dtwtype,
                                                                     mdtwseed=seedlist[ep])
            savename = os.path.join(normsavepath, tailname)
            np.save(savename, M_dtw)

def isdb_conversion_train(dtwsize = 28, epochs = 5, maxl=400, scaler='minmax', dtwtype='fixed'):
    stdict, normdict, distdict, span = isdb.isdbload_filter()

    trainpath = r'/home/kexin/phdwork/ddcls2021/data/train/isdb_dtwmatrix_multiscale'
    if not os.path.exists(trainpath):
        os.makedirs(trainpath)
    ts = str(maxl)
    tspath = os.path.join(trainpath, ts)
    if not os.path.exists(tspath):
        os.makedirs(tspath)
    shutil.rmtree(tspath)
    os.makedirs(tspath)
    normsavepath = os.path.join(trainpath, ts, 'norm')
    sticsavepath = os.path.join(trainpath, ts, 'stic')
    os.makedirs(normsavepath)
    os.makedirs(sticsavepath)
    seedlist = np.random.randint(0, 1000, epochs)

    for ep in range(epochs):
        epo = str(ep)
        print('-------------epo:', ep)
        stloopname = stdict.keys()

        for k, v in stdict.items():
            if k not in badsticloop:
                tailname = k + '_' + epo
                data = get_loop_datanew(v, maxl=maxl, scaler=scaler)
                M_dtw = dtwmatrix_conversion.multivariate_dtwmatrix_fast(data, mdtwsize=dtwsize,
                                                                         dtwtpye=dtwtype, mdtwseed=seedlist[ep])
                savename = os.path.join(sticsavepath, tailname)
                np.save(savename, M_dtw)

        for k, v in normdict.items():
            if k not in badnormloop:
                tailname = k + '_' + epo
                data = get_loop_datanew(v, maxl=maxl, scaler=scaler)
                M_dtw = dtwmatrix_conversion.multivariate_dtwmatrix_fast(data, mdtwsize=dtwsize,
                                                                         dtwtpye=dtwtype, mdtwseed=seedlist[ep])
                savename = os.path.join(normsavepath, tailname)
                np.save(savename, M_dtw)
    print('Finish ISDB Training Conversion')

def isdb_conversion_test(dtwsize = 28, epochs = 1, maxl=400, scaler='minmax', dtwtype='fixed'):
    stdict, normdict, distdict, span = isdb.isdbload_filter()
    trainpath = r'/home/kexin/phdwork/ddcls2021/data/test/dtwmatrix_multiscale'
    if not os.path.exists(trainpath):
        os.makedirs(trainpath)
    ts = str(maxl)
    tspath = os.path.join(trainpath, ts)
    if not os.path.exists(tspath):
        os.makedirs(tspath)
    shutil.rmtree(tspath)
    os.makedirs(tspath)
    normsavepath = os.path.join(trainpath, ts, 'norm')
    sticsavepath = os.path.join(trainpath, ts, 'stic')
    os.makedirs(normsavepath)
    os.makedirs(sticsavepath)
    seedlist = np.random.randint(0, 1000, epochs)

    for ep in range(epochs):
        epo = str(ep)
        print('-------------epo:', ep)
        for k, v in stdict.items():
            if k not in badsticloop:
                tailname = k + '_' + epo
                data = get_loop_datanew(v, maxl=maxl, scaler=scaler)
                M_dtw = dtwmatrix_conversion.multivariate_dtwmatrix_fast(data, mdtwsize=dtwsize,
                                                                         dtwtpye=dtwtype, mdtwseed=seedlist[ep])
                savename = os.path.join(sticsavepath, tailname)
                np.save(savename, M_dtw)

        for k, v in normdict.items():
            if k not in badnormloop:
                tailname = k + '_' + epo
                data = get_loop_datanew(v, maxl=maxl, scaler=scaler)
                M_dtw = dtwmatrix_conversion.multivariate_dtwmatrix_fast(data, mdtwsize=dtwsize,
                                                                         dtwtpye=dtwtype, mdtwseed=seedlist[ep])
                savename = os.path.join(normsavepath, tailname)
                np.save(savename, M_dtw)
    print('Finish ISDB Test Conversion')

if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: isdb_conversion.py')
    print('---------------------------------------')

    dtwsize = 28
    epochs = 5
    isdbflag = False
    if isdbflag:
        timescale = [75, 100, 150, 200, 400]
        convtype = 'random'
        trainpath = r'/home/kexin/phdwork/ddcls2021/data/train/isdb_dtwmatrix_multiscale'
        if not os.path.exists(trainpath):
            os.makedirs(trainpath)
        else:
            shutil.rmtree(trainpath)
            os.makedirs(trainpath)
        for ts in timescale:
            print('timescale:', ts)
            isdb_conversion_train(dtwsize=dtwsize, epochs=epochs, maxl=ts, scaler='minmax', dtwtype=convtype)

    matlabflag = False
    if matlabflag:
        trainpath = r'/home/kexin/phdwork/ddcls2021/data/train/matlab_dtwmatrix_multiscale'
        if not os.path.exists(trainpath):
            os.makedirs(trainpath)
        else:
            shutil.rmtree(trainpath)
            os.makedirs(trainpath)
        convtype = 'random'
        timescale = [200, 400, 600]
        for ts in timescale:
            print('timescale:', ts)
            matlab_conversion_train(dtwsize=dtwsize, epochs=epochs, maxl=ts, dtwtype=convtype)

    testpath = r'/home/kexin/phdwork/ddcls2021/data/test/dtwmatrix_multiscale'
    if not os.path.exists(testpath):
        os.makedirs(testpath)
    else:
        shutil.rmtree(testpath)
        os.makedirs(testpath)
    convtype = 'random'
    timescale = [50, 60, 70, 75, 80, 90, 100, 110, 120, 125, 150, 175, 200, 300, 400, 600]
    for ts in timescale:
        print('timescale:', ts)
        isdb_conversion_test(dtwsize=dtwsize, epochs=int(epochs/epochs), maxl=ts, scaler='minmax', dtwtype=convtype)

    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')