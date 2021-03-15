import numpy as np
import pandas as pd
import random
import os
import re
import time
from sklearn import preprocessing
from scipy import signal

def isdb_prepare():
    '''
    :param testnum:
    '''
    #
    # ------------读取粘滞回路数据和回路名: stictiondata/stictionloop---------------
    filepath = r'/home/kexin/data/isdb/stiction_loops'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    stictiondata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        stictiondata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                      engine='python',
                                                                      names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    stictionloop = list(map(lambda x: x[0:-3], list(stictiondata.keys())))
    stictionloop = list(set(stictionloop))
    print('Number of Stiction Loops: ', len(stictionloop))

    # 读取正常回路数据和回路名:normaldata/nomalloop
    filepath = r'/home/kexin/data/isdb/normal_loops'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    normaldata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        normaldata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                    engine='python',
                                                                    names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    normalloop = list(map(lambda x: x[0:-3], list(normaldata.keys())))
    normalloop = list(set(normalloop))
    print('Number of Normal Loops: ', len(normalloop))

    # 读取正常回路数据和回路名:normaldata/nomalloop
    filepath = r'/home/kexin/data/isdb/disturbance_loops'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    disturbancedata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        disturbancedata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                     engine='python',
                                                                     names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    disturbanceloop = list(map(lambda x: x[0:-3], list(disturbancedata.keys())))
    disturbanceloop = list(set(disturbanceloop))
    print('Number of Disturbance Loops: ', len(disturbanceloop))

    # ----读取回路信息，包括回路样本数量和采样周期: loop1info-------
    filepath = r'/home/kexin/data/isdb/loopinfo'
    filelist = os.listdir(filepath)
    loopinfo = pd.DataFrame(columns=['loopname', 'ts', 'samplenum'])
    pattern1 = re.compile('Ts: \d.+|Ts: \d+')
    pattern2 = re.compile(r'\d.+|\d+')
    pattern3 = re.compile(r'PV: \[\d+')
    pattern4 = re.compile(r'\d+')
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        with open(data_path, 'r') as f:
            tstemp = f.readlines()
            numtemp = tstemp.copy()
            # 查询ts
            tstemp = list(map(lambda x: pattern1.search(x), tstemp))
            tstemp = list(filter(None, tstemp))[0]
            ts = pattern2.search(tstemp.group(0)).group(0)
            # 查询样本数量
            numtemp = list(map(lambda x: pattern3.search(x), numtemp))
            numtemp = list(filter(None, numtemp))[0]
            num = pattern4.search(numtemp.group(0)).group(0)
        loopinfo.loc[i, 'loopname'] = os.path.basename(data_path)[0:-4]
        loopinfo.loc[i, 'ts'] = ts
        loopinfo.loc[i, 'samplenum'] = num
    loopinfo.to_csv(r'/home/kexin/phdwork/ddcls2021/data/loopinfo.csv')

    # 将粘滞回路数据按照dict格式存放：stictiondata
    datalist = list(stictiondata.keys())
    # stictionloop
    stictiondatadict = {}
    for loop in stictionloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = stictiondata[loop + '.OP'].values.flatten()
        df.pv = stictiondata[loop + '.PV'].values.flatten()
        df.sp = stictiondata[loop + '.SP'].values.flatten()
        stictiondatadict[loop] = df.values
    print('Number of Siction Loops:', len(list(stictiondatadict.keys())))
    stictioninfo = loopinfo.loc[loopinfo['loopname'].isin(stictionloop)]

    # 将正常回路数据按照dict格式存放： normaldatadict
    datalist = list(normaldata.keys())
    # stictionloop
    normaldatadict = {}
    for loop in normalloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = normaldata[loop + '.OP'].values.flatten()
        df.pv = normaldata[loop + '.PV'].values.flatten()
        df.sp = normaldata[loop + '.SP'].values.flatten()
        normaldatadict[loop] = df.values
    print('Number of Normal Loops:', len(list(normaldatadict.keys())))
    normalinfo = loopinfo.loc[loopinfo['loopname'].isin(normalloop)]

    # 将扰动回路数据按照dict格式存放： disturbancedatadict
    datalist = list(disturbancedata.keys())
    # stictionloop
    disturbancedatadict = {}
    for loop in disturbanceloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = disturbancedata[loop + '.OP'].values.flatten()
        df.pv = disturbancedata[loop + '.PV'].values.flatten()
        df.sp = disturbancedata[loop + '.SP'].values.flatten()
        disturbancedatadict[loop] = df.values
    print('Number of Distrubance Loops:', len(list(disturbancedatadict.keys())))
    disturbanceinfo = loopinfo.loc[loopinfo['loopname'].isin(disturbanceloop)]
    # Save as np
    np.save(r'/home/kexin/phdwork/ddcls2021/data/sticdict.npy', stictiondatadict)
    np.save(r'/home/kexin/phdwork/ddcls2021/data/normdict.npy', normaldatadict)
    np.save(r'/home/kexin/phdwork/ddcls2021/data/distdict.npy', disturbancedatadict)

def isdbload():
    path = r'/home/kexin/phdwork/ddcls2021/data/sticdict.npy'
    stdict = np.load(path, allow_pickle=True).item()
    path = r'/home/kexin/phdwork/ddcls2021/data/normdict.npy'
    normdict = np.load(path, allow_pickle=True).item()
    path = r'/home/kexin/phdwork/ddcls2021/data/distdict.npy'
    distdict = np.load(path, allow_pickle=True).item()
    infopath = r'/home/kexin/phdwork/ddcls2021/data/loopinfo.csv'
    info = pd.read_csv(infopath, index_col = 0)
    return stdict, normdict, distdict, info

def isdb_filter():
    stdict, normdict, distdict, info = isdbload()
    spanpath = '/home/kexin/phdwork/ddcls2021/data/isdbspan.csv'
    span = pd.read_csv(spanpath)
    span = span.set_index('loopname')
    stdictnew = {}
    for k, v in stdict.items():
        start = span.loc[k].values[0]
        end = span.loc[k].values[1]
        start = int(start.split('/')[0])
        end = int(end.split('/')[0])
        #print(start, end)
        stdictnew[k] = v[start:end, :]
    normdictnew = {}
    for k, v in normdict.items():
        start = span.loc[k].values[0]
        end = span.loc[k].values[1]
        start = int(start.split('/')[0])
        end = int(end.split('/')[0])
        #print(start, end)
        normdictnew[k] = v[start:end, :]
    distdictnew = {}
    for k, v in distdict.items():
        start = span.loc[k].values[0]
        end = span.loc[k].values[1]
        start = int(start.split('/')[0])
        end = int(end.split('/')[0])
        #print(start, end)
        distdictnew[k] = v[start:end, :]

    np.save(r'/home/kexin/phdwork/ddcls2021/data/sticdictfilter.npy', stdictnew)
    np.save(r'/home/kexin/phdwork/ddcls2021/data/normdictfilter.npy', normdictnew)
    np.save(r'/home/kexin/phdwork/ddcls2021/data/distdictfilter.npy', distdictnew)

def isdbload_filter():
    path = r'/home/kexin/phdwork/ddcls2021/data/sticdictfilter.npy'
    stdict = np.load(path, allow_pickle=True).item()
    path = r'/home/kexin/phdwork/ddcls2021/data/normdictfilter.npy'
    normdict = np.load(path, allow_pickle=True).item()
    path = r'/home/kexin/phdwork/ddcls2021/data/distdictfilter.npy'
    distdict = np.load(path, allow_pickle=True).item()
    spanpath = '/home/kexin/phdwork/ddcls2021/data/isdbspan.csv'
    span = pd.read_csv(spanpath,index_col = 0)
    return stdict, normdict, distdict, span

def matlab_prepare():
    sticpath = '/home/kexin/data/isdb/matlab/valvedata.csv'
    df = pd.read_csv(sticpath)
    idx = np.array(range(0, len(df) ,1))
    stdata = df.iloc[idx, 0:2000]

    weakpath = '/home/kexin/data/isdb/matlab/weakdata.csv'

    sticpath = '/home/kexin/data/isdb/matlab/normaldata.csv'
    df = pd.read_csv(sticpath)
    idx = np.array(range(0, len(df), 3))
    normdata = df.iloc[idx, 0:2000]
    return stdata, normdata


if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: isdb_dtwmatrix.py')
    print('---------------------------------------')

    #### Main_func
    isdb_prepare()
    stdict, normdict, distdict, info = isdbload()
    isdb_filter()
    stdictnew, normdictnew, distdictnew, span = isdbload_filter()
    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')

