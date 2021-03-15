import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans
import pickle

def kmeans_train(data, N_class=2):
    '''
    K-means clustering
    :param data: 2d array, size(N, M), N: number of data sets; M: number of features
    :param N_class: Desired number of clusters or classes
    :return: a trained kmeans clf
    '''
    clusterdata = data
    kmeans = KMeans(n_clusters=N_class, random_state=0).fit(clusterdata)
    with open(r'/home/kexin/phdwork/ddcls2021/code/saves/kmeansclf.pickle', 'wb') as f:
        pickle.dump(kmeans, f)
    return kmeans

def kmeans_test(test_data):
    with open(r'/home/kexin/phdwork/ddcls2021/code/saves/kmeansclf.pickle', 'rb') as f:
        kmeans_reload = pickle.load(f)
    pred = kmeans_reload.predict(test_data)
    return pred


def fcm_train(data, N_class=2):
    '''
    FCM clustering
    :param data: 2d array, size(N, M), N: number of data sets; M: number of features
    :param N_class:
    :return: cntr: Cluster centers
             u_orig: Final fuzzy c-partitioned matrix
    '''
    clusterdata = data.T
    cntr, u_orig, _, _, _, _, _ = fuzz.cmeans(clusterdata, c=N_class, m=2, error=0.005, maxiter=1000)
    savecntr = r'/home/kexin/phdwork/ddcls2021/code/saves/cntr.npy'
    np.save(savecntr, cntr)
    saveu = r'/home/kexin/phdwork/ddcls2021/code/saves/u.npy'
    np.save(saveu, u_orig)
    return cntr, u_orig

def fcm_test(test_data):
    clusterdata = test_data.T
    cntr = np.load(r'/home/kexin/phdwork/ddcls2021/code/saves/cntr.npy')
    u, u0, d, jm, p, fpc = fuzz.cmeans_predict(clusterdata, cntr, 2, error=0.005, maxiter=1000)
    pred = []
    for i in range(u.shape[1]):
        idx = np.where(u[:, i] == max(u[:, i]))[0][0]
        pred.append(idx)
    return np.array(pred)