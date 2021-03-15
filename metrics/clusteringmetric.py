import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from typing import Optional
from scipy.optimize import linear_sum_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def acc(y_true, y_predicted):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    cluster_number = (max(y_predicted.max(), y_true.max()) + 1)
    # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    #reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return accuracy