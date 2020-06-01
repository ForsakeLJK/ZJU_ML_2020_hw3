import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful
    # YOUR CODE HERE
    
    # begin answer
    
    P = x_train.shape[1]
    N = x_train.shape[0]
    N_test = x.shape[0]
    y = np.zeros((N_test, ))

    for i in range(N_test):
        tmp1 = np.abs(x_train - x[i, :])
        tmp2 = np.square(tmp1)
        # shape (N, )
        distances = np.sum(tmp2, axis=1)
        # find k nearest neighbors, shape (k, )
        nearestNeighborsIndices = np.argsort(distances)[:k]
        y[i] = np.argmax(np.bincount(y_train[nearestNeighborsIndices]))

    # end answer

    return y
