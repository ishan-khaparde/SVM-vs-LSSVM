import numpy as np
import time
from SVM import SVM
from LSSVM import LSSVM

def gen_non_lin_separable_data(numpoints):
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, numpoints)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, numpoints)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, numpoints)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, numpoints)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def split_train(X1, y1, X2, y2,num):

    X1_train = X1[:num]
    y1_train = y1[:num]
    X2_train = X2[:num]
    y2_train = y2[:num]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train


def split_test(X1, y1, X2, y2,num):

    X1_test = X1[num:]
    y1_test = y1[num:]
    X2_test = X2[num:]
    y2_test = y2[num:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

if __name__ == '__main__':

    numpoints = 1000
    X1,y1,X2,y2 = gen_non_lin_separable_data(numpoints)
    X_train,y_train = split_train(X1,y1,X2,y2,int(numpoints*0.5))
    X_test,y_test = split_test(X1,y1,X2,y2,int(numpoints*0.5))

    svm = SVM(kernel = 'rbf',C=0.01,gamma = 0.1)
    start_time = time.time()
    svm.fit(X_train,y_test)
    end_time = time.time()

    print("IT TOOK",end_time-start_time,"seconds to train.")
    print("BENCHMARK")