from __future__ import division, print_function
import csv, os, sys
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from SVM import SVM
from LSSVM import LSSVM
filepath = os.path.dirname(os.path.abspath(__file__))

def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'r',encoding = 'utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)

def gen_lin_separable_data():
     mean1 = np.array([0, 2])
     mean2 = np.array([2, 0])
     cov = np.array([[0.8, 0.6], [0.6, 0.8]])
     X1 = np.random.multivariate_normal(mean1, cov, 1200000)
     y1 = np.ones(len(X1))
     X2 = np.random.multivariate_normal(mean2, cov, 1200000)
     y2 = np.ones(len(X2)) * -1
     return X1, y1, X2, y2
def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 30000)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 30000)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 30000)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 30000)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def split_train(X1, y1, X2, y2):

    X1_train = X1[:25000]
    y1_train = y1[:25000]
    X2_train = X2[:25000]
    y2_train = y2[:25000]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train


def split_test(X1, y1, X2, y2):

    X1_test = X1[25001:29999]
    y1_test = y1[25001:29999]
    X2_test = X2[25001:29999]
    y2_test = y2[25001:29999]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

def main(filename='data/iris-versicolor.txt'):
    # Load data
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    data = data.astype(float)

    # Split data
    X, y = data[:,0:-1], data[:,-1].astype(int)
    print(X)
    print(y)
    # Initialize model
    model = LSSVM(kernel = 'linear', C = 0.01) #kernel = linear, C = 0.01 96%
    model.fit(X,y)
    # Fit model
    #support_vectors, iterations, training_time = model.fit(X, y)

    # Support vector count
    #sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(X)

    # Calculate accuracy
    #acc = calc_acc(y, y_hat)
    acc = accuracy_score(y,y_hat)

    #print("Support vector count: %d" % (sv_count))
    #print("bias:\t\t%.3f" % (model.b))
    #print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc*100),"%")
    #print("Kernel", model.kernel_type)
    #print("C",model.C)
    #print("Converged after %d iterations" % (iterations))
    print("Model took %f seconds to train" % (training_time))

if __name__ == '__main__':
    
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("")
        print("Trains a support vector machine.")
        print("Usage: %s FILENAME C kernel eps" % (sys.argv[0]))
        print("")
        print("FILENAME: Relative path of data file.")
        print("C:        Value of regularization parameter C.")
        print("kernel:   Kernel type to use in training.")
        print("          'linear' use linear kernel function.")
        print("          'quadratic' use quadratic kernel function.")
        print("eps:      Convergence value.")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['filename'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['C'] = float(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['kernel_type'] = sys.argv[3]
        if len(sys.argv) > 4:
            kwargs['epsilon'] = float(sys.argv[4])
        if len(sys.argv) > 5:
            sys.exit("Not correct arguments provided. Use %s -h for more information"
                     % (sys.argv[0]))
        main(**kwargs)
    
    '''
    iris = pd.read_csv('iris copy.csv')
    iris = iris[iris.species != 'virginica']
   # iris = iris.drop(labels = "ID",axis = 1)
    label_encoder = LabelEncoder()
    label_encoder.fit(iris['species'])
    iris['species'] = label_encoder.transform(iris['species'])
    
    labels = iris['species']
    del iris['species']
    features = iris.as_matrix().astype(int)
    labels = labels.as_matrix().astype(int)
    #print(features)
    #print(labels)

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size = 0.1)

    clf = SVM(kernel_type = 'rbf',C = 0.01,gamma = 0.1)
    clf.fit(X_train,y_train)
    predictions = []
    for i in range(0,50):
        predictions.append(accuracy_score(y_test,clf.predict(X_test)))
    #pred = clf.predict(X_test)
    #pred_train = clf.predict(X_train)
    #print(accuracy_score(y_test,pred)*100)
    #print(accuracy_score(y_train,pred_train)*100)
    print(np.mean(predictions)*100) # 90% mean, rbf, C = 0.01, gamma = 0.1
    
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    x1,y1,x2,y2 = gen_non_lin_separable_data()
    X_train,y_train = split_train(x1,y1,x2,y2)
    X_test,y_test = split_test(x1,y1,x2,y2)

    clf = SVM(kernel_type='linear',C=0.1,gamma = 0.02,d = 4)
    
    clf_sk = SVC(kernel = 'linear',C=0.1)
    clf_sk.fit(X_train,y_train)
    pred_sk = clf_sk.predict(X_test)
    print(X_train.shape)
    print("SKLEARN",accuracy_score(pred_sk,y_test)*100)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    pred_t = clf.predict(X_train)
    print("APNA",accuracy_score(y_test,pred)*100)
    #print(accuracy_score(y_train,pred_t)*100)
    '''