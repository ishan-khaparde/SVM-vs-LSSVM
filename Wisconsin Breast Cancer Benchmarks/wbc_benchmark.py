import pandas as pd 
import numpy as np 
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from SVM import SVM 
from LSSVM import LSSVM 


if __name__ == '__main__':

    wbc = pd.read_csv('wbc.csv')
    wbc['diagnosis'] = wbc['diagnosis'].map({'M': -1, 'B': 1})

    del wbc['id']
    del wbc['Unnamed: 32']
    labels = wbc['diagnosis']
    del wbc['diagnosis']

    wbc = wbc.as_matrix()
    labels = labels.as_matrix()
    X_train , X_test , y_train, y_test = train_test_split(wbc,labels,test_size = 0.33)
    start_time = time.time()
    clf = SVM(kernel = 'rbf',C = 0.01, gamma = 0.01)
    clf.fit(X_train,y_train)
    end_time = time.time()
    pred = clf.predict(X_test)
    print("SVM accuracy",accuracy_score(y_test,pred)*100)
    print("It took",end_time-start_time,"seconds for SVM to train")

    start_time = time.time()
    clf = SVC(kernel = 'rbf',C = 0.01, gamma = 0.01)
    clf.fit(X_train,y_train)
    end_time = time.time()
    pred = clf.predict(X_test)
    print("SVC accuracy",accuracy_score(y_test,pred)*100)
    print("It took",end_time-start_time,"seconds to train SVC")

    start_time = time.time()
    clf = LSSVM(kernel = 'rbf',C = 0.01, gamma = 0.01)
    clf.fit(X_train,y_train)
    end_time = time.time()
    pred = clf.predict(X_test)
    print("LS-SVM",accuracy_score(y_test,pred)*100)
    print("It took",end_time-start_time,"seconds for LS SVM  to train")