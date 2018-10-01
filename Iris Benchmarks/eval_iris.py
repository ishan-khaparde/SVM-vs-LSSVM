import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC 
from SVM import SVM
from LSSVM import LSSVM

if __name__ == "__main__":
    iris = pd.read_csv('iris - setosa.csv')
  
    labels = iris.species
    del iris['species']

    labels = labels.as_matrix()

    features = iris.as_matrix()
    
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size = 0.333)

    svm_start = time.time()
    clf = SVM(kernel = 'linear', C = 0.01)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)  
    svm_end = time.time()
    print("SVM",accuracy_score(y_test,pred)*100)
    print("It took",svm_end - svm_start,"seconds for SVM to train")
    lssvm_start = time.time()
    clf = LSSVM(kernel = 'linear', C = 0.01)
    clf.fit(X_train,y_train)
    lssvm_end = time.time()
    pred_ls = clf.predict(X_test)

    pred_ls_train = clf.predict(X_train)
    print("LS TEST",accuracy_score(y_test,pred_ls)*100)
    print("LS TRAIN",accuracy_score(y_train,pred_ls_train)*100)
    print("It took",lssvm_end - lssvm_start,"seconds for LS-SVM to train" )

    start_time = time.time()
    clf_sk = SVC(kernel = 'linear', C = 0.01)
    clf_sk.fit(X_train,y_train)
    end_time = time.time()
    pred = clf_sk.predict(X_test)
    print(accuracy_score(y_test,pred)*100)
    print(end_time-start_time)
#SVM -> 90%/ linear / C = 0.01, acc : 94%/ 0.027 seconds
#LS-SVM -> 74% / linear / C = 0.01/ acc: 0.015 seconds 
    
