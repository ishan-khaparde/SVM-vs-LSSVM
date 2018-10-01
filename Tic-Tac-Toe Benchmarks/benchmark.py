import pandas as pd
import time

from SVM import SVM
from LSSVM import LSSVM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

ttt = pd.read_csv('tic-tac-toe.csv')

ttt['result'] = ttt['result'].map({'negative': -1, 'positive': 1})

col = ['top_left','top_middle','top_right','mid_left','mid_mid','mid_right','bot_left','bot_mid','bot_right']
for c in col:
    label_encoder = LabelEncoder()
    label_encoder.fit(ttt[c])
    ttt[c] = label_encoder.transform(ttt[c])

labels = ttt['result']

del ttt['result']

labels = labels.as_matrix()
ttt = ttt.as_matrix()

X_train,X_test,y_train,y_test = train_test_split(ttt,labels,test_size = 0.3)
start_time = time.time()

clf = SVM(kernel = 'rbf',C = 1,gamma = 0.1)
clf.fit(X_train,y_train)
end_time = time.time()
pred = clf.predict(X_test)
print("SVM ACC",accuracy_score(y_test,pred))
print("SVM TIME",end_time-start_time)

start_time = time.time()
clf_ls = LSSVM(kernel = 'rbf', C = 1, gamma = 0.1)
clf_ls.fit(X_train,y_train)
end_time = time.time()
print("LSSVM TIME",end_time - start_time)
pred_ls = clf_ls.predict(X_test)
print("LSSVM acc",accuracy_score(y_test,pred_ls))


start_time = time.time()
clf_sk = SVC(kernel = 'rbf',C=1.5,gamma = 0.15)
clf_sk.fit(X_train,y_train)
end_time = time.time()
pred_sk = clf_sk.predict(X_test)
print("SVC accuracy",accuracy_score(y_test,pred_sk))
print("SVC time",end_time - start_time)
