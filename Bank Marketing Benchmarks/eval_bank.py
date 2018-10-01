import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from SVM import SVM
from LSSVM import LSSVM

bank = pd.read_csv('bank-modified.csv')

bank['y'] = bank['y'].map({'no': -1, 'yes': 1})

col = ['job','marital','education','default','housing','loan','contact','month','poutcome']
for c in col:
    label_encoder = LabelEncoder()
    label_encoder.fit(bank[c])
    bank[c] = label_encoder.transform(bank[c])

labels = bank.y

del bank['y']
bank = bank.as_matrix()
labels = labels.as_matrix()
labels = labels[:1000]
bank = bank[:1000]
X_train,X_test,y_train,y_test = train_test_split(bank,labels,test_size = 0.3)
start_time = time.time()

clf = SVM(kernel = 'rbf',C = 0.1,gamma = 1) #74.33%/rbf/891.3169 seconds/ C = 0.1 gamma = 0.01
#clf = SVC(kernel = 'rbf',C = 0.001,gamma = 0.01,verbose = True)#87.875/rbf/same hp/0.5 seconds
#clf_ls = LSSVM(kernel = 'rbf', C = 0.1, gamma = 1)

print("Now beginning training...")
clf.fit(X_train,y_train)
#clf_ls.fit(X_train,y_train)

end_time = time.time()

print("Done Training!")
pred = clf.predict(X_test)
#pred_ls = clf_ls.predict(X_test)
print(accuracy_score(y_test,pred)* 100)
#print(accuracy_score(y_test,pred_ls)* 100)
print("It took",end_time-start_time,"seconds to train for Bank Marketing dataset")