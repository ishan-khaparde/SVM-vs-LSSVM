from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from SVM import SVM
from LSSVM import LSSVM
import time

train = fetch_20newsgroups(subset = 'train',categories = ['talk.politics.misc','sci.electronics'])
test = fetch_20newsgroups(subset = 'test',categories = ['talk.politics.misc','sci.electronics'])

y_train = train.target
y_test = test.target
#The labels are 0, 1 but we need them to be -1 or 1. 0,1 mapping will yield erroneous results. 
for i,x in enumerate(y_train):
    if x == 0:
        y_train[i] = -1
for i,x in enumerate(y_test):
    if x == 0:
        y_test[i] = -1
#Feature engineering and preprocessing
vector_transformer = TfidfVectorizer(sublinear_tf = True, stop_words = 'english')
#Use the Tf-idf representation for text 
X_train = vector_transformer.fit_transform(train.data)

X_test = vector_transformer.transform(test.data)
start_time = time.time()

print("Training SVC started")
clf = SVC(kernel = 'linear',C = 1)
clf.fit(X_train,y_train)
end_time = time.time()
print("SVC took",end_time-start_time)
pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy SVC ",accuracy_score(y_test,pred)*100,"%")
print("Now Training SVM")
clf_mine = SVM(kernel = 'linear', C = 1)

start_time = time.time()

X_train = X_train.toarray()
X_test = X_test.toarray()

print(X_train.shape,X_test.shape)
clf_mine.fit(X_train,y_train)

end_time = time.time()
pred_min = clf_mine.predict(X_test)
print("Accuracy ",accuracy_score(y_test,pred_min)*100,"%")
print("It took",end_time - start_time,"seconds")
print("Now training LS SVM")
clf_mine_ls = LSSVM(kernel = 'linear', C = 0.1)
start_time = time.time()
clf_mine_ls.fit(X_train,y_train)
end_time = time.time()
print("LS SVM acc",accuracy_score(y_test,pred)*100,"%")
print("LS SVM time",end_time-start_time)

#SVC -> 97.58%, 29.768 seconds to train
#SVM -> 95.87%, 108.35 seconds to train
#LS-SVM -> 55% , 17.011 seconds to train