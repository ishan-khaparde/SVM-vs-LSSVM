import numpy as np
from scipy.sparse import linalg
from sklearn.metrics import accuracy_score

class LSSVM:
    def __init__(self, kernel = 'linear', C = 1.0,gamma = 1.0):
        kernels = {
            'rbf':self.rbf,
            'poly':self.polynomial,
            'linear':self.linear
        }
        self.kernel = kernels[kernel]
        self.C = C
        self.gamma = 1.0
        

    def build_kernel_matrix(self, X, y):
        instances, dimensions = X.shape

        gram_matrix = np.zeros((instances,instances))

        for i in range(0, instances):
            for j in range(0, instances):
                gram_matrix[i, j] = self.kernel(X[i], X[j])
        return gram_matrix

    def fit(self, X, y):

        self.kernel_matrix = self.build_kernel_matrix(X,y)
        identity_matrix = np.identity(X.shape[0])
        A = np.zeros((X.shape[0]+1, X.shape[0]+1))
        B = np.ones(((X.shape[0]+1,1)))

        A[0][0] = 0
        A[0,1:X.shape[0]+1] = np.hstack((np.ones(X.shape[0])))
        A[1:X.shape[0]+1,0] = np.ones(X.shape[0])
        A[1:X.shape[0]+1,1:X.shape[0]+1] = self.kernel_matrix + identity_matrix / self.C

        B[0][0] = 0
        B[1:X.shape[0]+1,0] = y

        solution = np.linalg.solve(A,B)
        
        #print(solution.shape)
        self.bias = solution[:-1]
        
        solution = solution[:-1]
        self.support_vector_alphas = []
        self.support_vector_labels = []
        self.support_vectors = []
        for index,alpha in enumerate(solution):
            if(alpha > 1e-3):
                self.support_vector_alphas.append(alpha)
                self.support_vector_labels.append(y[index])
                self.support_vectors.append(X[index])

    def linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def polynomial(self, x1, x2):
        return (np.dot(x1, x2.T) ** self.d)
    
    def rbf(self,xi,xj):
        return np.exp(-self.gamma * np.linalg.norm(xi-xj)**2)
    
    def predict(self, X_test):
        predictions = []

        class_1_pred = 0
        class_2_pred = 0
        
        for instance in X_test:
            for index, sv in enumerate(self.support_vectors):
                prediction = np.sign(self.support_vector_alphas[index] * self.support_vector_labels[index] * self.kernel(sv,instance))
                if(prediction == 1):
                    class_1_pred +=1
                else:
                    class_2_pred +=1        
            if(class_1_pred > class_2_pred):
                predictions.append(1)
            else:
                predictions.append(-1)
            class_1_pred = 0
            class_2_pred = 0

        return np.array(predictions)

if __name__ == "__main__":

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2
    
    def split_train(X1, y1, X2, y2): 
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train


    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)
        #print(X_test.shape)
        #print(y_test.shape)
        clf = LSSVM(kernel = 'linear',C=1,gamma = 0.1)
        clf.fit(X_train, y_train)
        
        y_predict = clf.predict(X_test)
        print(len(y_predict))
        print(accuracy_score(y_test,y_predict))
        #plot_margin(X_train[y_train == 1], X_train[y_train == -1], clf)

#test_linear()
