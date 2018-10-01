import numpy as np
import scipy
from scipy.sparse import linalg
from sklearn.metrics import accuracy_score

class LSSVM:
    def __init__(self, kernel = 'linear', C = 1.0,gamma = 1.0, d = 2.0):
        kernels = {
            'rbf':self.rbf,
            'poly':self.polynomial,
            'linear':self.linear
        }
        
        self.kernel = kernels[kernel]
        self.C = C
        self.gamma = 1.0
        self.d = d
        
    #Build the gram matrix
    def build_kernel_matrix(self, X, y):
        instances, dimensions = X.shape

        gram_matrix = np.zeros((instances,instances))
        #computing the gram matrix, involves going over the dataset and computing pairwise kernel function
        for i in range(0, instances):
            for j in range(0, instances):
                
                gram_matrix[i, j] = self.kernel(X[i], X[j])
        return gram_matrix

    def fit(self, X, y):

        self.kernel_matrix = self.build_kernel_matrix(X,y)
        identity_matrix = np.identity(X.shape[0])
        #We wish to solve Ax = B, so we begin by defining the matrices A, B
        A = np.zeros((X.shape[0]+1, X.shape[0]+1))
        B = np.ones(((X.shape[0]+1,1)))

        A[0][0] = 0
        A[0,1:X.shape[0]+1] = np.hstack((np.ones(X.shape[0])))
        A[1:X.shape[0]+1,0] = np.ones(X.shape[0])
        A[1:X.shape[0]+1,1:X.shape[0]+1] = self.kernel_matrix + identity_matrix / self.C
        
        #B is a column vector. 
        B[0][0] = 0
        B[1:X.shape[0]+1,0] = y

        #The numpy and scipy package uses Conjugate Gradient to solve system of linear equations
        solution = np.linalg.solve(A,B)
        
        #print(solution.shape)
        self.bias = solution[:-1]
        
        solution = solution[:-1]
        self.support_vector_alphas = []
        self.support_vector_labels = []
        self.support_vectors = []
        for index,alpha in enumerate(solution):
            #We must have a threshold for alpha, so we are pruning the alpha vector to an extent, however setting such a threshold is not a good idea, better algorithms have been proposed.
            if(alpha > 1e-3): 
                self.support_vector_alphas.append(alpha)
                self.support_vector_labels.append(y[index])
                self.support_vectors.append(X[index])
    #define kernels
    def linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def polynomial(self, x1, x2):
        return (np.dot(x1, x2.T) ** self.d)
    
    def rbf(self,xi,xj):
        return np.exp(-self.gamma * np.linalg.norm(xi-xj)**2)

    def predict(self,X_test):
        predictions = []
        
        for instance in X_test:

            for index, sv in enumerate(self.support_vectors):
                prediction = np.sum(self.support_vector_alphas[index] * self.support_vector_labels[index] * self.kernel(sv,instance) + self.bias)
                
            predictions.append(np.sign(prediction).astype(int))

        return np.array(predictions)