"""
    Author: Lasse Regin Nielsen
"""

from __future__ import division, print_function
import os
import numpy as np
import random as rnd
import time 
filepath = os.path.dirname(os.path.abspath(__file__))

class SVM():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """
    def __init__(self, max_iter=10000, gamma = 1.0,kernel='rbf', C=1.0, epsilon=0.001,d = 2):
        self.kernels = {
            'linear' : self.linear,
            'poly' : self.polynomial,
            'rbf' : self.rbf
        }
        self.max_iter = max_iter
        self.kernel_type = self.kernels[kernel]
        self.C = C
        self.epsilon = epsilon
        self.d = d
        self.gamma = gamma
        

    def fit(self, X, y):
        # Initialization
        start_time = time.time()
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernel_type
        #print(kernel)
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.generate_random_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.computeWeights(alpha, y, X)
                self.b = self.computeBias(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return
        # Compute final model parameters
        self.b = self.computeBias(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.computeWeights(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        end_time = time.time()
        return support_vectors, count , end_time - start_time
    def predict(self, X):
        #return self.helper(X, self.w, self.b)
        return np.sign(np.dot(self.w.T , X.T) + b).astype(int)
    def computeBias(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def computeWeights(self, alpha, y, X):
        return np.dot(alpha * y, X)
    # Prediction
    def helper(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.helper(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
    def generate_random_int(self, a,b,z):
        i = z
        counter=0
        while i == z and counter<1000:
            i = rnd.randint(a,b)
            counter=counter+1
        return i

    # Define kernels
    def linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def polynomial(self, x1, x2):
        return (np.dot(x1, x2.T) ** self.d)
    def rbf(self,xi,xj):
        return np.exp(-self.gamma * np.linalg.norm(xi-xj)**2)