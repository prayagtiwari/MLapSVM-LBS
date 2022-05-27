import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity

class Cons():
    def __init__(self, k=2):
        self.k = k
        
    def construct_g_rbf(self,data):
        W=rbf_kernel(data, data, 0.1)
        x=np.sum(W,axis=1)
        D=np.diag(x)
        L=D-W
        L=fractional_matrix_power(D,-0.5)@L@fractional_matrix_power(D,-0.5)
        return L
    
    def construct_g_cos(self,data):
        W=cosine_similarity(data)
        x=np.sum(W,axis=1)
        D=np.diag(x)
        L=D-W
        L=fractional_matrix_power(D,-0.5)@L@fractional_matrix_power(D,-0.5)
        return L

    def construct_LBS(self, data, label):
        W = euclidean_distances(data, data)
        a = data.shape[0]
        l = np.zeros((self.k+1, a))
        for i in range(a):
            b = np.argpartition(W[:, i], self.k)[:self.k+1]
            l[:, i] = W[:, i][b]
        s = np.matrix(np.sum(np.array(l), axis=0)/self.k)
        W = np.exp(-np.square(W)/(s.T@s))
        W = np.array(W)
        label = label@label.T

        W[np.where(label == 1)] = 1/(3*np.sqrt(10/9-W[np.where(label == 1)]))
        W[np.where(label == -1)] = 1/(3*np.sqrt(1/W[np.where(label == -1)]))
        W[np.where(label == 0)] = 2/(3*(np.sqrt(1 -
                                                W[np.where(label == 0)])+np.sqrt(1/W[np.where(label == 0)])))

        x = np.sum(np.array(W), axis=1)
        D = np.diag(x)

        L = D-W
        L = fractional_matrix_power(D, -0.5)@L@fractional_matrix_power(D, -0.5)
        return L

    def intergrate(self, data, label):
        ls = [2,4,8,16,32]
        L1 = []
        for i in ls:
            self.k = i
            L = self.construct_LBS(data, label).tolist()
            L1.append(L)
        return np.array(L1)