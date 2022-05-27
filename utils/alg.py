from libsvm.svm import *
from libsvm.svmutil import *
import math
import numpy as np


class Alg():
    def __init__(self, c=0.03125, lamda2=1, lamda1=1,e=2,iter_num=4, q=5):
        self.c = c                           #l2正则项
        self.lamda1 = lamda1                 #经验损失
        self.lamda2 = lamda2                 #图正则项
        self.e = e
        self.iter_num = iter_num             #算法迭代次数
        self.q = q                           #图的数量

    #计算融合后的图
    def constructL(self, L1, w):
        L = np.zeros(L1[0].shape)
        for i in range(len(w)):
            L = L+w[i]*L1[i]
        return L
    
    
    def solve_alpha(self, L, K, trainlabel):
        l = trainlabel.shape[0]
        I = np.eye(K.shape[0])
        G = np.linalg.inv(2*self.lamda1*I+2*self.lamda2*L@K)
        Gram = K@G
        Gram = Gram[0:l, 0:l]
        option = '-s 3 -t 4 -c '+str(self.c)+' -p 0.01 -q' 

        model = svm_train(trainlabel.flatten(), add_index(Gram), option)
        b = np.matrix(read_B(model, l)).T
        alpha = G[:, :l]@b
        score = K@alpha
        return score

    def solve_w(self, L1, score, w):
        for i in range(self.q):
            w[i] = math.pow(1/(score.T@L1[i]@score), 1/(self.e-1))
        s = sum(w)
        for i in range(self.q):
            w[i] /= s
        return w

    def train(self, K, L1, trainlabel):
        w = [0., 0., 0., 0., 1]
        L = self.constructL(L1, w)
        score = self.solve_alpha(L, K, trainlabel)
        for i in range(self.iter_num):
            print(w)
            w = self.solve_w(L1, score, w)
            L = self.constructL(L1, w)
            score = self.solve_alpha(L, K, trainlabel)
        return score


def read_B(model, l):
    index = np.array(model.get_sv_indices())
    index = index-1
    value = np.array(model.get_sv_coef()).T
    b = np.zeros(l)
    b[index] = value
    return b


def add_index(Gram):
    l = Gram.shape[0]
    a = np.matrix(np.array(range(1, l+1))).T
    return np.concatenate((a, Gram), axis=1)
