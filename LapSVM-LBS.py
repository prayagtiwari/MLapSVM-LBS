import numpy as np
import sklearn
from sklearn.metrics.pairwise import rbf_kernel
from utils.alg import Alg
from utils.cons import Cons
from sklearn import preprocessing
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


class LapSVM():
    def __init__(self, gamma=0.1, c=0.03125, lamda2=0.7,lamda1=0.9, k=8):
        self.gamma = gamma  # 核矩阵gamma
        self.alg = Alg(c, lamda1, lamda2)
        self.cons = Cons(k)
    
    #单图
    def train(self,traindata, testdata, trainlabel):
        l = traindata.shape[0]
        u = testdata.shape[0]
        testlabel = np.zeros((u, 1))
        label = np.concatenate((trainlabel, testlabel), axis=0)
        data = np.concatenate((traindata, testdata), axis=0)
        K = rbf_kernel(data, data, self.gamma)
        L=self.cons.construct_LBS(data,label)
        score=self.alg.solve_alpha(L,K,trainlabel)
        pred_score = score[l:]
        return pred_score

def loaddata(filename1,filename2):
    train_data=sio.loadmat(filename1)
    test_data=sio.loadmat(filename2)
    GE_train=train_data['GE_1075']
    NMBAC_train=train_data['NMBAC_1075']
    Pse_train=train_data['PSSM_Pse_1075']
    trainlabel=train_data['label_1075']
    traindata = np.concatenate((GE_train,NMBAC_train,Pse_train),axis=1)
    GE_test=test_data['GE_186']
    NMBAC_test=test_data['NMBAC_186']
    Pse_test=test_data['PSSM_Pse_186']
    testlabel=test_data['label_186']
    testdata = np.concatenate((GE_test,NMBAC_test,Pse_test),axis=1)
    data =np.concatenate((traindata,testdata),axis=0)
    min_max_scaler=preprocessing.MinMaxScaler()
    data=min_max_scaler.fit_transform(data)
    traindata=data[:1075,:]
    testdata=data[1075:,:]
    traindata,trainlabel = sklearn.utils.shuffle(traindata, trainlabel,random_state=1)
    testdata,testlabel =sklearn.utils.shuffle(testdata,testlabel,random_state=1)
    traindata,x=np.unique(traindata,axis=0,return_index=True)
    ls=trainlabel[x]
    return traindata,ls,testdata,testlabel



if __name__ == '__main__':
    lapsvm = LapSVM(gamma=0.125,c=8,lamda2=1,lamda1=0.3)
    traindata,trainlabel,testdata,testlabel=loaddata("dataset/PDB1075_feature.mat","dataset/PDB186_feature.mat")
    predlabel=lapsvm.train(traindata,testdata,trainlabel)
    auc=roc_auc_score(testlabel,predlabel)
    predlabel[np.where(predlabel>=0)]=1
    predlabel[np.where(predlabel<0)]=0
    testlabel[np.where(testlabel<0)]=0
    tn, fp, fn, tp = confusion_matrix(testlabel,predlabel).ravel()
    SN=tp/(tp+fn)
    SP=tn/(tn+fp)
    acc=(tp+tn)/(tp+fn+tn+fp)
    mcc=matthews_corrcoef(testlabel,predlabel)
    print(acc)
