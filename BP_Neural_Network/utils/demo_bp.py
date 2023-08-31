from cv2 import normalize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pair_confusion_matrix

class bp_neural:
    imgs = np.empty((0,0))
    labels = np.empty((0,0))
    normalize = None
    onehot = None

    def __init__(self,train_imgs,train_labels,n_hidden=100,method='softmax'):
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.n_hidden = n_hidden
        self.method = method
        self.o_train_labels = train_labels.copy()

    def normalize_images(self):
        if self.normalize:
            print('Already normalized')
            return None
        self.imgs = self.train_imgs.astype('float32')
        self.imgs = self.imgs / 255.0
        self.imgs = self.imgs.reshape(self.imgs.shape[0],-1).T
        self.normalize = True

    def one_hot_labels(self):
        if self.onehot:
            print('Already one-hot')
            return None
        self.labels = self.train_labels
        self.labels = np.zeros((self.labels.shape[0],self.labels.max()+1))
        self.labels[np.arange(0,self.labels.shape[0]),self.train_labels] = 1
        self.labels = self.labels.T
        self.onehot = True

    def init_params(self):
        if self.normalize and self.onehot:
            self.w1 = np.random.randn(self.imgs.shape[0],self.n_hidden)
            self.b1 = np.zeros((self.n_hidden,1))
            self.w2 = np.random.randn(self.n_hidden,self.labels.shape[0])
            self.b2 = np.zeros((self.labels.shape[0],1))
        else:
            self.normalize_images()
            self.one_hot_labels()
            self.init_params()

    def front_prop(self):
        self.z1 = np.dot(self.w1.T,self.imgs) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.w2.T,self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)

    def back_prop(self):
        self.da2 = self.labels / self.a2
        self.dz2 = self.da2 * self.sigmoid_d(self.z2)
        self.dw2 = -1/self.imgs.shape[1] * np.dot(self.a1,self.dz2.T)
        self.db2 = -1/self.imgs.shape[1] * np.sum(self.dz2,axis=1,keepdims=True)
        self.da1 = np.dot(self.dz2.T,self.w2.T).T
        self.dz1 = self.da1 * self.sigmoid_d(self.z1)
        self.dw1 = -1/self.imgs.shape[1] * np.dot(self.dz1,self.imgs.T).T
        self.db1 = -1/self.imgs.shape[1] * np.sum(self.dz1,axis=1,keepdims=True)


    def sigmoid(self,z):
        result = 1/(1+np.exp(-z))
        return result
    
    def sigmoid_d(self,z):
        result = self.sigmoid(z) * (1-self.sigmoid(z))
        return result

    def node_info(self):
        print('Imgs Shape: ',self.imgs.shape)
