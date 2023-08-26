import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pair_confusion_matrix

class bp_neural:
    imgs = np.empty((0,0))
    labels = np.empty((0,0))

    def __init__(self,train_imgs,train_labels,n_hidden=100,method='softmax'):
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.n_hidden = n_hidden
        self.method = method
        self.o_train_labels = train_labels.copy()

    def normalize_images(self):
        self.imgs = self.train_imgs.astype('float32')
        self.imgs = self.imgs / 255.0
        self.imgs = self.imgs.reshape(self.imgs.shape[0],-1)

    def one_hot_labels(self):
        self.labels = self.train_labels
        self.labels = np.zeros((self.labels.shape[0],self.labels.max()+1))
        self.labels[np.arange(0,self.labels.shape[0]),self.train_labels] = 1


    def node_info(self):
        print('Imgs Shape: ',self.imgs.shape)
