import numpy as np

def normalize_images(images):
    images = images.astype('float32')
    images = images / 255.0
    images = images.reshape(images.shape[0],-1)
    return images

def one_hot(labels):
    result = np.zeros((labels.size,labels.max()+1))
    result[np.arange(labels.size),labels] = 1
    return result

def initialize(tr_imgs,tr_lables,n_hidden=100):
    n_input = tr_imgs.shape[0]
    n_output = tr_lables.shape[0]
    n_hidden = n_hidden
    W1 = np.random.randn(n_input,n_hidden)
    b1 = np.zeros((n_hidden,1))
    W2 = np.random.randn(n_hidden,n_output)
    b2 = np.zeros((n_output,1))
    return W1,b1,W2,b2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_d(z):
    return sigmoid(z)*(1-sigmoid(z))

class bp_neural():
    n_input = None
    n_output = None
    normalize = None
    onehot = None

    def __init__(self,train_imgs,train_labels,n_hidden=100):
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.n_hidden = n_hidden
        self.node_info()

    def normalize_images(self,inplace=False):
        if self.normalize:
            print('Already normalized')
            self.node_info()
            return None
        tr_imgs = self.train_imgs.astype('float32')
        tr_imgs = tr_imgs / 255.0
        tr_imgs = tr_imgs.reshape(tr_imgs.shape[0],-1)
        self.n_input = tr_imgs.shape[1]
        if inplace:
            self.normalize = True
            self.train_imgs = tr_imgs.T
            self.node_info()
            return None
        self.node_info()
        return tr_imgs.T
    
    def one_hot(self,inplace=False):
        if self.onehot:
            print('Already one-hot')
            self.node_info()
            return None
        tr_labels = self.train_labels
        result = np.zeros((tr_labels.size,tr_labels.max()+1))
        result[np.arange(tr_labels.size),tr_labels] = 1
        self.n_output = result.shape[1]
        if inplace:
            self.onehot = True
            self.train_labels = result.T
            self.node_info()
            return None
        self.node_info()
        return result.T
    
    def init_params(self):
        self.w1 = np.random.randn(self.n_input,self.n_hidden)
        self.b1 = np.zeros((self.n_hidden,1))
        self.w2 = np.random.randn(self.n_hidden,self.n_output)
        self.b2 = np.zeros((self.n_output,1))

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def sigmoid_d(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def front_prop(self,output = False):
        self.z1 = np.dot(self.w1.T,self.train_imgs) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.w2.T,self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        if output:
            return self.z1,self.a1,self.z2,self.a2
        
    def back_prop(self):
        self.batch = self.train_labels.shape[1]
        self.dz2 = -self.train_labels*(1-self.a2)
        pass

    def predict(self,imgs):
        p_z1 = np.dot(self.w1.T,imgs) + self.b1
        p_a1 = self.sigmoid(p_z1)
        p_z2 = np.dot(self.w2.T,p_a1) + self.b2
        p_a2 = self.sigmoid(p_z2)
        result = np.argmax(p_a2,axis=0)
        return result
    
    def node_info(self):
        print(f'Train_images shape: {self.train_imgs.shape}')
        print(f'Train_lables shape: {self.train_labels.shape}')
        print(f'Input layer nodes: {self.n_input}')
        print(f'Hidden layer nodes: {self.n_hidden}')
        print(f'Output layer nodes: {self.n_output}')