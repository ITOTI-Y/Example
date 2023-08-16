import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class bp_neural():
    """
    该类实现了一个简单的BP神经网络，包括前向传播、反向传播、参数更新、预测、准确率计算、混淆矩阵绘制等功能。
    """
    n_input = 0
    n_output = 0
    normalize = None
    onehot = None
    frontprop = None
    initparams = None
    backprop = None
    a1,a2,z1,z2 = np.empty((4,0))

    def __init__(self,train_imgs,train_labels,n_hidden=100,method = 'softmax'):
        """
        初始化神经网络

        Args:
        train_imgs: numpy.ndarray, 训练图像数据，形状为 (n_samples, n_features)
        train_labels: numpy.ndarray, 训练标签数据，形状为 (n_samples,)
        n_hidden: int, 隐藏层节点数，默认为 100
        method: str, 激活函数类型，可选 'sigmoid' 或 'softmax'，默认为 'softmax'
        """
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.n_hidden = n_hidden
        self.o_train_labels = train_labels.copy()
        self.act_function = method

    def normalize_images(self,inplace=False):
        """
        对图像数据进行归一化处理

        Args:
        inplace: bool, 是否在原数据上进行操作，默认为 False

        Returns:
        numpy.ndarray, 归一化后的图像数据，形状为 (n_features, n_samples)
        """
        if self.normalize:
            print('Already normalized')
            return None
        tr_imgs = self.train_imgs.astype('float32')
        tr_imgs = tr_imgs / 255.0
        tr_imgs = tr_imgs.reshape(tr_imgs.shape[0],-1)
        self.n_input = tr_imgs.shape[1]
        if inplace:
            self.normalize = True
            self.train_imgs = tr_imgs.T
            return None
        return tr_imgs.T
    
    def one_hot(self,inplace=False):
        """
        对标签数据进行 one-hot 编码

        Args:
        inplace: bool, 是否在原数据上进行操作，默认为 False

        Returns:
        numpy.ndarray, one-hot 编码后的标签数据，形状为 (n_classes, n_samples)
        """
        if self.onehot:
            print('Already one-hot')
            return None
        tr_labels = self.train_labels
        result = np.zeros((tr_labels.size,tr_labels.max()+1))
        result[np.arange(tr_labels.size),tr_labels] = 1
        self.n_output = result.shape[1]
        if inplace:
            self.onehot = True
            self.train_labels = result.T
            return None
        return result.T
    
    def init_params(self):
        """
        初始化神经网络参数
        """
        if self.initparams:
            print('Already init params')
            return None
        self.w1 = np.random.randn(self.n_input,self.n_hidden)
        self.b1 = np.zeros((self.n_hidden,1))
        self.w2 = np.random.randn(self.n_hidden,self.n_output)
        self.b2 = np.zeros((self.n_output,1))
        self.initparams = True

    def sigmoid(self,z):
        """
        sigmoid 激活函数

        Args:
        z: numpy.ndarray, 输入数据

        Returns:
        numpy.ndarray, 经过 sigmoid 函数处理后的数据
        """
        return 1/(1+np.exp(-z))
    
    def sigmoid_d(self,z):
        """
        sigmoid 函数的导数

        Args:
        z: numpy.ndarray, 输入数据

        Returns:
        numpy.ndarray, sigmoid 函数的导数
        """
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def front_prop(self,output = False):
        """
        前向传播

        Args:
        output: bool, 是否返回前向传播的结果，默认为 False

        Returns:
        tuple, 前向传播的结果，包括 z1, a1, z2, a2 四个 numpy.ndarray
        """
        if self.frontprop:
            print('Already frontprop')
            return None
        self.z1 = np.dot(self.w1.T,self.train_imgs) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.w2.T,self.a1) + self.b2

        if self.act_function == 'sigmoid':
            self.a2 = self.sigmoid(self.z2)
        if self.act_function == 'softmax':
            self.a2 = self.softmax(self.z2)

        self.frontprop = True
        self.backprop = None
        if output:
            return self.z1,self.a1,self.z2,self.a2
        
    def back_prop(self,output=False):
        """
        反向传播

        Args:
        output: bool, 是否返回反向传播的结果，默认为 False

        Returns:
        tuple, 反向传播的结果，包括 dw1, db1, dw2, db2 四个 numpy.ndarray
        """
        if self.backprop:
            print('Already backprop')
            return None
        batch = self.train_imgs.shape[1]
        if self.act_function == 'sigmoid':
            self.dz2 = -self.train_labels/self.a2 * self.sigmoid_d(self.z2)
        if self.act_function == 'softmax':
            self.dz2 = self.a2 - self.train_labels

        self.dw2 = 1/batch * np.dot(self.dz2,self.a1.T).T
        self.db2 = 1/batch * np.sum(self.dz2,axis=1,keepdims=True)
        self.da1 = np.dot(self.dz2.T,self.w2.T).T
        self.dz1 = self.da1 * self.sigmoid_d(self.z1)
        self.dw1 = 1/batch * np.dot(self.dz1,self.train_imgs.T).T
        self.db1 = 1/batch * np.sum(self.dz1,axis=1,keepdims=True)
        
        self.backprop = True
        if output:
            return self.dw1,self.db1,self.dw2,self.db2
        
    def update_params(self,lr:float=1):
        """
        更新神经网络参数

        Args:
        lr: float, 学习率，默认为 1
        """
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2
        self.frontprop = None

    def predict(self,imgs):
        """
        预测

        Args:
        imgs: numpy.ndarray, 待预测的图像数据，形状为 (n_samples, n_features)

        Returns:
        numpy.ndarray, 预测结果，形状为 (n_samples,)
        """
        p_z1 = np.dot(self.w1.T,imgs) + self.b1
        p_a1 = self.sigmoid(p_z1)
        p_z2 = np.dot(self.w2.T,p_a1) + self.b2
        if self.act_function == 'sigmoid':
            p_a2 = self.sigmoid(p_z2)
        else:
            p_a2 = self.softmax(p_z2)
        result = np.argmax(p_a2,axis=0)
        return result
    
    def softmax(self,z):
        """
        softmax 激活函数

        Args:
        z: numpy.ndarray, 输入数据

        Returns:
        numpy.ndarray, 经过 softmax 函数处理后的数据
        """
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z,axis=0,keepdims=True)
        result = exp_z / sum_exp_z
        return result
    
    def accuracy(self):
        """
        计算准确率

        Returns:
        float, 准确率
        """
        result = np.mean(self.predict(self.train_imgs) == self.o_train_labels)
        return result
    
    def plot(self):
        """
        绘制混淆矩阵
        """
        train_preds = self.predict(self.train_imgs)
        cm = confusion_matrix(train_preds,self.o_train_labels,)
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False)
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.title('Confusion Matrix')
        plt.show()
    
    def node_info(self):
        """
        输出神经网络节点信息
        """
        print(f'Train_images shape: {self.train_imgs.shape}')
        print(f'Train_lables shape: {self.train_labels.shape}')
        print(f'Input layer nodes: {self.n_input}')
        print(f'Hidden layer nodes: {self.n_hidden}')
        print(f'Output layer nodes: {self.n_output}')
        print(f'Activation function: {self.act_function}')
        print(f'A1 shape: {self.a1.shape}')
        print(f'A2 shape: {self.a2.shape}')
        print(f'Z1 shape: {self.z1.shape}')
        print(f'Z2 shape: {self.z2.shape}')