import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from itertools import islice

class cnn():

    def __init__(self,image_path='./data'):
        self.image_path = image_path

    def dowmload_data(self):
        self.train_data = torchvision.datasets.MNIST(self.image_path,train=True,download=True)
        self.test_data = torchvision.datasets.MNIST(self.image_path,train=False,download=True)

    def show_data(self):
        fig = plt.figure(figsize=(15,6))
        for i,(image,label) in islice(enumerate(self.train_data),10): #type: ignore
            ax = fig.add_subplot(2,5,i+1)
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'Label:{label}')
            ax.imshow(image)
