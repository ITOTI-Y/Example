import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import v2

class Image_dataset(Dataset):
    transform = v2.Compose([
        v2.Resize((300,400),antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean=[0.5],std=[0.5]),
    ])

    def __init__(self,data_path,mode='train'):
        self.data_path = data_path
        self.mode = mode
        if self.mode == 'train':
            self.train_image_list = os.listdir(self.data_path + '/train_images')
            self.train_label = pd.read_csv(self.data_path + '/train.csv')
        elif self.mode == 'test':
            self.test_image_list = os.listdir(self.data_path + '/test_images')

    def __len__(self):
        return len(self.train_image_list)
    
    def __getitem__(self,idx):
        if self.mode == 'train':
            image_name = self.train_image_list[idx]
            image_path = self.data_path + '/train_images/' + image_name
            label = self.train_label.iloc[idx,1]
        elif self.mode == 'test':
            image_name = self.test_image_list[idx]
            image_path = self.data_path + '/test_images/' + image_name
            label = None
        else:
            raise Exception('mode must be train or test')
        image = Image.open(image_path)
        image = self.transform(image)
        return image,label