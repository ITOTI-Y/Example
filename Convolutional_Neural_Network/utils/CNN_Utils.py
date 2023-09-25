import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


class DogDataset(Dataset):

    def __init__(self,data_dir,label_list,transform=transforms.Compose([])):
        super().__init__()
        self.data_dir = data_dir
        self.label_list = label_list
        self.transform = transform

        self.image_paths = []
        for label in self.label_list:
            label_dir = os.path.join(self.data_dir, label)
            self.image_paths.extend([os.path.join(label_dir,image) for image in os.listdir(label_dir)])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        label = self.label_list.index(os.path.split(os.path.dirname(image_path))[1])
        return image,label
    
class CNN(torch.nn.Module):
    feature_maps_id = {}
    feature_maps = {}

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,16,3,padding=1)
        self.conv2 = torch.nn.Conv2d(16,32,3,padding=1)
        self.fc1 = torch.nn.Linear(32*56*56,6)
        self.conv1.register_forward_hook(self.hook_fn) # 创建conv1的钩子函数
        self.conv2.register_forward_hook(self.hook_fn) # 创建conv2的钩子函数

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.softmax(x,dim=1)
        return x
    
    def hook_fn(self,m:torch.nn.Module,i,o):
        layer_id = id(m)
        self.feature_maps_id[layer_id] = o

    def show_feature_map(self,images:torch.Tensor): # 可视化特征图
        images = images.detach()
        size = images.size(0)
        fig,axes = plt.subplots(size//4,4,figsize=(20,20))

        for i,image in enumerate(images):
            row = i//4
            col = i%4

            ax = axes[row,col]
            ax.imshow(image,cmap = 'gray')
            ax.axis('off')
        
        plt.show()

    def get_feature_maps(self,layer_name:str):
        try:
            layer_id = self.name2id()[f'{layer_name}']
            return self.feature_maps_id[layer_id].detach()
        except KeyError:
            print(f'No such layer named {layer_name}')
        return torch.tensor([])
    
    def name2id(self):
        module_id = {}
        for name,module in self.named_modules():
            module_id[name] = id(module)
        return module_id