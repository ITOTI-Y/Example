import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class DogDataset(Dataset):

    def __init__(self,data_dir,label_list):
        super(DogDataset,self).__init__()
        self.data_dir = data_dir
        self.label_list = label_list

        self.image_paths = []
        for label in self.label_list:
            label_dir = os.path.join(self.data_dir, label)
            self.image_paths.extend([os.path.join(label_dir,image) for image in os.listdir(label_dir)])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = transforms.ToTensor()(image)
        label = self.label_list.index(os.path.split(os.path.dirname(image_path))[1])
        return image,label