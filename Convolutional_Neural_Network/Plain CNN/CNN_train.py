import torch
import os
from torchsummary import summary
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from utils.CNN_Utils import DogDataset,CNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.RandomRotation(10), # 随机旋转
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 正则化,
])

label_name_list = os.listdir('Data/Dog Breed Image Classification Dataset')
dog_dataset = DogDataset('Data/Dog Breed Image Classification Dataset', label_name_list,transform=data_transform)
train_dataset,val_dataset,test_dataset = random_split(dog_dataset,[0.8,0.1,0.1])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 模型训练
model = CNN()
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

# 保存模型
torch.save(model.state_dict(),'Model/Plain CNN.pth')