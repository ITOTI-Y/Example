import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

class ConvBlock(nn.Module):

    def __init__(self,in_channels:int,out_channels:int,kernel_size:int = 3 ,stride:int = 1, padding:int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int = 3 ,stride:int = 1, padding:int = 1,num:int = 3):
        super().__init__()
        self.num = num
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self,x):
        x = self.conv1(x)
        for i in range(self.num):
            x = F.relu(self.conv(x) + x)
        return x
    

class Output(nn.Module):

    def __init__(self,in_channels:int,out_channels:int,kernel_size:int = 3 ,stride:int = 1, padding:int = 1):
        super().__init__()
        self.linear = nn.Linear(in_channels,out_channels)
        self.out = nn.Sequential(
            self.linear,
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.out(x)



class Resnet(nn.Module):

    def __init__(self,num_classes:int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.inc = ConvBlock(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.resblock1 = ResnetBlock(in_channels=8,out_channels=16,num=2)
        self.resblock2 = ResnetBlock(in_channels=16,out_channels=32,num=3)
        self.resblock3 = ResnetBlock(in_channels=32,out_channels=64,num=5)
        self.resblock4 = ResnetBlock(in_channels=64,out_channels=128,num=2)
        self.avgpool = nn.AvgPool2d(7,1,0)
        self.out = None

    def forward(self,x):
        x = self.inc(x)
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        if self.out is None:
            self.out = Output(x.shape[1],self.num_classes).to(x.device)
        x = self.out(x)
        return x
    
    def summary(self,input:list[int] = [4,3,600,800]):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device=device)  # type: ignore
        input_size = torch.tensor(input,device=device)
        print(torchinfo.summary(self,input_size=input,device=device))