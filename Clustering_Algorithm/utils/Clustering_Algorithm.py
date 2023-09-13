import torch
import plotly.express as px 
import pandas as pd

class Cluster:
    process_centers = torch.empty(0)
    process_node = torch.empty(0)

    def __init__(self,data,k,max_iters,tol=1e-4):
        self.data = data
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.data_tensor = torch.tensor(data.values)
        self.lenght = self.data_tensor.shape[0]
        self.init_centers()
    
    def init_centers(self):
        random_index = torch.randperm(self.data_tensor.shape[0])
        random_index = random_index[:self.k]
        self.centers = self.data_tensor[random_index]

    def cal_distance(self,x1,x2):
        result = torch.sqrt(torch.sum((x1-x2)**2,dim=1))
        return result
    
    def distances(self):
        centers = self.centers
        temp = []
        for i in centers:
            temp.append(self.cal_distance(self.data_tensor,i))
        result = torch.stack(temp,dim=0)
        return result
    
    def clu(self,):
        result = torch.argmin(self.distances(),dim=0)
        return result

    def new_centers(self):
        clu = self.clu()
        temp = []
        for i in range(self.k):
            temp.append(self.data_tensor[clu == i])
        for j in range(self.k):
            temp[j] = torch.mean(temp[j],dim=0)
        result = torch.stack(temp,dim=0)

        return result
    
    def train(self,process = None):
        for i in range(self.max_iters):
            new_centers = self.new_centers()
            if process:
                self.process(i)
            if torch.all(torch.abs(new_centers-self.centers) < self.tol):
                break
            self.centers = new_centers

    def process(self,index):
        temp_node = torch.cat([torch.ones(self.lenght,1)*index,self.clu().reshape(-1,1),self.data_tensor],dim=1)
        self.process_node = torch.cat([self.process_node,temp_node],dim=0)
        temp_centers = torch.cat([torch.ones(self.k,1)*index,self.centers],dim=1)
        self.process_centers = torch.cat([self.process_centers,temp_centers],dim=0)

    def process_plot(self):
        pd_node = pd.DataFrame(self.process_node[:,:5],columns=['index','category','x','y','z']) # type:ignore
        pd_node.sort_values(by=['index','category'],inplace=True)
        pd_node['category'] = pd_node['category'].astype('str')
        fig = px.scatter_3d(pd_node,x='x',y='y',z='z',animation_frame='index',
                            animation_group='category',color='category',width=600,height=600,opacity=0.8
                            )
        fig.show()