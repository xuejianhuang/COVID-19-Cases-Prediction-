import  torch
from torch import nn
from torch.nn import  Sequential
from torch.utils.data import  DataLoader

class BaseNNet(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BaseNNet, self).__init__()
        self.net=Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )

    def forward(self,x):
        return self.net(x)