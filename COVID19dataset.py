import pandas as pd
import torch
from torch.utils.data import  Dataset,DataLoader
import pandas as pd
import  random
import numpy as np

class COVID19Dataset(Dataset):
    def __init__(self,mode="train"):
        super(COVID19Dataset, self).__init__()
        self.mode=mode
        self.features=[]
        self.target=[]

        if mode=="train" or mode=="dev":
            ratio=0.9
            train_data=pd.read_csv("data/covid.train.csv")
            features_=train_data.iloc[:,1:-1]
            target_=train_data.iloc[:,-1]
            if mode=="train":
                indices = random.sample(range(train_data.shape[0]), int(ratio * train_data.shape[0]))
                self.features=features_.loc[indices].values
                self.target=target_.loc[indices].values
            else:
                indices = random.sample(range(train_data.shape[0]), int((1-ratio) * train_data.shape[0]))
                self.features = features_.loc[indices].values
                self.target = target_.loc[indices].values
        elif mode=="test":
            train_data = pd.read_csv("data/covid.test.csv")
            self.features = train_data.iloc[:, 1:].values
            self.target=[-1.0]*len(self.features)
        self.features[:,40:]=  (self.features[:, 40:] - self.features[:, 40:].mean(axis=0)) / self.features[:, 40:].std(axis=0) #正则化，normalizatin，如果不加normalizatin很难训练，l
    def __getitem__(self, item):
            return torch.tensor(self.features[item]).to(torch.float32),torch.tensor(self.target[item])
    def __len__(self):
        return len(self.features)

if __name__ == '__main__':

    a=[[1.0,0],[2,2.0]]
    b=np.array(a)
    print(b.mean(axis=0))
    c=torch.tensor(a)
    print((c.mean(dim=0)))
    data=COVID19Dataset(mode="train")
    dl=DataLoader(data,batch_size=64,shuffle=True)
    for i ,(features,target) in enumerate(dl):
        print(target[0])
