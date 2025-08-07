import os.path
import random
import torch
from  torch.utils.data import Dataset
import numpy as np

def max_min_norm(x):
    max_ = np.max(x)
    min_ = np.min(x)
    return (x - min_)/(max_ - min_)

def mean_norm(x):
    mean = np.mean(x)
    std  = x - mean
    max_ = np.max(std) if -np.min(std) < np.max(std) else -np.min(std)
    return std / max_

def DropoutMin(x):
    min = np.min(x)
    return x - min

class Sequence(Dataset):
    def __init__(self,datas,labels, sq=400)->None:
        super(Sequence,self).__init__()
        self.datas  = datas
        self.labels = labels
        self.sq = sq

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.datas[index],dtype=np.float32).reshape(1,self.sq)),torch.from_numpy(np.array(self.labels[index],dtype=np.float32).reshape(1,1))



