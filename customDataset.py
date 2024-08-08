
import pandas as pd
from torch.utils.data import Dataset
import torch
import ast


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.trace = self.data['trace'].values
        self.label = self.data['label'].values
        self.evelabel = self.data['evelabel']


        self.label[self.label=='normal']=0
        self.label[self.label=='anomalous']=1


        
    def __len__(self):
        return len(self.data)
    
    def get_batch(self, indexes):
        return self.trace[indexes], self.label[indexes]
    
    def get_label(self):

        return self.label, self.evelabel
