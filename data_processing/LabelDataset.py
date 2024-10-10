import torch
from torch.utils.data import Dataset

class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.labels[index]