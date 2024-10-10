import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, graph_feature, pattern_feature, labels):
        super().__init__()
        self.graph_feature = graph_feature
        self.pattern_feature = pattern_feature
        self.labels = labels

    def __len__(self):
        return len(self.graph_feature)
    
    def __getitem__(self, index):
        graph_item = self.graph_feature[index]
        pattern_item = self.pattern_feature[index]
        label = self.labels[index]
        return graph_item, pattern_item, label
