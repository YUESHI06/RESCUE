import torch
from torch.utils.data import Dataset


class ClientDataset(Dataset):
    def __init__(self, graph_feature, pattern_feature, noise_labels, global_labels):
        super().__init__()
        self.graph_feature = graph_feature
        self.pattern_feature = pattern_feature
        self.noise_labels = noise_labels
        self.global_labels = global_labels

    def __len__(self):
        return len(self.graph_feature)
    
    def __getitem__(self, index):
        graph_item = self.graph_feature[index]
        pattern_item = self.pattern_feature[index]
        noise_label = self.noise_labels[index]
        global_label = self.global_labels[index]
        return graph_item, pattern_item, noise_label, global_label
    