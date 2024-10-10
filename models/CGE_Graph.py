import torch
import torch.nn as nn
import torch.nn.functional as F

class CGE_Graph(nn.Module):
    def __init__(self):
        super().__init__()
        # graph training
        self.inter_outputs = None
        self.con1 = nn.Conv1d(1, 200, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(1, 1)

        self.fc1 = nn.Linear(200*256, 10)
        self.fc2 = nn.Linear(10, 2)

        self.fc1.register_forward_hook(self.get_intermediate_outputs)


    def forward(self, graph_train, pattern_train):
        # graph_training
        graph_train = F.relu(self.con1(graph_train))
        graph_train = self.pool1(graph_train)

        # flatten
        graph_train = torch.flatten(graph_train, 1)

        # joint training
        joint_feature = F.relu(self.fc1(graph_train))
        predict = self.fc2(joint_feature)
        
        return predict
    
    # Get h(x) from main model
    def get_intermediate_outputs(self, module, input, output):
        self.inter_outputs = output.detach()