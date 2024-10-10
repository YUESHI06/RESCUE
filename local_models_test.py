import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from models.CGE_Variants import CGEVariant
from models.LCN import LCN
from trainers.fed_avg_trainer import ClientFedAvg
from data_processing.preprocessing import get_graph_feature, get_pattern_feature
from data_processing.CustomDataset import CustomDataset
from options import parse_args
from CGE_test import CGE_test

# 这份代码需要修改的地方很多

if __name__ == "__main__":
    args = parse_args()
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pure_graph_train, pure_graph_test, pure_labels_train, pure_labels_test,_ = get_graph_feature(args.vul, noise_valid=False)
    pure_pattern_train, pure_pattern_test, _, _ =  get_pattern_feature(args.vul)
    noise_graph_train, noise_graph_test, noise_labels_train, noise_labels_test,_ = get_graph_feature(args.vul, noise_valid=True)
    noise_pattern_train, noise_pattern_test, _, _ = get_pattern_feature(args.vul)
    pure_train_ds = CustomDataset(pure_graph_train, pure_pattern_train, pure_labels_train)
    noise_train_ds = CustomDataset(noise_graph_train, noise_pattern_train, noise_labels_train)
    pure_train_dl = DataLoader(pure_train_ds, batch_size=args.batch, shuffle=True)
    noise_train_dl = DataLoader(noise_train_ds,batch_size=args.batch, shuffle=False)
    test_ds = CustomDataset(pure_graph_test, pure_pattern_test,pure_labels_test)
    test_dl = DataLoader(test_ds, shuffle=False)

    global_model = CGEVariant()
    inner_model = CGEVariant()
    outer_model = LCN(in_channels=args.input_channels)
    global_model, inner_model, outer_model = global_model.to(device), inner_model.to(device), outer_model.to(device)
    client = ClientFedAvg(args, criterion, device, inner_model, outer_model, noise_train_dl, pure_train_dl)

    for epoch in range(args.epoch):
        global_labels = []
        print(epoch)
        for graph_data, pattern_data, _ in noise_train_dl:
            graph_data, pattern_data = graph_data.to(device), pattern_data.to(device)
            gl_outputs = global_model(graph_data, pattern_data)
            gl_outputs = torch.sigmoid(gl_outputs)
            gl_label = gl_outputs.round().float()
            global_labels.append(gl_label)
        conc_labels = torch.cat(global_labels, dim=0)
        client.meta_train(conc_labels)
        global_model.load_state_dict(client.inner_model.state_dict())
    
    CGE_test(global_model, test_dl, criterion, device)
    
    


        


