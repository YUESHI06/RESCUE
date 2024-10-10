import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CGE_Graph import CGE_Graph
from trainers.CGE_client import CGE_client
from trainers.server import Server
from data_processing.dataloader_manager import gen_client_dataloader, gen_test_dataloader
from models.RCELoss import RCELoss
from options import parse_args
from CGE_test import CGE_test


if __name__ == '__main__':
    args = parse_args()
    criterion = nn.CrossEntropyLoss()
    re_criterion = RCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datloader_dict = dict()
    datloader_dict['train'] = list() 
    for i in range(args.client_num):
        train_dl = gen_client_dataloader(i, args.vul, args.noise_type, args.noise_rate)
        datloader_dict['train'].append(train_dl)
    
    global_model = CGE_Graph()
    global_model = global_model.to(device)
    server = Server(
        args,
        global_model,
        device,
        criterion
    )
    
    client_list = list()
    for i in range(args.client_num):
        model = CGE_Graph()
        model = model.to(device)
        client = CGE_client(
            args,
            criterion,
            re_criterion,
            device,
            model,
            datloader_dict['train'][i],
            i
        )
        client_list.append(client)

    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)

        for client_id in range(args.client_num):
            client = client_list[client_id]
            client.model = copy.deepcopy(server.global_model)
            client.RCE_train()
            server.save_train_updates(
                    copy.deepcopy(client.get_parameters()),
                    client.result['sample'],
                    client.result
            )
            print(f"client:{client_id}")
            client.print_loss()
    
        server.average_weights()
        print(epoch)

test_dl = gen_test_dataloader(args.vul)
CGE_test(server.global_model, test_dl, criterion, device, args, 'Fed_CGE_Graph')

    